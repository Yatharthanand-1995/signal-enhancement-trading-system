"""
Enhanced Database Management with Connection Pooling and Performance Monitoring
"""
import asyncio
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Generator
from datetime import datetime, timedelta
import psycopg2
from psycopg2 import pool, extras
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool
import pandas as pd

from config.enhanced_config import enhanced_config
from src.utils.error_handling import DatabaseError, ErrorSeverity, ErrorCategory, handle_errors, safe_execute
from src.utils.logging_setup import get_logger, perf_logger
from src.utils.validation import input_validator, ValidationRule, ValidationType

logger = get_logger(__name__)

class ConnectionPool:
    """Enhanced PostgreSQL connection pool with monitoring"""
    
    def __init__(self):
        self.pool = None
        self.engine = None
        self.session_factory = None
        self.connection_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'failed_connections': 0,
            'query_count': 0,
            'average_query_time': 0.0,
            'slow_query_count': 0
        }
        self.slow_query_threshold = 2.0  # seconds
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool and SQLAlchemy engine"""
        try:
            db_config = enhanced_config.db
            
            # Create psycopg2 connection pool
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=db_config.max_connections,
                host=db_config.host,
                port=db_config.port,
                database=db_config.database,
                user=db_config.user,
                password=db_config.password,
                cursor_factory=extras.RealDictCursor
            )
            
            # Create SQLAlchemy engine with connection pooling
            self.engine = create_engine(
                db_config.connection_string,
                poolclass=QueuePool,
                pool_size=db_config.pool_size,
                max_overflow=db_config.max_connections - db_config.pool_size,
                pool_recycle=db_config.pool_recycle,
                pool_pre_ping=True,  # Verify connections before use
                echo=enhanced_config.logging.log_sql_queries
            )
            
            # Create session factory
            self.session_factory = sessionmaker(bind=self.engine)
            
            logger.info(
                "Database connection pool initialized",
                component='database',
                pool_size=db_config.pool_size,
                max_connections=db_config.max_connections
            )
            
        except Exception as e:
            logger.error("Failed to initialize database connection pool", exception=e)
            raise DatabaseError(
                "Failed to initialize database connection pool",
                severity=ErrorSeverity.CRITICAL,
                details={'error': str(e)},
                original_error=e
            )
    
    @contextmanager
    def get_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """Get a database connection from the pool"""
        connection = None
        start_time = time.time()
        
        try:
            connection = self.pool.getconn()
            self.connection_stats['total_connections'] += 1
            self.connection_stats['active_connections'] += 1
            
            logger.debug("Database connection acquired", component='database')
            yield connection
            
        except Exception as e:
            self.connection_stats['failed_connections'] += 1
            logger.error("Database connection error", exception=e, component='database')
            if connection:
                connection.rollback()
            raise DatabaseError(
                "Database connection failed",
                severity=ErrorSeverity.HIGH,
                details={'error': str(e)},
                original_error=e
            )
        finally:
            if connection:
                try:
                    connection.commit()
                    self.pool.putconn(connection)
                    self.connection_stats['active_connections'] -= 1
                    
                    duration = time.time() - start_time
                    perf_logger.log_execution_time('database_connection', duration)
                    
                    logger.debug(
                        "Database connection released",
                        component='database',
                        duration_seconds=duration
                    )
                except Exception as e:
                    logger.error("Error releasing database connection", exception=e)
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a SQLAlchemy session"""
        session = self.session_factory()
        start_time = time.time()
        
        try:
            logger.debug("SQLAlchemy session created", component='database')
            yield session
            session.commit()
            
        except Exception as e:
            session.rollback()
            logger.error("Database session error", exception=e, component='database')
            raise DatabaseError(
                "Database session failed",
                severity=ErrorSeverity.HIGH,
                details={'error': str(e)},
                original_error=e
            )
        finally:
            try:
                session.close()
                duration = time.time() - start_time
                perf_logger.log_execution_time('database_session', duration)
                
                logger.debug(
                    "SQLAlchemy session closed",
                    component='database',
                    duration_seconds=duration
                )
            except Exception as e:
                logger.error("Error closing database session", exception=e)
    
    def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        fetch_results: bool = True
    ) -> Optional[List[Dict[str, Any]]]:
        """Execute a SQL query with performance monitoring"""
        start_time = time.time()
        query_hash = hash(query)
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Log query start
                    logger.debug(
                        "Executing SQL query",
                        component='database',
                        query_hash=query_hash,
                        query=query[:100] + "..." if len(query) > 100 else query
                    )
                    
                    # Execute query
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)
                    
                    results = None
                    if fetch_results:
                        results = cursor.fetchall()
                        # Convert to list of dicts for easier handling
                        if results:
                            results = [dict(row) for row in results]
                    
                    # Update statistics
                    duration = time.time() - start_time
                    self._update_query_stats(duration)
                    
                    # Log slow queries
                    if duration > self.slow_query_threshold:
                        self.connection_stats['slow_query_count'] += 1
                        logger.warning(
                            f"Slow query detected: {duration:.3f}s",
                            component='database',
                            query_hash=query_hash,
                            duration_seconds=duration,
                            query=query[:200] + "..." if len(query) > 200 else query
                        )
                    
                    perf_logger.log_database_query('SELECT' if query.strip().upper().startswith('SELECT') else 'OTHER', duration, len(results) if results else 0)
                    
                    logger.debug(
                        f"Query completed in {duration:.3f}s",
                        component='database',
                        query_hash=query_hash,
                        duration_seconds=duration,
                        row_count=len(results) if results else 0
                    )
                    
                    return results
                    
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Query failed after {duration:.3f}s: {str(e)[:200]}",
                exception=e,
                component='database',
                query_hash=query_hash,
                duration_seconds=duration
            )
            raise DatabaseError(
                f"Query execution failed: {str(e)[:100]}",
                severity=ErrorSeverity.HIGH,
                details={
                    'query_hash': query_hash,
                    'duration': duration,
                    'error': str(e)
                },
                original_error=e
            )
    
    def execute_dataframe_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute query and return results as pandas DataFrame"""
        try:
            with self.get_session() as session:
                if params:
                    df = pd.read_sql(text(query), session.bind, params=params)
                else:
                    df = pd.read_sql(text(query), session.bind)
                
                logger.info(
                    f"DataFrame query returned {len(df)} rows",
                    component='database',
                    row_count=len(df),
                    columns=len(df.columns)
                )
                
                return df
                
        except Exception as e:
            logger.error("DataFrame query failed", exception=e, component='database')
            raise DatabaseError(
                f"DataFrame query failed: {str(e)[:100]}",
                severity=ErrorSeverity.HIGH,
                original_error=e
            )
    
    def bulk_insert(self, table_name: str, data: List[Dict[str, Any]], batch_size: int = 1000):
        """Bulk insert data with batching for performance"""
        if not data:
            return
            
        start_time = time.time()
        total_rows = len(data)
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Get column names from first row
                    columns = list(data[0].keys())
                    placeholders = ', '.join(['%s'] * len(columns))
                    insert_query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                    
                    # Process in batches
                    for i in range(0, total_rows, batch_size):
                        batch = data[i:i + batch_size]
                        batch_data = [tuple(row[col] for col in columns) for row in batch]
                        
                        cursor.executemany(insert_query, batch_data)
                        
                        logger.debug(
                            f"Inserted batch {i//batch_size + 1}: {len(batch)} rows",
                            component='database',
                            table=table_name,
                            batch_size=len(batch)
                        )
                    
                    duration = time.time() - start_time
                    perf_logger.log_database_query('BULK_INSERT', duration, total_rows)
                    
                    logger.info(
                        f"Bulk insert completed: {total_rows} rows in {duration:.3f}s",
                        component='database',
                        table=table_name,
                        total_rows=total_rows,
                        duration_seconds=duration,
                        rows_per_second=total_rows / duration if duration > 0 else 0
                    )
                    
        except Exception as e:
            logger.error("Bulk insert failed", exception=e, component='database', table=table_name)
            raise DatabaseError(
                f"Bulk insert failed for table {table_name}",
                severity=ErrorSeverity.HIGH,
                details={'table': table_name, 'rows': total_rows},
                original_error=e
            )
    
    def _update_query_stats(self, duration: float):
        """Update query performance statistics"""
        self.connection_stats['query_count'] += 1
        
        # Update rolling average
        current_avg = self.connection_stats['average_query_time']
        query_count = self.connection_stats['query_count']
        new_avg = ((current_avg * (query_count - 1)) + duration) / query_count
        self.connection_stats['average_query_time'] = round(new_avg, 4)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            **self.connection_stats,
            'pool_size': enhanced_config.db.pool_size,
            'max_connections': enhanced_config.db.max_connections,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        try:
            start_time = time.time()
            result = self.execute_query("SELECT 1 as health_check")
            duration = time.time() - start_time
            
            health_status = {
                'healthy': True,
                'response_time': round(duration, 4),
                'connection_pool_active': True,
                'stats': self.get_stats()
            }
            
            logger.debug("Database health check passed", component='database', **health_status)
            return health_status
            
        except Exception as e:
            health_status = {
                'healthy': False,
                'error': str(e),
                'connection_pool_active': False,
                'stats': self.get_stats()
            }
            
            logger.error("Database health check failed", exception=e, component='database')
            return health_status
    
    def close(self):
        """Close all connections and cleanup"""
        try:
            if self.pool:
                self.pool.closeall()
            if self.engine:
                self.engine.dispose()
            logger.info("Database connection pool closed", component='database')
        except Exception as e:
            logger.error("Error closing database pool", exception=e)

class DatabaseManager:
    """High-level database management interface"""
    
    def __init__(self):
        self.pool = ConnectionPool()
    
    @handle_errors(category=ErrorCategory.DATABASE, severity=ErrorSeverity.HIGH)
    def get_stock_prices(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Get stock price data with validation"""
        # Validate inputs
        validated_symbol = input_validator.validate_field(
            symbol, 
            ValidationRule(ValidationType.SYMBOL, required=True),
            'symbol'
        )
        
        # Build query
        query = """
        SELECT dp.trade_date, dp.open, dp.high, dp.low, dp.close, dp.adj_close, dp.volume, dp.dollar_volume
        FROM daily_prices dp
        JOIN securities s ON dp.symbol_id = s.id
        WHERE s.symbol = %(symbol)s
        """
        
        params = {'symbol': validated_symbol}
        
        if start_date:
            query += " AND dp.trade_date >= %(start_date)s"
            params['start_date'] = start_date
        
        if end_date:
            query += " AND dp.trade_date <= %(end_date)s"
            params['end_date'] = end_date
        
        query += " ORDER BY dp.trade_date DESC"
        
        if limit:
            query += " LIMIT %(limit)s"
            params['limit'] = limit
        
        return self.pool.execute_dataframe_query(query, params)
    
    @handle_errors(category=ErrorCategory.DATABASE, severity=ErrorSeverity.HIGH)
    def store_trading_signals(self, signals: List[Dict[str, Any]]) -> bool:
        """Store trading signals with validation"""
        if not signals:
            return True
        
        # Validate each signal
        validated_signals = []
        for signal in signals:
            try:
                validated_signal = input_validator.validate_dict(signal, {
                    'symbol': ValidationRule(ValidationType.SYMBOL, required=True),
                    'signal_type': ValidationRule(ValidationType.STRING, required=True),
                    'direction': ValidationRule(ValidationType.STRING, required=True, allowed_values=['BUY', 'SELL', 'HOLD']),
                    'strength': ValidationRule(ValidationType.NUMERIC, required=True, min_value=0, max_value=1),
                    'price_at_signal': ValidationRule(ValidationType.PRICE, required=True),
                    'confidence': ValidationRule(ValidationType.NUMERIC, required=True, min_value=0, max_value=1)
                })
                
                # Add timestamp if not present
                if 'signal_date' not in validated_signal:
                    validated_signal['signal_date'] = datetime.utcnow()
                
                validated_signals.append(validated_signal)
                
            except ValidationError as e:
                logger.warning(f"Invalid signal data: {e}")
                continue
        
        if validated_signals:
            self.pool.bulk_insert('trading_signals', validated_signals)
            logger.info(f"Stored {len(validated_signals)} trading signals", component='database')
        
        return True
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get database health status"""
        return self.pool.health_check()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get database performance statistics"""
        return self.pool.get_stats()
    
    def close(self):
        """Close database connections"""
        self.pool.close()

# Global database manager instance
db_manager = DatabaseManager()

# Convenience functions
def get_db_connection():
    """Get database connection context manager"""
    return db_manager.pool.get_connection()

def get_db_session():
    """Get SQLAlchemy session context manager"""
    return db_manager.pool.get_session()

def execute_query(query: str, params: Optional[Dict[str, Any]] = None, fetch_results: bool = True):
    """Execute query using global database manager"""
    return db_manager.pool.execute_query(query, params, fetch_results)