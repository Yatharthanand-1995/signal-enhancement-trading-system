"""
Warm Data Manager - PostgreSQL-based warm data storage
Manages historical indicators, backtesting data, and ML features
Part of the tiered data storage architecture
"""
import psycopg2
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager
from dataclasses import dataclass
import json
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

from config.config import config

logger = logging.getLogger(__name__)

@dataclass
class WarmDataConfig:
    """Configuration for warm data management"""
    technical_indicators_retention: int = 90    # days
    historical_data_retention: int = 730        # 2 years
    ml_features_retention: int = 180            # 6 months
    backtest_results_retention: int = 365       # 1 year
    batch_size: int = 1000
    connection_timeout: int = 30
    max_connections: int = 20

class WarmDataManager:
    """
    Manages warm data in PostgreSQL for analytical and historical access
    Implements the 'Warm' tier of the data storage architecture
    
    Data Types:
    - Technical indicators: 90-day retention
    - Historical OHLCV: 2-year retention 
    - ML features: 6-month retention
    - Backtest results: 1-year retention
    """
    
    def __init__(self, db_config: Optional[Dict] = None, warm_config: Optional[WarmDataConfig] = None):
        self.db_config = db_config or {
            'host': config.db.host,
            'port': config.db.port,
            'database': config.db.database,
            'user': config.db.user,
            'password': config.db.password
        }
        
        self.config = warm_config or WarmDataConfig()
        self.connection_string = self._build_connection_string()
        
        # SQLAlchemy engine with connection pooling
        self.engine = None
        self._initialize_engine()
        
        # Table schemas
        self.table_schemas = {
            'technical_indicators': self._get_technical_indicators_schema(),
            'historical_ohlcv': self._get_historical_ohlcv_schema(),
            'ml_features': self._get_ml_features_schema(),
            'backtest_results': self._get_backtest_results_schema(),
            'signal_history': self._get_signal_history_schema()
        }
    
    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string"""
        return (
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}"
            f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
    
    def _initialize_engine(self) -> None:
        """Initialize SQLAlchemy engine with connection pooling"""
        try:
            self.engine = create_engine(
                self.connection_string,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=self.config.max_connections - 10,
                pool_timeout=self.config.connection_timeout,
                pool_recycle=3600,  # Recycle connections after 1 hour
                echo=False  # Set to True for SQL debugging
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("PostgreSQL warm data connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL engine: {str(e)}")
            self.engine = None
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        if not self.engine:
            raise Exception("Database engine not initialized")
        
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()
    
    def initialize_tables(self) -> bool:
        """Initialize all warm data tables with partitioning"""
        try:
            with self._get_connection() as conn:
                for table_name, schema in self.table_schemas.items():
                    # Create table
                    conn.execute(text(schema))
                    
                    # Create partitions for time-series data
                    if table_name in ['technical_indicators', 'historical_ohlcv', 'signal_history']:
                        self._create_monthly_partitions(conn, table_name)
                    
                    logger.info(f"Initialized table: {table_name}")
                
                conn.commit()
                logger.info("All warm data tables initialized successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error initializing tables: {str(e)}")
            return False
    
    def _create_monthly_partitions(self, conn, table_name: str) -> None:
        """Create monthly partitions for time-series tables"""
        try:
            # Create partitions for current and next 12 months
            start_date = datetime.now().replace(day=1)
            
            for i in range(12):
                partition_date = start_date + timedelta(days=32 * i)
                partition_date = partition_date.replace(day=1)
                
                partition_name = f"{table_name}_{partition_date.strftime('%Y_%m')}"
                next_month = partition_date + timedelta(days=32)
                next_month = next_month.replace(day=1)
                
                partition_sql = f"""
                CREATE TABLE IF NOT EXISTS {partition_name} 
                PARTITION OF {table_name}
                FOR VALUES FROM ('{partition_date.strftime('%Y-%m-%d')}') 
                TO ('{next_month.strftime('%Y-%m-%d')}');
                """
                
                conn.execute(text(partition_sql))
                
        except Exception as e:
            logger.warning(f"Error creating partitions for {table_name}: {str(e)}")
    
    def store_technical_indicators(self, symbol: str, date: datetime, 
                                  indicators: Dict[str, float]) -> bool:
        """
        Store technical indicators for a symbol and date
        
        Args:
            symbol: Stock symbol
            date: Date for the indicators
            indicators: Dictionary of indicator values
            
        Returns:
            bool: Success status
        """
        try:
            with self._get_connection() as conn:
                # Prepare data
                indicator_data = {
                    'symbol': symbol,
                    'date': date.date(),
                    'created_at': datetime.now(),
                    **indicators
                }
                
                # Convert to DataFrame for easier insertion
                df = pd.DataFrame([indicator_data])
                
                # Insert using pandas to_sql with conflict handling
                df.to_sql(
                    'technical_indicators',
                    conn,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                
                logger.debug(f"Stored technical indicators for {symbol} on {date.date()}")
                return True
                
        except Exception as e:
            logger.error(f"Error storing technical indicators for {symbol}: {str(e)}")
            return False
    
    def get_technical_indicators(self, symbol: str, start_date: datetime, 
                               end_date: datetime, indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Retrieve technical indicators for a symbol and date range
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date  
            indicators: List of specific indicators to retrieve (None for all)
            
        Returns:
            DataFrame with indicator data
        """
        try:
            with self._get_connection() as conn:
                # Build column selection
                if indicators:
                    indicator_columns = ', '.join([f'"{ind}"' for ind in indicators])
                    columns = f'symbol, date, {indicator_columns}'
                else:
                    columns = '*'
                
                query = f"""
                SELECT {columns}
                FROM technical_indicators
                WHERE symbol = :symbol 
                AND date >= :start_date 
                AND date <= :end_date
                ORDER BY date
                """
                
                df = pd.read_sql(
                    query,
                    conn,
                    params={
                        'symbol': symbol,
                        'start_date': start_date.date(),
                        'end_date': end_date.date()
                    }
                )
                
                logger.debug(f"Retrieved {len(df)} indicator records for {symbol}")
                return df
                
        except Exception as e:
            logger.error(f"Error retrieving technical indicators for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def store_historical_ohlcv(self, symbol: str, ohlcv_data: pd.DataFrame) -> bool:
        """
        Store historical OHLCV data
        
        Args:
            symbol: Stock symbol
            ohlcv_data: DataFrame with OHLCV data (must have date index or date column)
            
        Returns:
            bool: Success status
        """
        try:
            with self._get_connection() as conn:
                # Prepare data
                df = ohlcv_data.copy()
                
                # Ensure date column exists
                if df.index.name == 'Date' or 'Date' in str(type(df.index)):
                    df = df.reset_index()
                    df['date'] = pd.to_datetime(df['Date']).dt.date
                elif 'date' not in df.columns:
                    logger.error("OHLCV data must have date index or date column")
                    return False
                
                # Add symbol and metadata
                df['symbol'] = symbol
                df['created_at'] = datetime.now()
                
                # Ensure required columns exist
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                missing_columns = set(required_columns) - set(df.columns.str.lower())
                
                if missing_columns:
                    logger.error(f"Missing required OHLCV columns: {missing_columns}")
                    return False
                
                # Standardize column names
                column_mapping = {
                    col: col.lower() for col in df.columns 
                    if col.lower() in required_columns
                }
                df = df.rename(columns=column_mapping)
                
                # Insert data
                df.to_sql(
                    'historical_ohlcv',
                    conn,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                
                logger.info(f"Stored {len(df)} OHLCV records for {symbol}")
                return True
                
        except Exception as e:
            logger.error(f"Error storing OHLCV data for {symbol}: {str(e)}")
            return False
    
    def get_historical_ohlcv(self, symbol: str, start_date: datetime, 
                            end_date: datetime) -> pd.DataFrame:
        """
        Retrieve historical OHLCV data
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            with self._get_connection() as conn:
                query = """
                SELECT date, open, high, low, close, volume
                FROM historical_ohlcv
                WHERE symbol = :symbol 
                AND date >= :start_date 
                AND date <= :end_date
                ORDER BY date
                """
                
                df = pd.read_sql(
                    query,
                    conn,
                    params={
                        'symbol': symbol,
                        'start_date': start_date.date(),
                        'end_date': end_date.date()
                    },
                    parse_dates=['date']
                )
                
                if not df.empty:
                    df.set_index('date', inplace=True)
                
                logger.debug(f"Retrieved {len(df)} OHLCV records for {symbol}")
                return df
                
        except Exception as e:
            logger.error(f"Error retrieving OHLCV data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def store_ml_features(self, symbol: str, date: datetime, 
                         features: Dict[str, Any]) -> bool:
        """
        Store ML features for a symbol and date
        
        Args:
            symbol: Stock symbol
            date: Date for the features
            features: Dictionary of feature values
            
        Returns:
            bool: Success status
        """
        try:
            with self._get_connection() as conn:
                feature_data = {
                    'symbol': symbol,
                    'date': date.date(),
                    'features': json.dumps(features, default=str),
                    'feature_count': len(features),
                    'created_at': datetime.now()
                }
                
                # Convert to DataFrame and insert
                df = pd.DataFrame([feature_data])
                df.to_sql(
                    'ml_features',
                    conn,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                
                logger.debug(f"Stored ML features for {symbol} on {date.date()}")
                return True
                
        except Exception as e:
            logger.error(f"Error storing ML features for {symbol}: {str(e)}")
            return False
    
    def get_ml_features(self, symbol: str, start_date: datetime, 
                       end_date: datetime) -> pd.DataFrame:
        """
        Retrieve ML features for a symbol and date range
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with feature data
        """
        try:
            with self._get_connection() as conn:
                query = """
                SELECT symbol, date, features, feature_count, created_at
                FROM ml_features
                WHERE symbol = :symbol 
                AND date >= :start_date 
                AND date <= :end_date
                ORDER BY date
                """
                
                df = pd.read_sql(
                    query,
                    conn,
                    params={
                        'symbol': symbol,
                        'start_date': start_date.date(),
                        'end_date': end_date.date()
                    },
                    parse_dates=['date', 'created_at']
                )
                
                # Parse JSON features back to dictionary columns
                if not df.empty and 'features' in df.columns:
                    df['features_dict'] = df['features'].apply(
                        lambda x: json.loads(x) if x else {}
                    )
                
                logger.debug(f"Retrieved {len(df)} ML feature records for {symbol}")
                return df
                
        except Exception as e:
            logger.error(f"Error retrieving ML features for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def store_backtest_results(self, strategy_name: str, 
                              results: Dict[str, Any]) -> bool:
        """
        Store backtest results
        
        Args:
            strategy_name: Name of the strategy
            results: Backtest results dictionary
            
        Returns:
            bool: Success status
        """
        try:
            with self._get_connection() as conn:
                result_data = {
                    'strategy_name': strategy_name,
                    'test_start_date': results.get('start_date'),
                    'test_end_date': results.get('end_date'),
                    'total_return': results.get('total_return', 0.0),
                    'annual_return': results.get('annual_return', 0.0),
                    'sharpe_ratio': results.get('sharpe_ratio', 0.0),
                    'max_drawdown': results.get('max_drawdown', 0.0),
                    'win_rate': results.get('win_rate', 0.0),
                    'profit_factor': results.get('profit_factor', 0.0),
                    'total_trades': results.get('total_trades', 0),
                    'results_json': json.dumps(results, default=str),
                    'created_at': datetime.now()
                }
                
                df = pd.DataFrame([result_data])
                df.to_sql(
                    'backtest_results',
                    conn,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                
                logger.info(f"Stored backtest results for strategy: {strategy_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error storing backtest results: {str(e)}")
            return False
    
    def get_backtest_results(self, strategy_name: Optional[str] = None, 
                           limit: int = 100) -> pd.DataFrame:
        """
        Retrieve backtest results
        
        Args:
            strategy_name: Specific strategy name (None for all)
            limit: Maximum number of results to return
            
        Returns:
            DataFrame with backtest results
        """
        try:
            with self._get_connection() as conn:
                if strategy_name:
                    query = """
                    SELECT * FROM backtest_results
                    WHERE strategy_name = :strategy_name
                    ORDER BY created_at DESC
                    LIMIT :limit
                    """
                    params = {'strategy_name': strategy_name, 'limit': limit}
                else:
                    query = """
                    SELECT * FROM backtest_results
                    ORDER BY created_at DESC
                    LIMIT :limit
                    """
                    params = {'limit': limit}
                
                df = pd.read_sql(
                    query,
                    conn,
                    params=params,
                    parse_dates=['test_start_date', 'test_end_date', 'created_at']
                )
                
                logger.debug(f"Retrieved {len(df)} backtest results")
                return df
                
        except Exception as e:
            logger.error(f"Error retrieving backtest results: {str(e)}")
            return pd.DataFrame()
    
    def cleanup_expired_data(self) -> Dict[str, int]:
        """
        Clean up expired data according to retention policies
        
        Returns:
            Dict with count of deleted records by table
        """
        cleanup_counts = {}
        current_date = datetime.now().date()
        
        cleanup_tasks = [
            ('technical_indicators', self.config.technical_indicators_retention),
            ('historical_ohlcv', self.config.historical_data_retention),
            ('ml_features', self.config.ml_features_retention),
            ('backtest_results', self.config.backtest_results_retention)
        ]
        
        try:
            with self._get_connection() as conn:
                for table_name, retention_days in cleanup_tasks:
                    cutoff_date = current_date - timedelta(days=retention_days)
                    
                    if table_name == 'backtest_results':
                        # Use created_at for backtest results
                        delete_query = f"""
                        DELETE FROM {table_name} 
                        WHERE created_at < :cutoff_date
                        """
                    else:
                        # Use date column for time-series data
                        delete_query = f"""
                        DELETE FROM {table_name} 
                        WHERE date < :cutoff_date
                        """
                    
                    result = conn.execute(text(delete_query), {'cutoff_date': cutoff_date})
                    cleanup_counts[table_name] = result.rowcount
                    
                    logger.info(f"Cleaned up {result.rowcount} expired records from {table_name}")
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error during data cleanup: {str(e)}")
        
        return cleanup_counts
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for warm data"""
        try:
            with self._get_connection() as conn:
                stats = {}
                
                # Get row counts for each table
                for table_name in self.table_schemas.keys():
                    try:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                        stats[f'{table_name}_rows'] = result.scalar()
                    except Exception as e:
                        logger.warning(f"Error counting rows in {table_name}: {str(e)}")
                        stats[f'{table_name}_rows'] = 0
                
                # Get database size
                try:
                    size_query = """
                    SELECT pg_size_pretty(pg_database_size(current_database())) as db_size
                    """
                    result = conn.execute(text(size_query))
                    stats['database_size'] = result.scalar()
                except Exception as e:
                    logger.warning(f"Error getting database size: {str(e)}")
                    stats['database_size'] = 'Unknown'
                
                # Get table sizes
                for table_name in self.table_schemas.keys():
                    try:
                        size_query = f"""
                        SELECT pg_size_pretty(pg_total_relation_size('{table_name}')) as table_size
                        """
                        result = conn.execute(text(size_query))
                        stats[f'{table_name}_size'] = result.scalar()
                    except Exception as e:
                        stats[f'{table_name}_size'] = 'Unknown'
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting storage stats: {str(e)}")
            return {}
    
    # Table schema definitions
    def _get_technical_indicators_schema(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS technical_indicators (
            id SERIAL,
            symbol VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            rsi_14 DECIMAL(8,4),
            rsi_30 DECIMAL(8,4),
            macd_line DECIMAL(10,6),
            macd_signal DECIMAL(10,6),
            macd_histogram DECIMAL(10,6),
            bb_upper DECIMAL(12,4),
            bb_middle DECIMAL(12,4),
            bb_lower DECIMAL(12,4),
            bb_position DECIMAL(6,4),
            sma_20 DECIMAL(12,4),
            sma_50 DECIMAL(12,4),
            sma_200 DECIMAL(12,4),
            ema_12 DECIMAL(12,4),
            ema_26 DECIMAL(12,4),
            atr_14 DECIMAL(10,4),
            volume_sma_20 BIGINT,
            obv BIGINT,
            cmf DECIMAL(6,4),
            mfi DECIMAL(6,2),
            stoch_k DECIMAL(6,2),
            stoch_d DECIMAL(6,2),
            williams_r DECIMAL(6,2),
            cci DECIMAL(8,2),
            adx DECIMAL(6,2),
            vwap DECIMAL(12,4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, date)
        ) PARTITION BY RANGE (date);
        
        CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_date 
        ON technical_indicators (symbol, date);
        
        CREATE INDEX IF NOT EXISTS idx_technical_indicators_date 
        ON technical_indicators (date);
        """
    
    def _get_historical_ohlcv_schema(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS historical_ohlcv (
            id SERIAL,
            symbol VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            open DECIMAL(12,4) NOT NULL,
            high DECIMAL(12,4) NOT NULL,
            low DECIMAL(12,4) NOT NULL,
            close DECIMAL(12,4) NOT NULL,
            volume BIGINT NOT NULL,
            adjusted_close DECIMAL(12,4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, date)
        ) PARTITION BY RANGE (date);
        
        CREATE INDEX IF NOT EXISTS idx_historical_ohlcv_symbol_date 
        ON historical_ohlcv (symbol, date);
        
        CREATE INDEX IF NOT EXISTS idx_historical_ohlcv_date 
        ON historical_ohlcv (date);
        """
    
    def _get_ml_features_schema(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS ml_features (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            features JSONB NOT NULL,
            feature_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, date)
        );
        
        CREATE INDEX IF NOT EXISTS idx_ml_features_symbol_date 
        ON ml_features (symbol, date);
        
        CREATE INDEX IF NOT EXISTS idx_ml_features_date 
        ON ml_features (date);
        
        CREATE INDEX IF NOT EXISTS idx_ml_features_gin 
        ON ml_features USING GIN (features);
        """
    
    def _get_backtest_results_schema(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS backtest_results (
            id SERIAL PRIMARY KEY,
            strategy_name VARCHAR(100) NOT NULL,
            test_start_date DATE,
            test_end_date DATE,
            total_return DECIMAL(10,4),
            annual_return DECIMAL(10,4),
            sharpe_ratio DECIMAL(8,4),
            max_drawdown DECIMAL(8,4),
            win_rate DECIMAL(6,4),
            profit_factor DECIMAL(8,4),
            total_trades INTEGER,
            results_json JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_backtest_results_strategy 
        ON backtest_results (strategy_name);
        
        CREATE INDEX IF NOT EXISTS idx_backtest_results_created_at 
        ON backtest_results (created_at);
        """
    
    def _get_signal_history_schema(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS signal_history (
            id SERIAL,
            symbol VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            signal_direction VARCHAR(20) NOT NULL,
            signal_strength DECIMAL(6,4),
            confidence DECIMAL(6,4),
            signal_components JSONB,
            market_regime VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, date)
        ) PARTITION BY RANGE (date);
        
        CREATE INDEX IF NOT EXISTS idx_signal_history_symbol_date 
        ON signal_history (symbol, date);
        
        CREATE INDEX IF NOT EXISTS idx_signal_history_signal_direction 
        ON signal_history (signal_direction);
        """