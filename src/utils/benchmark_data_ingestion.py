"""
Benchmark Performance Data Ingestion Module
Fetches and stores historical data for benchmark ETFs (SPY, QQQ, IWM, RSP, VTI)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from src.utils.backtesting_schema_sqlite import sqlite_backtesting_schema
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

class BenchmarkDataIngestion:
    """Handles fetching and storing benchmark performance data"""
    
    def __init__(self, db_path: str = "data/historical_stocks.db"):
        self.db_path = db_path
        self.benchmark_symbols = ['SPY', 'QQQ', 'IWM', 'RSP', 'VTI']
        
        # Define data range - 5 years of history
        self.end_date = datetime.now().date()
        self.start_date = self.end_date - timedelta(days=5*365)  # 5 years
        
    def get_connection(self):
        """Get database connection"""
        return sqlite_backtesting_schema.get_connection()
        
    def fetch_benchmark_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data for a single benchmark symbol"""
        
        try:
            logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
            
            # Fetch data using yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date, auto_adjust=False)
            
            if hist.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Reset index to get date as a column
            hist.reset_index(inplace=True)
            
            # Rename columns to match our schema
            hist.columns = [col.lower().replace(' ', '_') for col in hist.columns]
            hist.rename(columns={
                'adj_close': 'adj_close_price',
                'open': 'open_price',
                'high': 'high_price', 
                'low': 'low_price',
                'close': 'close_price'
            }, inplace=True)
            
            # Add symbol column
            hist['symbol'] = symbol
            
            # Calculate returns and performance metrics
            hist = self.calculate_performance_metrics(hist)
            
            logger.info(f"Successfully fetched {len(hist)} records for {symbol}")
            return hist
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_performance_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate performance metrics for benchmark data"""
        
        if df.empty:
            return df
            
        try:
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            # Calculate daily returns
            df['daily_return'] = df['adj_close_price'].pct_change()
            
            # Calculate cumulative returns
            df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
            
            # Calculate rolling 20-day volatility
            df['volatility_20d'] = df['daily_return'].rolling(window=20).std() * np.sqrt(252)
            
            # Calculate running maximum and drawdown
            df['running_max'] = df['adj_close_price'].expanding().max()
            df['drawdown'] = (df['adj_close_price'] / df['running_max'] - 1)
            df['max_drawdown'] = df['drawdown'].expanding().min()
            
            # Fill NaN values for first rows
            df['daily_return'].iloc[0] = 0
            df['cumulative_return'].iloc[0] = 0
            df['volatility_20d'].fillna(0, inplace=True)
            df['max_drawdown'].fillna(0, inplace=True)
            
            logger.debug(f"Calculated performance metrics for {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {str(e)}")
            return df
    
    def store_benchmark_data(self, df: pd.DataFrame, symbol: str):
        """Store benchmark data in the database"""
        
        if df.empty:
            logger.warning(f"No data to store for {symbol}")
            return
        
        try:
            with self.get_connection() as conn:
                # Check if data already exists
                existing_count = conn.execute("""
                    SELECT COUNT(*) as count 
                    FROM benchmark_performance 
                    WHERE symbol = ?
                """, (symbol,)).fetchone()['count']
                
                if existing_count > 0:
                    logger.info(f"Benchmark data for {symbol} already exists ({existing_count} records). Updating...")
                    # Delete existing data for this symbol
                    conn.execute("DELETE FROM benchmark_performance WHERE symbol = ?", (symbol,))
                
                # Prepare data for insertion
                records = []
                for _, row in df.iterrows():
                    records.append((
                        symbol,
                        row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                        float(row.get('open_price', 0)),
                        float(row.get('high_price', 0)),
                        float(row.get('low_price', 0)),
                        float(row.get('close_price', 0)),
                        float(row.get('adj_close_price', 0)),
                        int(row.get('volume', 0)),
                        float(row.get('daily_return', 0)),
                        float(row.get('cumulative_return', 0)),
                        float(row.get('volatility_20d', 0)),
                        float(row.get('max_drawdown', 0))
                    ))
                
                # Insert data
                conn.executemany("""
                    INSERT INTO benchmark_performance 
                    (symbol, date, open_price, high_price, low_price, close_price, 
                     adj_close_price, volume, daily_return, cumulative_return, 
                     volatility_20d, max_drawdown)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records)
                
                conn.commit()
                logger.info(f"Successfully stored {len(records)} records for {symbol}")
                
        except Exception as e:
            logger.error(f"Failed to store benchmark data for {symbol}: {str(e)}")
            raise
    
    def fetch_all_benchmarks(self, max_workers: int = 5) -> Dict[str, pd.DataFrame]:
        """Fetch data for all benchmark symbols in parallel"""
        
        logger.info(f"Fetching benchmark data for {len(self.benchmark_symbols)} symbols")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        
        benchmark_data = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all fetch tasks
            future_to_symbol = {
                executor.submit(
                    self.fetch_benchmark_data, 
                    symbol, 
                    str(self.start_date), 
                    str(self.end_date)
                ): symbol 
                for symbol in self.benchmark_symbols
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if not data.empty:
                        benchmark_data[symbol] = data
                        logger.info(f"✅ Successfully fetched data for {symbol}")
                    else:
                        logger.warning(f"⚠️ No data received for {symbol}")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to fetch data for {symbol}: {str(e)}")
        
        logger.info(f"Successfully fetched data for {len(benchmark_data)} out of {len(self.benchmark_symbols)} benchmarks")
        return benchmark_data
    
    def store_all_benchmark_data(self, benchmark_data: Dict[str, pd.DataFrame]):
        """Store all benchmark data in the database"""
        
        logger.info("Storing benchmark data in database...")
        
        success_count = 0
        for symbol, data in benchmark_data.items():
            try:
                self.store_benchmark_data(data, symbol)
                success_count += 1
                logger.info(f"✅ Stored data for {symbol}")
            except Exception as e:
                logger.error(f"❌ Failed to store data for {symbol}: {str(e)}")
        
        logger.info(f"Successfully stored data for {success_count} out of {len(benchmark_data)} benchmarks")
    
    def validate_benchmark_data(self) -> Dict[str, Dict]:
        """Validate the stored benchmark data"""
        
        logger.info("Validating stored benchmark data...")
        validation_results = {}
        
        try:
            with self.get_connection() as conn:
                for symbol in self.benchmark_symbols:
                    # Get basic statistics
                    stats_query = """
                    SELECT 
                        COUNT(*) as record_count,
                        MIN(date) as start_date,
                        MAX(date) as end_date,
                        AVG(daily_return) as avg_daily_return,
                        AVG(volatility_20d) as avg_volatility,
                        MIN(max_drawdown) as max_drawdown,
                        MAX(cumulative_return) as max_cumulative_return
                    FROM benchmark_performance 
                    WHERE symbol = ?
                    """
                    
                    result = conn.execute(stats_query, (symbol,)).fetchone()
                    
                    if result and result['record_count'] > 0:
                        validation_results[symbol] = {
                            'record_count': result['record_count'],
                            'date_range': f"{result['start_date']} to {result['end_date']}",
                            'avg_daily_return': round(float(result['avg_daily_return']) * 100, 4),
                            'annualized_return': round(float(result['avg_daily_return']) * 252 * 100, 2),
                            'avg_volatility': round(float(result['avg_volatility']) * 100, 2),
                            'max_drawdown': round(float(result['max_drawdown']) * 100, 2),
                            'total_return': round(float(result['max_cumulative_return']) * 100, 2),
                            'status': 'valid'
                        }
                    else:
                        validation_results[symbol] = {
                            'record_count': 0,
                            'status': 'missing'
                        }
        
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {}
        
        # Log validation results
        logger.info("Benchmark data validation results:")
        for symbol, stats in validation_results.items():
            if stats['status'] == 'valid':
                logger.info(f"{symbol}: {stats['record_count']} records, "
                          f"Return: {stats['annualized_return']}%, "
                          f"Vol: {stats['avg_volatility']}%, "
                          f"Max DD: {stats['max_drawdown']}%")
            else:
                logger.warning(f"{symbol}: {stats['status']}")
        
        return validation_results
    
    def ingest_benchmark_data(self) -> bool:
        """Main method to ingest all benchmark data"""
        
        logger.info("🚀 Starting benchmark data ingestion process...")
        start_time = time.time()
        
        try:
            # Fetch all benchmark data
            benchmark_data = self.fetch_all_benchmarks()
            
            if not benchmark_data:
                logger.error("No benchmark data was fetched")
                return False
            
            # Store all data
            self.store_all_benchmark_data(benchmark_data)
            
            # Validate stored data
            validation_results = self.validate_benchmark_data()
            
            # Check if all benchmarks have valid data
            valid_benchmarks = sum(1 for stats in validation_results.values() if stats.get('status') == 'valid')
            
            execution_time = time.time() - start_time
            
            if valid_benchmarks == len(self.benchmark_symbols):
                logger.info(f"✅ Benchmark data ingestion completed successfully in {execution_time:.2f} seconds")
                logger.info(f"All {valid_benchmarks} benchmarks have valid data")
                return True
            else:
                logger.warning(f"⚠️ Partial success: {valid_benchmarks}/{len(self.benchmark_symbols)} benchmarks have valid data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Benchmark data ingestion failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

# Global benchmark data ingestion instance
benchmark_ingestion = BenchmarkDataIngestion()

# Convenience function
def ingest_all_benchmark_data():
    """Ingest all benchmark performance data"""
    return benchmark_ingestion.ingest_benchmark_data()