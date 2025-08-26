"""
Top 100 US Stocks Data Management System
Fetches and manages stock data for the top 100 US stocks by market cap
"""
import yfinance as yf
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime, timedelta
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import redis
import json
from config.config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Top100StocksDataManager:
    """Manages data for the top 100 US stocks"""
    
    def __init__(self, db_config=None, redis_config=None):
        # Database connection
        if db_config is None:
            db_config = config.db
            
        self.db_config = {
            'host': db_config.host,
            'port': db_config.port,
            'database': db_config.database,
            'user': db_config.user,
            'password': db_config.password
        }
        
        self.conn = psycopg2.connect(**self.db_config)
        self.cursor = self.conn.cursor()
        
        # Redis connection for caching
        if redis_config is None:
            redis_config = config.redis
            
        self.redis_client = redis.Redis(
            host=redis_config.host,
            port=redis_config.port,
            db=redis_config.db,
            password=redis_config.password if redis_config.password else None,
            decode_responses=True
        )
        
        self.logger = logger
        
    def get_sp500_constituents(self) -> pd.DataFrame:
        """Fetch S&P 500 constituents from Wikipedia"""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_df = tables[0]
            
            # Clean up column names
            sp500_df.columns = ['Symbol', 'Security', 'GICS_Sector', 'GICS_Sub_Industry', 
                               'Headquarters_Location', 'Date_Added', 'CIK', 'Founded']
            
            self.logger.info(f"Fetched {len(sp500_df)} S&P 500 constituents")
            return sp500_df
            
        except Exception as e:
            self.logger.error(f"Error fetching S&P 500 constituents: {str(e)}")
            # Fallback to hardcoded list of top stocks
            return self._get_fallback_stock_list()
    
    def _get_fallback_stock_list(self) -> pd.DataFrame:
        """Fallback list of top US stocks"""
        stocks = [
            ['AAPL', 'Apple Inc.', 'Information Technology', 'Technology Hardware, Storage & Peripherals'],
            ['MSFT', 'Microsoft Corporation', 'Information Technology', 'Systems Software'],
            ['GOOGL', 'Alphabet Inc.', 'Communication Services', 'Interactive Media & Services'],
            ['AMZN', 'Amazon.com Inc.', 'Consumer Discretionary', 'Internet & Direct Marketing Retail'],
            ['NVDA', 'NVIDIA Corporation', 'Information Technology', 'Semiconductors & Semiconductor Equipment'],
            ['TSLA', 'Tesla Inc.', 'Consumer Discretionary', 'Automobiles'],
            ['META', 'Meta Platforms Inc.', 'Communication Services', 'Interactive Media & Services'],
            ['BRK.B', 'Berkshire Hathaway Inc.', 'Financials', 'Multi-Sector Holdings'],
            ['UNH', 'UnitedHealth Group Incorporated', 'Health Care', 'Health Care Services'],
            ['JNJ', 'Johnson & Johnson', 'Health Care', 'Pharmaceuticals']
        ]
        
        df = pd.DataFrame(stocks, columns=['Symbol', 'Security', 'GICS_Sector', 'GICS_Sub_Industry'])
        return df
        
    def get_market_cap_data(self, symbols: List[str], batch_size: int = 10) -> Dict[str, float]:
        """Fetch market cap data for given symbols"""
        market_caps = {}
        
        # Check cache first
        cached_caps = {}
        for symbol in symbols:
            cache_key = f"market_cap:{symbol}"
            cached_value = self.redis_client.get(cache_key)
            if cached_value:
                cached_caps[symbol] = float(cached_value)
                
        # Get symbols that need fresh data
        symbols_to_fetch = [s for s in symbols if s not in cached_caps]
        market_caps.update(cached_caps)
        
        if not symbols_to_fetch:
            return market_caps
            
        self.logger.info(f"Fetching market cap for {len(symbols_to_fetch)} symbols")
        
        # Process in batches to avoid overwhelming the API
        for i in range(0, len(symbols_to_fetch), batch_size):
            batch = symbols_to_fetch[i:i + batch_size]
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(self._get_single_market_cap, symbol): symbol
                    for symbol in batch
                }
                
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        market_cap = future.result()
                        if market_cap:
                            market_caps[symbol] = market_cap
                            # Cache for 1 hour
                            cache_key = f"market_cap:{symbol}"
                            self.redis_client.setex(cache_key, 3600, str(market_cap))
                    except Exception as e:
                        self.logger.warning(f"Error getting market cap for {symbol}: {str(e)}")
                        
            # Rate limiting
            time.sleep(1)
            
        return market_caps
        
    def _get_single_market_cap(self, symbol: str) -> Optional[float]:
        """Get market cap for a single symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            market_cap = info.get('marketCap')
            if market_cap and market_cap > 0:
                return float(market_cap)
                
            # Alternative: calculate from shares outstanding and price
            shares_outstanding = info.get('sharesOutstanding')
            current_price = info.get('currentPrice')
            
            if shares_outstanding and current_price:
                return float(shares_outstanding * current_price)
                
            return None
            
        except Exception as e:
            self.logger.warning(f"Error fetching market cap for {symbol}: {str(e)}")
            return None
    
    def get_top_100_stocks(self) -> List[str]:
        """Get top 100 US stocks by market cap"""
        # Check if we have recent data in cache
        cache_key = "top_100_stocks"
        cached_stocks = self.redis_client.get(cache_key)
        
        if cached_stocks:
            self.logger.info("Using cached top 100 stocks")
            return json.loads(cached_stocks)
            
        # Fetch S&P 500 constituents
        sp500_df = self.get_sp500_constituents()
        symbols = sp500_df['Symbol'].tolist()
        
        # Get market cap data
        market_caps = self.get_market_cap_data(symbols[:200])  # Get more than needed
        
        if not market_caps:
            self.logger.warning("No market cap data available, using fallback list")
            fallback_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK.B', 'UNH', 'JNJ']
            return fallback_stocks
            
        # Sort by market cap and get top 100
        sorted_stocks = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)[:100]
        top_100 = [stock[0] for stock in sorted_stocks]
        
        # Update database with stock info
        self._update_securities_table(sp500_df, market_caps, top_100)
        
        # Cache for 4 hours
        self.redis_client.setex(cache_key, 14400, json.dumps(top_100))
        
        self.logger.info(f"Retrieved top 100 stocks by market cap")
        return top_100
        
    def _update_securities_table(self, sp500_df: pd.DataFrame, market_caps: Dict[str, float], 
                               top_100: List[str]) -> None:
        """Update the securities table with stock information"""
        try:
            records = []
            
            for symbol in top_100:
                # Find company info
                company_info = sp500_df[sp500_df['Symbol'] == symbol]
                if not company_info.empty:
                    info = company_info.iloc[0]
                    records.append((
                        symbol,
                        info['Security'],
                        info['GICS_Sector'],
                        info['GICS_Sub_Industry'],
                        market_caps.get(symbol, 0)
                    ))
                else:
                    # For stocks not in S&P 500, use basic info
                    records.append((
                        symbol,
                        f"{symbol} Corp",  # Placeholder
                        "Unknown",
                        "Unknown",
                        market_caps.get(symbol, 0)
                    ))
            
            # Bulk upsert
            execute_batch(self.cursor, """
                INSERT INTO securities (symbol, company_name, sector, industry, market_cap)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (symbol) 
                DO UPDATE SET 
                    company_name = EXCLUDED.company_name,
                    sector = EXCLUDED.sector,
                    industry = EXCLUDED.industry,
                    market_cap = EXCLUDED.market_cap,
                    last_updated = CURRENT_TIMESTAMP
            """, records, page_size=100)
            
            self.conn.commit()
            self.logger.info(f"Updated securities table with {len(records)} stocks")
            
        except Exception as e:
            self.logger.error(f"Error updating securities table: {str(e)}")
            self.conn.rollback()
    
    def fetch_historical_data(self, symbols: List[str], start_date: str = '2020-01-01',
                             end_date: Optional[str] = None) -> None:
        """Fetch and store historical data for given symbols"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        self.logger.info(f"Fetching historical data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Process symbols in parallel batches
        batch_size = 20
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(self._fetch_single_stock_data, symbol, start_date, end_date): symbol
                    for symbol in batch
                }
                
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        success = future.result()
                        if success:
                            self.logger.info(f"Successfully fetched data for {symbol}")
                        else:
                            self.logger.warning(f"Failed to fetch data for {symbol}")
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol}: {str(e)}")
                        
            # Rate limiting between batches
            if i + batch_size < len(symbols):
                time.sleep(2)
    
    def _fetch_single_stock_data(self, symbol: str, start_date: str, end_date: str) -> bool:
        """Fetch historical data for a single stock"""
        try:
            # Get symbol_id from database
            self.cursor.execute("SELECT id FROM securities WHERE symbol = %s", (symbol,))
            result = self.cursor.fetchone()
            if not result:
                self.logger.warning(f"Symbol {symbol} not found in securities table")
                return False
                
            symbol_id = result[0]
            
            # Check what data we already have
            self.cursor.execute("""
                SELECT MAX(trade_date) FROM daily_prices WHERE symbol_id = %s
            """, (symbol_id,))
            
            last_date_result = self.cursor.fetchone()
            last_date = last_date_result[0] if last_date_result[0] else None
            
            # Determine fetch start date
            fetch_start = start_date
            if last_date:
                fetch_start = max(start_date, (last_date + timedelta(days=1)).strftime('%Y-%m-%d'))
                
            if fetch_start >= end_date:
                self.logger.info(f"Data for {symbol} is up to date")
                return True
            
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=fetch_start, end=end_date, auto_adjust=False)
            
            if df.empty:
                self.logger.warning(f"No data returned for {symbol}")
                return False
            
            # Prepare data for insertion
            records = []
            for date, row in df.iterrows():
                # Skip if any essential data is missing
                if pd.isna(row['Open']) or pd.isna(row['Close']) or row['Volume'] == 0:
                    continue
                    
                records.append((
                    symbol_id,
                    date.date(),
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    float(row['Adj Close']),
                    int(row['Volume'])
                ))
            
            if not records:
                self.logger.warning(f"No valid records for {symbol}")
                return False
            
            # Bulk insert
            execute_batch(self.cursor, """
                INSERT INTO daily_prices 
                (symbol_id, trade_date, open, high, low, close, adj_close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol_id, trade_date) DO NOTHING
            """, records, page_size=1000)
            
            self.conn.commit()
            
            # Update cache to indicate fresh data
            cache_key = f"last_update:{symbol}"
            self.redis_client.setex(cache_key, 3600, datetime.now().isoformat())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            self.conn.rollback()
            return False
    
    def update_daily_data(self, symbols: Optional[List[str]] = None) -> None:
        """Update daily data for all active stocks or specified symbols"""
        if symbols is None:
            # Get all active symbols
            self.cursor.execute("""
                SELECT id, symbol FROM securities WHERE is_active = true
            """)
            stocks = [(row[0], row[1]) for row in self.cursor.fetchall()]
        else:
            # Get specific symbols
            symbols_str = "','".join(symbols)
            self.cursor.execute(f"""
                SELECT id, symbol FROM securities 
                WHERE symbol IN ('{symbols_str}') AND is_active = true
            """)
            stocks = [(row[0], row[1]) for row in self.cursor.fetchall()]
        
        self.logger.info(f"Updating daily data for {len(stocks)} stocks")
        
        today = datetime.now().date()
        updated_count = 0
        
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = {
                executor.submit(self._update_single_stock_daily, symbol_id, symbol, today): symbol
                for symbol_id, symbol in stocks
            }
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    success = future.result()
                    if success:
                        updated_count += 1
                except Exception as e:
                    self.logger.error(f"Error updating {symbol}: {str(e)}")
        
        self.logger.info(f"Updated daily data for {updated_count}/{len(stocks)} stocks")
    
    def _update_single_stock_daily(self, symbol_id: int, symbol: str, today: datetime.date) -> bool:
        """Update daily data for a single stock"""
        try:
            # Check if we already have today's data
            self.cursor.execute("""
                SELECT trade_date FROM daily_prices 
                WHERE symbol_id = %s AND trade_date = %s
            """, (symbol_id, today))
            
            if self.cursor.fetchone():
                return True  # Already have today's data
            
            # Check last update time from cache
            cache_key = f"last_update:{symbol}"
            last_update = self.redis_client.get(cache_key)
            
            if last_update:
                last_update_time = datetime.fromisoformat(last_update)
                if datetime.now() - last_update_time < timedelta(hours=1):
                    return True  # Updated recently
            
            # Fetch latest data (last 5 days to ensure we don't miss anything)
            start_date = today - timedelta(days=5)
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=today + timedelta(days=1), auto_adjust=False)
            
            if df.empty:
                return False
            
            records = []
            for date, row in df.iterrows():
                if pd.isna(row['Open']) or pd.isna(row['Close']) or row['Volume'] == 0:
                    continue
                    
                records.append((
                    symbol_id,
                    date.date(),
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    float(row['Adj Close']),
                    int(row['Volume'])
                ))
            
            if records:
                execute_batch(self.cursor, """
                    INSERT INTO daily_prices 
                    (symbol_id, trade_date, open, high, low, close, adj_close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol_id, trade_date) DO NOTHING
                """, records)
                
                self.conn.commit()
                
                # Update cache
                self.redis_client.setex(cache_key, 3600, datetime.now().isoformat())
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating daily data for {symbol}: {str(e)}")
            self.conn.rollback()
            return False
    
    def get_data_quality_report(self) -> Dict[str, any]:
        """Generate a data quality report"""
        report = {}
        
        # Check data completeness
        self.cursor.execute("""
            WITH expected_dates AS (
                SELECT generate_series(
                    CURRENT_DATE - INTERVAL '30 days',
                    CURRENT_DATE,
                    '1 day'::interval
                )::date AS trade_date
                WHERE EXTRACT(DOW FROM generate_series(
                    CURRENT_DATE - INTERVAL '30 days',
                    CURRENT_DATE,
                    '1 day'::interval
                )) NOT IN (0, 6)  -- Exclude weekends
            ),
            stock_completeness AS (
                SELECT 
                    s.symbol,
                    COUNT(DISTINCT dp.trade_date) as actual_days,
                    COUNT(DISTINCT ed.trade_date) as expected_days
                FROM securities s
                CROSS JOIN expected_dates ed
                LEFT JOIN daily_prices dp ON s.id = dp.symbol_id AND dp.trade_date = ed.trade_date
                WHERE s.is_active = true
                GROUP BY s.symbol
            )
            SELECT 
                symbol,
                actual_days,
                expected_days,
                ROUND(actual_days::numeric / expected_days * 100, 2) as completeness_pct
            FROM stock_completeness
            WHERE actual_days::numeric / expected_days < 0.95
            ORDER BY completeness_pct ASC
        """)
        
        incomplete_data = self.cursor.fetchall()
        report['incomplete_data'] = [
            {
                'symbol': row[0],
                'actual_days': row[1],
                'expected_days': row[2],
                'completeness_pct': float(row[3])
            }
            for row in incomplete_data
        ]
        
        # Check for price anomalies
        self.cursor.execute("""
            SELECT 
                s.symbol,
                dp.trade_date,
                dp.close,
                LAG(dp.close) OVER (PARTITION BY s.symbol ORDER BY dp.trade_date) as prev_close,
                ABS((dp.close - LAG(dp.close) OVER (PARTITION BY s.symbol ORDER BY dp.trade_date)) / 
                    LAG(dp.close) OVER (PARTITION BY s.symbol ORDER BY dp.trade_date)) as pct_change
            FROM securities s
            JOIN daily_prices dp ON s.id = dp.symbol_id
            WHERE dp.trade_date >= CURRENT_DATE - INTERVAL '7 days'
              AND s.is_active = true
            HAVING ABS((dp.close - LAG(dp.close) OVER (PARTITION BY s.symbol ORDER BY dp.trade_date)) / 
                       LAG(dp.close) OVER (PARTITION BY s.symbol ORDER BY dp.trade_date)) > 0.15
            ORDER BY pct_change DESC
        """)
        
        price_anomalies = self.cursor.fetchall()
        report['price_anomalies'] = [
            {
                'symbol': row[0],
                'date': row[1],
                'close': float(row[2]),
                'prev_close': float(row[3]) if row[3] else None,
                'pct_change': float(row[4]) if row[4] else None
            }
            for row in price_anomalies
        ]
        
        # Overall statistics
        self.cursor.execute("""
            SELECT 
                COUNT(DISTINCT s.symbol) as total_stocks,
                COUNT(DISTINCT dp.trade_date) as total_dates,
                COUNT(*) as total_records,
                MIN(dp.trade_date) as earliest_date,
                MAX(dp.trade_date) as latest_date
            FROM securities s
            JOIN daily_prices dp ON s.id = dp.symbol_id
            WHERE s.is_active = true
        """)
        
        stats = self.cursor.fetchone()
        report['statistics'] = {
            'total_stocks': stats[0],
            'total_dates': stats[1],
            'total_records': stats[2],
            'earliest_date': stats[3],
            'latest_date': stats[4],
            'data_quality_score': 100 - len(incomplete_data) - len(price_anomalies)
        }
        
        return report
    
    def optimize_database(self) -> None:
        """Run database optimization tasks"""
        try:
            self.logger.info("Starting database optimization")
            
            # Update table statistics
            self.cursor.execute("ANALYZE daily_prices;")
            self.cursor.execute("ANALYZE technical_indicators;")
            self.cursor.execute("ANALYZE securities;")
            
            # Refresh materialized views
            self.cursor.execute("SELECT refresh_materialized_views();")
            
            # Create next month partition if needed
            next_month = (datetime.now().replace(day=1) + timedelta(days=32)).replace(day=1)
            partition_name = f"daily_prices_{next_month.strftime('%Y_%m')}"
            
            self.cursor.execute(f"""
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = '{partition_name}'
            """)
            
            if not self.cursor.fetchone():
                self.cursor.execute(f"""
                    CREATE TABLE {partition_name} PARTITION OF daily_prices
                    FOR VALUES FROM ('{next_month.strftime('%Y-%m-01')}') 
                    TO ('{(next_month + timedelta(days=32)).replace(day=1).strftime('%Y-%m-01')}');
                """)
                self.logger.info(f"Created partition {partition_name}")
            
            self.conn.commit()
            self.logger.info("Database optimization completed")
            
        except Exception as e:
            self.logger.error(f"Error during database optimization: {str(e)}")
            self.conn.rollback()
    
    def close(self) -> None:
        """Close database connections"""
        if self.conn:
            self.conn.close()
        if self.redis_client:
            self.redis_client.close()

# Example usage and testing
if __name__ == "__main__":
    # Test the data manager
    manager = Top100StocksDataManager()
    
    try:
        # Get top 100 stocks
        top_stocks = manager.get_top_100_stocks()
        print(f"Top 10 stocks: {top_stocks[:10]}")
        
        # Fetch historical data for a sample
        sample_stocks = top_stocks[:5]
        manager.fetch_historical_data(sample_stocks, start_date='2023-01-01')
        
        # Update daily data
        manager.update_daily_data(sample_stocks)
        
        # Get data quality report
        quality_report = manager.get_data_quality_report()
        print(f"Data quality score: {quality_report['statistics']['data_quality_score']}")
        
        # Optimize database
        manager.optimize_database()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        
    finally:
        manager.close()