"""
Historical Data Manager - Efficient storage and retrieval of stock data
Stores 5-year historical data once, only fetches live data on refresh
"""
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import os

class HistoricalDataManager:
    def __init__(self, db_path="/Users/yatharthanand/SIgnal - US/data/historical_stocks.db"):
        self.db_path = db_path
        self.ensure_db_directory()
        self.init_database()
        self.lock = threading.Lock()
    
    def ensure_db_directory(self):
        """Ensure database directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Historical OHLCV data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
            ''')
            
            # Stock metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_metadata (
                    symbol TEXT PRIMARY KEY,
                    company_name TEXT,
                    sector TEXT,
                    industry TEXT,
                    market_cap BIGINT,
                    last_historical_update TIMESTAMP,
                    last_live_update TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Live data cache table (latest values only)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS live_data_cache (
                    symbol TEXT PRIMARY KEY,
                    current_price REAL,
                    volume INTEGER,
                    change_1d REAL,
                    change_5d REAL,
                    change_20d REAL,
                    rsi_14 REAL,
                    macd_line REAL,
                    macd_signal REAL,
                    macd_histogram REAL,
                    bb_upper REAL,
                    bb_middle REAL,
                    bb_lower REAL,
                    bb_position REAL,
                    sma_20 REAL,
                    sma_50 REAL,
                    volatility_20d REAL,
                    volume_ratio REAL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_historical_symbol_date ON historical_data(symbol, date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metadata_update ON stock_metadata(last_historical_update)')
            
            conn.commit()
            print("Database initialized successfully")
    
    def get_stocks_needing_historical_update(self, symbols, days_threshold=7):
        """Get list of stocks that need historical data update"""
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ','.join(['?'] * len(symbols))
            query = f'''
                SELECT symbol FROM stock_metadata 
                WHERE symbol IN ({placeholders})
                AND (last_historical_update IS NULL 
                     OR last_historical_update < datetime('now', '-{days_threshold} days'))
            '''
            df = pd.read_sql_query(query, conn, params=symbols)
            
            # Also include symbols not in metadata table
            existing_symbols = set(df['symbol'].tolist())
            all_symbols = set(symbols)
            missing_symbols = all_symbols - existing_symbols
            
            return list(df['symbol']) + list(missing_symbols)
    
    def fetch_historical_data_parallel(self, symbols, max_workers=10, period="5y"):
        """Fetch historical data for multiple symbols in parallel"""
        def fetch_single_stock(symbol):
            try:
                print(f"Fetching {symbol}...")
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if hist.empty:
                    return symbol, None, None, f"No historical data - possibly delisted"
                
                # Validate minimum data requirements
                if len(hist) < 50:  # Need at least 50 days for meaningful analysis
                    return symbol, None, None, f"Insufficient data: {len(hist)} days"
                
                # Get company info with error handling
                try:
                    info = ticker.info
                except:
                    info = {}
                
                # Prepare historical data
                hist_data = hist.reset_index()
                hist_data['symbol'] = symbol
                hist_data['Date'] = pd.to_datetime(hist_data['Date']).dt.date
                
                # Prepare metadata with safe extraction
                metadata = {
                    'symbol': symbol,
                    'company_name': str(info.get('longName', symbol))[:100] if info.get('longName') else symbol,
                    'sector': str(info.get('sector', 'Unknown'))[:50] if info.get('sector') else 'Unknown',
                    'industry': str(info.get('industry', 'Unknown'))[:100] if info.get('industry') else 'Unknown',
                    'market_cap': int(info.get('marketCap', 0)) if info.get('marketCap') and isinstance(info.get('marketCap'), (int, float)) else 0
                }
                
                return symbol, hist_data, metadata, None
                
            except Exception as e:
                print(f"ERROR fetching {symbol}: {str(e)}")
                return symbol, None, None, str(e)
        
        results = {'success': [], 'failed': []}
        
        print(f"Starting parallel fetch for {len(symbols)} symbols with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {executor.submit(fetch_single_stock, symbol): symbol 
                              for symbol in symbols}
            
            # Process completed tasks
            for future in as_completed(future_to_symbol):
                symbol, hist_data, metadata, error = future.result()
                
                if error:
                    print(f"❌ {symbol}: {error}")
                    results['failed'].append({'symbol': symbol, 'error': error})
                else:
                    print(f"✅ {symbol}: {len(hist_data)} days of data")
                    self.store_historical_data(hist_data, metadata)
                    results['success'].append(symbol)
        
        return results
    
    def store_historical_data(self, hist_data, metadata):
        """Store historical data and metadata in database"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                try:
                    # Store metadata
                    metadata['last_historical_update'] = datetime.now().isoformat()
                    
                    conn.execute('''
                        INSERT OR REPLACE INTO stock_metadata 
                        (symbol, company_name, sector, industry, market_cap, last_historical_update)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (metadata['symbol'], metadata['company_name'], metadata['sector'], 
                          metadata['industry'], metadata['market_cap'], metadata['last_historical_update']))
                    
                    # Store historical data
                    hist_data_clean = hist_data[['symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
                    hist_data_clean.columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
                    
                    # Delete existing data for this symbol first to avoid UNIQUE constraint issues
                    conn.execute('DELETE FROM historical_data WHERE symbol = ?', (metadata['symbol'],))
                    
                    # Insert new data
                    hist_data_clean.to_sql('historical_data', conn, if_exists='append', index=False, method='multi')
                    
                    conn.commit()
                    
                except Exception as e:
                    print(f"Error storing data for {metadata['symbol']}: {e}")
                    conn.rollback()
    
    def get_historical_data(self, symbol, days=252):
        """Retrieve historical data for a symbol"""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT date, open, high, low, close, volume
                FROM historical_data 
                WHERE symbol = ? 
                ORDER BY date DESC 
                LIMIT ?
            '''
            df = pd.read_sql_query(query, conn, params=[symbol, days])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df.sort_index()  # Return in ascending order
    
    def calculate_technical_indicators(self, symbol):
        """Calculate technical indicators from stored historical data"""
        hist_data = self.get_historical_data(symbol, days=100)
        
        if len(hist_data) < 20:
            return None
        
        # Calculate all indicators
        indicators = {}
        
        # RSI
        delta = hist_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        indicators['rsi_14'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        # MACD
        ema_fast = hist_data['close'].ewm(span=12).mean()
        ema_slow = hist_data['close'].ewm(span=26).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=9).mean()
        macd_histogram = macd_line - macd_signal
        
        indicators['macd_line'] = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0
        indicators['macd_signal'] = macd_signal.iloc[-1] if not pd.isna(macd_signal.iloc[-1]) else 0
        indicators['macd_histogram'] = macd_histogram.iloc[-1] if not pd.isna(macd_histogram.iloc[-1]) else 0
        
        # Bollinger Bands
        sma_20 = hist_data['close'].rolling(20).mean()
        sma_20_std = hist_data['close'].rolling(20).std()
        bb_upper = sma_20 + (sma_20_std * 2)
        bb_lower = sma_20 - (sma_20_std * 2)
        
        indicators['bb_upper'] = bb_upper.iloc[-1] if not pd.isna(bb_upper.iloc[-1]) else hist_data['close'].iloc[-1]
        indicators['bb_middle'] = sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else hist_data['close'].iloc[-1]
        indicators['bb_lower'] = bb_lower.iloc[-1] if not pd.isna(bb_lower.iloc[-1]) else hist_data['close'].iloc[-1]
        
        # BB Position
        current_price = hist_data['close'].iloc[-1]
        indicators['bb_position'] = ((current_price - indicators['bb_lower']) / 
                                   (indicators['bb_upper'] - indicators['bb_lower']) * 100)
        
        # Moving Averages
        indicators['sma_20'] = sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else current_price
        sma_50 = hist_data['close'].rolling(50).mean()
        indicators['sma_50'] = sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else current_price
        
        # Returns
        indicators['change_1d'] = ((hist_data['close'].iloc[-1] - hist_data['close'].iloc[-2]) / 
                                 hist_data['close'].iloc[-2] * 100) if len(hist_data) > 1 else 0
        indicators['change_5d'] = ((hist_data['close'].iloc[-1] - hist_data['close'].iloc[-6]) / 
                                 hist_data['close'].iloc[-6] * 100) if len(hist_data) > 5 else 0
        indicators['change_20d'] = ((hist_data['close'].iloc[-1] - hist_data['close'].iloc[-21]) / 
                                  hist_data['close'].iloc[-21] * 100) if len(hist_data) > 20 else 0
        
        # Volatility
        returns = hist_data['close'].pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252) * 100
        indicators['volatility_20d'] = volatility.iloc[-1] if not pd.isna(volatility.iloc[-1]) else 20
        
        # Volume metrics
        volume_sma = hist_data['volume'].rolling(20).mean()
        indicators['volume_ratio'] = (hist_data['volume'].iloc[-1] / volume_sma.iloc[-1] 
                                    if volume_sma.iloc[-1] > 0 else 1.0)
        
        return indicators
    
    def fetch_live_data_parallel(self, symbols, max_workers=20):
        """Fetch only current day data for symbols in parallel"""
        def fetch_live_single(symbol):
            try:
                ticker = yf.Ticker(symbol)
                # Get just today's data
                hist = ticker.history(period="2d")  # 2 days to ensure we have latest
                
                if hist.empty:
                    return symbol, None, f"No live data - possibly delisted"
                
                # Validate price data
                latest = hist.iloc[-1]
                if pd.isna(latest['Close']) or latest['Close'] <= 0:
                    return symbol, None, f"Invalid price data"
                
                live_data = {
                    'symbol': symbol,
                    'current_price': float(latest['Close']),
                    'volume': int(latest['Volume']) if not pd.isna(latest['Volume']) else 0,
                }
                
                # Calculate indicators from stored historical data
                try:
                    indicators = self.calculate_technical_indicators(symbol)
                    if indicators:
                        # Validate indicator values
                        for key, value in indicators.items():
                            if pd.isna(value) or not isinstance(value, (int, float)):
                                indicators[key] = 0.0 if 'change' in key else 50.0 if key == 'rsi_14' else live_data['current_price']
                        live_data.update(indicators)
                    else:
                        # Provide default indicators if calculation fails
                        live_data.update({
                            'change_1d': 0.0, 'change_5d': 0.0, 'change_20d': 0.0,
                            'rsi_14': 50.0, 'macd_line': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0,
                            'bb_upper': live_data['current_price'] * 1.02,
                            'bb_middle': live_data['current_price'],
                            'bb_lower': live_data['current_price'] * 0.98,
                            'bb_position': 50.0, 'sma_20': live_data['current_price'],
                            'sma_50': live_data['current_price'], 'volatility_20d': 20.0,
                            'volume_ratio': 1.0
                        })
                except Exception as indicator_error:
                    print(f"Indicator calculation failed for {symbol}: {indicator_error}")
                    # Provide safe defaults
                    live_data.update({
                        'change_1d': 0.0, 'change_5d': 0.0, 'change_20d': 0.0,
                        'rsi_14': 50.0, 'macd_line': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0,
                        'bb_upper': live_data['current_price'] * 1.02,
                        'bb_middle': live_data['current_price'],
                        'bb_lower': live_data['current_price'] * 0.98,
                        'bb_position': 50.0, 'sma_20': live_data['current_price'],
                        'sma_50': live_data['current_price'], 'volatility_20d': 20.0,
                        'volume_ratio': 1.0
                    })
                
                return symbol, live_data, None
                
            except Exception as e:
                print(f"ERROR fetching live data for {symbol}: {str(e)}")
                return symbol, None, str(e)
        
        results = {'success': [], 'failed': []}
        
        print(f"Fetching live data for {len(symbols)} symbols with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(fetch_live_single, symbol): symbol 
                              for symbol in symbols}
            
            for future in as_completed(future_to_symbol):
                symbol, live_data, error = future.result()
                
                if error:
                    print(f"❌ Live data {symbol}: {error}")
                    results['failed'].append({'symbol': symbol, 'error': error})
                else:
                    print(f"✅ Live data {symbol}: ${live_data['current_price']:.2f}")
                    self.store_live_data(live_data)
                    results['success'].append(symbol)
        
        return results
    
    def store_live_data(self, live_data):
        """Store live data in cache table"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                placeholders = ', '.join(['?' for _ in live_data.keys()])
                columns = ', '.join(live_data.keys())
                
                conn.execute(f'''
                    INSERT OR REPLACE INTO live_data_cache ({columns}, last_updated)
                    VALUES ({placeholders}, ?)
                ''', list(live_data.values()) + [datetime.now().isoformat()])
                
                conn.commit()
    
    def get_complete_stock_data(self, symbols):
        """Get complete dataset combining metadata and live data"""
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ','.join(['?' for _ in symbols])
            
            query = f'''
                SELECT 
                    m.symbol,
                    m.company_name,
                    m.sector,
                    m.industry,
                    m.market_cap,
                    l.current_price,
                    l.volume,
                    l.change_1d,
                    l.change_5d,
                    l.change_20d,
                    l.rsi_14,
                    l.macd_line,
                    l.macd_signal,
                    l.macd_histogram,
                    l.bb_upper,
                    l.bb_middle,
                    l.bb_lower,
                    l.bb_position,
                    l.sma_20,
                    l.sma_50,
                    l.volatility_20d,
                    l.volume_ratio,
                    m.last_historical_update,
                    l.last_updated as live_last_updated
                FROM stock_metadata m
                LEFT JOIN live_data_cache l ON m.symbol = l.symbol
                WHERE m.symbol IN ({placeholders})
            '''
            
            df = pd.read_sql_query(query, conn, params=symbols)
            return df
    
    def get_database_stats(self):
        """Get statistics about stored data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count of stocks with historical data
            cursor.execute('SELECT COUNT(DISTINCT symbol) FROM historical_data')
            stocks_with_historical = cursor.fetchone()[0]
            
            # Count of stocks with live data
            cursor.execute('SELECT COUNT(*) FROM live_data_cache')
            stocks_with_live = cursor.fetchone()[0]
            
            # Total historical records
            cursor.execute('SELECT COUNT(*) FROM historical_data')
            total_historical_records = cursor.fetchone()[0]
            
            # Date range of historical data
            cursor.execute('SELECT MIN(date), MAX(date) FROM historical_data')
            date_range = cursor.fetchone()
            
            return {
                'stocks_with_historical': stocks_with_historical,
                'stocks_with_live': stocks_with_live,
                'total_historical_records': total_historical_records,
                'historical_date_range': date_range,
                'database_path': self.db_path
            }

# Usage example and testing
if __name__ == "__main__":
    # Initialize manager
    manager = HistoricalDataManager()
    
    # Test symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    # Check which need historical update
    needs_update = manager.get_stocks_needing_historical_update(test_symbols)
    print(f"Stocks needing historical update: {needs_update}")
    
    if needs_update:
        # Fetch historical data
        print("Fetching historical data...")
        hist_results = manager.fetch_historical_data_parallel(needs_update, max_workers=5)
        print(f"Historical fetch results: {hist_results}")
    
    # Fetch live data
    print("Fetching live data...")
    live_results = manager.fetch_live_data_parallel(test_symbols, max_workers=10)
    print(f"Live fetch results: {live_results}")
    
    # Get complete data
    complete_data = manager.get_complete_stock_data(test_symbols)
    print(f"Complete data shape: {complete_data.shape}")
    
    # Show stats
    stats = manager.get_database_stats()
    print(f"Database stats: {stats}")