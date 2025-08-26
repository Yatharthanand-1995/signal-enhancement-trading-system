"""
Technical Indicators Calculation System
Optimized for 2-15 day trading strategies with RSI, MACD, Bollinger Bands, and other indicators
"""
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_batch
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
from config.config import config

# Import volume indicators calculator
try:
    from .volume_indicators import VolumeIndicatorCalculator
    VOLUME_INDICATORS_AVAILABLE = True
except ImportError:
    VOLUME_INDICATORS_AVAILABLE = False
    logging.warning("Volume indicators not available - install volume_indicators module")

logger = logging.getLogger(__name__)

class TechnicalIndicatorCalculator:
    """Calculate and store technical indicators optimized for medium-term trading"""
    
    def __init__(self, db_config=None):
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
        self.logger = logger
        
        # Initialize volume indicator calculator
        if VOLUME_INDICATORS_AVAILABLE:
            self.volume_calculator = VolumeIndicatorCalculator()
            self.logger.info("Volume indicators enabled")
        else:
            self.volume_calculator = None
            self.logger.warning("Volume indicators disabled - some features unavailable")
        
        # Optimized parameters for 2-15 day holding periods
        self.params = {
            'rsi_periods': [9, 14],  # 9 for faster signals, 14 for standard
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2.0,
            'sma_periods': [20, 50],
            'ema_periods': [12, 26],
            'atr_period': 14,
            'volume_sma': 20,
            'stoch_k': 14,
            'stoch_d': 3,
            'williams_r': 14,
            # Volume indicator parameters
            'cmf_period': 20,
            'mfi_period': 14,
            'volume_profile_bins': 20
        }
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use exponential moving average for smoother RSI
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal line, and Histogram"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal
        
        return macd, macd_signal, macd_histogram
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                 std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series,
                           k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    def calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series,
                           period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        return williams_r
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for a given price dataframe"""
        if len(df) < 50:  # Need sufficient data for calculations
            self.logger.warning("Insufficient data for indicator calculations")
            return pd.DataFrame()
        
        result_df = df.copy()
        
        # RSI calculations
        for period in self.params['rsi_periods']:
            result_df[f'rsi_{period}'] = self.calculate_rsi(df['close'], period)
        
        # MACD calculations
        macd, macd_signal, macd_histogram = self.calculate_macd(
            df['close'], 
            self.params['macd_fast'],
            self.params['macd_slow'],
            self.params['macd_signal']
        )
        result_df['macd_value'] = macd
        result_df['macd_signal'] = macd_signal
        result_df['macd_histogram'] = macd_histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(
            df['close'], 
            self.params['bb_period'], 
            self.params['bb_std']
        )
        result_df['bb_upper'] = bb_upper
        result_df['bb_middle'] = bb_middle
        result_df['bb_lower'] = bb_lower
        
        # Moving Averages
        for period in self.params['sma_periods']:
            result_df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        for period in self.params['ema_periods']:
            result_df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # ATR
        result_df['atr_14'] = self.calculate_atr(df['high'], df['low'], df['close'], 
                                               self.params['atr_period'])
        
        # Volume indicators
        result_df['volume_sma_20'] = df['volume'].rolling(window=self.params['volume_sma']).mean()
        
        # Stochastic Oscillator
        stoch_k, stoch_d = self.calculate_stochastic(
            df['high'], df['low'], df['close'],
            self.params['stoch_k'], self.params['stoch_d']
        )
        result_df['stoch_k'] = stoch_k
        result_df['stoch_d'] = stoch_d
        
        # Williams %R
        result_df['williams_r'] = self.calculate_williams_r(
            df['high'], df['low'], df['close'], self.params['williams_r']
        )
        
        # NEW: Advanced Volume Indicators
        if self.volume_calculator is not None:
            try:
                self.logger.info("Calculating advanced volume indicators...")
                volume_df = self.volume_calculator.calculate_all_volume_indicators(df)
                
                # Merge volume indicators into result
                volume_cols = [
                    'obv', 'cmf', 'mfi', 'vwap', 'accumulation_distribution',
                    'price_volume_trend', 'volume_ratio_20', 'volume_sma_10', 
                    'volume_ema_20', 'unusual_volume', 'volume_profile'
                ]
                
                for col in volume_cols:
                    if col in volume_df.columns:
                        result_df[col] = volume_df[col]
                
                self.logger.info(f"Added {len(volume_cols)} volume indicators")
                
            except Exception as e:
                self.logger.error(f"Error calculating volume indicators: {str(e)}")
                # Continue without volume indicators
        else:
            self.logger.warning("Volume indicators not available - skipping volume calculations")
        
        return result_df
    
    def calculate_and_store_indicators(self, symbols: Optional[List[str]] = None,
                                     lookback_days: int = 100) -> None:
        """Calculate and store indicators for specified symbols or all active symbols"""
        if symbols:
            symbols_condition = f"AND s.symbol IN ({','.join(['%s'] * len(symbols))})"
            query_params = symbols
        else:
            symbols_condition = ""
            query_params = []
        
        # Get symbols that need indicator updates
        self.cursor.execute(f"""
            SELECT DISTINCT s.id, s.symbol
            FROM securities s
            JOIN daily_prices dp ON s.id = dp.symbol_id
            WHERE s.is_active = true
              AND dp.trade_date >= CURRENT_DATE - INTERVAL '{lookback_days} days'
              {symbols_condition}
        """, query_params)
        
        stocks_to_process = self.cursor.fetchall()
        self.logger.info(f"Processing indicators for {len(stocks_to_process)} stocks")
        
        for symbol_id, symbol in stocks_to_process:
            try:
                self._process_single_stock_indicators(symbol_id, symbol, lookback_days)
                self.logger.info(f"Completed indicators for {symbol}")
            except Exception as e:
                self.logger.error(f"Error processing indicators for {symbol}: {str(e)}")
                self.conn.rollback()
                continue
    
    def _process_single_stock_indicators(self, symbol_id: int, symbol: str, 
                                       lookback_days: int) -> None:
        """Process indicators for a single stock"""
        # Fetch price data
        df = pd.read_sql("""
            SELECT trade_date, open, high, low, close, volume
            FROM daily_prices
            WHERE symbol_id = %s
              AND trade_date >= CURRENT_DATE - INTERVAL '%s days'
            ORDER BY trade_date
        """, self.conn, params=[symbol_id, lookback_days])
        
        if len(df) < 50:
            self.logger.warning(f"Insufficient data for {symbol} - only {len(df)} records")
            return
        
        # Calculate all indicators
        df_with_indicators = self.calculate_all_indicators(df)
        
        if df_with_indicators.empty:
            return
        
        # Prepare records for database insertion
        records = []
        
        # Start from row 50 to avoid NaN values from rolling calculations
        for idx, row in df_with_indicators.iloc[50:].iterrows():
            if pd.notna(row['rsi_14']):  # Only insert if key indicators are valid
                record = (
                    symbol_id,
                    row['trade_date'],
                    self._safe_float(row['rsi_9']),
                    self._safe_float(row['rsi_14']),
                    self._safe_float(row['macd_value']),
                    self._safe_float(row['macd_signal']),
                    self._safe_float(row['macd_histogram']),
                    self._safe_float(row['bb_upper']),
                    self._safe_float(row['bb_middle']),
                    self._safe_float(row['bb_lower']),
                    self._safe_float(row['sma_20']),
                    self._safe_float(row['sma_50']),
                    self._safe_float(row['ema_12']),
                    self._safe_float(row['ema_26']),
                    self._safe_float(row['atr_14']),
                    self._safe_int(row['volume_sma_20']),
                    self._safe_float(row['stoch_k']),
                    self._safe_float(row['stoch_d']),
                    self._safe_float(row['williams_r']),
                    # NEW: Volume indicators
                    self._safe_int(row.get('obv')),
                    self._safe_float(row.get('cmf')),
                    self._safe_float(row.get('mfi')),
                    self._safe_float(row.get('vwap')),
                    self._safe_int(row.get('accumulation_distribution')),
                    self._safe_float(row.get('price_volume_trend')),
                    self._safe_float(row.get('volume_ratio_20')),
                    self._safe_int(row.get('volume_sma_10')),
                    self._safe_int(row.get('volume_ema_20')),
                    bool(row.get('unusual_volume', False)),
                    json.dumps(row.get('volume_profile', {})) if isinstance(row.get('volume_profile'), dict) else '{}'
                )
                records.append(record)
        
        if not records:
            self.logger.warning(f"No valid indicator records for {symbol}")
            return
        
        # Bulk upsert indicators
        execute_batch(self.cursor, """
            INSERT INTO technical_indicators
            (symbol_id, trade_date, rsi_9, rsi_14, macd_value, macd_signal, 
             macd_histogram, bb_upper, bb_middle, bb_lower, sma_20, sma_50, 
             ema_12, ema_26, atr_14, volume_sma_20, stoch_k, stoch_d, williams_r,
             obv, cmf, mfi, vwap, accumulation_distribution, price_volume_trend,
             volume_ratio, volume_sma_10, volume_ema_20, unusual_volume_flag, volume_profile)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol_id, trade_date) 
            DO UPDATE SET
                rsi_9 = EXCLUDED.rsi_9,
                rsi_14 = EXCLUDED.rsi_14,
                macd_value = EXCLUDED.macd_value,
                macd_signal = EXCLUDED.macd_signal,
                macd_histogram = EXCLUDED.macd_histogram,
                bb_upper = EXCLUDED.bb_upper,
                bb_middle = EXCLUDED.bb_middle,
                bb_lower = EXCLUDED.bb_lower,
                sma_20 = EXCLUDED.sma_20,
                sma_50 = EXCLUDED.sma_50,
                ema_12 = EXCLUDED.ema_12,
                ema_26 = EXCLUDED.ema_26,
                atr_14 = EXCLUDED.atr_14,
                volume_sma_20 = EXCLUDED.volume_sma_20,
                stoch_k = EXCLUDED.stoch_k,
                stoch_d = EXCLUDED.stoch_d,
                williams_r = EXCLUDED.williams_r,
                -- NEW: Volume indicator updates
                obv = EXCLUDED.obv,
                cmf = EXCLUDED.cmf,
                mfi = EXCLUDED.mfi,
                vwap = EXCLUDED.vwap,
                accumulation_distribution = EXCLUDED.accumulation_distribution,
                price_volume_trend = EXCLUDED.price_volume_trend,
                volume_ratio = EXCLUDED.volume_ratio,
                volume_sma_10 = EXCLUDED.volume_sma_10,
                volume_ema_20 = EXCLUDED.volume_ema_20,
                unusual_volume_flag = EXCLUDED.unusual_volume_flag,
                volume_profile = EXCLUDED.volume_profile
        """, records, page_size=500)
        
        self.conn.commit()
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert to float, return None for NaN"""
        if pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _safe_int(self, value) -> Optional[int]:
        """Safely convert to int, return None for NaN"""
        if pd.isna(value):
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    
    def get_latest_indicators(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get latest indicators for a specific symbol"""
        query = """
        SELECT 
            s.symbol,
            dp.trade_date,
            dp.close,
            dp.volume,
            ti.rsi_9,
            ti.rsi_14,
            ti.macd_value,
            ti.macd_signal,
            ti.macd_histogram,
            ti.bb_upper,
            ti.bb_middle,
            ti.bb_lower,
            ti.sma_20,
            ti.sma_50,
            ti.ema_12,
            ti.ema_26,
            ti.atr_14,
            ti.volume_sma_20,
            ti.stoch_k,
            ti.stoch_d,
            ti.williams_r
        FROM securities s
        JOIN daily_prices dp ON s.id = dp.symbol_id
        LEFT JOIN technical_indicators ti ON s.id = ti.symbol_id 
            AND dp.trade_date = ti.trade_date
        WHERE s.symbol = %s
          AND dp.trade_date >= CURRENT_DATE - INTERVAL '%s days'
        ORDER BY dp.trade_date DESC
        """
        
        return pd.read_sql(query, self.conn, params=[symbol, days])
    
    def generate_signal_features(self, symbol: str) -> Dict[str, float]:
        """Generate trading signal features based on latest indicators"""
        df = self.get_latest_indicators(symbol, days=5)
        
        if df.empty or len(df) < 2:
            return {}
        
        latest = df.iloc[0]
        prev = df.iloc[1] if len(df) > 1 else latest
        
        features = {}
        
        # RSI signals
        features['rsi_14'] = latest['rsi_14'] if pd.notna(latest['rsi_14']) else 50
        features['rsi_oversold'] = 1 if features['rsi_14'] < 30 else 0
        features['rsi_overbought'] = 1 if features['rsi_14'] > 70 else 0
        
        # MACD signals
        if pd.notna(latest['macd_histogram']) and pd.notna(prev['macd_histogram']):
            features['macd_histogram'] = latest['macd_histogram']
            features['macd_bullish_crossover'] = 1 if (
                latest['macd_histogram'] > 0 and prev['macd_histogram'] <= 0
            ) else 0
            features['macd_bearish_crossover'] = 1 if (
                latest['macd_histogram'] < 0 and prev['macd_histogram'] >= 0
            ) else 0
        else:
            features['macd_histogram'] = 0
            features['macd_bullish_crossover'] = 0
            features['macd_bearish_crossover'] = 0
        
        # Bollinger Bands signals
        if all(pd.notna(latest[col]) for col in ['bb_upper', 'bb_lower', 'close']):
            bb_position = (latest['close'] - latest['bb_lower']) / (
                latest['bb_upper'] - latest['bb_lower']
            )
            features['bb_position'] = bb_position
            features['bb_squeeze'] = 1 if (
                (latest['bb_upper'] - latest['bb_lower']) / latest['close'] < 0.1
            ) else 0
        else:
            features['bb_position'] = 0.5
            features['bb_squeeze'] = 0
        
        # Moving Average signals
        if pd.notna(latest['sma_20']) and pd.notna(latest['sma_50']):
            features['sma_trend'] = 1 if latest['sma_20'] > latest['sma_50'] else 0
            features['price_above_sma20'] = 1 if latest['close'] > latest['sma_20'] else 0
        else:
            features['sma_trend'] = 0
            features['price_above_sma20'] = 0
        
        # Volume indicators
        if pd.notna(latest['volume_sma_20']):
            features['volume_ratio'] = (
                latest['volume'] / latest['volume_sma_20'] 
                if latest['volume_sma_20'] > 0 else 1
            )
        else:
            features['volume_ratio'] = 1
        
        # Stochastic signals
        if pd.notna(latest['stoch_k']):
            features['stoch_k'] = latest['stoch_k']
            features['stoch_oversold'] = 1 if latest['stoch_k'] < 20 else 0
            features['stoch_overbought'] = 1 if latest['stoch_k'] > 80 else 0
        else:
            features['stoch_k'] = 50
            features['stoch_oversold'] = 0
            features['stoch_overbought'] = 0
        
        # Volatility indicator
        if pd.notna(latest['atr_14']):
            features['volatility'] = latest['atr_14'] / latest['close'] if latest['close'] > 0 else 0
        else:
            features['volatility'] = 0
        
        return features
    
    def get_indicator_summary(self) -> Dict[str, any]:
        """Get summary statistics for all indicators"""
        query = """
        SELECT 
            COUNT(DISTINCT symbol_id) as stocks_with_indicators,
            COUNT(*) as total_indicator_records,
            MIN(trade_date) as earliest_date,
            MAX(trade_date) as latest_date,
            AVG(rsi_14) as avg_rsi,
            AVG(ABS(macd_histogram)) as avg_macd_histogram,
            COUNT(CASE WHEN rsi_14 < 30 THEN 1 END) as oversold_count,
            COUNT(CASE WHEN rsi_14 > 70 THEN 1 END) as overbought_count
        FROM technical_indicators
        WHERE trade_date >= CURRENT_DATE - INTERVAL '30 days'
        """
        
        self.cursor.execute(query)
        result = self.cursor.fetchone()
        
        return {
            'stocks_with_indicators': result[0],
            'total_records': result[1],
            'earliest_date': result[2],
            'latest_date': result[3],
            'average_rsi': float(result[4]) if result[4] else None,
            'average_macd_histogram': float(result[5]) if result[5] else None,
            'oversold_stocks': result[6],
            'overbought_stocks': result[7]
        }
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

# Example usage
if __name__ == "__main__":
    calculator = TechnicalIndicatorCalculator()
    
    try:
        # Calculate indicators for top 10 stocks as a test
        test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        calculator.calculate_and_store_indicators(symbols=test_symbols)
        
        # Get summary
        summary = calculator.get_indicator_summary()
        print(f"Indicator summary: {summary}")
        
        # Test signal features for AAPL
        features = calculator.generate_signal_features('AAPL')
        print(f"AAPL signal features: {features}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        
    finally:
        calculator.close()