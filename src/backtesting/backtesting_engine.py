#!/usr/bin/env python3
"""
Comprehensive Backtesting Engine for Signal Logic Validation
Backtests our signal generation system across different market regimes (2020-2025)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import logging
from typing import Dict, List, Tuple, Optional
import sqlite3
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class BacktestingEngine:
    """
    Comprehensive backtesting engine for validating signal logic performance
    """
    
    def __init__(self, start_date: str = "2020-01-01", end_date: str = "2025-08-30"):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.data_cache_dir = "cache/"
        self.results_dir = "results/"
        
        # Ensure directories exist
        os.makedirs(self.data_cache_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Portfolio parameters
        self.initial_capital = 1_000_000  # $1M starting capital
        self.max_positions = 20
        self.transaction_cost = 0.001  # 10 bps per trade
        self.rebalance_frequency = "W-FRI"  # Weekly on Fridays
        
        # Risk management parameters
        self.max_position_pct = 0.05  # 5% max per position
        self.max_sector_pct = 0.25    # 25% max per sector
        self.stop_loss_pct = 0.10     # 10% stop loss
        
        logger.info(f"BacktestingEngine initialized: {start_date} to {end_date}")
        
    def get_sp500_top100(self, date: pd.Timestamp = None) -> List[str]:
        """
        Get top 100 S&P 500 stocks by market cap for a given date
        For simplicity, using current top 100 - in production would use historical constituents
        """
        # Top 100 largest S&P 500 companies (approximate)
        top100_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH',
            'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC', 'ABBV',
            'PFE', 'AVGO', 'KO', 'COST', 'DIS', 'TMO', 'WMT', 'LIN', 'ACN', 'MRK',
            'DHR', 'VZ', 'ADBE', 'NFLX', 'CRM', 'TXN', 'NKE', 'RTX', 'QCOM', 'AMD',
            'NEE', 'PM', 'UPS', 'LOW', 'T', 'SPGI', 'INTU', 'HON', 'IBM', 'CAT',
            'AMGN', 'AMAT', 'GS', 'BKNG', 'ELV', 'ISRG', 'SYK', 'TJX', 'AXP', 'VRTX',
            'DE', 'BLK', 'MDLZ', 'ADI', 'GILD', 'LRCX', 'C', 'CVS', 'TMUS', 'SCHW',
            'BMY', 'SO', 'ETN', 'BSX', 'MU', 'ZTS', 'FIS', 'REGN', 'SLB', 'PYPL',
            'MMC', 'DUK', 'CSX', 'AON', 'EQIX', 'CL', 'SNPS', 'ITW', 'WM', 'APD',
            'CME', 'ICE', 'FDX', 'NOC', 'PLD', 'SHW', 'GD', 'EMR', 'USB', 'NSC'
        ]
        
        logger.info(f"Retrieved {len(top100_symbols)} stocks for backtesting")
        return top100_symbols
    
    def download_historical_data(self, symbols: List[str], force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Download historical data for all symbols with caching
        """
        cache_file = os.path.join(self.data_cache_dir, f"historical_data_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}.pkl")
        
        # Load from cache if exists and not forcing refresh
        if os.path.exists(cache_file) and not force_refresh:
            logger.info("Loading historical data from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        logger.info(f"Downloading historical data for {len(symbols)} symbols...")
        historical_data = {}
        failed_symbols = []
        
        def download_symbol(symbol):
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=self.start_date, end=self.end_date)
                
                if len(hist) > 0:
                    # Add basic info
                    info = ticker.info
                    hist['symbol'] = symbol
                    hist['sector'] = info.get('sector', 'Unknown')
                    hist['industry'] = info.get('industry', 'Unknown')
                    hist['market_cap'] = info.get('marketCap', 0)
                    
                    return symbol, hist
                else:
                    return symbol, None
                    
            except Exception as e:
                logger.warning(f"Failed to download {symbol}: {e}")
                return symbol, None
        
        # Download data in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(download_symbol, symbols))
        
        for symbol, data in results:
            if data is not None:
                historical_data[symbol] = data
            else:
                failed_symbols.append(symbol)
        
        logger.info(f"Successfully downloaded {len(historical_data)} symbols")
        if failed_symbols:
            logger.warning(f"Failed to download: {failed_symbols}")
        
        # Cache the results
        with open(cache_file, 'wb') as f:
            pickle.dump(historical_data, f)
        
        return historical_data
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators needed for signal generation
        This replicates the exact logic from our dashboard
        """
        df = data.copy()
        
        # Basic returns
        df['returns_1d'] = df['Close'].pct_change() * 100
        df['returns_5d'] = df['Close'].pct_change(5) * 100
        
        # Volume analysis
        df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
        
        # RSI (14-period)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['Close'].rolling(window=bb_period).mean()
        df['bb_std'] = df['Close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * bb_std)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * bb_std)
        df['bb_position'] = ((df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])) * 100
        df['bb_position'] = df['bb_position'].fillna(50)
        
        # Volatility (20-day annualized)
        df['volatility_20d'] = df['returns_1d'].rolling(window=20).std() * np.sqrt(252)
        
        # Clean up any NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def get_market_environment(self, date: pd.Timestamp) -> Dict:
        """
        Get market environment data for a specific date
        This includes VIX, sentiment, breadth, etc.
        """
        # For backtesting, we'll use simplified market environment
        # In production, this would fetch real historical VIX, sentiment data
        
        # Simplified market environment based on S&P 500 behavior
        # This is a placeholder - would be enhanced with real data
        market_env = {
            'vix_level': 20,  # Default VIX level
            'breadth_health': 'Good',  # Default breadth
            'fear_greed_index': 50,  # Neutral sentiment
            'risk_environment': 'Normal'  # Default risk
        }
        
        # Adjust based on historical periods (simplified)
        if date >= pd.Timestamp('2020-02-15') and date <= pd.Timestamp('2020-04-15'):
            # COVID crash period
            market_env.update({
                'vix_level': 50,
                'breadth_health': 'Poor',
                'fear_greed_index': 10,
                'risk_environment': 'High Risk'
            })
        elif date >= pd.Timestamp('2021-12-01') and date <= pd.Timestamp('2022-10-31'):
            # Bear market 2022
            market_env.update({
                'vix_level': 28,
                'breadth_health': 'Poor',
                'fear_greed_index': 25,
                'risk_environment': 'Elevated Risk'
            })
        elif date >= pd.Timestamp('2024-01-01'):
            # Current AI boom period - elevated valuations
            market_env.update({
                'vix_level': 16,
                'breadth_health': 'Moderate',
                'fear_greed_index': 70,
                'risk_environment': 'Normal'
            })
        
        return market_env
    
    def classify_market_regime(self, date: pd.Timestamp, market_data: Dict = None) -> str:
        """
        Classify the market regime for a given date
        """
        # Normalize timezone for comparison
        if date.tz is not None:
            date = date.tz_localize(None)
            
        # Historical regime classification
        if date >= pd.Timestamp('2020-02-15') and date <= pd.Timestamp('2020-03-31'):
            return "COVID_CRASH"
        elif date >= pd.Timestamp('2020-04-01') and date <= pd.Timestamp('2021-12-31'):
            return "COVID_RECOVERY"
        elif date >= pd.Timestamp('2021-01-01') and date <= pd.Timestamp('2021-11-30'):
            return "INFLATION_PERIOD"
        elif date >= pd.Timestamp('2021-12-01') and date <= pd.Timestamp('2022-10-31'):
            return "BEAR_MARKET"
        elif date >= pd.Timestamp('2022-11-01') and date <= pd.Timestamp('2023-12-31'):
            return "FED_PIVOT_RECOVERY"
        elif date >= pd.Timestamp('2024-01-01'):
            return "AI_BOOM_CURRENT"
        else:
            return "NORMAL"
    
    def calculate_individual_signals(self, row: pd.Series, market_env: Dict) -> Dict:
        """
        Calculate individual signal components - exact copy from dashboard logic
        """
        signals = {}
        
        # 1. RSI Signal (15% weight)
        rsi = row['rsi_14']
        if rsi < 25:
            signals['rsi'] = {'value': 0.9, 'interpretation': 'Extremely Oversold', 'color': 'positive'}
        elif rsi < 30:
            signals['rsi'] = {'value': 0.8, 'interpretation': 'Oversold', 'color': 'positive'}
        elif rsi < 40:
            signals['rsi'] = {'value': 0.7, 'interpretation': 'Moderately Oversold', 'color': 'positive'}
        elif rsi > 75:
            signals['rsi'] = {'value': 0.1, 'interpretation': 'Extremely Overbought', 'color': 'negative'}
        elif rsi > 70:
            signals['rsi'] = {'value': 0.2, 'interpretation': 'Overbought', 'color': 'negative'}
        elif rsi > 60:
            signals['rsi'] = {'value': 0.3, 'interpretation': 'Moderately Overbought', 'color': 'negative'}
        else:
            signals['rsi'] = {'value': 0.5, 'interpretation': 'Neutral', 'color': 'neutral'}
        
        # 2. MACD Signal (13% weight)
        macd_hist = row['macd_histogram']
        if macd_hist > 0.5:
            signals['macd'] = {'value': 0.8, 'interpretation': 'Strong Bullish Momentum', 'color': 'positive'}
        elif macd_hist > 0.1:
            signals['macd'] = {'value': 0.7, 'interpretation': 'Bullish Momentum', 'color': 'positive'}
        elif macd_hist > 0:
            signals['macd'] = {'value': 0.6, 'interpretation': 'Weak Bullish', 'color': 'positive'}
        elif macd_hist < -0.5:
            signals['macd'] = {'value': 0.2, 'interpretation': 'Strong Bearish Momentum', 'color': 'negative'}
        elif macd_hist < -0.1:
            signals['macd'] = {'value': 0.3, 'interpretation': 'Bearish Momentum', 'color': 'negative'}
        elif macd_hist < 0:
            signals['macd'] = {'value': 0.4, 'interpretation': 'Weak Bearish', 'color': 'negative'}
        else:
            signals['macd'] = {'value': 0.5, 'interpretation': 'Neutral', 'color': 'neutral'}
        
        # 3. Volume Signal (12% weight)
        vol_ratio = row['volume_ratio']
        if vol_ratio > 2.5:
            signals['volume'] = {'value': 0.85, 'interpretation': 'Very High Volume', 'color': 'positive'}
        elif vol_ratio > 1.8:
            signals['volume'] = {'value': 0.75, 'interpretation': 'High Volume', 'color': 'positive'}
        elif vol_ratio > 1.3:
            signals['volume'] = {'value': 0.65, 'interpretation': 'Above Average Volume', 'color': 'positive'}
        elif vol_ratio < 0.7:
            signals['volume'] = {'value': 0.35, 'interpretation': 'Low Volume Concern', 'color': 'negative'}
        elif vol_ratio < 0.5:
            signals['volume'] = {'value': 0.25, 'interpretation': 'Very Low Volume', 'color': 'negative'}
        else:
            signals['volume'] = {'value': 0.5, 'interpretation': 'Normal Volume', 'color': 'neutral'}
        
        # 4. Bollinger Bands Signal (11% weight)
        bb_pos = row['bb_position']
        if bb_pos < 5:
            signals['bb'] = {'value': 0.8, 'interpretation': 'Below Lower Band', 'color': 'positive'}
        elif bb_pos < 25:
            signals['bb'] = {'value': 0.7, 'interpretation': 'Near Lower Band', 'color': 'positive'}
        elif bb_pos > 95:
            signals['bb'] = {'value': 0.2, 'interpretation': 'Above Upper Band', 'color': 'negative'}
        elif bb_pos > 75:
            signals['bb'] = {'value': 0.3, 'interpretation': 'Near Upper Band', 'color': 'negative'}
        else:
            signals['bb'] = {'value': 0.5, 'interpretation': 'Middle of Bands', 'color': 'neutral'}
        
        # 5. Moving Average Signal (10% weight)
        if row['Close'] > row['sma_20'] > row['sma_50']:
            signals['ma'] = {'value': 0.8, 'interpretation': 'Strong Uptrend', 'color': 'positive'}
        elif row['Close'] > row['sma_20']:
            signals['ma'] = {'value': 0.65, 'interpretation': 'Above SMA20', 'color': 'positive'}
        elif row['Close'] < row['sma_20'] < row['sma_50']:
            signals['ma'] = {'value': 0.2, 'interpretation': 'Strong Downtrend', 'color': 'negative'}
        elif row['Close'] < row['sma_20']:
            signals['ma'] = {'value': 0.35, 'interpretation': 'Below SMA20', 'color': 'negative'}
        else:
            signals['ma'] = {'value': 0.5, 'interpretation': 'Around Moving Averages', 'color': 'neutral'}
        
        # 6. Momentum Signal (8% weight)
        momentum = row['returns_1d']
        if momentum > 5:
            signals['momentum'] = {'value': 0.8, 'interpretation': 'Strong Positive Momentum', 'color': 'positive'}
        elif momentum > 2:
            signals['momentum'] = {'value': 0.65, 'interpretation': 'Positive Momentum', 'color': 'positive'}
        elif momentum < -5:
            signals['momentum'] = {'value': 0.2, 'interpretation': 'Strong Negative Momentum', 'color': 'negative'}
        elif momentum < -2:
            signals['momentum'] = {'value': 0.35, 'interpretation': 'Negative Momentum', 'color': 'negative'}
        else:
            signals['momentum'] = {'value': 0.5, 'interpretation': 'Neutral Momentum', 'color': 'neutral'}
        
        # 7. Volatility Signal (6% weight)
        vol = row['volatility_20d']
        vix_level = market_env['vix_level']
        if vol > vix_level * 2:
            signals['volatility'] = {'value': 0.3, 'interpretation': 'Very High Volatility', 'color': 'negative'}
        elif vol > vix_level * 1.5:
            signals['volatility'] = {'value': 0.4, 'interpretation': 'High Volatility', 'color': 'negative'}
        elif vol < vix_level * 0.5:
            signals['volatility'] = {'value': 0.6, 'interpretation': 'Low Volatility', 'color': 'positive'}
        else:
            signals['volatility'] = {'value': 0.5, 'interpretation': 'Normal Volatility', 'color': 'neutral'}
        
        # 8. ML Signal (20% weight)
        ml_score = (
            signals['rsi']['value'] * 0.3 +
            signals['macd']['value'] * 0.25 +
            signals['volume']['value'] * 0.2 +
            signals['bb']['value'] * 0.15 +
            signals['ma']['value'] * 0.1
        )
        
        if ml_score > 0.75:
            signals['ml_signal'] = {'value': 0.9, 'interpretation': 'ML Model: Strong Buy', 'color': 'positive'}
        elif ml_score > 0.6:
            signals['ml_signal'] = {'value': 0.75, 'interpretation': 'ML Model: Buy', 'color': 'positive'}
        elif ml_score < 0.25:
            signals['ml_signal'] = {'value': 0.1, 'interpretation': 'ML Model: Strong Sell', 'color': 'negative'}
        elif ml_score < 0.4:
            signals['ml_signal'] = {'value': 0.25, 'interpretation': 'ML Model: Sell', 'color': 'negative'}
        else:
            signals['ml_signal'] = {'value': 0.5, 'interpretation': 'ML Model: Neutral', 'color': 'neutral'}
        
        # 9. Other (5% weight)
        signals['other'] = {'value': 0.5, 'interpretation': 'Reserved', 'color': 'neutral'}
        
        return signals
    
    def generate_signal_for_date(self, symbol: str, data: pd.DataFrame, date: pd.Timestamp) -> Dict:
        """
        Generate signal for a specific stock on a specific date
        """
        try:
            # Get data for this date
            if date not in data.index:
                return None
            
            row = data.loc[date]
            market_env = self.get_market_environment(date)
            
            # Calculate individual signals
            individual_signals = self.calculate_individual_signals(row, market_env)
            
            # Define weights
            weights = {
                'rsi': 0.15, 'macd': 0.13, 'volume': 0.12, 'bb': 0.11,
                'ma': 0.10, 'momentum': 0.08, 'volatility': 0.06, 'ml_signal': 0.20, 'other': 0.05
            }
            
            # Calculate raw score
            raw_score = sum(individual_signals[indicator]['value'] * weight 
                          for indicator, weight in weights.items())
            
            # Apply market environment filters (simplified for backtesting)
            vix_adjustment = 0.85 if market_env['vix_level'] > 25 else 0.92 if market_env['vix_level'] > 20 else 1.0
            breadth_adjustment = 0.8 if market_env['breadth_health'] == 'Poor' else 0.95 if market_env['breadth_health'] == 'Moderate' else 1.0
            
            final_score = raw_score * vix_adjustment * breadth_adjustment
            
            # Dynamic thresholds based on market environment
            if market_env['vix_level'] > 25:
                thresholds = {'strong_buy': 0.70, 'buy': 0.58, 'sell': 0.42, 'strong_sell': 0.35}
            elif market_env['vix_level'] > 20:
                thresholds = {'strong_buy': 0.68, 'buy': 0.55, 'sell': 0.45, 'strong_sell': 0.38}
            else:
                thresholds = {'strong_buy': 0.65, 'buy': 0.52, 'sell': 0.48, 'strong_sell': 0.40}
            
            # Determine signal
            if final_score > thresholds['strong_buy']:
                direction, strength = "STRONG_BUY", "Strong"
            elif final_score > thresholds['buy']:
                direction, strength = "BUY", "Moderate"
            elif final_score < thresholds['strong_sell']:
                direction, strength = "STRONG_SELL", "Strong"
            elif final_score < thresholds['sell']:
                direction, strength = "SELL", "Moderate"
            else:
                direction, strength = "HOLD", "Neutral"
            
            # Calculate confidence
            confidence = min(0.95, abs(final_score - 0.5) * 1.8)
            
            return {
                'symbol': symbol,
                'date': date,
                'signal': direction,
                'strength': strength,
                'raw_score': raw_score,
                'final_score': final_score,
                'confidence': confidence,
                'price': row['Close'],
                'volume': row['Volume'],
                'market_regime': self.classify_market_regime(date),
                'market_env': market_env,
                'individual_signals': individual_signals
            }
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol} on {date}: {e}")
            return None
    
    def run_signal_generation(self, historical_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate signals for all stocks across all dates
        """
        logger.info("Generating signals for all stocks...")
        all_signals = []
        
        # Generate signals for each stock
        for symbol, data in historical_data.items():
            logger.info(f"Processing signals for {symbol}...")
            
            # Generate signals for each date
            for date in data.index:
                # Normalize timezone
                normalized_date = date.tz_localize(None) if date.tz is not None else date
                
                if normalized_date.weekday() == 4:  # Only Fridays for weekly rebalancing
                    signal = self.generate_signal_for_date(symbol, data, date)
                    if signal:
                        all_signals.append(signal)
        
        signals_df = pd.DataFrame(all_signals)
        logger.info(f"Generated {len(signals_df)} signals")
        
        return signals_df

def main():
    """
    Main backtesting execution
    """
    logger.info("Starting backtesting engine...")
    
    # Initialize backtesting engine
    engine = BacktestingEngine(start_date="2020-01-01", end_date="2025-08-30")
    
    # Get stock universe
    symbols = engine.get_sp500_top100()
    
    # Download historical data
    historical_data = engine.download_historical_data(symbols, force_refresh=False)
    
    # Calculate technical indicators for all stocks
    logger.info("Calculating technical indicators...")
    for symbol in historical_data:
        historical_data[symbol] = engine.calculate_technical_indicators(historical_data[symbol])
    
    # Generate signals
    signals_df = engine.run_signal_generation(historical_data)
    
    # Save signals for further analysis
    signals_file = os.path.join(engine.results_dir, "generated_signals.csv")
    signals_df.to_csv(signals_file, index=False)
    logger.info(f"Signals saved to {signals_file}")
    
    # Quick analysis
    signal_counts = signals_df['signal'].value_counts()
    regime_analysis = signals_df.groupby('market_regime')['signal'].value_counts()
    
    logger.info("Signal Generation Summary:")
    logger.info(f"Total signals: {len(signals_df)}")
    logger.info(f"Signal distribution:\n{signal_counts}")
    logger.info(f"Regime analysis:\n{regime_analysis}")

if __name__ == "__main__":
    main()