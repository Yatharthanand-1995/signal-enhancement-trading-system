#!/usr/bin/env python3
"""
Simple test of the backtesting framework
Tests basic functionality without database dependencies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_mock_data(symbols, start_date, end_date, seed=42):
    """Generate mock market data for testing"""
    
    np.random.seed(seed)
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    dates = pd.date_range(start_dt, end_dt, freq='D')
    
    # Filter to business days
    dates = dates[dates.dayofweek < 5]
    
    mock_data = []
    
    for symbol in symbols:
        # Generate realistic price series
        initial_price = np.random.uniform(50, 200)
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        for i, date in enumerate(dates):
            price = prices[i]
            
            # Generate OHLC
            daily_vol = abs(np.random.normal(0, 0.015))
            high = price * (1 + daily_vol)
            low = price * (1 - daily_vol)
            open_price = price * (1 + np.random.normal(0, 0.005))
            close_price = prices[i]
            
            # Generate volume
            volume = int(np.random.uniform(1000000, 10000000))
            
            # Generate technical indicators
            rsi = np.random.uniform(20, 80)
            macd_hist = np.random.normal(0, 0.5)
            sma_20 = price * np.random.uniform(0.95, 1.05)
            sma_50 = price * np.random.uniform(0.90, 1.10)
            
            mock_data.append({
                'symbol': symbol,
                'trade_date': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume,
                'rsi_14': rsi,
                'rsi_9': rsi * 0.9,
                'macd_value': np.random.normal(0, 1),
                'macd_signal': np.random.normal(0, 1),
                'macd_histogram': macd_hist,
                'bb_upper': price * 1.1,
                'bb_middle': price,
                'bb_lower': price * 0.9,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'ema_12': price * 0.98,
                'ema_26': price * 0.97,
                'atr_14': price * 0.02,
                'volume_sma_20': volume * 0.8,
                'stoch_k': np.random.uniform(0, 100),
                'stoch_d': np.random.uniform(0, 100),
                'williams_r': np.random.uniform(-100, 0)
            })
    
    return pd.DataFrame(mock_data)

def test_basic_strategy():
    """Test basic strategy functionality"""
    
    logger.info("Testing basic strategy functionality...")
    
    try:
        from src.backtesting.enhanced_signal_strategy import BaselineStrategy
        
        # Create mock data
        mock_data = generate_mock_data(['TEST'], '2023-01-01', '2023-01-31')
        
        # Test strategy
        strategy = BaselineStrategy()
        signals = strategy.generate_signals(mock_data, datetime(2023, 1, 15))
        
        logger.info(f"Generated {len(signals)} signals")
        
        # Test position sizing
        if signals:
            symbol, signal = list(signals.items())[0]
            position_size = strategy.get_position_size(signal, 100000, 'Low_Volatility')
            logger.info(f"Position size for {symbol}: {position_size}")
        
        # Test exit rules
        exit_rules = strategy.get_exit_rules()
        logger.info(f"Exit rules: {exit_rules}")
        
        logger.info("✅ Basic strategy test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic strategy test failed: {str(e)}")
        return False

def test_enhanced_strategy():
    """Test enhanced strategy with fallback handling"""
    
    logger.info("Testing enhanced strategy functionality...")
    
    try:
        from src.backtesting.enhanced_signal_strategy import EnhancedSignalStrategy
        
        # Create mock data
        mock_data = generate_mock_data(['TEST'], '2023-01-01', '2023-02-28')
        
        # Test strategy (should work with fallback implementations)
        strategy = EnhancedSignalStrategy()
        signals = strategy.generate_signals(mock_data, datetime(2023, 2, 15))
        
        logger.info(f"Generated {len(signals)} enhanced signals")
        
        if signals:
            symbol, signal = list(signals.items())[0]
            logger.info(f"Enhanced signal for {symbol}: {signal}")
            
            # Test position sizing
            position_size = strategy.get_position_size(signal, 100000, 'Low_Volatility')
            logger.info(f"Enhanced position size: {position_size}")
        
        logger.info("✅ Enhanced strategy test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced strategy test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_mock_backtest():
    """Test backtesting with mock data (no database)"""
    
    logger.info("Testing mock backtest functionality...")
    
    try:
        from src.backtesting.enhanced_signal_strategy import BaselineStrategy
        
        # Create a simple mock backtest engine
        class MockBacktestEngine:
            def __init__(self, initial_capital=100000):
                self.initial_capital = initial_capital
            
            def run_simple_test(self, strategy, mock_data):
                """Simple backtest simulation"""
                
                signals_generated = 0
                total_return = 0
                trades = 0
                
                # Get unique dates
                dates = sorted(mock_data['trade_date'].unique())
                
                for date in dates[20:]:  # Skip first 20 days for indicators
                    historical_data = mock_data[mock_data['trade_date'] <= date]
                    signals = strategy.generate_signals(historical_data, date)
                    
                    signals_generated += len(signals)
                    
                    # Simple return calculation
                    if signals:
                        trades += len(signals)
                        # Mock some returns
                        total_return += np.random.normal(0.001, 0.02) * len(signals)
                
                return {
                    'signals_generated': signals_generated,
                    'trades': trades,
                    'total_return': total_return,
                    'avg_return_per_signal': total_return / max(signals_generated, 1)
                }
        
        # Create mock data
        mock_data = generate_mock_data(['AAPL', 'MSFT'], '2023-01-01', '2023-03-31')
        
        # Test with baseline strategy
        engine = MockBacktestEngine()
        strategy = BaselineStrategy()
        
        results = engine.run_simple_test(strategy, mock_data)
        
        logger.info(f"Mock backtest results:")
        logger.info(f"  Signals generated: {results['signals_generated']}")
        logger.info(f"  Total trades: {results['trades']}")
        logger.info(f"  Total return: {results['total_return']:.2%}")
        logger.info(f"  Avg return per signal: {results['avg_return_per_signal']:.2%}")
        
        logger.info("✅ Mock backtest test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Mock backtest test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_framework_tests():
    """Run all framework tests"""
    
    logger.info("=" * 50)
    logger.info("RUNNING BACKTEST FRAMEWORK TESTS")
    logger.info("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Basic strategy
    if test_basic_strategy():
        tests_passed += 1
    
    # Test 2: Enhanced strategy  
    if test_enhanced_strategy():
        tests_passed += 1
    
    # Test 3: Mock backtest
    if test_mock_backtest():
        tests_passed += 1
    
    logger.info("=" * 50)
    logger.info(f"TESTS COMPLETED: {tests_passed}/{total_tests} passed")
    logger.info("=" * 50)
    
    if tests_passed == total_tests:
        logger.info("🎉 All tests passed! Framework is ready for full backtesting.")
        return True
    else:
        logger.warning(f"⚠️ {total_tests - tests_passed} tests failed. Check logs for details.")
        return False

if __name__ == "__main__":
    success = run_framework_tests()
    exit(0 if success else 1)