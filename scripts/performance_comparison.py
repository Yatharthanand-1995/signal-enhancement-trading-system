#!/usr/bin/env python3
"""
Performance Comparison: Enhanced vs Baseline Strategy
Runs simplified performance comparison with mock data to demonstrate improvements
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def generate_realistic_market_data(symbols, start_date, end_date, seed=42):
    """Generate realistic market data with trends and volatility"""
    
    np.random.seed(seed)
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    dates = pd.date_range(start_dt, end_dt, freq='D')
    dates = dates[dates.dayofweek < 5]  # Business days only
    
    data = []
    
    for symbol in symbols:
        # Symbol-specific characteristics
        initial_price = np.random.uniform(80, 150)
        volatility = np.random.uniform(0.15, 0.30)  # Annual volatility
        trend = np.random.uniform(-0.05, 0.15)  # Annual trend
        
        # Generate price series with realistic characteristics
        daily_returns = []
        prices = [initial_price]
        
        for i in range(len(dates)):
            # Base return with trend
            base_return = trend / 252  # Daily trend
            
            # Add volatility
            vol_return = np.random.normal(0, volatility / np.sqrt(252))
            
            # Add mean reversion
            price_vs_ma = (prices[-1] / np.mean(prices[-20:]) if len(prices) > 20 else 1) - 1
            mean_reversion = -0.1 * price_vs_ma  # Weak mean reversion
            
            # Add momentum component
            recent_returns = daily_returns[-5:] if len(daily_returns) >= 5 else [0]
            momentum = 0.05 * np.mean(recent_returns)
            
            total_return = base_return + vol_return + mean_reversion + momentum
            daily_returns.append(total_return)
            
            new_price = prices[-1] * (1 + total_return)
            prices.append(max(new_price, 1.0))  # Prevent negative prices
        
        # Generate full OHLCV data
        for i, date in enumerate(dates):
            price = prices[i + 1]  # +1 because we added initial_price
            daily_vol = abs(np.random.normal(0, 0.01))
            
            high = price * (1 + daily_vol)
            low = price * (1 - daily_vol)
            open_price = prices[i] * (1 + np.random.normal(0, 0.005))
            
            volume = int(np.random.uniform(500000, 5000000))
            
            # Generate technical indicators with some realism
            price_series = pd.Series(prices[max(0, i-20):i+2])
            
            # RSI-like oscillator
            if len(price_series) > 14:
                gains = price_series.diff().clip(lower=0).rolling(14).mean()
                losses = (-price_series.diff().clip(upper=0)).rolling(14).mean()
                rs = gains / (losses + 1e-10)
                rsi = 100 - (100 / (1 + rs.iloc[-1]))
            else:
                rsi = 50 + np.random.normal(0, 15)
            
            rsi = max(0, min(100, rsi))
            
            # MACD-like momentum
            ema_12 = price * np.random.uniform(0.98, 1.02)
            ema_26 = price * np.random.uniform(0.96, 1.04)
            macd_line = ema_12 - ema_26
            macd_signal = macd_line * 0.9  # Simplified signal line
            macd_hist = macd_line - macd_signal
            
            # Moving averages
            sma_20 = np.mean(prices[max(0, i-19):i+2])
            sma_50 = np.mean(prices[max(0, i-49):i+2])
            
            # Bollinger Bands
            bb_middle = sma_20
            bb_std = np.std(prices[max(0, i-19):i+2])
            bb_upper = bb_middle + 2 * bb_std
            bb_lower = bb_middle - 2 * bb_std
            
            data.append({
                'symbol': symbol,
                'trade_date': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume,
                'rsi_14': rsi,
                'rsi_9': rsi * 0.95,
                'macd_value': macd_line,
                'macd_signal': macd_signal,
                'macd_histogram': macd_hist,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'ema_12': ema_12,
                'ema_26': ema_26,
                'atr_14': abs(high - low) * 14,  # Simplified ATR
                'volume_sma_20': volume * np.random.uniform(0.8, 1.2),
                'stoch_k': np.random.uniform(0, 100),
                'stoch_d': np.random.uniform(0, 100),
                'williams_r': np.random.uniform(-100, 0)
            })
    
    return pd.DataFrame(data)

class SimpleBacktester:
    """Simplified backtester for performance comparison"""
    
    def __init__(self, initial_capital=100000, commission=0.005):
        self.initial_capital = initial_capital
        self.commission = commission
    
    def run_backtest(self, strategy, data, symbols):
        """Run simplified backtest"""
        
        portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'trades': [],
            'equity_history': [],
            'dates': []
        }
        
        dates = sorted(data['trade_date'].unique())
        
        for current_date in dates[20:]:  # Skip first 20 days for indicators
            
            # Get historical data up to current date
            historical = data[data['trade_date'] <= current_date]
            current_day = data[data['trade_date'] == current_date]
            
            # Generate signals
            try:
                signals = strategy.generate_signals(historical, current_date)
            except:
                signals = {}
            
            # Calculate current portfolio value
            portfolio_value = portfolio['cash']
            current_prices = {}
            
            for _, row in current_day.iterrows():
                current_prices[row['symbol']] = row['close']
                if row['symbol'] in portfolio['positions']:
                    portfolio_value += portfolio['positions'][row['symbol']]['quantity'] * row['close']
            
            # Record equity
            portfolio['equity_history'].append(portfolio_value)
            portfolio['dates'].append(current_date)
            
            # Process new signals
            for symbol, signal in signals.items():
                if (signal.get('direction') == 'BUY' and 
                    symbol not in portfolio['positions'] and
                    signal.get('strength', 0) > 0.5):
                    
                    # Get price
                    symbol_data = current_day[current_day['symbol'] == symbol]
                    if symbol_data.empty:
                        continue
                    
                    price = symbol_data.iloc[0]['close']
                    
                    # Calculate position size
                    try:
                        position_size = strategy.get_position_size(signal, portfolio_value, 'Normal')
                        
                        if position_size > 0:
                            cost = position_size * price * (1 + self.commission)
                            
                            if cost <= portfolio['cash']:
                                portfolio['cash'] -= cost
                                portfolio['positions'][symbol] = {
                                    'quantity': position_size,
                                    'entry_price': price,
                                    'entry_date': current_date
                                }
                    except:
                        pass
            
            # Check exit conditions (simplified)
            positions_to_close = []
            
            for symbol in list(portfolio['positions'].keys()):
                if symbol in current_prices:
                    position = portfolio['positions'][symbol]
                    current_price = current_prices[symbol]
                    
                    # Simple exit rules
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                    days_held = (current_date - position['entry_date']).days
                    
                    if pnl_pct >= 0.08 or pnl_pct <= -0.04 or days_held >= 20:
                        positions_to_close.append(symbol)
            
            # Close positions
            for symbol in positions_to_close:
                if symbol in portfolio['positions'] and symbol in current_prices:
                    position = portfolio['positions'][symbol]
                    price = current_prices[symbol]
                    
                    proceeds = position['quantity'] * price * (1 - self.commission)
                    portfolio['cash'] += proceeds
                    
                    # Record trade
                    pnl = proceeds - (position['quantity'] * position['entry_price'] * (1 + self.commission))
                    portfolio['trades'].append({
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'exit_date': current_date,
                        'pnl': pnl,
                        'pnl_pct': pnl / (position['quantity'] * position['entry_price'])
                    })
                    
                    del portfolio['positions'][symbol]
        
        # Calculate final metrics
        if portfolio['equity_history']:
            final_value = portfolio['equity_history'][-1]
            total_return = (final_value - self.initial_capital) / self.initial_capital
            
            equity_series = pd.Series(portfolio['equity_history'])
            returns = equity_series.pct_change().dropna()
            
            if len(returns) > 0:
                volatility = returns.std() * np.sqrt(252)
                sharpe = returns.mean() * 252 / (volatility + 1e-10)
                
                # Max drawdown
                peak = equity_series.expanding().max()
                drawdown = (equity_series - peak) / peak
                max_drawdown = drawdown.min()
            else:
                volatility = sharpe = max_drawdown = 0
            
            # Trade statistics
            trades = portfolio['trades']
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            win_rate = winning_trades / len(trades) if trades else 0
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'volatility': volatility,
                'max_drawdown': abs(max_drawdown),
                'win_rate': win_rate,
                'total_trades': len(trades),
                'final_value': final_value
            }
        else:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'volatility': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'total_trades': 0,
                'final_value': self.initial_capital
            }

def run_performance_comparison():
    """Run comprehensive performance comparison"""
    
    logger.info("="*60)
    logger.info("ENHANCED SIGNAL SYSTEM PERFORMANCE COMPARISON")
    logger.info("="*60)
    
    # Test parameters
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'TSLA']
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    logger.info(f"Testing period: {start_date} to {end_date}")
    logger.info(f"Universe: {symbols}")
    logger.info("")
    
    # Generate realistic market data
    logger.info("Generating realistic market data...")
    market_data = generate_realistic_market_data(symbols, start_date, end_date)
    logger.info(f"Generated {len(market_data)} data points")
    
    # Initialize backtester
    backtester = SimpleBacktester(initial_capital=100000)
    
    # Test 1: Baseline Strategy
    logger.info("\n" + "-"*40)
    logger.info("TESTING BASELINE STRATEGY")
    logger.info("-"*40)
    
    try:
        from src.backtesting.enhanced_signal_strategy import BaselineStrategy
        
        baseline_strategy = BaselineStrategy()
        baseline_results = backtester.run_backtest(baseline_strategy, market_data, symbols)
        
        logger.info("Baseline Results:")
        logger.info(f"  Total Return:     {baseline_results['total_return']:.2%}")
        logger.info(f"  Sharpe Ratio:     {baseline_results['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown:     {baseline_results['max_drawdown']:.2%}")
        logger.info(f"  Win Rate:         {baseline_results['win_rate']:.2%}")
        logger.info(f"  Total Trades:     {baseline_results['total_trades']}")
        logger.info(f"  Final Value:      ${baseline_results['final_value']:,.2f}")
        
    except Exception as e:
        logger.error(f"Error testing baseline strategy: {str(e)}")
        baseline_results = None
    
    # Test 2: Enhanced Strategy
    logger.info("\n" + "-"*40)
    logger.info("TESTING ENHANCED STRATEGY")
    logger.info("-"*40)
    
    try:
        from src.backtesting.enhanced_signal_strategy import EnhancedSignalStrategy
        
        enhanced_strategy = EnhancedSignalStrategy()
        enhanced_results = backtester.run_backtest(enhanced_strategy, market_data, symbols)
        
        logger.info("Enhanced Results:")
        logger.info(f"  Total Return:     {enhanced_results['total_return']:.2%}")
        logger.info(f"  Sharpe Ratio:     {enhanced_results['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown:     {enhanced_results['max_drawdown']:.2%}")
        logger.info(f"  Win Rate:         {enhanced_results['win_rate']:.2%}")
        logger.info(f"  Total Trades:     {enhanced_results['total_trades']}")
        logger.info(f"  Final Value:      ${enhanced_results['final_value']:,.2f}")
        
    except Exception as e:
        logger.error(f"Error testing enhanced strategy: {str(e)}")
        enhanced_results = None
    
    # Comparison Analysis
    if baseline_results and enhanced_results:
        logger.info("\n" + "="*40)
        logger.info("PERFORMANCE COMPARISON ANALYSIS")
        logger.info("="*40)
        
        return_improvement = enhanced_results['total_return'] - baseline_results['total_return']
        sharpe_improvement = enhanced_results['sharpe_ratio'] - baseline_results['sharpe_ratio']
        dd_improvement = baseline_results['max_drawdown'] - enhanced_results['max_drawdown']
        wr_improvement = enhanced_results['win_rate'] - baseline_results['win_rate']
        
        logger.info("IMPROVEMENTS (Enhanced - Baseline):")
        logger.info(f"  Return Improvement:    {return_improvement:+.2%}")
        logger.info(f"  Sharpe Improvement:    {sharpe_improvement:+.2f}")
        logger.info(f"  Drawdown Improvement:  {dd_improvement:+.2%}")
        logger.info(f"  Win Rate Improvement:  {wr_improvement:+.2%}")
        logger.info("")
        
        # Calculate relative improvements
        if baseline_results['total_return'] != 0:
            relative_return = return_improvement / abs(baseline_results['total_return'])
            logger.info(f"  Relative Return Improvement: {relative_return:.1%}")
        
        if baseline_results['sharpe_ratio'] != 0:
            relative_sharpe = sharpe_improvement / abs(baseline_results['sharpe_ratio'])
            logger.info(f"  Relative Sharpe Improvement: {relative_sharpe:.1%}")
        
        logger.info("")
        
        # Generate conclusions
        logger.info("CONCLUSIONS:")
        
        if return_improvement > 0.02:  # 2% improvement
            logger.info("✅ Enhanced system shows SIGNIFICANT return improvement")
        elif return_improvement > 0:
            logger.info("✅ Enhanced system shows positive return improvement")
        else:
            logger.info("⚠️  Enhanced system underperformed on returns")
        
        if sharpe_improvement > 0.2:
            logger.info("✅ Enhanced system shows SIGNIFICANT risk-adjusted improvement")
        elif sharpe_improvement > 0:
            logger.info("✅ Enhanced system shows positive risk-adjusted improvement")
        else:
            logger.info("⚠️  Enhanced system underperformed on risk-adjusted returns")
        
        if dd_improvement > 0.01:  # 1% better drawdown
            logger.info("✅ Enhanced system demonstrates better risk management")
        elif dd_improvement > 0:
            logger.info("✅ Enhanced system shows improved risk management")
        else:
            logger.info("⚠️  Enhanced system had higher drawdown")
        
        # Overall assessment
        improvements = sum([
            return_improvement > 0,
            sharpe_improvement > 0,
            dd_improvement > 0,
            wr_improvement > 0
        ])
        
        logger.info("")
        logger.info(f"OVERALL ASSESSMENT: {improvements}/4 metrics improved")
        
        if improvements >= 3:
            logger.info("🎉 RECOMMENDATION: Enhanced system ready for deployment")
        elif improvements >= 2:
            logger.info("✅ RECOMMENDATION: Enhanced system shows promise, consider further optimization")
        else:
            logger.info("⚠️  RECOMMENDATION: Enhanced system needs further refinement")
    
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE COMPARISON COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    run_performance_comparison()