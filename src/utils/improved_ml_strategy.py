#!/usr/bin/env python3
"""
Improved ML Strategy - Based on Correlation Analysis
Fix ML signals to work WITH the market patterns, not against them
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class ImprovedMLStrategy:
    """ML strategy based on actual market correlations"""
    
    def __init__(self):
        self.name = "Correlation-Based ML Strategy"
        
    def generate_signals(self, data):
        """Generate improved ML signals based on correlation findings"""
        
        if len(data) < 25:
            return 0.0, 0.5, "Insufficient data"
        
        # Technical indicators
        price = data['close'].iloc[-1]
        sma_5 = data['close'].rolling(5).mean().iloc[-1]
        sma_10 = data['close'].rolling(10).mean().iloc[-1]
        sma_20 = data['close'].rolling(20).mean().iloc[-1]
        vol_avg = data['volume'].rolling(10).mean().iloc[-1]
        vol_current = data['volume'].iloc[-1]
        
        # INSIGHT 1: Momentum had negative correlation = Use CONTRARIAN signals
        # Instead of buying momentum, sell into strength and buy weakness
        
        # Short-term mean reversion (opposite of momentum)
        short_momentum = (price / data['close'].iloc[-6] - 1) if len(data) >= 6 else 0
        contrarian_short = -np.tanh(short_momentum * 15) * 0.3  # Invert the signal
        
        # Long-term mean reversion (strongest negative correlation)
        long_momentum = (price / data['close'].iloc[-21] - 1) if len(data) >= 21 else 0
        contrarian_long = -np.tanh(long_momentum * 8) * 0.4   # Invert and weight heavily
        
        # INSIGHT 2: Volume anomaly had positive correlation = Keep as is
        volume_ratio = vol_current / vol_avg if vol_avg > 0 else 1
        volume_signal = np.tanh((volume_ratio - 1) * 2) * 0.2
        
        # INSIGHT 3: Price position had negative correlation = Use contrarian
        price_position = (price - sma_20) / sma_20 if sma_20 > 0 else 0
        position_contrarian = -np.tanh(price_position * 8) * 0.3  # Invert
        
        # NEW INSIGHT 4: Add trend-following for longer timeframes
        # Use SMA crossover (but be careful about timing)
        trend_signal = 0
        if sma_5 > sma_10 > sma_20:
            trend_signal = 0.2  # Weak bullish
        elif sma_5 < sma_10 < sma_20:
            trend_signal = -0.2  # Weak bearish
        
        # Combine ML components
        ml_signal = contrarian_short + contrarian_long + volume_signal + position_contrarian + trend_signal
        ml_signal = np.clip(ml_signal, -1.0, 1.0)
        
        # Calculate confidence based on signal strength and volume confirmation
        base_confidence = 0.6
        strength_boost = min(0.3, abs(ml_signal) * 0.4)
        volume_boost = min(0.1, abs(volume_ratio - 1) * 0.1)
        
        confidence = base_confidence + strength_boost + volume_boost
        confidence = min(confidence, 0.95)
        
        explanation = f"Contrarian ML: short={contrarian_short:.3f}, long={contrarian_long:.3f}, vol={volume_signal:.3f}"
        
        return ml_signal, confidence, explanation

def test_improved_strategy():
    """Test the improved strategy against baseline"""
    
    print("🚀 IMPROVED ML STRATEGY TEST")
    print("=" * 50)
    print("Using correlation insights: contrarian signals + volume confirmation")
    print()
    
    # Create test data (multiple scenarios)
    scenarios = [
        {"name": "Trending Market", "trend": 0.0005, "vol": 0.015, "seed": 42},
        {"name": "Sideways Market", "trend": 0.0001, "vol": 0.020, "seed": 123},
        {"name": "Volatile Market", "trend": 0.0003, "vol": 0.025, "seed": 456},
    ]
    
    improved_strategy = ImprovedMLStrategy()
    
    total_results = []
    
    for scenario in scenarios:
        print(f"📊 Testing: {scenario['name']}")
        print("-" * 30)
        
        # Create scenario data
        data = create_scenario_data(
            days=252,
            trend=scenario['trend'],
            volatility=scenario['vol'],
            seed=scenario['seed']
        )
        
        # Test strategies
        baseline_result = run_baseline_strategy(data)
        improved_result = run_improved_ml_strategy(data, improved_strategy)
        
        # Compare results
        return_improvement = (improved_result['return'] / baseline_result['return'] - 1) * 100 if baseline_result['return'] != 0 else 0
        sharpe_improvement = (improved_result['sharpe'] / baseline_result['sharpe'] - 1) * 100 if baseline_result['sharpe'] != 0 else 0
        
        print(f"Baseline: {baseline_result['return']*100:+.2f}% return, {baseline_result['sharpe']:+.2f} Sharpe")
        print(f"Improved: {improved_result['return']*100:+.2f}% return, {improved_result['sharpe']:+.2f} Sharpe")
        print(f"Return improvement: {return_improvement:+.1f}%")
        print(f"Sharpe improvement: {sharpe_improvement:+.1f}%")
        
        success = return_improvement > 5 and sharpe_improvement > 5
        print(f"Success: {'✅ YES' if success else '❌ NO'}")
        print()
        
        total_results.append({
            'scenario': scenario['name'],
            'return_improvement': return_improvement,
            'sharpe_improvement': sharpe_improvement,
            'success': success
        })
    
    # Overall results
    print("🎯 OVERALL RESULTS")
    print("=" * 30)
    
    successful_scenarios = sum(1 for r in total_results if r['success'])
    avg_return_improvement = np.mean([r['return_improvement'] for r in total_results])
    avg_sharpe_improvement = np.mean([r['sharpe_improvement'] for r in total_results])
    
    print(f"Successful scenarios: {successful_scenarios}/{len(scenarios)}")
    print(f"Average return improvement: {avg_return_improvement:+.1f}%")
    print(f"Average Sharpe improvement: {avg_sharpe_improvement:+.1f}%")
    
    if successful_scenarios >= 2:
        print(f"\n✅ IMPROVED ML STRATEGY SUCCESS!")
        print("Strategy shows consistent improvement across market conditions")
        return True
    else:
        print(f"\n⚠️ NEEDS FURTHER REFINEMENT")
        print("Strategy shows mixed results - continue optimization")
        return False

def create_scenario_data(days=252, trend=0.0003, volatility=0.015, seed=42):
    """Create market data for specific scenario"""
    
    np.random.seed(seed)
    
    prices = [100]
    volumes = []
    
    for i in range(days):
        # Trend component
        trend_return = trend
        
        # Mean reversion component
        mean_reversion = -0.02 * (prices[-1]/prices[0] - 1.05) if len(prices) > 20 else 0
        
        # Random component
        noise = np.random.normal(0, volatility)
        
        # Occasional regime changes
        if i % 60 == 0:  # Every ~3 months
            regime_shift = np.random.normal(0, 0.01)
        else:
            regime_shift = 0
        
        daily_return = trend_return + mean_reversion + noise + regime_shift
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
        
        # Volume (higher on big moves)
        base_volume = 75000000
        volume_mult = 1 + abs(daily_return) * 5  # Higher volume on big moves
        volume = int(base_volume * volume_mult * np.random.uniform(0.7, 1.3))
        volumes.append(volume)
    
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=days+1),
        'close': prices,
        'volume': [75000000] + volumes
    })

def run_baseline_strategy(data):
    """Run simple baseline strategy"""
    
    returns = []
    
    for i in range(20, len(data)-1):
        current_data = data.iloc[:i+1]
        
        # Simple technical baseline
        price = current_data['close'].iloc[-1]
        sma_10 = current_data['close'].rolling(10).mean().iloc[-1]
        sma_20 = current_data['close'].rolling(20).mean().iloc[-1]
        
        signal = 0
        if price > sma_10: signal += 0.3
        if sma_10 > sma_20: signal += 0.2
        
        signal = np.clip(signal, -0.5, 0.5)
        
        # Calculate return
        next_return = (data['close'].iloc[i+1] / data['close'].iloc[i] - 1)
        returns.append(signal * next_return)
    
    total_return = sum(returns)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
    
    return {'return': total_return, 'sharpe': sharpe}

def run_improved_ml_strategy(data, ml_strategy):
    """Run improved ML strategy"""
    
    returns = []
    
    for i in range(20, len(data)-1):
        current_data = data.iloc[:i+1]
        
        # Get baseline signal
        price = current_data['close'].iloc[-1]
        sma_10 = current_data['close'].rolling(10).mean().iloc[-1]
        sma_20 = current_data['close'].rolling(20).mean().iloc[-1]
        
        baseline_signal = 0
        if price > sma_10: baseline_signal += 0.3
        if sma_10 > sma_20: baseline_signal += 0.2
        baseline_signal = np.clip(baseline_signal, -0.5, 0.5)
        
        # Get ML signal
        ml_signal, ml_confidence, _ = ml_strategy.generate_signals(current_data)
        
        # Conservative combination (80% baseline, 20% ML)
        combined_signal = 0.8 * baseline_signal + 0.2 * ml_signal * ml_confidence
        combined_signal = np.clip(combined_signal, -1.0, 1.0)
        
        # Calculate return
        next_return = (data['close'].iloc[i+1] / data['close'].iloc[i] - 1)
        returns.append(combined_signal * next_return)
    
    total_return = sum(returns)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
    
    return {'return': total_return, 'sharpe': sharpe}

if __name__ == "__main__":
    print("🛠️ IMPROVED ML STRATEGY BASED ON CORRELATION ANALYSIS")
    print("=" * 70)
    print("Key insights: Contrarian signals work better than momentum")
    print("Strategy: 80% baseline + 20% correlation-optimized ML")
    print()
    
    success = test_improved_strategy()
    
    if success:
        print(f"\n🚀 READY FOR PRODUCTION TESTING")
        print("Improved ML strategy shows consistent benefits")
        print("Next step: Implement in actual backtesting system")
    else:
        print(f"\n🔧 CONTINUE OPTIMIZATION")
        print("Strategy needs further refinement before production")
        print("Focus on individual component tuning")
    
    print(f"\n📊 Dashboard: http://localhost:8504")
    print("🔬 Monitor performance in Backtesting tab")