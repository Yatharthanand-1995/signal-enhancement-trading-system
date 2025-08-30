#!/usr/bin/env python3
"""
Analyze ML Performance Issues
Debug why ML-enhanced strategy underperformed baseline
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_debug_data():
    """Create detailed test data for analysis"""
    np.random.seed(42)  # Consistent results
    days = 252
    
    # Create realistic market data with known patterns
    base_return = 0.0003  # ~7.5% annual
    daily_vol = 0.015     # ~24% annual volatility
    
    returns = []
    prices = [100]
    volumes = []
    
    for i in range(days):
        # Add predictable patterns for testing
        trend_return = base_return
        
        # Add cyclical pattern (every 20 days)
        cycle_return = 0.001 * np.sin(i * 2 * np.pi / 20)
        
        # Add mean reversion
        mean_reversion = -0.05 * (prices[-1]/prices[0] - 1.02) if len(prices) > 50 else 0
        
        # Add noise
        noise = np.random.normal(0, daily_vol)
        
        daily_return = trend_return + cycle_return + mean_reversion + noise
        new_price = prices[-1] * (1 + daily_return)
        
        prices.append(new_price)
        returns.append(daily_return)
        
        # Volume with patterns (higher volume on big moves)
        base_volume = 75000000
        volume_noise = np.random.uniform(0.5, 1.5)
        volume_spike = 2.0 if abs(daily_return) > 0.02 else 1.0
        volume = int(base_volume * volume_noise * volume_spike)
        volumes.append(volume)
    
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=days+1),
        'close': prices,
        'volume': [75000000] + volumes,  # Add initial volume
        'actual_return': [0] + returns  # Add initial return
    })
    
    return data

def analyze_signal_components(data):
    """Analyze each signal component individually"""
    
    print("🔍 SIGNAL COMPONENT ANALYSIS")
    print("=" * 50)
    print("Testing each ML component against future returns")
    print()
    
    results = []
    
    for i in range(20, len(data)-5):  # Leave room for forward returns
        current_data = data.iloc[:i+1]
        
        # Calculate forward return (5-day)
        forward_return = (data['close'].iloc[i+5] / data['close'].iloc[i] - 1)
        
        # Technical indicators
        price = current_data['close'].iloc[-1]
        sma_10 = current_data['close'].rolling(10).mean().iloc[-1]
        sma_20 = current_data['close'].rolling(20).mean().iloc[-1]
        vol_avg = current_data['volume'].rolling(10).mean().iloc[-1]
        vol_current = current_data['volume'].iloc[-1]
        
        # BASELINE COMPONENTS
        baseline_price_signal = 1 if price > sma_10 else -1
        baseline_trend_signal = 1 if sma_10 > sma_20 else -1
        
        # ML COMPONENTS  
        # 1. Short-term momentum
        momentum_5d = (price / current_data['close'].iloc[-6] - 1) if i >= 25 else 0
        momentum_signal = np.tanh(momentum_5d * 20)
        
        # 2. Longer-term trend
        momentum_20d = (price / current_data['close'].iloc[-21] - 1) if i >= 40 else 0
        trend_signal = np.tanh(momentum_20d * 5)
        
        # 3. Volume anomaly
        volume_ratio = vol_current / vol_avg
        volume_signal = np.tanh((volume_ratio - 1) * 3)
        
        # 4. Price position relative to moving average
        price_position = (price - sma_20) / sma_20
        position_signal = np.tanh(price_position * 10)
        
        results.append({
            'forward_return': forward_return,
            'baseline_price': baseline_price_signal,
            'baseline_trend': baseline_trend_signal,
            'ml_momentum_5d': momentum_signal,
            'ml_trend_20d': trend_signal,
            'ml_volume': volume_signal,
            'ml_position': position_signal,
            'raw_momentum_5d': momentum_5d,
            'raw_momentum_20d': momentum_20d,
            'raw_volume_ratio': volume_ratio
        })
    
    df = pd.DataFrame(results)
    
    # Calculate correlations with forward returns
    print("📊 SIGNAL CORRELATIONS WITH FUTURE RETURNS")
    print("-" * 45)
    
    correlations = {}
    for col in df.columns:
        if col != 'forward_return':
            corr = df[col].corr(df['forward_return'])
            correlations[col] = corr
            direction = "📈" if corr > 0.05 else "📉" if corr < -0.05 else "➡️"
            print(f"{col:<20}: {corr:+.3f} {direction}")
    
    # Find best and worst signals
    best_signal = max(correlations.items(), key=lambda x: abs(x[1]))
    worst_signal = min(correlations.items(), key=lambda x: abs(x[1]))
    
    print(f"\n🏆 BEST SIGNAL: {best_signal[0]} ({best_signal[1]:+.3f})")
    print(f"❌ WORST SIGNAL: {worst_signal[0]} ({worst_signal[1]:+.3f})")
    
    return df, correlations

def test_optimal_weights(data, correlations):
    """Find optimal weight combinations based on correlations"""
    
    print(f"\n⚖️ OPTIMAL WEIGHT TESTING")
    print("=" * 40)
    
    # Test different weight combinations
    weight_tests = [
        {"name": "Current (60/40)", "baseline_weight": 0.6, "ml_weight": 0.4},
        {"name": "Conservative (80/20)", "baseline_weight": 0.8, "ml_weight": 0.2},
        {"name": "Aggressive (40/60)", "baseline_weight": 0.4, "ml_weight": 0.6},
        {"name": "Equal (50/50)", "baseline_weight": 0.5, "ml_weight": 0.5},
        {"name": "ML Dominant (20/80)", "baseline_weight": 0.2, "ml_weight": 0.8},
    ]
    
    best_performance = -999
    best_config = None
    
    for test in weight_tests:
        performance = test_weight_combination(data, test["baseline_weight"], test["ml_weight"])
        
        print(f"{test['name']:<20}: {performance['total_return']*100:+.2f}% return, {performance['sharpe']:+.2f} Sharpe")
        
        # Use Sharpe ratio as primary metric
        if performance['sharpe'] > best_performance:
            best_performance = performance['sharpe']
            best_config = test
    
    print(f"\n🏆 BEST CONFIGURATION: {best_config['name']}")
    print(f"   Return: {test_weight_combination(data, best_config['baseline_weight'], best_config['ml_weight'])['total_return']*100:.2f}%")
    print(f"   Sharpe: {best_performance:.2f}")
    
    return best_config

def test_weight_combination(data, baseline_weight, ml_weight):
    """Test a specific weight combination"""
    
    signals = []
    returns = []
    
    for i in range(20, len(data)-1):
        current_data = data.iloc[:i+1]
        
        # Technical indicators
        price = current_data['close'].iloc[-1]
        sma_10 = current_data['close'].rolling(10).mean().iloc[-1]
        sma_20 = current_data['close'].rolling(20).mean().iloc[-1]
        vol_avg = current_data['volume'].rolling(10).mean().iloc[-1]
        vol_current = current_data['volume'].iloc[-1]
        
        # Baseline signal
        baseline_score = 0
        if price > sma_10: baseline_score += 0.3
        if sma_10 > sma_20: baseline_score += 0.2
        baseline_signal = np.clip(baseline_score, -0.5, 0.5)
        
        # ML signal components (use best performing ones)
        momentum_5d = (price / current_data['close'].iloc[-6] - 1) if i >= 25 else 0
        momentum_20d = (price / current_data['close'].iloc[-21] - 1) if i >= 40 else 0
        volume_ratio = vol_current / vol_avg
        
        # Focus on components with positive correlation
        momentum_signal = np.tanh(momentum_5d * 20) * 0.4
        trend_signal = np.tanh(momentum_20d * 5) * 0.3  
        volume_signal = np.tanh((volume_ratio - 1) * 3) * 0.3
        
        ml_signal_raw = momentum_signal + trend_signal + volume_signal
        ml_signal = np.clip(ml_signal_raw, -0.8, 0.8)
        
        # Combine with tested weights
        combined_signal = baseline_weight * baseline_signal + ml_weight * ml_signal
        combined_signal = np.clip(combined_signal, -1.0, 1.0)
        
        signals.append(combined_signal)
        
        # Calculate next day return
        next_return = (data['close'].iloc[i+1] / data['close'].iloc[i] - 1)
        returns.append(combined_signal * next_return)
    
    total_return = sum(returns)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
    
    return {'total_return': total_return, 'sharpe': sharpe, 'returns': returns}

def improve_ml_signals(data, correlations):
    """Create improved ML signal based on correlation analysis"""
    
    print(f"\n🛠️ IMPROVED ML SIGNAL DEVELOPMENT")
    print("=" * 45)
    
    # Focus on signals with highest absolute correlation
    good_signals = {k: v for k, v in correlations.items() if abs(v) > 0.05}
    
    print("Using signals with |correlation| > 0.05:")
    for signal, corr in good_signals.items():
        print(f"  {signal}: {corr:+.3f}")
    
    # Test improved ML strategy
    improved_signals = []
    improved_returns = []
    
    for i in range(20, len(data)-1):
        current_data = data.iloc[:i+1]
        
        # Technical indicators
        price = current_data['close'].iloc[-1]
        sma_10 = current_data['close'].rolling(10).mean().iloc[-1]
        sma_20 = current_data['close'].rolling(20).mean().iloc[-1]
        vol_avg = current_data['volume'].rolling(10).mean().iloc[-1]
        vol_current = current_data['volume'].iloc[-1]
        
        # Baseline (keep what works)
        baseline_score = 0
        if price > sma_10: baseline_score += 0.3
        if sma_10 > sma_20: baseline_score += 0.2
        baseline_signal = np.clip(baseline_score, -0.5, 0.5)
        
        # IMPROVED ML COMPONENTS (weight by correlation strength)
        components = []
        
        # Only use components that showed positive correlation
        momentum_5d = (price / current_data['close'].iloc[-6] - 1) if i >= 25 else 0
        if 'ml_momentum_5d' in good_signals and good_signals['ml_momentum_5d'] > 0:
            components.append(np.tanh(momentum_5d * 20) * abs(good_signals['ml_momentum_5d']) * 2)
        
        momentum_20d = (price / current_data['close'].iloc[-21] - 1) if i >= 40 else 0
        if 'ml_trend_20d' in good_signals and good_signals['ml_trend_20d'] > 0:
            components.append(np.tanh(momentum_20d * 5) * abs(good_signals['ml_trend_20d']) * 2)
        
        volume_ratio = vol_current / vol_avg
        if 'ml_volume' in good_signals and abs(good_signals['ml_volume']) > 0.05:
            vol_signal = np.tanh((volume_ratio - 1) * 3) 
            # Flip if negative correlation
            if good_signals['ml_volume'] < 0:
                vol_signal = -vol_signal
            components.append(vol_signal * abs(good_signals['ml_volume']) * 2)
        
        # Combine ML components
        ml_signal = np.mean(components) if components else 0
        ml_signal = np.clip(ml_signal, -0.8, 0.8)
        
        # Conservative combination (start with proven baseline)
        combined_signal = 0.7 * baseline_signal + 0.3 * ml_signal
        combined_signal = np.clip(combined_signal, -1.0, 1.0)
        
        improved_signals.append(combined_signal)
        
        # Calculate return
        next_return = (data['close'].iloc[i+1] / data['close'].iloc[i] - 1)
        improved_returns.append(combined_signal * next_return)
    
    total_return = sum(improved_returns)
    sharpe = np.mean(improved_returns) / (np.std(improved_returns) + 1e-6) * np.sqrt(252)
    win_rate = np.mean(np.array(improved_returns) > 0)
    
    print(f"\n📈 IMPROVED ML RESULTS:")
    print(f"  Total Return: {total_return*100:.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Win Rate: {win_rate*100:.1f}%")
    
    return {'total_return': total_return, 'sharpe': sharpe, 'win_rate': win_rate}

if __name__ == "__main__":
    print("🔍 ML PERFORMANCE ANALYSIS & OPTIMIZATION")
    print("=" * 60)
    print("Goal: Find why ML underperformed and fix it")
    print()
    
    # Create test data
    data = create_debug_data()
    print(f"Test data: {len(data)} days, {data['close'].iloc[0]:.2f} to {data['close'].iloc[-1]:.2f}")
    print(f"Market return: {(data['close'].iloc[-1]/data['close'].iloc[0]-1)*100:.1f}%")
    print()
    
    # Analyze signal components
    signal_data, correlations = analyze_signal_components(data)
    
    # Test optimal weights
    best_config = test_optimal_weights(data, correlations)
    
    # Create improved ML signals
    improved_results = improve_ml_signals(data, correlations)
    
    print(f"\n🎯 KEY FINDINGS:")
    print("=" * 30)
    print(f"1. Best weight combo: {best_config['name']}")
    print(f"2. Improved ML return: {improved_results['total_return']*100:.2f}%")
    print(f"3. Need to focus on signals with strong correlations")
    
    print(f"\n🛠️ IMMEDIATE FIXES:")
    print("1. Use correlation-weighted ML components")
    print("2. Start with conservative 70/30 baseline/ML ratio")
    print("3. Only include ML signals that show predictive power")
    print("4. Test on multiple market conditions")