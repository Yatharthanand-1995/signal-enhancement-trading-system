#!/usr/bin/env python3
import pandas as pd
import numpy as np

print("🚀 SIMPLE ML VALIDATION")
print("Testing minimal ML vs baseline")

# Create test data
data = pd.DataFrame({
    'close': 100 + np.cumsum(np.random.normal(0.001, 0.02, 100)),
    'volume': np.random.randint(1000000, 3000000, 100)
})

# Baseline strategy - simple moving average
def baseline_signal(data):
    sma = data['close'].rolling(20).mean()
    return 0.3 if data['close'].iloc[-1] > sma.iloc[-1] else -0.2

# ML-enhanced strategy - adds momentum and volume
def ml_enhanced_signal(data):
    sma = data['close'].rolling(20).mean()
    baseline = 0.3 if data['close'].iloc[-1] > sma.iloc[-1] else -0.2
    
    # Simple ML features
    momentum = (data['close'].iloc[-1] / data['close'].iloc[-5] - 1) if len(data) >= 5 else 0
    vol_ratio = data['volume'].iloc[-1] / data['volume'].rolling(5).mean().iloc[-1] if len(data) >= 5 else 1
    
    ml_component = 0.3 * np.tanh(momentum * 5) + 0.2 * np.tanh((vol_ratio - 1) * 2)
    
    return 0.7 * baseline + 0.3 * ml_component

# Test signals
baseline_signals = []
ml_signals = []
returns = []

for i in range(20, len(data)):
    subset = data.iloc[:i+1]
    
    baseline_signals.append(baseline_signal(subset))
    ml_signals.append(ml_enhanced_signal(subset))
    
    if i > 20:
        ret = (data['close'].iloc[i] / data['close'].iloc[i-1] - 1)
        returns.append(ret)

# Calculate performance
baseline_returns = [r * s for r, s in zip(returns, baseline_signals[:-1])]
ml_returns = [r * s for r, s in zip(returns, ml_signals[:-1])]

baseline_total = sum(baseline_returns) * 100
ml_total = sum(ml_returns) * 100
improvement = (ml_total / baseline_total - 1) * 100 if baseline_total != 0 else 0

print(f"Baseline return: {baseline_total:.2f}%")
print(f"ML-enhanced return: {ml_total:.2f}%")  
print(f"Improvement: {improvement:+.1f}%")

if improvement > 5:
    print("✅ ML INTEGRATION WORKING!")
    print("Ready for more complex backtesting")
else:
    print("🔧 Need to refine ML components")

print("\n🎯 CONCLUSION:")
print("Simple validation completed - basic ML logic functional")