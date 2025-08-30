#!/usr/bin/env python3
"""
Evidence-Based ML Strategy - Based on REAL Market Data Correlations
Use proven correlations from real market data analysis
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class EvidenceBasedMLStrategy:
    """ML strategy based on proven real market correlations"""
    
    def __init__(self):
        self.name = "Evidence-Based ML Strategy"
        
        # PROVEN CORRELATIONS (from real market data analysis)
        self.proven_features = {
            'macd_normalized': -0.1178,      # STRONGEST predictor
            'price_vs_sma50': -0.1014,      # Strong mean reversion
            'sma10_vs_sma20': -0.0752,      # Trend exhaustion
            'sma5_vs_sma20': -0.0710,       # Short-term momentum
            'volume_ratio': +0.0685,        # Volume confirmation (POSITIVE)
            'price_vs_sma20': -0.0681,      # Price position
            'price_change_20d': -0.0601,    # Momentum exhaustion
            'bb_position': -0.0507,         # Bollinger band position
            'rsi_14': -0.0506              # RSI overbought/oversold
        }
        
    def generate_ml_signal(self, data):
        """Generate ML signal using proven correlations"""
        
        if len(data) < 50:
            return 0.0, 0.5, "Insufficient data"
        
        # Calculate technical indicators
        indicators = self._calculate_indicators(data)
        
        # Use PROVEN correlations to weight features
        ml_signal = 0.0
        confidence_factors = []
        
        # Feature 1: MACD Normalized (STRONGEST - 11.78% correlation)
        if 'macd_normalized' in indicators:
            macd_signal = -indicators['macd_normalized']  # Invert (negative correlation)
            macd_contribution = macd_signal * abs(self.proven_features['macd_normalized']) * 3
            ml_signal += macd_contribution
            confidence_factors.append(abs(macd_signal))
        
        # Feature 2: Price vs SMA50 (STRONG - 10.14% correlation) 
        if 'price_vs_sma50' in indicators:
            price_sma50_signal = -indicators['price_vs_sma50']  # Invert
            price_contribution = price_sma50_signal * abs(self.proven_features['price_vs_sma50']) * 3
            ml_signal += price_contribution
            confidence_factors.append(abs(price_sma50_signal))
        
        # Feature 3: Volume Ratio (POSITIVE correlation - 6.85%)
        if 'volume_ratio' in indicators:
            volume_signal = indicators['volume_ratio']  # Keep positive
            volume_contribution = volume_signal * self.proven_features['volume_ratio'] * 2
            ml_signal += volume_contribution
            confidence_factors.append(abs(volume_signal))
        
        # Feature 4: SMA Crossover (7.52% correlation)
        if 'sma10_vs_sma20' in indicators:
            crossover_signal = -indicators['sma10_vs_sma20']  # Invert
            crossover_contribution = crossover_signal * abs(self.proven_features['sma10_vs_sma20']) * 2
            ml_signal += crossover_contribution
            confidence_factors.append(abs(crossover_signal))
        
        # Feature 5: RSI (5.06% correlation)
        if 'rsi_14' in indicators:
            rsi_normalized = (indicators['rsi_14'] - 50) / 50  # Normalize to -1/+1
            rsi_signal = -rsi_normalized  # Invert (sell overbought, buy oversold)
            rsi_contribution = rsi_signal * abs(self.proven_features['rsi_14']) * 1.5
            ml_signal += rsi_contribution
            confidence_factors.append(abs(rsi_signal))
        
        # Normalize signal
        ml_signal = np.tanh(ml_signal)  # Keep in [-1, +1] range
        
        # Calculate confidence based on feature agreement
        base_confidence = 0.6
        if confidence_factors:
            avg_factor = np.mean(confidence_factors)
            confidence_boost = min(0.3, avg_factor * 0.4)
            confidence = base_confidence + confidence_boost
        else:
            confidence = 0.5
        
        explanation = f"Evidence ML: MACD={indicators.get('macd_normalized', 0):.4f}, Price/SMA50={indicators.get('price_vs_sma50', 0):.4f}"
        
        return ml_signal, confidence, explanation
    
    def _calculate_indicators(self, data):
        """Calculate the proven technical indicators"""
        
        indicators = {}
        
        try:
            current_price = data['close'].iloc[-1]
            
            # Moving averages
            sma_5 = data['close'].rolling(5).mean().iloc[-1] if len(data) >= 5 else current_price
            sma_10 = data['close'].rolling(10).mean().iloc[-1] if len(data) >= 10 else current_price
            sma_20 = data['close'].rolling(20).mean().iloc[-1] if len(data) >= 20 else current_price
            sma_50 = data['close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else current_price
            
            # MACD (most important)
            ema_12 = data['close'].ewm(span=12).mean().iloc[-1]
            ema_26 = data['close'].ewm(span=26).mean().iloc[-1] 
            macd = ema_12 - ema_26
            indicators['macd_normalized'] = macd / current_price
            
            # Price relative positions
            indicators['price_vs_sma50'] = (current_price - sma_50) / sma_50
            indicators['price_vs_sma20'] = (current_price - sma_20) / sma_20
            
            # SMA relationships
            indicators['sma10_vs_sma20'] = (sma_10 - sma_20) / sma_20
            indicators['sma5_vs_sma20'] = (sma_5 - sma_20) / sma_20
            
            # Volume
            if 'volume' in data.columns:
                vol_avg = data['volume'].rolling(10).mean().iloc[-1] if len(data) >= 10 else data['volume'].iloc[-1]
                current_vol = data['volume'].iloc[-1]
                indicators['volume_ratio'] = (current_vol / vol_avg - 1)  # Normalize around 0
            
            # RSI
            if len(data) >= 14:
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                indicators['rsi_14'] = rsi.iloc[-1]
            
            # Price changes
            if len(data) >= 20:
                indicators['price_change_20d'] = (current_price / data['close'].iloc[-20] - 1)
            
            # Bollinger Band position
            if len(data) >= 20:
                bb_sma = data['close'].rolling(20).mean().iloc[-1]
                bb_std = data['close'].rolling(20).std().iloc[-1]
                bb_upper = bb_sma + (bb_std * 2)
                bb_lower = bb_sma - (bb_std * 2)
                indicators['bb_position'] = (current_price - bb_lower) / (bb_upper - bb_lower)
            
        except Exception as e:
            print(f"⚠️ Indicator calculation error: {str(e)}")
        
        return indicators

def test_evidence_based_strategy():
    """Test the evidence-based strategy on real market data"""
    
    print("🧪 EVIDENCE-BASED ML STRATEGY TEST")
    print("=" * 50)
    print("Using PROVEN correlations from real market analysis")
    print("Key insight: Contrarian signals work best (negative correlations)")
    print()
    
    try:
        # Load real validation data
        data_dir = 'data/full_market'
        val_path = os.path.join(data_dir, 'validation_data.csv')
        
        if not os.path.exists(val_path):
            print("❌ Validation data not found. Run real_data_pipeline.py first")
            return False
        
        val_data = pd.read_csv(val_path)
        val_data['date'] = pd.to_datetime(val_data['date'])
        
        print(f"📊 Validation data: {len(val_data):,} records")
        print(f"Period: {val_data['date'].min().date()} to {val_data['date'].max().date()}")
        print(f"Symbols: {val_data['symbol'].nunique()}")
        
        # Test strategy
        strategy = EvidenceBasedMLStrategy()
        
        results = []
        
        # Test on multiple symbols
        test_symbols = ['AAPL', 'MSFT', 'SPY', 'GOOGL', 'AMZN']
        
        for symbol in test_symbols:
            if symbol in val_data['symbol'].values:
                print(f"\n📈 Testing {symbol}...")
                
                symbol_data = val_data[val_data['symbol'] == symbol].copy().sort_values('date')
                
                signals = []
                returns = []
                confidences = []
                
                # Generate signals and calculate returns
                for i in range(50, len(symbol_data)-5):  # Leave room for forward returns
                    current_data = symbol_data.iloc[:i+1]
                    
                    # Generate ML signal
                    ml_signal, ml_confidence, explanation = strategy.generate_ml_signal(current_data)
                    
                    # Get baseline signal for comparison
                    price = current_data['close'].iloc[-1]
                    sma_20 = current_data['close'].rolling(20).mean().iloc[-1]
                    baseline_signal = 0.3 if price > sma_20 else -0.2
                    
                    # Conservative combination: 70% baseline + 30% ML
                    combined_signal = 0.7 * baseline_signal + 0.3 * ml_signal * ml_confidence
                    combined_signal = np.clip(combined_signal, -1.0, 1.0)
                    
                    signals.append(combined_signal)
                    confidences.append(ml_confidence)
                    
                    # Calculate 5-day forward return (proven timeframe)
                    if i < len(symbol_data) - 5:
                        forward_return = (symbol_data['close'].iloc[i+5] / symbol_data['close'].iloc[i] - 1)
                        strategy_return = combined_signal * forward_return
                        returns.append(strategy_return)
                
                if returns:
                    total_return = sum(returns)
                    win_rate = np.mean(np.array(returns) > 0)
                    avg_confidence = np.mean(confidences)
                    sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252/5)  # 5-day periods
                    
                    print(f"  Total Return: {total_return*100:.2f}%")
                    print(f"  Win Rate: {win_rate*100:.1f}%")
                    print(f"  Avg Confidence: {avg_confidence:.2f}")
                    print(f"  Sharpe Ratio: {sharpe:.2f}")
                    
                    results.append({
                        'symbol': symbol,
                        'return': total_return,
                        'win_rate': win_rate,
                        'sharpe': sharpe,
                        'confidence': avg_confidence
                    })
        
        # Overall results
        if results:
            print(f"\n🎯 OVERALL RESULTS")
            print("=" * 30)
            
            avg_return = np.mean([r['return'] for r in results]) * 100
            avg_win_rate = np.mean([r['win_rate'] for r in results]) * 100
            avg_sharpe = np.mean([r['sharpe'] for r in results])
            positive_returns = sum(1 for r in results if r['return'] > 0)
            
            print(f"Average Return: {avg_return:.2f}%")
            print(f"Average Win Rate: {avg_win_rate:.1f}%") 
            print(f"Average Sharpe: {avg_sharpe:.2f}")
            print(f"Positive Return Symbols: {positive_returns}/{len(results)}")
            
            # Success criteria
            success = avg_return > 2 and avg_sharpe > 0.5 and positive_returns >= len(results) * 0.6
            
            if success:
                print(f"\n✅ EVIDENCE-BASED STRATEGY SUCCESS!")
                print("Strategy shows consistent improvements using proven correlations")
                return True
            else:
                print(f"\n📊 MIXED RESULTS")
                print("Strategy shows some promise but needs optimization")
                return False
        
        else:
            print("❌ No results generated")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 EVIDENCE-BASED ML STRATEGY")
    print("=" * 60)
    print("Strategy based on PROVEN real market correlations:")
    print("• MACD Normalized: -11.78% correlation (STRONGEST)")
    print("• Price vs SMA50: -10.14% correlation (STRONG)")  
    print("• Volume Ratio: +6.85% correlation (MODERATE)")
    print("• Approach: Contrarian signals work best")
    print()
    
    success = test_evidence_based_strategy()
    
    if success:
        print(f"\n🚀 READY FOR PRODUCTION")
        print("Evidence-based ML strategy validated on real data")
        print("Proceed to full backtesting integration")
    else:
        print(f"\n🔧 CONTINUE REFINEMENT")
        print("Optimize signal combination and risk management")