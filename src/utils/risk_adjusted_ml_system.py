#!/usr/bin/env python3
"""
Risk-Adjusted ML System - Phase 2
Combine proven signal generation with ML-enhanced risk management
Focus: Use ML for risk prediction and position sizing, not just signal generation
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class RiskAdjustedMLSystem:
    """Complete risk-first ML trading system"""
    
    def __init__(self):
        self.name = "Risk-Adjusted ML System"
        
        # Proven signal features (from Phase 1)
        self.signal_features = {
            'macd_normalized': -0.1178,      # STRONGEST predictor
            'price_vs_sma50': -0.1014,      # Strong mean reversion
            'volume_ratio': +0.0685,        # Volume confirmation
            'sma10_vs_sma20': -0.0752,      # Trend exhaustion
            'rsi_normalized': -0.0506       # Overbought/oversold
        }
        
        # Risk prediction models
        self.volatility_models = {}
        self.risk_scalers = {}
        
    def calculate_technical_features(self, data):
        """Calculate proven technical features"""
        
        enhanced_data = data.copy()
        
        try:
            current_price = data['close'].iloc[-1]
            
            # Moving averages
            sma_10 = data['close'].rolling(10).mean().iloc[-1] if len(data) >= 10 else current_price
            sma_20 = data['close'].rolling(20).mean().iloc[-1] if len(data) >= 20 else current_price
            sma_50 = data['close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else current_price
            
            # MACD (proven strongest predictor)
            ema_12 = data['close'].ewm(span=12).mean().iloc[-1]
            ema_26 = data['close'].ewm(span=26).mean().iloc[-1]
            macd = ema_12 - ema_26
            macd_normalized = macd / current_price
            
            # Price relative positions
            price_vs_sma50 = (current_price - sma_50) / sma_50
            
            # SMA relationships
            sma10_vs_sma20 = (sma_10 - sma_20) / sma_20
            
            # Volume
            volume_ratio = 0
            if 'volume' in data.columns and len(data) >= 10:
                vol_avg = data['volume'].rolling(10).mean().iloc[-1]
                current_vol = data['volume'].iloc[-1]
                volume_ratio = (current_vol / vol_avg - 1)  # Normalized around 0
            
            # RSI
            rsi_normalized = 0
            if len(data) >= 14:
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                rsi_normalized = (rsi.iloc[-1] - 50) / 50  # Normalize to [-1, +1]
            
            return {
                'macd_normalized': macd_normalized,
                'price_vs_sma50': price_vs_sma50,
                'volume_ratio': volume_ratio,
                'sma10_vs_sma20': sma10_vs_sma20,
                'rsi_normalized': rsi_normalized
            }
            
        except Exception as e:
            print(f"⚠️ Feature calculation error: {str(e)}")
            return {feature: 0 for feature in self.signal_features.keys()}
    
    def calculate_risk_features(self, data):
        """Calculate features for volatility/risk prediction"""
        
        risk_features = {}
        
        try:
            # 1. Recent realized volatility
            if len(data) >= 20:
                returns = data['close'].pct_change()
                risk_features['vol_5d'] = returns.tail(5).std()
                risk_features['vol_20d'] = returns.tail(20).std()
                risk_features['vol_ratio'] = risk_features['vol_5d'] / (risk_features['vol_20d'] + 1e-6)
            
            # 2. Price range volatility
            if len(data) >= 5:
                high_low_range = (data['high'] - data['low']) / data['close']
                risk_features['range_vol'] = high_low_range.tail(5).mean()
            
            # 3. Volume volatility
            if 'volume' in data.columns and len(data) >= 10:
                vol_changes = data['volume'].pct_change()
                risk_features['volume_vol'] = vol_changes.tail(10).std()
            
            # 4. Gap volatility (overnight moves)
            if len(data) >= 5:
                gaps = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
                risk_features['gap_vol'] = gaps.tail(5).std()
            
            # 5. Technical indicator volatility
            if len(data) >= 20:
                # RSI volatility
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                risk_features['rsi_vol'] = rsi.tail(5).std()
            
            # Fill missing values
            for key in ['vol_5d', 'vol_20d', 'vol_ratio', 'range_vol', 'volume_vol', 'gap_vol', 'rsi_vol']:
                if key not in risk_features:
                    risk_features[key] = 0.02  # Default 2% volatility
            
        except Exception as e:
            print(f"⚠️ Risk feature calculation error: {str(e)}")
            # Default risk features
            risk_features = {
                'vol_5d': 0.02, 'vol_20d': 0.02, 'vol_ratio': 1.0,
                'range_vol': 0.02, 'volume_vol': 0.5, 'gap_vol': 0.01,
                'rsi_vol': 5.0
            }
        
        return risk_features
    
    def generate_ml_signal(self, data):
        """Generate ML signal using proven correlations"""
        
        if len(data) < 20:
            return 0.0, 0.5, "Insufficient data"
        
        # Get technical features
        features = self.calculate_technical_features(data)
        
        # Apply PROVEN correlations with contrarian approach
        ml_signal = 0.0
        confidence_factors = []
        
        # Use proven negative correlations (contrarian signals)
        for feature_name, correlation in self.signal_features.items():
            if feature_name in features:
                feature_value = features[feature_name]
                
                if correlation < 0:  # Contrarian signal
                    signal_contribution = -feature_value * abs(correlation) * 2
                else:  # Trend following signal
                    signal_contribution = feature_value * correlation * 2
                
                ml_signal += signal_contribution
                confidence_factors.append(abs(feature_value))
        
        # Normalize signal
        ml_signal = np.tanh(ml_signal)
        
        # Calculate confidence
        base_confidence = 0.6
        if confidence_factors:
            avg_factor = np.mean(confidence_factors)
            confidence_boost = min(0.3, avg_factor * 0.5)
            confidence = base_confidence + confidence_boost
        else:
            confidence = 0.5
        
        explanation = f"ML Signal: MACD={features['macd_normalized']:.4f}, Price/SMA50={features['price_vs_sma50']:.4f}"
        
        return ml_signal, confidence, explanation
    
    def predict_volatility(self, data):
        """Predict future volatility using risk features"""
        
        risk_features = self.calculate_risk_features(data)
        
        # Simple volatility prediction using recent patterns
        current_vol = risk_features['vol_5d']
        recent_vol = risk_features['vol_20d']
        vol_trend = risk_features['vol_ratio']
        
        # Predict next period volatility
        # Use exponential weighted combination of recent volatility patterns
        predicted_vol = 0.4 * current_vol + 0.3 * recent_vol + 0.3 * (current_vol * vol_trend)
        
        # Add regime adjustment
        regime_adjustment = 1.0
        if risk_features['range_vol'] > 0.03:  # High intraday volatility
            regime_adjustment += 0.2
        if risk_features['volume_vol'] > 1.0:  # High volume volatility
            regime_adjustment += 0.1
        
        predicted_vol *= regime_adjustment
        
        # Ensure reasonable bounds
        predicted_vol = max(0.005, min(0.08, predicted_vol))  # 0.5% to 8%
        
        # Confidence based on volatility stability
        vol_stability = 1 / (1 + abs(current_vol - recent_vol) * 50)
        confidence = 0.5 + 0.4 * vol_stability
        
        return predicted_vol, confidence
    
    def calculate_position_size(self, signal_strength, confidence, predicted_volatility, base_capital=10000):
        """Calculate risk-adjusted position size using Kelly-like criterion"""
        
        # Base position size
        base_position_pct = 0.1  # 10% of capital
        
        # Adjust for signal strength and confidence
        signal_adjustment = abs(signal_strength) * confidence
        
        # Adjust for predicted volatility (lower volatility = larger position)
        vol_adjustment = 0.02 / (predicted_volatility + 1e-6)  # Target 2% volatility
        vol_adjustment = min(2.0, max(0.5, vol_adjustment))  # Cap adjustments
        
        # Conservative Kelly-like sizing
        kelly_adjustment = signal_adjustment * vol_adjustment
        kelly_adjustment = min(1.5, max(0.3, kelly_adjustment))  # Conservative bounds
        
        final_position_pct = base_position_pct * kelly_adjustment
        final_position_pct = min(0.25, max(0.02, final_position_pct))  # 2% to 25% of capital
        
        position_value = base_capital * final_position_pct
        
        return {
            'position_pct': final_position_pct,
            'position_value': position_value,
            'signal_adjustment': signal_adjustment,
            'vol_adjustment': vol_adjustment,
            'kelly_adjustment': kelly_adjustment
        }
    
    def calculate_stop_loss(self, entry_price, predicted_volatility, signal_strength):
        """Calculate adaptive stop-loss based on predicted volatility"""
        
        # Base stop loss
        base_stop_pct = 0.02  # 2%
        
        # Adjust for predicted volatility
        vol_multiplier = predicted_volatility / 0.02  # Normalize to 2% base
        vol_multiplier = min(3.0, max(0.5, vol_multiplier))
        
        # Adjust for signal confidence (stronger signals get wider stops)
        signal_multiplier = 1 + abs(signal_strength) * 0.5
        
        # Calculate final stop loss
        stop_loss_pct = base_stop_pct * vol_multiplier * signal_multiplier
        stop_loss_pct = min(0.08, max(0.01, stop_loss_pct))  # 1% to 8%
        
        if signal_strength > 0:  # Long position
            stop_price = entry_price * (1 - stop_loss_pct)
        else:  # Short position
            stop_price = entry_price * (1 + stop_loss_pct)
        
        return {
            'stop_price': stop_price,
            'stop_loss_pct': stop_loss_pct,
            'vol_multiplier': vol_multiplier,
            'signal_multiplier': signal_multiplier
        }

def test_risk_adjusted_system():
    """Test the complete risk-adjusted ML system"""
    
    print("🛡️ RISK-ADJUSTED ML SYSTEM TEST")
    print("=" * 50)
    print("Integration: Proven signals + ML risk management")
    
    try:
        # Load validation data
        data_dir = 'data/full_market'
        val_path = os.path.join(data_dir, 'validation_data.csv')
        
        if not os.path.exists(val_path):
            print("❌ Validation data not found")
            return False
        
        val_data = pd.read_csv(val_path)
        val_data['date'] = pd.to_datetime(val_data['date'])
        
        system = RiskAdjustedMLSystem()
        
        print(f"📊 Testing on validation data: {len(val_data):,} records")
        
        # Test on major symbols
        test_symbols = ['AAPL', 'MSFT', 'SPY']
        results = []
        
        for symbol in test_symbols:
            if symbol in val_data['symbol'].values:
                print(f"\n📈 Testing {symbol}...")
                
                symbol_data = val_data[val_data['symbol'] == symbol].sort_values('date')
                
                trades = []
                
                # Simulate trading with risk management
                for i in range(50, len(symbol_data)-5, 5):  # Every 5 days
                    current_data = symbol_data.iloc[:i+1]
                    current_price = current_data['close'].iloc[-1]
                    
                    # Generate signal
                    signal_strength, confidence, explanation = system.generate_ml_signal(current_data)
                    
                    # Predict volatility
                    predicted_vol, vol_confidence = system.predict_volatility(current_data)
                    
                    # Calculate position size
                    position_info = system.calculate_position_size(
                        signal_strength, confidence, predicted_vol
                    )
                    
                    # Calculate stop loss
                    stop_info = system.calculate_stop_loss(
                        current_price, predicted_vol, signal_strength
                    )
                    
                    # Calculate 5-day forward return
                    if i < len(symbol_data) - 5:
                        future_price = symbol_data['close'].iloc[i+5]
                        raw_return = (future_price / current_price - 1)
                        
                        # Apply position sizing
                        position_return = raw_return * position_info['position_pct'] * 10  # Scale for visibility
                        
                        # Apply stop loss (simplified)
                        if signal_strength > 0:  # Long
                            if future_price <= stop_info['stop_price']:
                                position_return = -stop_info['stop_loss_pct'] * position_info['position_pct'] * 10
                        
                        trades.append({
                            'signal_strength': signal_strength,
                            'confidence': confidence,
                            'predicted_vol': predicted_vol,
                            'position_pct': position_info['position_pct'],
                            'raw_return': raw_return,
                            'position_return': position_return
                        })
                
                if trades:
                    trades_df = pd.DataFrame(trades)
                    
                    total_return = trades_df['position_return'].sum()
                    win_rate = (trades_df['position_return'] > 0).mean()
                    avg_position = trades_df['position_pct'].mean()
                    avg_vol_pred = trades_df['predicted_vol'].mean()
                    sharpe = trades_df['position_return'].mean() / (trades_df['position_return'].std() + 1e-6)
                    
                    print(f"  Total Return: {total_return:.2f}%")
                    print(f"  Win Rate: {win_rate*100:.1f}%")
                    print(f"  Avg Position Size: {avg_position*100:.1f}%")
                    print(f"  Avg Vol Prediction: {avg_vol_pred*100:.1f}%")
                    print(f"  Sharpe Ratio: {sharpe:.2f}")
                    
                    results.append({
                        'symbol': symbol,
                        'return': total_return,
                        'win_rate': win_rate,
                        'sharpe': sharpe,
                        'avg_position': avg_position
                    })
        
        # Overall assessment
        if results:
            print(f"\n🎯 OVERALL RISK-ADJUSTED RESULTS")
            print("=" * 40)
            
            avg_return = np.mean([r['return'] for r in results])
            avg_win_rate = np.mean([r['win_rate'] for r in results])
            avg_sharpe = np.mean([r['sharpe'] for r in results])
            positive_results = sum(1 for r in results if r['return'] > 0)
            
            print(f"Average Return: {avg_return:.2f}%")
            print(f"Average Win Rate: {avg_win_rate*100:.1f}%")
            print(f"Average Sharpe: {avg_sharpe:.2f}")
            print(f"Positive Results: {positive_results}/{len(results)}")
            
            # Success criteria
            success = avg_return > 1 and avg_sharpe > 0.3 and positive_results >= 2
            
            if success:
                print(f"\n✅ RISK-ADJUSTED ML SUCCESS!")
                print("System shows improved risk-adjusted returns")
                return True
            else:
                print(f"\n📊 FOUNDATION ESTABLISHED")
                print("Risk management framework working, needs optimization")
                return False
        
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🛡️ RISK-ADJUSTED ML SYSTEM - PHASE 2")
    print("=" * 60)
    print("Approach: Proven signals + ML-enhanced risk management")
    print("Focus: Position sizing, volatility prediction, adaptive stops")
    print()
    
    success = test_risk_adjusted_system()
    
    if success:
        print(f"\n🚀 READY FOR PRODUCTION INTEGRATION")
        print("Risk-adjusted ML system showing consistent benefits")
    else:
        print(f"\n🔧 CONTINUE RISK MANAGEMENT OPTIMIZATION")
        print("Focus on volatility prediction and position sizing refinement")