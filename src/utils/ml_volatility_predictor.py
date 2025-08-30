#!/usr/bin/env python3
"""
ML Volatility Predictor - Phase 2, Day 1
Build ML system for volatility forecasting (more predictable than returns)
Focus: Risk prediction is more reliable than return prediction
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class MLVolatilityPredictor:
    """ML system for predicting future volatility"""
    
    def __init__(self):
        self.name = "ML Volatility Predictor"
        self.models = {}  # Symbol-specific models
        self.scalers = {}
        self.feature_names = []
        
        # Volatility prediction is more reliable than return prediction
        self.volatility_features = [
            'realized_vol_5d', 'realized_vol_20d', 'vol_ratio_5_20',
            'price_range_5d', 'volume_vol_5d', 'return_autocorr',
            'high_low_ratio', 'gap_volatility', 'macd_volatility',
            'rsi_volatility', 'volume_surge', 'price_acceleration'
        ]
    
    def load_and_prepare_data(self):
        """Load real market data and prepare volatility features"""
        
        print("📊 VOLATILITY PREDICTION DATA PREPARATION")
        print("=" * 55)
        
        data_dir = 'data/full_market'
        train_path = os.path.join(data_dir, 'train_data.csv')
        
        if not os.path.exists(train_path):
            raise FileNotFoundError("Run real_data_pipeline.py first")
        
        data = pd.read_csv(train_path)
        data['date'] = pd.to_datetime(data['date'])
        
        print(f"Raw data: {len(data):,} records")
        print(f"Symbols: {data['symbol'].nunique()}")
        
        # Prepare volatility features for each symbol
        enhanced_data = []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy().sort_values('date')
            print(f"Processing {symbol}... ", end="")
            
            # Create volatility features
            symbol_data = self._create_volatility_features(symbol_data)
            
            # Create volatility targets (what we want to predict)
            symbol_data = self._create_volatility_targets(symbol_data)
            
            enhanced_data.append(symbol_data)
            print(f"✅ {len(symbol_data)} records")
        
        combined_data = pd.concat(enhanced_data, ignore_index=True)
        
        # Remove rows with missing targets
        combined_data = combined_data.dropna(subset=['target_vol_5d'])
        
        print(f"\nEnhanced dataset: {len(combined_data):,} records")
        print(f"Features created: {len(self.volatility_features)} volatility features")
        
        return combined_data
    
    def _create_volatility_features(self, data):
        """Create features that predict volatility"""
        
        data = data.copy()
        
        # 1. Realized volatility (rolling standard deviation of returns)
        data['daily_return'] = data['close'].pct_change()
        data['realized_vol_5d'] = data['daily_return'].rolling(5).std()
        data['realized_vol_20d'] = data['daily_return'].rolling(20).std()
        data['vol_ratio_5_20'] = data['realized_vol_5d'] / (data['realized_vol_20d'] + 1e-6)
        
        # 2. Price range volatility
        data['high_low_range'] = (data['high'] - data['low']) / data['close']
        data['price_range_5d'] = data['high_low_range'].rolling(5).mean()
        
        # 3. Volume volatility
        data['volume_change'] = data['volume'].pct_change()
        data['volume_vol_5d'] = data['volume_change'].rolling(5).std()
        
        # 4. Return autocorrelation (momentum persistence)
        data['return_autocorr'] = data['daily_return'].rolling(10).apply(
            lambda x: x.autocorr() if len(x.dropna()) > 5 else 0, raw=False
        )
        
        # 5. High/Low ratio (intraday volatility)
        data['high_low_ratio'] = data['high'] / data['low']
        
        # 6. Gap volatility (overnight moves)
        data['gap_return'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        data['gap_volatility'] = data['gap_return'].rolling(5).std()
        
        # 7. Technical indicator volatility
        # MACD volatility
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        data['macd_volatility'] = (macd / data['close']).rolling(5).std()
        
        # RSI volatility
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        data['rsi_volatility'] = rsi.rolling(5).std()
        
        # 8. Volume surge indicator
        vol_sma_20 = data['volume'].rolling(20).mean()
        data['volume_surge'] = data['volume'] / (vol_sma_20 + 1)
        
        # 9. Price acceleration (second derivative)
        data['price_velocity'] = data['close'].pct_change()
        data['price_acceleration'] = data['price_velocity'].diff()
        
        return data
    
    def _create_volatility_targets(self, data):
        """Create forward-looking volatility targets"""
        
        data = data.copy()
        
        # Target: 5-day forward realized volatility
        data['target_vol_5d'] = data['daily_return'].rolling(5).std().shift(-5)
        
        # Target: 20-day forward realized volatility  
        data['target_vol_20d'] = data['daily_return'].rolling(20).std().shift(-20)
        
        # Target: Maximum 5-day drawdown (risk measure)
        data['target_max_drawdown_5d'] = data['daily_return'].rolling(5).apply(
            lambda x: (1 + x).cumprod().drawdown().max() if len(x) == 5 else np.nan, 
            raw=False
        ).shift(-5)
        
        return data
    
    def train_volatility_models(self, data):
        """Train ML models to predict volatility for each symbol"""
        
        print(f"\n🤖 TRAINING VOLATILITY PREDICTION MODELS")
        print("=" * 55)
        
        results = {}
        feature_cols = [f for f in self.volatility_features if f in data.columns]
        self.feature_names = feature_cols
        
        print(f"Features used: {len(feature_cols)}")
        for feature in feature_cols:
            print(f"  • {feature}")
        
        print(f"\nTraining models...")
        
        for symbol in data['symbol'].unique():
            print(f"\n📊 Training {symbol} volatility model...")
            
            symbol_data = data[data['symbol'] == symbol].copy()
            
            # Prepare features and targets
            X = symbol_data[feature_cols].fillna(0)
            y = symbol_data['target_vol_5d'].fillna(0)
            
            # Remove rows where target is 0 (invalid)
            valid_mask = (y > 0) & np.isfinite(y)
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 100:
                print(f"  ⚠️ Insufficient data for {symbol}: {len(X)} samples")
                continue
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Use time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            val_scores = []
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_train, y_train)
                
                # Validate
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                val_scores.append(score)
            
            avg_val_score = np.mean(val_scores)
            
            if avg_val_score > 0.1:  # Minimum acceptable R²
                # Train final model on all data
                final_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                
                final_model.fit(X_scaled, y)
                
                # Store model and scaler
                self.models[symbol] = final_model
                self.scalers[symbol] = scaler
                
                # Calculate feature importance
                feature_importance = dict(zip(feature_cols, final_model.feature_importances_))
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                
                print(f"  ✅ R² Score: {avg_val_score:.3f}")
                print(f"  📊 Top features: {', '.join([f[0] for f in top_features])}")
                
                results[symbol] = {
                    'r2_score': avg_val_score,
                    'feature_importance': feature_importance,
                    'sample_size': len(X)
                }
            else:
                print(f"  ❌ Poor performance: R² = {avg_val_score:.3f}")
        
        print(f"\n📋 TRAINING SUMMARY")
        print("-" * 30)
        print(f"Models trained: {len(self.models)}")
        print(f"Average R² score: {np.mean([r['r2_score'] for r in results.values()]):.3f}")
        
        successful_models = len([r for r in results.values() if r['r2_score'] > 0.2])
        print(f"Good models (R² > 0.2): {successful_models}/{len(results)}")
        
        return results
    
    def predict_volatility(self, symbol, recent_data):
        """Predict future volatility for a given symbol"""
        
        if symbol not in self.models:
            # Use general model or return default
            available_symbols = list(self.models.keys())
            if available_symbols:
                symbol = available_symbols[0]  # Use first available model
            else:
                return 0.02, 0.5  # Default 2% volatility, low confidence
        
        try:
            # Create features
            featured_data = self._create_volatility_features(recent_data)
            
            # Extract features
            feature_values = []
            for feature in self.feature_names:
                if feature in featured_data.columns:
                    value = featured_data[feature].iloc[-1]
                    feature_values.append(value if pd.notna(value) else 0)
                else:
                    feature_values.append(0)
            
            # Scale and predict
            X = np.array(feature_values).reshape(1, -1)
            X_scaled = self.scalers[symbol].transform(X)
            
            predicted_vol = self.models[symbol].predict(X_scaled)[0]
            
            # Ensure reasonable bounds
            predicted_vol = max(0.005, min(0.1, predicted_vol))  # 0.5% to 10%
            
            # Calculate confidence based on recent volatility stability
            recent_vol = featured_data['realized_vol_5d'].iloc[-1] if 'realized_vol_5d' in featured_data.columns else 0.02
            confidence = 0.6 + 0.3 * (1 / (1 + abs(predicted_vol - recent_vol) * 20))
            
            return predicted_vol, confidence
            
        except Exception as e:
            print(f"⚠️ Volatility prediction error: {str(e)}")
            return 0.02, 0.5

def test_volatility_prediction():
    """Test the volatility prediction system"""
    
    print("🧪 VOLATILITY PREDICTION TESTING")
    print("=" * 50)
    
    try:
        predictor = MLVolatilityPredictor()
        
        # Load and prepare data
        data = predictor.load_and_prepare_data()
        
        # Train models
        results = predictor.train_volatility_models(data)
        
        if not results:
            print("❌ No models trained successfully")
            return False
        
        # Test predictions on validation data
        print(f"\n🔮 TESTING PREDICTIONS")
        print("-" * 30)
        
        val_data_path = 'data/full_market/validation_data.csv'
        if os.path.exists(val_data_path):
            val_data = pd.read_csv(val_data_path)
            val_data['date'] = pd.to_datetime(val_data['date'])
            
            test_symbols = ['AAPL', 'MSFT', 'SPY']
            
            for symbol in test_symbols:
                if symbol in predictor.models and symbol in val_data['symbol'].values:
                    symbol_data = val_data[val_data['symbol'] == symbol].sort_values('date')
                    
                    if len(symbol_data) > 50:
                        # Test prediction on recent data
                        recent_data = symbol_data.tail(50)
                        predicted_vol, confidence = predictor.predict_volatility(symbol, recent_data)
                        
                        # Calculate actual recent volatility for comparison
                        actual_vol = recent_data['close'].pct_change().std()
                        
                        error = abs(predicted_vol - actual_vol) / actual_vol if actual_vol > 0 else 1
                        
                        print(f"{symbol}: Predicted {predicted_vol:.3f}, Actual {actual_vol:.3f}, Error {error:.1%}, Confidence {confidence:.2f}")
        
        # Success criteria
        good_models = len([r for r in results.values() if r['r2_score'] > 0.2])
        avg_r2 = np.mean([r['r2_score'] for r in results.values()])
        
        success = good_models >= 3 and avg_r2 > 0.15
        
        if success:
            print(f"\n✅ VOLATILITY PREDICTION SUCCESS!")
            print(f"Models show predictive power for risk management")
            return True
        else:
            print(f"\n⚠️ MIXED VOLATILITY RESULTS")
            print(f"Models need improvement but foundation established")
            return False
        
    except Exception as e:
        print(f"❌ Volatility testing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🌊 ML VOLATILITY PREDICTOR - PHASE 2")
    print("=" * 60)
    print("Goal: Build ML system for volatility forecasting")
    print("Approach: Risk prediction more reliable than return prediction")
    print("Application: Risk-adjusted position sizing and stop losses")
    print()
    
    success = test_volatility_prediction()
    
    if success:
        print(f"\n🚀 READY FOR RISK MANAGEMENT INTEGRATION")
        print("Volatility prediction working - proceed to position sizing")
    else:
        print(f"\n🔧 CONTINUE VOLATILITY MODEL REFINEMENT")
        print("Improve volatility features and model architecture")