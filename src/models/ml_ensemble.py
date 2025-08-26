"""
LSTM-XGBoost Ensemble Model for Stock Price Prediction
Optimized for 2-15 day holding periods with 93%+ accuracy target
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import psycopg2
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from config.config import config

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Advanced feature engineering for stock prediction"""
    
    def __init__(self, lookback_window: int = 30):
        self.lookback_window = lookback_window
        self.scalers = {}
        
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features"""
        result_df = df.copy()
        
        # Price returns
        result_df['returns_1d'] = df['close'].pct_change(1)
        result_df['returns_2d'] = df['close'].pct_change(2)
        result_df['returns_5d'] = df['close'].pct_change(5)
        result_df['returns_10d'] = df['close'].pct_change(10)
        result_df['returns_20d'] = df['close'].pct_change(20)
        
        # Log returns (more stable)
        result_df['log_returns_1d'] = np.log(df['close'] / df['close'].shift(1))
        result_df['log_returns_5d'] = np.log(df['close'] / df['close'].shift(5))
        
        # Price momentum
        result_df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
        result_df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
        result_df['momentum_20d'] = df['close'] / df['close'].shift(20) - 1
        
        # Price acceleration
        result_df['acceleration_5d'] = result_df['momentum_5d'].diff()
        result_df['acceleration_10d'] = result_df['momentum_10d'].diff()
        
        # High-Low spread
        result_df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        result_df['hl_ratio_ma'] = result_df['hl_ratio'].rolling(window=20).mean()
        
        return result_df
        
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features"""
        result_df = df.copy()
        
        # Volume ratios
        result_df['volume_ratio_5d'] = df['volume'] / df['volume'].rolling(window=5).mean()
        result_df['volume_ratio_20d'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # Volume price trend
        result_df['vpt'] = (df['volume'] * (df['close'] - df['close'].shift(1)) / df['close'].shift(1)).cumsum()
        result_df['vpt_sma'] = result_df['vpt'].rolling(window=20).mean()
        
        # On Balance Volume
        obv = (df['volume'] * np.where(df['close'] > df['close'].shift(1), 1, 
                                      np.where(df['close'] < df['close'].shift(1), -1, 0))).cumsum()
        result_df['obv'] = obv
        result_df['obv_sma'] = obv.rolling(window=20).mean()
        
        return result_df
        
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from technical indicators"""
        result_df = df.copy()
        
        # RSI momentum
        result_df['rsi_momentum'] = df['rsi_14'].diff()
        result_df['rsi_divergence'] = (df['rsi_14'].diff() * df['returns_1d']).rolling(window=5).mean()
        
        # MACD features
        result_df['macd_ratio'] = df['macd_value'] / df['macd_signal'].replace(0, np.nan)
        result_df['macd_momentum'] = df['macd_histogram'].diff()
        
        # Bollinger Bands features
        bb_width = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        result_df['bb_width'] = bb_width
        result_df['bb_width_ratio'] = bb_width / bb_width.rolling(window=20).mean()
        
        bb_position = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        result_df['bb_position'] = bb_position
        result_df['bb_position_momentum'] = bb_position.diff()
        
        # Moving average convergence
        result_df['sma_convergence'] = (df['sma_20'] - df['sma_50']) / df['close']
        result_df['ema_convergence'] = (df['ema_12'] - df['ema_26']) / df['close']
        
        # ATR features
        result_df['atr_ratio'] = df['atr_14'] / df['close']
        result_df['atr_momentum'] = df['atr_14'].pct_change()
        
        return result_df
        
    def create_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market structure features"""
        result_df = df.copy()
        
        # Higher highs, lower lows
        result_df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        result_df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # Price gaps
        result_df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
        result_df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
        result_df['gap_size'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Support and resistance levels
        result_df['resistance_20d'] = df['high'].rolling(window=20).max()
        result_df['support_20d'] = df['low'].rolling(window=20).min()
        result_df['resistance_distance'] = (result_df['resistance_20d'] - df['close']) / df['close']
        result_df['support_distance'] = (df['close'] - result_df['support_20d']) / df['close']
        
        return result_df
        
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        result_df = df.copy()
        
        # Apply all feature engineering steps
        result_df = self.create_price_features(result_df)
        result_df = self.create_volume_features(result_df)
        result_df = self.create_technical_features(result_df)
        result_df = self.create_market_structure_features(result_df)
        
        # Add time-based features
        result_df['day_of_week'] = pd.to_datetime(result_df['trade_date']).dt.dayofweek
        result_df['month'] = pd.to_datetime(result_df['trade_date']).dt.month
        result_df['quarter'] = pd.to_datetime(result_df['trade_date']).dt.quarter
        
        # Cyclical encoding for time features
        result_df['day_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
        result_df['day_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)
        result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
        result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
        
        return result_df

class LSTMXGBoostEnsemble:
    """Ensemble model combining LSTM and XGBoost for stock prediction"""
    
    def __init__(self, config_params=None):
        if config_params is None:
            config_params = config.ml
            
        self.config = config_params
        self.lookback_window = config_params.feature_lookback
        
        # Model components
        self.lstm_model = None
        self.xgb_model = None
        self.meta_model = None  # Meta-learner for combining predictions
        
        # Data preprocessing
        self.feature_engineer = FeatureEngineer(self.lookback_window)
        self.price_scaler = RobustScaler()
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # Feature importance tracking
        self.feature_importance = {}
        self.feature_names = []
        
        # Training history
        self.training_history = []
        
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build optimized LSTM model architecture"""
        inputs = Input(shape=input_shape)
        
        # First LSTM layer
        lstm1 = LSTM(self.config.lstm_units[0], return_sequences=True, 
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.dropout_rate)(inputs)
        lstm1 = Dropout(self.config.dropout_rate)(lstm1)
        
        # Second LSTM layer
        lstm2 = LSTM(self.config.lstm_units[1], return_sequences=True,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.dropout_rate)(lstm1)
        lstm2 = Dropout(self.config.dropout_rate)(lstm2)
        
        # Third LSTM layer
        lstm3 = LSTM(self.config.lstm_units[2], return_sequences=False,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.dropout_rate)(lstm2)
        lstm3 = Dropout(self.config.dropout_rate)(lstm3)
        
        # Dense layers
        dense1 = Dense(self.config.lstm_units[3], activation='relu')(lstm3)
        dense1 = Dropout(self.config.dropout_rate)(dense1)
        
        dense2 = Dense(25, activation='relu')(dense1)
        outputs = Dense(1, activation='linear')(dense2)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile with custom optimizer
        optimizer = Adam(learning_rate=self.config.learning_rate, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
        
    def prepare_lstm_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for LSTM training"""
        # Select features for LSTM (price and volume data work best)
        lstm_features = [
            'open', 'high', 'low', 'close', 'volume',
            'returns_1d', 'returns_5d', 'log_returns_1d',
            'momentum_5d', 'momentum_10d', 'hl_ratio',
            'volume_ratio_5d', 'volume_ratio_20d'
        ]
        
        # Ensure all features exist
        available_features = [col for col in lstm_features if col in df.columns]
        
        if len(available_features) < 5:
            raise ValueError(f"Insufficient features available: {available_features}")
        
        # Scale features
        feature_data = df[available_features].fillna(method='ffill').fillna(0)
        scaled_features = self.price_scaler.fit_transform(feature_data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback_window, len(scaled_features)):
            X.append(scaled_features[i-self.lookback_window:i])
            # Predict next day's return
            next_return = df['returns_1d'].iloc[i]
            y.append(next_return if not pd.isna(next_return) else 0)
        
        X, y = np.array(X), np.array(y)
        
        # Scale targets
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        return X, y_scaled, available_features
    
    def prepare_xgb_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for XGBoost training"""
        # Select features for XGBoost (technical indicators work best)
        xgb_features = [
            'rsi_9', 'rsi_14', 'macd_value', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'sma_20', 'sma_50',
            'ema_12', 'ema_26', 'atr_14', 'volume_sma_20',
            'stoch_k', 'stoch_d', 'williams_r',
            'rsi_momentum', 'macd_momentum', 'bb_width', 'bb_position',
            'sma_convergence', 'ema_convergence', 'atr_ratio',
            'higher_high', 'lower_low', 'gap_up', 'gap_down',
            'day_sin', 'day_cos', 'month_sin', 'month_cos'
        ]
        
        # Include lagged features
        for lag in [1, 2, 3, 5]:
            for feature in ['returns_1d', 'volume_ratio_20d', 'rsi_14', 'macd_histogram']:
                if feature in df.columns:
                    lagged_col = f'{feature}_lag_{lag}'
                    df[lagged_col] = df[feature].shift(lag)
                    xgb_features.append(lagged_col)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            for feature in ['returns_1d', 'volume_ratio_20d']:
                if feature in df.columns:
                    rolling_mean_col = f'{feature}_rolling_mean_{window}'
                    rolling_std_col = f'{feature}_rolling_std_{window}'
                    df[rolling_mean_col] = df[feature].rolling(window=window).mean()
                    df[rolling_std_col] = df[feature].rolling(window=window).std()
                    xgb_features.extend([rolling_mean_col, rolling_std_col])
        
        # Filter available features
        available_features = [col for col in xgb_features if col in df.columns]
        
        if len(available_features) < 10:
            raise ValueError(f"Insufficient XGBoost features: {len(available_features)}")
        
        # Prepare data
        feature_data = df[available_features].fillna(method='ffill').fillna(0)
        
        # Scale features
        scaled_features = self.feature_scaler.fit_transform(feature_data)
        
        # Target: predict 5-day forward return (optimal for 2-15 day strategies)
        target = df['close'].pct_change(5).shift(-5).fillna(0)
        
        # Remove NaN rows
        valid_idx = ~np.isnan(scaled_features).any(axis=1) & ~np.isnan(target)
        X = scaled_features[valid_idx]
        y = target[valid_idx]
        
        return X, y, available_features
    
    def train(self, df: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the ensemble model"""
        logger.info("Starting ensemble model training...")
        
        # Feature engineering
        df_features = self.feature_engineer.create_all_features(df)
        
        # Prepare data for both models
        try:
            # LSTM data preparation
            X_lstm, y_lstm, lstm_features = self.prepare_lstm_data(df_features)
            logger.info(f"LSTM data shape: X={X_lstm.shape}, y={y_lstm.shape}")
            
            # XGBoost data preparation  
            X_xgb, y_xgb, xgb_features = self.prepare_xgb_data(df_features)
            logger.info(f"XGBoost data shape: X={X_xgb.shape}, y={y_xgb.shape}")
            
            self.feature_names = {
                'lstm': lstm_features,
                'xgb': xgb_features
            }
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            return {'success': False, 'error': str(e)}
        
        # Train-validation split
        lstm_split = int(len(X_lstm) * (1 - validation_split))
        xgb_split = int(len(X_xgb) * (1 - validation_split))
        
        X_lstm_train, X_lstm_val = X_lstm[:lstm_split], X_lstm[lstm_split:]
        y_lstm_train, y_lstm_val = y_lstm[:lstm_split], y_lstm[lstm_split:]
        
        X_xgb_train, X_xgb_val = X_xgb[:xgb_split], X_xgb[xgb_split:]
        y_xgb_train, y_xgb_val = y_xgb[:xgb_split], y_xgb[xgb_split:]
        
        training_results = {}
        
        # Train LSTM model
        try:
            logger.info("Training LSTM model...")
            self.lstm_model = self.build_lstm_model((X_lstm.shape[1], X_lstm.shape[2]))
            
            # Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            model_checkpoint = ModelCheckpoint('models/lstm_best.h5', save_best_only=True, monitor='val_loss')
            
            # Train
            history = self.lstm_model.fit(
                X_lstm_train, y_lstm_train,
                validation_data=(X_lstm_val, y_lstm_val),
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                callbacks=[early_stopping, reduce_lr, model_checkpoint],
                verbose=1
            )
            
            training_results['lstm'] = {
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'epochs_trained': len(history.history['loss'])
            }
            
            logger.info(f"LSTM training completed. Final loss: {training_results['lstm']['final_loss']:.4f}")
            
        except Exception as e:
            logger.error(f"LSTM training failed: {str(e)}")
            training_results['lstm'] = {'error': str(e)}
        
        # Train XGBoost model
        try:
            logger.info("Training XGBoost model...")
            
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.xgb_learning_rate,
                subsample=self.config.subsample,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=1
            )
            
            # Train with early stopping
            self.xgb_model.fit(
                X_xgb_train, y_xgb_train,
                eval_set=[(X_xgb_val, y_xgb_val)],
                early_stopping_rounds=20,
                verbose=False
            )
            
            # Feature importance
            self.feature_importance['xgb'] = dict(zip(
                xgb_features, 
                self.xgb_model.feature_importances_
            ))
            
            # Validation score
            val_score = self.xgb_model.score(X_xgb_val, y_xgb_val)
            
            training_results['xgb'] = {
                'validation_r2': val_score,
                'n_estimators': self.xgb_model.n_estimators,
                'best_iteration': getattr(self.xgb_model, 'best_iteration', None)
            }
            
            logger.info(f"XGBoost training completed. Validation R²: {val_score:.4f}")
            
        except Exception as e:
            logger.error(f"XGBoost training failed: {str(e)}")
            training_results['xgb'] = {'error': str(e)}
        
        # Train meta-model (simple weighted average for now)
        if 'error' not in training_results.get('lstm', {}) and 'error' not in training_results.get('xgb', {}):
            logger.info("Training ensemble meta-model...")
            
            # Get predictions from both models on validation data
            lstm_val_pred = self.lstm_model.predict(X_lstm_val)
            xgb_val_pred = self.xgb_model.predict(X_xgb_val)
            
            # Align predictions (they might have different lengths)
            min_len = min(len(lstm_val_pred), len(xgb_val_pred), len(y_lstm_val))
            lstm_val_pred = lstm_val_pred[:min_len]
            xgb_val_pred = xgb_val_pred[:min_len]
            y_val_aligned = y_lstm_val[:min_len]
            
            # Simple weighted combination (can be enhanced with a neural network)
            weights = [0.6, 0.4]  # LSTM gets higher weight initially
            ensemble_pred = weights[0] * lstm_val_pred.flatten() + weights[1] * xgb_val_pred
            
            # Calculate ensemble performance
            ensemble_mse = np.mean((ensemble_pred - y_val_aligned) ** 2)
            ensemble_r2 = 1 - (ensemble_mse / np.var(y_val_aligned))
            
            self.meta_model = {'weights': weights, 'type': 'weighted_average'}
            
            training_results['ensemble'] = {
                'validation_r2': ensemble_r2,
                'validation_mse': ensemble_mse,
                'weights': weights
            }
            
            logger.info(f"Ensemble training completed. Validation R²: {ensemble_r2:.4f}")
        
        # Save models
        self.save_models()
        
        # Store training history
        self.training_history.append({
            'timestamp': datetime.now(),
            'results': training_results,
            'data_shape': df.shape,
            'features': self.feature_names
        })
        
        return {'success': True, 'results': training_results}
    
    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate predictions using the ensemble model"""
        if self.lstm_model is None or self.xgb_model is None:
            raise ValueError("Models not trained. Call train() first.")
        
        # Feature engineering
        df_features = self.feature_engineer.create_all_features(df)
        
        predictions = {}
        
        try:
            # LSTM predictions
            X_lstm, _, _ = self.prepare_lstm_data(df_features)
            lstm_pred = self.lstm_model.predict(X_lstm)
            predictions['lstm'] = self.target_scaler.inverse_transform(lstm_pred.reshape(-1, 1)).flatten()
            
            # XGBoost predictions
            X_xgb, _, _ = self.prepare_xgb_data(df_features)
            xgb_pred = self.xgb_model.predict(X_xgb)
            predictions['xgb'] = xgb_pred
            
            # Ensemble prediction
            if self.meta_model:
                min_len = min(len(predictions['lstm']), len(predictions['xgb']))
                ensemble_pred = (
                    self.meta_model['weights'][0] * predictions['lstm'][:min_len] +
                    self.meta_model['weights'][1] * predictions['xgb'][:min_len]
                )
                predictions['ensemble'] = ensemble_pred
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {'error': str(e)}
        
        return predictions
    
    def predict_single(self, df: pd.DataFrame) -> Tuple[float, float, Dict[str, Any]]:
        """Predict for a single stock with confidence and explanation"""
        predictions = self.predict(df)
        
        if 'error' in predictions:
            return 0.0, 0.0, {'error': predictions['error']}
        
        # Get latest prediction
        ensemble_pred = predictions.get('ensemble', predictions.get('lstm', [0]))[-1]
        
        # Calculate confidence based on model agreement
        lstm_pred = predictions.get('lstm', [ensemble_pred])[-1]
        xgb_pred = predictions.get('xgb', [ensemble_pred])[-1]
        
        # Confidence based on agreement between models
        agreement = 1 - abs(lstm_pred - xgb_pred) / (abs(lstm_pred) + abs(xgb_pred) + 1e-8)
        confidence = min(max(agreement, 0.1), 0.95)  # Clamp between 0.1 and 0.95
        
        # Explanation
        explanation = {
            'lstm_prediction': float(lstm_pred),
            'xgb_prediction': float(xgb_pred),
            'ensemble_prediction': float(ensemble_pred),
            'model_agreement': float(agreement),
            'prediction_direction': 'bullish' if ensemble_pred > 0 else 'bearish',
            'prediction_strength': abs(ensemble_pred)
        }
        
        return float(ensemble_pred), float(confidence), explanation
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance from trained models"""
        importance = {}
        
        if self.xgb_model and hasattr(self.xgb_model, 'feature_importances_'):
            xgb_importance = dict(zip(
                self.feature_names.get('xgb', []), 
                self.xgb_model.feature_importances_
            ))
            # Sort by importance
            importance['xgb'] = dict(sorted(xgb_importance.items(), 
                                          key=lambda x: x[1], reverse=True)[:20])
        
        if self.feature_importance:
            importance.update(self.feature_importance)
            
        return importance
    
    def save_models(self, base_path: str = 'models/') -> None:
        """Save trained models and scalers"""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Save LSTM model
            if self.lstm_model:
                self.lstm_model.save(f'{base_path}/lstm_model_{timestamp}.h5')
                
            # Save XGBoost model
            if self.xgb_model:
                joblib.dump(self.xgb_model, f'{base_path}/xgb_model_{timestamp}.pkl')
                
            # Save scalers and metadata
            joblib.dump({
                'price_scaler': self.price_scaler,
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler,
                'feature_names': self.feature_names,
                'meta_model': self.meta_model,
                'config': self.config.__dict__,
                'training_history': self.training_history
            }, f'{base_path}/ensemble_metadata_{timestamp}.pkl')
            
            logger.info(f"Models saved with timestamp: {timestamp}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def load_models(self, model_path: str) -> None:
        """Load trained models and scalers"""
        try:
            # Load metadata
            metadata = joblib.load(f'{model_path}/ensemble_metadata.pkl')
            self.price_scaler = metadata['price_scaler']
            self.feature_scaler = metadata['feature_scaler']
            self.target_scaler = metadata['target_scaler']
            self.feature_names = metadata['feature_names']
            self.meta_model = metadata['meta_model']
            self.training_history = metadata.get('training_history', [])
            
            # Load models
            self.lstm_model = tf.keras.models.load_model(f'{model_path}/lstm_model.h5')
            self.xgb_model = joblib.load(f'{model_path}/xgb_model.pkl')
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

# Example usage and testing
if __name__ == "__main__":
    # Test the ensemble model
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from data_management.stock_data_manager import Top100StocksDataManager
    
    try:
        # Get sample data
        data_manager = Top100StocksDataManager()
        
        # Fetch data for AAPL as test
        data_manager.cursor.execute("""
            SELECT dp.trade_date, dp.open, dp.high, dp.low, dp.close, dp.volume,
                   ti.rsi_9, ti.rsi_14, ti.macd_value, ti.macd_signal, ti.macd_histogram,
                   ti.bb_upper, ti.bb_middle, ti.bb_lower, ti.sma_20, ti.sma_50,
                   ti.ema_12, ti.ema_26, ti.atr_14, ti.volume_sma_20,
                   ti.stoch_k, ti.stoch_d, ti.williams_r
            FROM daily_prices dp
            LEFT JOIN technical_indicators ti ON dp.symbol_id = ti.symbol_id 
                AND dp.trade_date = ti.trade_date
            JOIN securities s ON dp.symbol_id = s.id
            WHERE s.symbol = 'AAPL'
              AND dp.trade_date >= '2022-01-01'
            ORDER BY dp.trade_date
        """)
        
        columns = [desc[0] for desc in data_manager.cursor.description]
        data = data_manager.cursor.fetchall()
        df = pd.DataFrame(data, columns=columns)
        
        if len(df) > 100:  # Need sufficient data
            # Initialize and train ensemble
            ensemble = LSTMXGBoostEnsemble()
            
            # Train on historical data
            results = ensemble.train(df)
            print(f"Training results: {results}")
            
            # Make predictions
            predictions = ensemble.predict(df.tail(50))  # Predict on recent data
            print(f"Recent predictions: {predictions.get('ensemble', 'No ensemble predictions')}")
            
            # Feature importance
            importance = ensemble.get_feature_importance()
            print("Top XGBoost features:")
            for feature, imp in list(importance.get('xgb', {}).items())[:10]:
                print(f"  {feature}: {imp:.4f}")
                
        else:
            print(f"Insufficient data for training: {len(df)} rows")
            
        data_manager.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()