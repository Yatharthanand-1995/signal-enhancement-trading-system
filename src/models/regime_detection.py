"""
Market Regime Detection using Hidden Markov Models
Implements 2-state and 3-state HMM for adaptive trading strategies
"""
import numpy as np
import pandas as pd
from hmmlearn import GaussianHMM
from sklearn.preprocessing import StandardScaler
import psycopg2
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

from config.config import config

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """Detect market regimes using Hidden Markov Models"""
    
    def __init__(self, n_regimes: int = 2, db_config=None):
        """
        Initialize regime detector
        
        Args:
            n_regimes: Number of market regimes (2 or 3)
                      2 = Low/High Volatility
                      3 = Bull/Bear/Sideways
        """
        self.n_regimes = n_regimes
        self.hmm_model = None
        self.scaler = StandardScaler()
        
        # Database connection
        if db_config is None:
            db_config = config.db
            
        self.db_config = {
            'host': db_config.host,
            'port': db_config.port,
            'database': db_config.database,
            'user': db_config.user,
            'password': db_config.password
        }
        
        # Regime definitions
        if n_regimes == 2:
            self.regime_names = {
                0: 'Low_Volatility',
                1: 'High_Volatility'
            }
        else:  # n_regimes == 3
            self.regime_names = {
                0: 'Bull_Market',
                1: 'Bear_Market', 
                2: 'Sideways_Market'
            }
            
        self.current_regime = None
        self.regime_probabilities = None
        self.regime_history = []
        
    def prepare_market_data(self, lookback_days: int = 252) -> pd.DataFrame:
        """Prepare market data for regime detection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            
            # Use SPY as market proxy, fall back to market-wide averages
            query = """
            WITH market_data AS (
                SELECT 
                    dp.trade_date,
                    dp.close,
                    dp.volume,
                    LAG(dp.close, 1) OVER (ORDER BY dp.trade_date) as prev_close,
                    LAG(dp.close, 20) OVER (ORDER BY dp.trade_date) as close_20d_ago
                FROM daily_prices dp
                JOIN securities s ON dp.symbol_id = s.id
                WHERE s.symbol = 'SPY'
                  AND dp.trade_date >= CURRENT_DATE - INTERVAL '%s days'
                ORDER BY dp.trade_date
            ),
            market_features AS (
                SELECT 
                    trade_date,
                    close,
                    volume,
                    (close - prev_close) / prev_close as daily_return,
                    (close - close_20d_ago) / close_20d_ago as return_20d
                FROM market_data
                WHERE prev_close IS NOT NULL
            ),
            volatility_features AS (
                SELECT 
                    trade_date,
                    close,
                    daily_return,
                    return_20d,
                    volume,
                    STDDEV(daily_return) OVER (ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as volatility_20d,
                    AVG(volume) OVER (ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as avg_volume_20d
                FROM market_features
            )
            SELECT 
                trade_date,
                close,
                daily_return,
                return_20d,
                volatility_20d * SQRT(252) as annualized_volatility,
                volume / avg_volume_20d as volume_ratio
            FROM volatility_features
            WHERE volatility_20d IS NOT NULL
            ORDER BY trade_date
            """ % (lookback_days + 50)  # Extra buffer for calculations
            
            df = pd.read_sql(query, conn, parse_dates=['trade_date'])
            conn.close()
            
            if df.empty:
                logger.warning("No market data found, creating synthetic features")
                return self._create_fallback_data(lookback_days)
                
            # Add technical features for regime detection
            df = self._add_regime_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing market data: {str(e)}")
            return self._create_fallback_data(lookback_days)
    
    def _create_fallback_data(self, lookback_days: int) -> pd.DataFrame:
        """Create fallback synthetic market data if real data unavailable"""
        dates = pd.date_range(end=datetime.now(), periods=lookback_days, freq='D')
        
        # Create synthetic market data with regime-like patterns
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.015, lookback_days)  # Typical market returns
        
        # Add regime changes
        regime_changes = [50, 150, 200]  # Change points
        volatilities = [0.01, 0.03, 0.015]  # Different volatility regimes
        
        for i, change_point in enumerate(regime_changes):
            if change_point < len(returns):
                vol = volatilities[i % len(volatilities)]
                returns[change_point:] = np.random.normal(0.0005, vol, 
                                                        len(returns) - change_point)
        
        prices = 100 * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            'trade_date': dates,
            'close': prices,
            'daily_return': returns,
            'return_20d': pd.Series(returns).rolling(20).sum(),
            'annualized_volatility': pd.Series(returns).rolling(20).std() * np.sqrt(252),
            'volume_ratio': np.random.uniform(0.5, 2.0, lookback_days)
        })
        
        df = self._add_regime_features(df)
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features specifically for regime detection"""
        # Rolling statistics for different windows
        for window in [5, 10, 20]:
            df[f'return_mean_{window}d'] = df['daily_return'].rolling(window).mean()
            df[f'return_std_{window}d'] = df['daily_return'].rolling(window).std()
            df[f'volume_mean_{window}d'] = df['volume_ratio'].rolling(window).mean()
        
        # Trend indicators
        df['price_sma_50'] = df['close'].rolling(50).mean()
        df['price_above_sma'] = (df['close'] > df['price_sma_50']).astype(int)
        
        # Momentum indicators
        df['momentum_5d'] = df['close'].pct_change(5)
        df['momentum_20d'] = df['close'].pct_change(20)
        
        # Volatility indicators
        df['volatility_ratio'] = (df['annualized_volatility'] / 
                                 df['annualized_volatility'].rolling(60).mean())
        
        # Market stress indicators
        df['drawdown'] = (df['close'] / df['close'].rolling(252).max() - 1)
        df['stress_indicator'] = (df['annualized_volatility'] > 0.25).astype(int)
        
        return df
    
    def fit(self, df: pd.DataFrame, features: Optional[List[str]] = None) -> Dict:
        """Fit HMM model to market data"""
        if features is None:
            features = [
                'daily_return', 'annualized_volatility', 'volume_ratio',
                'return_20d', 'momentum_20d', 'volatility_ratio'
            ]
        
        # Select and prepare features
        available_features = [f for f in features if f in df.columns]
        if len(available_features) < 3:
            raise ValueError(f"Insufficient features available: {available_features}")
        
        feature_data = df[available_features].fillna(method='ffill').dropna()
        
        if len(feature_data) < 50:
            raise ValueError("Insufficient data for regime detection")
        
        # Scale features
        scaled_features = self.scaler.fit_transform(feature_data)
        
        # Initialize and fit HMM model
        try:
            self.hmm_model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=42,
                verbose=False
            )
            
            # Fit model
            self.hmm_model.fit(scaled_features)
            
            # Predict regimes for training data
            regime_sequence = self.hmm_model.predict(scaled_features)
            regime_probs = self.hmm_model.predict_proba(scaled_features)
            
            # Analyze regime characteristics
            regime_analysis = self._analyze_regimes(df.iloc[-len(regime_sequence):].copy(), 
                                                  regime_sequence, regime_probs)
            
            # Store results
            results = {
                'success': True,
                'n_regimes': self.n_regimes,
                'log_likelihood': self.hmm_model.score(scaled_features),
                'aic': -2 * self.hmm_model.score(scaled_features) + 2 * self._get_n_parameters(),
                'regime_analysis': regime_analysis,
                'features_used': available_features,
                'training_samples': len(feature_data)
            }
            
            logger.info(f"HMM model fitted successfully. Log-likelihood: {results['log_likelihood']:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error fitting HMM model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _get_n_parameters(self) -> int:
        """Calculate number of parameters in HMM model"""
        if self.hmm_model is None:
            return 0
        
        n_features = self.hmm_model.means_.shape[1]
        n_states = self.hmm_model.n_components
        
        # Transition matrix: n_states * (n_states - 1)
        # Initial state probs: n_states - 1
        # Means: n_states * n_features
        # Covariances: n_states * n_features * (n_features + 1) / 2 (for full covariance)
        
        n_params = (n_states * (n_states - 1) + 
                   (n_states - 1) + 
                   n_states * n_features +
                   n_states * n_features * (n_features + 1) // 2)
        
        return n_params
    
    def _analyze_regimes(self, df: pd.DataFrame, regime_sequence: np.ndarray, 
                        regime_probs: np.ndarray) -> Dict:
        """Analyze characteristics of detected regimes"""
        analysis = {}
        
        df['regime'] = regime_sequence
        df['regime_prob'] = regime_probs.max(axis=1)
        
        for regime in range(self.n_regimes):
            regime_data = df[df['regime'] == regime]
            
            if len(regime_data) == 0:
                continue
                
            analysis[regime] = {
                'name': self.regime_names[regime],
                'frequency': len(regime_data) / len(df),
                'avg_return': regime_data['daily_return'].mean(),
                'volatility': regime_data['daily_return'].std() * np.sqrt(252),
                'avg_volume_ratio': regime_data['volume_ratio'].mean(),
                'max_drawdown': regime_data['drawdown'].min() if 'drawdown' in regime_data else None,
                'avg_duration': self._calculate_avg_regime_duration(regime_sequence, regime)
            }
        
        return analysis
    
    def _calculate_avg_regime_duration(self, regime_sequence: np.ndarray, regime: int) -> float:
        """Calculate average duration of a specific regime"""
        durations = []
        current_duration = 0
        
        for r in regime_sequence:
            if r == regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        # Add final duration if sequence ends in target regime
        if current_duration > 0:
            durations.append(current_duration)
            
        return np.mean(durations) if durations else 0
    
    def predict_regime(self, df: pd.DataFrame, features: Optional[List[str]] = None) -> Dict:
        """Predict current market regime"""
        if self.hmm_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if features is None:
            features = [
                'daily_return', 'annualized_volatility', 'volume_ratio',
                'return_20d', 'momentum_20d', 'volatility_ratio'
            ]
        
        # Prepare recent data
        available_features = [f for f in features if f in df.columns]
        feature_data = df[available_features].fillna(method='ffill').tail(1)
        
        if feature_data.empty:
            return {'error': 'No valid data for prediction'}
        
        # Scale features
        scaled_features = self.scaler.transform(feature_data)
        
        # Predict regime and probabilities
        regime = self.hmm_model.predict(scaled_features)[0]
        regime_probs = self.hmm_model.predict_proba(scaled_features)[0]
        
        # Store current state
        self.current_regime = regime
        self.regime_probabilities = regime_probs
        
        result = {
            'current_regime': int(regime),
            'regime_name': self.regime_names[regime],
            'probabilities': {
                self.regime_names[i]: float(prob) 
                for i, prob in enumerate(regime_probs)
            },
            'confidence': float(regime_probs.max()),
            'timestamp': datetime.now()
        }
        
        return result
    
    def get_regime_sequence(self, df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
        """Get full regime sequence for historical data"""
        if self.hmm_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if features is None:
            features = [
                'daily_return', 'annualized_volatility', 'volume_ratio',
                'return_20d', 'momentum_20d', 'volatility_ratio'
            ]
        
        # Prepare data
        available_features = [f for f in features if f in df.columns]
        feature_data = df[available_features].fillna(method='ffill').dropna()
        
        # Scale features
        scaled_features = self.scaler.transform(feature_data)
        
        # Predict regimes
        regime_sequence = self.hmm_model.predict(scaled_features)
        regime_probs = self.hmm_model.predict_proba(scaled_features)
        
        # Create result dataframe
        result_df = df.iloc[-len(regime_sequence):].copy()
        result_df['regime'] = regime_sequence
        result_df['regime_name'] = [self.regime_names[r] for r in regime_sequence]
        result_df['regime_confidence'] = regime_probs.max(axis=1)
        
        # Add individual regime probabilities
        for i, name in self.regime_names.items():
            result_df[f'prob_{name}'] = regime_probs[:, i]
        
        return result_df
    
    def detect_regime_changes(self, df: pd.DataFrame, min_duration: int = 5) -> List[Dict]:
        """Detect regime change points"""
        regime_df = self.get_regime_sequence(df)
        
        changes = []
        current_regime = None
        regime_start = None
        
        for idx, row in regime_df.iterrows():
            if row['regime'] != current_regime:
                # End previous regime
                if current_regime is not None and regime_start is not None:
                    duration = (idx - regime_start).days if hasattr(idx - regime_start, 'days') else idx - regime_start
                    if duration >= min_duration:
                        changes.append({
                            'from_regime': self.regime_names[current_regime],
                            'to_regime': self.regime_names[row['regime']],
                            'change_date': row['trade_date'] if 'trade_date' in row else idx,
                            'duration_days': duration,
                            'confidence': row['regime_confidence']
                        })
                
                current_regime = row['regime']
                regime_start = idx
        
        return changes
    
    def get_trading_adjustments(self) -> Dict[str, float]:
        """Get trading parameter adjustments based on current regime"""
        if self.current_regime is None:
            return {'error': 'No current regime detected'}
        
        regime_name = self.regime_names[self.current_regime]
        confidence = self.regime_probabilities.max() if self.regime_probabilities is not None else 0.5
        
        # Base adjustments for different regimes
        adjustments = {
            'Low_Volatility': {
                'position_size_multiplier': 1.2,  # Increase position sizes
                'stop_loss_multiplier': 1.0,      # Normal stops
                'profit_target_multiplier': 1.1,  # Slightly higher targets
                'holding_period_bias': 1.15       # Hold slightly longer
            },
            'High_Volatility': {
                'position_size_multiplier': 0.7,  # Decrease position sizes
                'stop_loss_multiplier': 0.8,      # Tighter stops
                'profit_target_multiplier': 1.3,  # Higher targets
                'holding_period_bias': 0.85       # Exit sooner
            },
            'Bull_Market': {
                'position_size_multiplier': 1.1,
                'stop_loss_multiplier': 1.1,
                'profit_target_multiplier': 1.2,
                'holding_period_bias': 1.1,
                'long_bias': 0.7  # 70% long bias
            },
            'Bear_Market': {
                'position_size_multiplier': 0.8,
                'stop_loss_multiplier': 0.9,
                'profit_target_multiplier': 1.1,
                'holding_period_bias': 0.9,
                'long_bias': 0.3  # 30% long bias
            },
            'Sideways_Market': {
                'position_size_multiplier': 0.9,
                'stop_loss_multiplier': 0.95,
                'profit_target_multiplier': 1.0,
                'holding_period_bias': 0.95,
                'long_bias': 0.5  # Neutral
            }
        }
        
        base_adjustments = adjustments.get(regime_name, adjustments['Low_Volatility'])
        
        # Adjust based on confidence
        confidence_factor = 0.5 + 0.5 * confidence  # Scale between 0.5 and 1.0
        
        final_adjustments = {}
        for key, value in base_adjustments.items():
            if key == 'long_bias':
                # Long bias doesn't scale with confidence
                final_adjustments[key] = value
            else:
                # Scale adjustment based on confidence
                adjustment = 1 + (value - 1) * confidence_factor
                final_adjustments[key] = adjustment
        
        final_adjustments.update({
            'regime': regime_name,
            'confidence': float(confidence),
            'recommendation': self._get_regime_recommendation(regime_name, confidence)
        })
        
        return final_adjustments
    
    def _get_regime_recommendation(self, regime_name: str, confidence: float) -> str:
        """Get trading recommendation based on regime"""
        recommendations = {
            'Low_Volatility': 'Increase position sizes, hold longer, normal risk management',
            'High_Volatility': 'Reduce position sizes, tighter stops, faster exits',
            'Bull_Market': 'Favor long positions, ride trends, wider stops',
            'Bear_Market': 'Defensive positioning, quick profits, tight risk control',
            'Sideways_Market': 'Range trading, quick scalps, mean reversion'
        }
        
        base_rec = recommendations.get(regime_name, 'Normal trading parameters')
        
        if confidence > 0.8:
            return f"HIGH CONFIDENCE: {base_rec}"
        elif confidence > 0.6:
            return f"MODERATE CONFIDENCE: {base_rec}"
        else:
            return f"LOW CONFIDENCE: Use conservative parameters, {base_rec.lower()}"
    
    def store_regime_detection(self, regime_data: Dict) -> None:
        """Store regime detection results in database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO market_regimes 
                (detection_date, regime_type, regime_name, volatility_level, confidence_score)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                datetime.now().date(),
                regime_data['current_regime'],
                regime_data['regime_name'],
                regime_data.get('volatility_level', 0),
                regime_data['confidence']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing regime detection: {str(e)}")
    
    def save_model(self, filepath: str) -> None:
        """Save trained model and scaler"""
        if self.hmm_model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'hmm_model': self.hmm_model,
            'scaler': self.scaler,
            'n_regimes': self.n_regimes,
            'regime_names': self.regime_names,
            'current_regime': self.current_regime,
            'regime_probabilities': self.regime_probabilities
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model and scaler"""
        model_data = joblib.load(filepath)
        
        self.hmm_model = model_data['hmm_model']
        self.scaler = model_data['scaler']
        self.n_regimes = model_data['n_regimes']
        self.regime_names = model_data['regime_names']
        self.current_regime = model_data.get('current_regime')
        self.regime_probabilities = model_data.get('regime_probabilities')
        
        logger.info(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    try:
        # Test 2-state regime detection (volatility-based)
        detector = MarketRegimeDetector(n_regimes=2)
        
        # Prepare market data
        market_data = detector.prepare_market_data(lookback_days=500)
        print(f"Market data shape: {market_data.shape}")
        
        if len(market_data) > 50:
            # Fit model
            results = detector.fit(market_data)
            print(f"Training results: {results}")
            
            # Predict current regime
            current_regime = detector.predict_regime(market_data.tail(10))
            print(f"Current regime: {current_regime}")
            
            # Get trading adjustments
            adjustments = detector.get_trading_adjustments()
            print(f"Trading adjustments: {adjustments}")
            
            # Detect regime changes
            changes = detector.detect_regime_changes(market_data.tail(100))
            print(f"Recent regime changes: {len(changes)}")
            
            # Test 3-state model
            detector3 = MarketRegimeDetector(n_regimes=3)
            results3 = detector3.fit(market_data)
            print(f"3-state model results: {results3}")
            
        else:
            print("Insufficient market data for testing")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()