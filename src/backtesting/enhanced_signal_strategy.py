"""
Enhanced Signal Strategy for Backtesting
Integrates all Phase 1-4 components for comprehensive signal generation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

# Import our enhanced components
from backtesting.backtest_engine import TradingStrategy
try:
    from data_management.volume_indicators import VolumeIndicatorCalculator
    from models.enhanced_regime_detector import EnhancedRegimeDetector
    from strategy.dynamic_signal_weighting import DynamicSignalWeighting
    from strategy.ensemble_signal_scoring import EnsembleSignalScoring
    from risk_management.dynamic_risk_manager import DynamicRiskManager
except ImportError as e:
    logging.warning(f"Import error: {e}. Using fallback implementations.")

logger = logging.getLogger(__name__)

@dataclass
class SignalComponents:
    """Container for all signal generation components"""
    technical_signals: Dict[str, float]
    volume_signals: Dict[str, float] 
    regime_signals: Dict[str, float]
    ensemble_score: float
    risk_metrics: Dict[str, float]
    regime_info: Dict[str, Any]

class EnhancedSignalStrategy(TradingStrategy):
    """
    Comprehensive strategy using all enhanced signal components:
    - Volume indicators (Phase 1)
    - Regime detection (Phase 2) 
    - Dynamic weighting (Phase 3)
    - Advanced risk management (Phase 4)
    """
    
    def __init__(self, 
                 # Volume parameters
                 volume_weight: float = 0.25,
                 
                 # Technical parameters  
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70,
                 macd_signal_threshold: float = 0.0,
                 
                 # Regime parameters
                 regime_weight: float = 0.30,
                 regime_confidence_threshold: float = 0.6,
                 
                 # Risk parameters
                 base_position_size: float = 0.04,  # 4% base allocation
                 max_position_size: float = 0.08,   # 8% max allocation
                 volatility_lookback: int = 20,
                 
                 # Signal thresholds
                 buy_threshold: float = 0.65,
                 sell_threshold: float = 0.35,
                 min_signal_strength: float = 0.5):
        
        # Store parameters
        self.volume_weight = volume_weight
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_signal_threshold = macd_signal_threshold
        self.regime_weight = regime_weight
        self.regime_confidence_threshold = regime_confidence_threshold
        self.base_position_size = base_position_size
        self.max_position_size = max_position_size
        self.volatility_lookback = volatility_lookback
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_signal_strength = min_signal_strength
        
        # Initialize components with fallback handling
        self._initialize_components()
        
        # Cache for recent calculations
        self._signal_cache = {}
        self._regime_cache = {}
        
    def _initialize_components(self):
        """Initialize all signal generation components with error handling"""
        try:
            self.volume_calculator = VolumeIndicatorCalculator()
        except:
            self.volume_calculator = None
            logger.warning("Volume calculator not available - using fallback")
            
        try:
            self.regime_detector = EnhancedRegimeDetector()
        except:
            self.regime_detector = None
            logger.warning("Regime detector not available - using fallback")
            
        try:
            self.signal_weighting = DynamicSignalWeighting()
        except:
            self.signal_weighting = None
            logger.warning("Signal weighting not available - using fallback")
            
        try:
            self.ensemble_scoring = EnsembleSignalScoring()
        except:
            self.ensemble_scoring = None
            logger.warning("Ensemble scoring not available - using fallback")
            
        try:
            self.risk_manager = DynamicRiskManager()
        except:
            self.risk_manager = None
            logger.warning("Risk manager not available - using fallback")
    
    def generate_signals(self, data: pd.DataFrame, current_date: datetime) -> Dict[str, Dict]:
        """Generate comprehensive trading signals"""
        
        signals = {}
        
        # Get current day data
        current_data = data[data['trade_date'] == current_date]
        
        if current_data.empty:
            return signals
            
        # Get historical data for each symbol
        for _, current_row in current_data.iterrows():
            symbol = current_row['symbol']
            
            try:
                # Get symbol historical data (last 60 days for calculations)
                symbol_historical = data[
                    (data['symbol'] == symbol) & 
                    (data['trade_date'] <= current_date)
                ].tail(60)
                
                if len(symbol_historical) < 20:  # Need minimum data
                    continue
                
                # Generate signal components
                signal_components = self._generate_signal_components(
                    symbol, symbol_historical, current_row
                )
                
                if signal_components is None:
                    continue
                
                # Calculate final signal
                final_signal = self._calculate_final_signal(signal_components)
                
                if final_signal['strength'] >= self.min_signal_strength:
                    signals[symbol] = final_signal
                    
            except Exception as e:
                logger.warning(f"Error generating signal for {symbol}: {str(e)}")
                continue
        
        return signals
    
    def _generate_signal_components(self, symbol: str, historical_data: pd.DataFrame, 
                                   current_row: pd.Series) -> Optional[SignalComponents]:
        """Generate all signal components for a symbol"""
        
        try:
            # 1. Technical Indicators Signals
            technical_signals = self._calculate_technical_signals(current_row)
            
            # 2. Volume Signals (Phase 1)
            volume_signals = self._calculate_volume_signals(historical_data)
            
            # 3. Regime Detection Signals (Phase 2)  
            regime_info = self._detect_market_regime(historical_data)
            regime_signals = self._calculate_regime_signals(regime_info)
            
            # 4. Ensemble Scoring (Phase 3)
            ensemble_score = self._calculate_ensemble_score(
                technical_signals, volume_signals, regime_signals
            )
            
            # 5. Risk Metrics (Phase 4)
            risk_metrics = self._calculate_risk_metrics(historical_data, regime_info)
            
            return SignalComponents(
                technical_signals=technical_signals,
                volume_signals=volume_signals,
                regime_signals=regime_signals,
                ensemble_score=ensemble_score,
                risk_metrics=risk_metrics,
                regime_info=regime_info
            )
            
        except Exception as e:
            logger.error(f"Error calculating signal components for {symbol}: {str(e)}")
            return None
    
    def _calculate_technical_signals(self, current_row: pd.Series) -> Dict[str, float]:
        """Calculate traditional technical indicator signals"""
        
        signals = {}
        
        # RSI signals
        if not pd.isna(current_row['rsi_14']):
            if current_row['rsi_14'] < self.rsi_oversold:
                signals['rsi'] = 0.8  # Strong buy
            elif current_row['rsi_14'] > self.rsi_overbought:
                signals['rsi'] = 0.2  # Strong sell
            else:
                # Normalize RSI to 0-1 scale
                signals['rsi'] = 1 - (current_row['rsi_14'] / 100)
        else:
            signals['rsi'] = 0.5  # Neutral
        
        # MACD signals
        if not pd.isna(current_row['macd_histogram']):
            if current_row['macd_histogram'] > self.macd_signal_threshold:
                signals['macd'] = 0.7
            else:
                signals['macd'] = 0.3
        else:
            signals['macd'] = 0.5
        
        # Bollinger Bands signals
        if (not pd.isna(current_row['bb_lower']) and 
            not pd.isna(current_row['bb_upper']) and 
            not pd.isna(current_row['close'])):
            
            bb_position = (current_row['close'] - current_row['bb_lower']) / \
                         (current_row['bb_upper'] - current_row['bb_lower'])
            signals['bollinger'] = max(0, min(1, bb_position))
        else:
            signals['bollinger'] = 0.5
        
        # Moving Average signals
        if (not pd.isna(current_row['sma_20']) and 
            not pd.isna(current_row['sma_50']) and
            not pd.isna(current_row['close'])):
            
            # Price vs SMA signals
            if current_row['close'] > current_row['sma_20'] > current_row['sma_50']:
                signals['ma_trend'] = 0.8  # Strong uptrend
            elif current_row['close'] < current_row['sma_20'] < current_row['sma_50']:
                signals['ma_trend'] = 0.2  # Strong downtrend
            else:
                signals['ma_trend'] = 0.5  # Neutral/mixed
        else:
            signals['ma_trend'] = 0.5
        
        return signals
    
    def _calculate_volume_signals(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-based signals using Phase 1 components"""
        
        signals = {'obv': 0.5, 'cmf': 0.5, 'mfi': 0.5, 'vwap': 0.5}
        
        if self.volume_calculator is None or len(historical_data) < 20:
            return signals
        
        try:
            # Calculate volume indicators
            volume_data = self.volume_calculator.calculate_all_indicators(historical_data)
            
            if not volume_data.empty:
                latest = volume_data.iloc[-1]
                
                # OBV signal (trend confirmation)
                if not pd.isna(latest.get('obv_signal', np.nan)):
                    if latest['obv_signal'] > 0:
                        signals['obv'] = 0.7
                    elif latest['obv_signal'] < 0:
                        signals['obv'] = 0.3
                
                # CMF signal (money flow strength)
                if not pd.isna(latest.get('cmf', np.nan)):
                    # CMF ranges from -1 to 1, normalize to 0-1
                    signals['cmf'] = max(0, min(1, (latest['cmf'] + 1) / 2))
                
                # MFI signal (momentum with volume)
                if not pd.isna(latest.get('mfi', np.nan)):
                    if latest['mfi'] < 20:
                        signals['mfi'] = 0.8  # Oversold
                    elif latest['mfi'] > 80:
                        signals['mfi'] = 0.2  # Overbought
                    else:
                        signals['mfi'] = 1 - (latest['mfi'] / 100)
                
                # VWAP signal (price vs volume-weighted average)
                close_price = historical_data['close'].iloc[-1]
                if not pd.isna(latest.get('vwap', np.nan)) and close_price > 0:
                    if close_price > latest['vwap']:
                        signals['vwap'] = 0.7
                    else:
                        signals['vwap'] = 0.3
                        
        except Exception as e:
            logger.warning(f"Error calculating volume signals: {str(e)}")
        
        return signals
    
    def _detect_market_regime(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect current market regime using Phase 2 components"""
        
        default_regime = {
            'regime': 'Low_Volatility',
            'confidence': 0.5,
            'volatility_regime': 'Normal',
            'trend_strength': 0.5,
            'regime_probability': {'Low_Volatility': 0.6, 'High_Volatility': 0.4}
        }
        
        if self.regime_detector is None or len(historical_data) < 30:
            return default_regime
        
        try:
            # Detect regime using enhanced detector
            regime_result = self.regime_detector.detect_regime(historical_data)
            
            if regime_result:
                return {
                    'regime': regime_result.get('regime_name', 'Low_Volatility'),
                    'confidence': regime_result.get('confidence', 0.5),
                    'volatility_regime': regime_result.get('volatility_regime', 'Normal'),
                    'trend_strength': regime_result.get('trend_strength', 0.5),
                    'regime_probability': regime_result.get('regime_probabilities', default_regime['regime_probability'])
                }
            
        except Exception as e:
            logger.warning(f"Error detecting regime: {str(e)}")
        
        return default_regime
    
    def _calculate_regime_signals(self, regime_info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate regime-based trading signals"""
        
        signals = {'regime_momentum': 0.5, 'regime_mean_reversion': 0.5, 'regime_volatility': 0.5}
        
        try:
            regime = regime_info.get('regime', 'Low_Volatility')
            confidence = regime_info.get('confidence', 0.5)
            
            if confidence < self.regime_confidence_threshold:
                return signals  # Low confidence, return neutral
            
            # Regime-specific signal adjustments
            if regime == 'High_Volatility':
                # In high volatility, favor mean reversion
                signals['regime_momentum'] = 0.3
                signals['regime_mean_reversion'] = 0.8
                signals['regime_volatility'] = 0.2  # Reduce position sizing signal
                
            elif regime == 'Low_Volatility': 
                # In low volatility, favor momentum
                signals['regime_momentum'] = 0.8
                signals['regime_mean_reversion'] = 0.3
                signals['regime_volatility'] = 0.8  # Increase position sizing signal
                
            elif regime == 'Crisis':
                # Crisis regime - very defensive
                signals['regime_momentum'] = 0.1
                signals['regime_mean_reversion'] = 0.2
                signals['regime_volatility'] = 0.1
                
            # Adjust based on trend strength
            trend_strength = regime_info.get('trend_strength', 0.5)
            signals['regime_momentum'] *= (0.5 + trend_strength * 0.5)
            
        except Exception as e:
            logger.warning(f"Error calculating regime signals: {str(e)}")
        
        return signals
    
    def _calculate_ensemble_score(self, technical_signals: Dict[str, float],
                                volume_signals: Dict[str, float], 
                                regime_signals: Dict[str, float]) -> float:
        """Calculate ensemble score using Phase 3 components"""
        
        try:
            if self.ensemble_scoring is not None:
                # Use ensemble scoring system
                all_signals = {
                    **technical_signals,
                    **volume_signals, 
                    **regime_signals
                }
                
                result = self.ensemble_scoring.calculate_ensemble_score(
                    signal_values=all_signals,
                    market_regime='Low_Volatility',  # Will be overridden
                    regime_confidence=0.7
                )
                
                return result.get('ensemble_score', 0.5)
            else:
                # Fallback calculation
                tech_avg = np.mean(list(technical_signals.values()))
                vol_avg = np.mean(list(volume_signals.values()))
                regime_avg = np.mean(list(regime_signals.values()))
                
                # Weighted average
                ensemble_score = (
                    tech_avg * 0.4 +  # 40% technical
                    vol_avg * self.volume_weight +  # Volume weight
                    regime_avg * self.regime_weight  # Regime weight
                )
                
                return max(0, min(1, ensemble_score))
                
        except Exception as e:
            logger.warning(f"Error calculating ensemble score: {str(e)}")
            return 0.5
    
    def _calculate_risk_metrics(self, historical_data: pd.DataFrame, 
                              regime_info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk metrics using Phase 4 components"""
        
        default_metrics = {
            'volatility': 0.02,
            'max_position_risk': 0.04,
            'stop_loss_distance': 0.03,
            'profit_target_distance': 0.06
        }
        
        if self.risk_manager is None or len(historical_data) < 20:
            return default_metrics
        
        try:
            # Calculate recent volatility
            returns = historical_data['close'].pct_change().dropna()
            if len(returns) >= self.volatility_lookback:
                recent_vol = returns.tail(self.volatility_lookbook).std() * np.sqrt(252)
            else:
                recent_vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.02
            
            # Regime-adjusted risk metrics
            regime = regime_info.get('regime', 'Low_Volatility')
            vol_multiplier = 1.0
            
            if regime == 'High_Volatility':
                vol_multiplier = 1.5
            elif regime == 'Crisis':
                vol_multiplier = 2.0
            elif regime == 'Low_Volatility':
                vol_multiplier = 0.8
            
            return {
                'volatility': recent_vol,
                'max_position_risk': min(self.max_position_size, 
                                       self.base_position_size * (1 / vol_multiplier)),
                'stop_loss_distance': recent_vol * 2 * vol_multiplier,
                'profit_target_distance': recent_vol * 3 * vol_multiplier
            }
            
        except Exception as e:
            logger.warning(f"Error calculating risk metrics: {str(e)}")
            return default_metrics
    
    def _calculate_final_signal(self, components: SignalComponents) -> Dict[str, Any]:
        """Calculate final trading signal from all components"""
        
        # Get ensemble score as primary signal
        primary_score = components.ensemble_score
        
        # Apply regime adjustments
        regime = components.regime_info.get('regime', 'Low_Volatility')
        regime_confidence = components.regime_info.get('confidence', 0.5)
        
        # Adjust for regime confidence
        if regime_confidence < 0.5:
            primary_score = 0.5 + (primary_score - 0.5) * 0.5  # Reduce signal strength
        
        # Determine direction and strength
        if primary_score >= self.buy_threshold:
            direction = 'BUY'
            strength = min(1.0, (primary_score - 0.5) * 2)  # Scale to 0-1
        elif primary_score <= self.sell_threshold:
            direction = 'SELL'  
            strength = min(1.0, (0.5 - primary_score) * 2)  # Scale to 0-1
        else:
            direction = 'HOLD'
            strength = 0.0
        
        # Risk-adjusted confidence
        risk_vol = components.risk_metrics.get('volatility', 0.02)
        vol_adjustment = max(0.5, min(1.2, 0.02 / risk_vol))  # Inverse vol relationship
        confidence = min(1.0, regime_confidence * vol_adjustment)
        
        return {
            'direction': direction,
            'strength': strength,
            'confidence': confidence,
            'ensemble_score': primary_score,
            'regime': regime,
            'risk_metrics': components.risk_metrics,
            'components': {
                'technical': components.technical_signals,
                'volume': components.volume_signals,
                'regime': components.regime_signals
            }
        }
    
    def get_position_size(self, signal: Dict, portfolio_value: float, regime: str) -> int:
        """Calculate position size using risk management"""
        
        try:
            # Get risk metrics from signal
            risk_metrics = signal.get('risk_metrics', {})
            max_risk = risk_metrics.get('max_position_risk', self.base_position_size)
            
            # Base allocation
            base_allocation = min(max_risk, self.max_position_size)
            
            # Adjust for signal strength
            signal_strength = signal.get('strength', 0.5)
            strength_adjustment = 0.5 + (signal_strength * 0.5)  # 0.5 to 1.0
            
            # Adjust for confidence
            confidence = signal.get('confidence', 0.7)
            confidence_adjustment = max(0.3, confidence)  # Minimum 30%
            
            # Final allocation
            final_allocation = base_allocation * strength_adjustment * confidence_adjustment
            position_value = portfolio_value * final_allocation
            
            # Convert to shares (estimate $100 per share)
            estimated_price = 100
            shares = max(1, int(position_value / estimated_price))
            
            return shares
            
        except Exception as e:
            logger.warning(f"Error calculating position size: {str(e)}")
            # Fallback to simple calculation
            allocation = min(0.02, self.base_position_size)
            return max(1, int(portfolio_value * allocation / 100))
    
    def get_exit_rules(self) -> Dict[str, Any]:
        """Get exit rules for positions"""
        return {
            'rsi_exit': True,
            'macd_exit': True,
            'regime_change_exit': True,
            'volume_divergence_exit': True,
            'profit_target': 0.12,  # 12% profit target
            'stop_loss': 0.06,      # 6% stop loss  
            'trailing_stop': True,
            'max_holding_days': 30
        }


# Baseline strategy for comparison
class BaselineStrategy(TradingStrategy):
    """Simple baseline strategy for performance comparison"""
    
    def __init__(self, rsi_oversold: float = 30, rsi_overbought: float = 70):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    def generate_signals(self, data: pd.DataFrame, current_date: datetime) -> Dict[str, Dict]:
        """Simple RSI-based signals"""
        
        signals = {}
        current_data = data[data['trade_date'] == current_date]
        
        for _, row in current_data.iterrows():
            symbol = row['symbol']
            
            if pd.isna(row['rsi_14']):
                continue
            
            if row['rsi_14'] < self.rsi_oversold:
                signals[symbol] = {
                    'direction': 'BUY',
                    'strength': 0.7,
                    'confidence': 0.6
                }
            elif row['rsi_14'] > self.rsi_overbought:
                signals[symbol] = {
                    'direction': 'SELL',
                    'strength': 0.7, 
                    'confidence': 0.6
                }
        
        return signals
    
    def get_position_size(self, signal: Dict, portfolio_value: float, regime: str) -> int:
        """Fixed 3% allocation"""
        return max(1, int(portfolio_value * 0.03 / 100))
    
    def get_exit_rules(self) -> Dict[str, Any]:
        """Basic exit rules"""
        return {
            'profit_target': 0.08,
            'stop_loss': 0.04
        }