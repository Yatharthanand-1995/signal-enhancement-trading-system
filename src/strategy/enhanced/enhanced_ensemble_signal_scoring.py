"""
Enhanced Ensemble Signal Scoring System

Advanced signal scoring with academic research-backed improvements:
- Regime-aware signal weighting and confidence adjustment
- Macro-economic signal overlay integration
- Dynamic position sizing with Kelly criterion
- Factor timing and attribution analysis
- Transaction cost optimization

Based on academic research:
- Regime Switching Models (Ang & Bekaert 2002)
- Multi-Factor Asset Pricing (Fama & French 2015)
- Kelly Criterion Optimization (Kelly 1956, Grinold & Kahn 1999)
- Dynamic Asset Allocation (Guidolin & Timmermann 2007)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from scipy import stats
from collections import defaultdict, deque

# Import existing system components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ensemble_signal_scoring import EnsembleSignalScorer, SignalDirection, EnsembleSignal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import MarketRegime and AdvancedRegimeDetector from regime detection module
from .regime_detection import MarketRegime, AdvancedRegimeDetector
from .macro_integration import MacroIntegrationEngine
from .position_sizing import DynamicPositionSizer

@dataclass
class EnhancedEnsembleSignal:
    """Enhanced signal with regime awareness and attribution"""
    # Base signal components
    symbol: str
    timestamp: datetime
    direction: SignalDirection
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    composite_score: float
    
    # Enhanced components
    market_regime: MarketRegime
    base_signal: Dict[str, Any] = field(default_factory=dict)
    regime_adjustment: float = 1.0
    macro_adjustment: float = 1.0  
    factor_adjustment: float = 1.0
    final_confidence: float = 0.0
    
    # Position sizing and risk
    optimal_position_size: float = 0.0
    expected_return: float = 0.0
    risk_level: str = "Medium"
    
    # Attribution analysis
    enhancement_attribution: Dict[str, float] = field(default_factory=dict)
    component_scores: Dict[str, float] = field(default_factory=dict)
    
    # Trading decision
    should_trade: bool = False
    trade_rationale: str = ""

class EnhancedEnsembleSignalScoring:
    """
    Enhanced ensemble signal scoring with regime awareness, macro integration,
    and academic research-backed improvements
    """
    
    def __init__(self, 
                 base_scorer: Optional[EnsembleSignalScorer] = None,
                 enable_regime_detection: bool = True,
                 enable_macro_integration: bool = True,
                 enable_factor_timing: bool = True,
                 enable_dynamic_sizing: bool = True):
        """
        Initialize enhanced ensemble signal scoring system
        
        Args:
            base_scorer: Existing ensemble signal scorer (will create if None)
            enable_regime_detection: Enable market regime detection
            enable_macro_integration: Enable macro-economic signal overlay
            enable_factor_timing: Enable factor timing adjustments
            enable_dynamic_sizing: Enable dynamic position sizing
        """
        # Base system
        self.base_scorer = base_scorer or EnsembleSignalScorer()
        
        # Enhancement flags
        self.enable_regime_detection = enable_regime_detection
        self.enable_macro_integration = enable_macro_integration  
        self.enable_factor_timing = enable_factor_timing
        self.enable_dynamic_sizing = enable_dynamic_sizing
        
        # Regime-specific indicator weights
        self.regime_weights = {
            MarketRegime.LOW_VOL_BULL: {
                'rsi': 0.20, 'macd': 0.25, 'bollinger_bands': 0.20, 
                'volume': 0.15, 'moving_averages': 0.15, 'momentum': 0.05
            },
            MarketRegime.HIGH_VOL_BULL: {
                'rsi': 0.25, 'macd': 0.15, 'bollinger_bands': 0.25, 
                'volume': 0.20, 'moving_averages': 0.10, 'momentum': 0.05
            },
            MarketRegime.BEAR_MARKET: {
                'rsi': 0.30, 'macd': 0.10, 'bollinger_bands': 0.30, 
                'volume': 0.15, 'moving_averages': 0.10, 'momentum': 0.05
            },
            MarketRegime.CRISIS: {
                'rsi': 0.35, 'macd': 0.05, 'bollinger_bands': 0.35, 
                'volume': 0.20, 'moving_averages': 0.05, 'momentum': 0.00
            }
        }
        
        # Regime-specific confidence thresholds
        self.regime_confidence_thresholds = {
            MarketRegime.LOW_VOL_BULL: 0.65,    # Standard threshold
            MarketRegime.HIGH_VOL_BULL: 0.70,   # Higher bar for volatile conditions
            MarketRegime.BEAR_MARKET: 0.75,     # Much higher bar for risk-off
            MarketRegime.CRISIS: 0.80           # Only highest conviction trades
        }
        
        # Position sizing parameters
        self.max_position_size = 0.02  # 2% maximum per position
        self.target_portfolio_volatility = 0.12  # 12% annual target
        
        # Regime history for smoothing
        self.regime_history = deque(maxlen=10)
        
        # Performance tracking
        self.signal_history = []
        self.enhancement_stats = defaultdict(list)
        
        # Initialize advanced regime detector
        self.regime_detector = AdvancedRegimeDetector() if self.enable_regime_detection else None
        
        # Initialize macro integration engine
        self.macro_engine = MacroIntegrationEngine() if self.enable_macro_integration else None
        
        # Initialize dynamic position sizer
        self.position_sizer = DynamicPositionSizer(
            max_position_size=self.max_position_size,
            kelly_fraction=0.25,
            min_position_size=0.001
        ) if self.enable_dynamic_sizing else None
        
        logger.info("Enhanced Ensemble Signal Scorer initialized with academic improvements")
    
    def calculate_enhanced_signal(self, symbol: str, data: pd.DataFrame) -> EnhancedEnsembleSignal:
        """
        Calculate enhanced signal with all academic improvements integrated
        
        Args:
            symbol: Stock symbol
            data: Historical price/volume data with technical indicators
            
        Returns:
            Enhanced signal with regime awareness and attribution
        """
        try:
            # Step 1: Create technical indicators from data 
            tech_indicators = self._extract_technical_indicators(data)
            volume_signals = self._extract_volume_signals(data)
            
            # Step 2: Get base signal from existing system
            base_signal = self.base_scorer.calculate_ensemble_score(
                symbol, data, tech_indicators, volume_signals
            )
            
            # Step 3: Detect current market regime using advanced detector
            if self.enable_regime_detection and self.regime_detector:
                regime_result = self.regime_detector.detect_market_regime(data)
                current_regime = regime_result.current_regime
            else:
                current_regime = MarketRegime.LOW_VOL_BULL
            
            # Step 3: Apply regime-specific technical indicator weights
            regime_adjusted_signal = self._apply_regime_weights(base_signal, current_regime)
            
            # Step 4: Apply macro overlay with advanced integration
            if self.enable_macro_integration and self.macro_engine:
                macro_result = self.macro_engine.analyze_macro_environment(symbol)
                macro_adjusted_signal = self._apply_macro_overlay(regime_adjusted_signal, symbol, macro_result)
            else:
                macro_adjusted_signal = regime_adjusted_signal
            
            # Step 5: Apply factor timing (placeholder for now)  
            factor_adjusted_signal = self._apply_factor_timing(macro_adjusted_signal, symbol) if self.enable_factor_timing else macro_adjusted_signal
            
            # Step 6: Calculate optimal position size using Kelly Criterion
            if self.enable_dynamic_sizing and self.position_sizer:
                # Get macro result if available
                macro_result_dict = None
                if hasattr(self, 'macro_engine') and self.macro_engine:
                    try:
                        macro_result_obj = self.macro_engine.analyze_macro_environment(symbol)
                        macro_result_dict = {
                            'equity_boost': macro_result_obj.equity_boost,
                            'risk_adjustment': macro_result_obj.risk_adjustment,
                            'signal_confidence': macro_result_obj.signal_confidence
                        }
                    except:
                        macro_result_dict = None
                
                # Calculate optimal position size
                sizing_result = self.position_sizer.calculate_optimal_position_size(
                    signal_strength=factor_adjusted_signal.strength,
                    signal_confidence=factor_adjusted_signal.confidence,
                    expected_return=factor_adjusted_signal.strength * 0.05,  # Scale to expected return
                    market_regime=current_regime.value,
                    macro_result=macro_result_dict,
                    historical_data=data,
                    symbol=symbol
                )
                position_size = sizing_result.optimal_size
            else:
                position_size = 0.01
            
            # Step 7: Determine if trade should be executed
            should_trade, rationale = self._should_execute_trade(factor_adjusted_signal, current_regime, position_size)
            
            # Step 8: Calculate enhancement attribution
            attribution = self._calculate_enhancement_attribution(base_signal, factor_adjusted_signal)
            
            # Create enhanced signal
            enhanced_signal = EnhancedEnsembleSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                direction=factor_adjusted_signal.direction,
                strength=factor_adjusted_signal.strength,
                confidence=factor_adjusted_signal.confidence,
                composite_score=factor_adjusted_signal.composite_score,
                market_regime=current_regime,
                base_signal=self._convert_signal_to_dict(base_signal),
                final_confidence=factor_adjusted_signal.confidence,
                optimal_position_size=position_size,
                should_trade=should_trade,
                trade_rationale=rationale,
                enhancement_attribution=attribution
            )
            
            # Store signal for performance tracking
            self.signal_history.append(enhanced_signal)
            
            # Log enhanced signal generation
            logger.info(f"Enhanced signal calculated for {symbol}: {enhanced_signal.direction.name} "
                       f"(strength: {enhanced_signal.strength:.2f}, confidence: {enhanced_signal.confidence:.2f}, "
                       f"regime: {current_regime.value})")
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Error calculating enhanced signal for {symbol}: {e}")
            raise
    
    def _detect_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        Detect current market regime based on VIX, correlations, and market conditions
        
        This is a simplified version - will be enhanced in regime_detection.py
        """
        try:
            # Simplified regime detection based on volatility
            if len(data) < 20:
                return MarketRegime.LOW_VOL_BULL
            
            # Calculate 20-day realized volatility as proxy for VIX
            returns = data['close'].pct_change().dropna()
            if len(returns) < 20:
                return MarketRegime.LOW_VOL_BULL
                
            realized_vol = returns.tail(20).std() * np.sqrt(252) * 100  # Annualized %
            
            # Simple regime classification
            if realized_vol > 50:
                regime = MarketRegime.CRISIS
            elif realized_vol > 35:
                regime = MarketRegime.BEAR_MARKET  
            elif realized_vol > 20:
                regime = MarketRegime.HIGH_VOL_BULL
            else:
                regime = MarketRegime.LOW_VOL_BULL
            
            # Apply regime smoothing
            regime = self._smooth_regime_transition(regime)
            
            return regime
            
        except Exception as e:
            logger.warning(f"Error detecting market regime: {e}, defaulting to LOW_VOL_BULL")
            return MarketRegime.LOW_VOL_BULL
    
    def _smooth_regime_transition(self, new_regime: MarketRegime) -> MarketRegime:
        """Apply smoothing to avoid rapid regime switches"""
        if len(self.regime_history) < 3:
            self.regime_history.append(new_regime)
            return new_regime
        
        # Count recent regime occurrences
        recent_regimes = list(self.regime_history)[-3:]
        regime_counts = {regime: recent_regimes.count(regime) for regime in set(recent_regimes)}
        
        # If new regime doesn't appear in majority of recent history, keep current
        if regime_counts.get(new_regime, 0) < 2:
            current_regime = self.regime_history[-1]
            self.regime_history.append(current_regime)
            return current_regime
        
        self.regime_history.append(new_regime)
        return new_regime
    
    def _apply_regime_weights(self, base_signal: EnsembleSignal, regime: MarketRegime) -> EnsembleSignal:
        """
        Apply regime-specific weights to technical indicators
        """
        try:
            # Get regime-specific weights
            weights = self.regime_weights.get(regime, self.regime_weights[MarketRegime.LOW_VOL_BULL])
            
            # Apply regime-specific confidence adjustment
            confidence_threshold = self.regime_confidence_thresholds[regime]
            regime_confidence_multiplier = min(1.2, max(0.8, base_signal.confidence / confidence_threshold))
            
            # Create adjusted signal
            adjusted_signal = EnsembleSignal(
                symbol=base_signal.symbol,
                timestamp=base_signal.timestamp,
                direction=base_signal.direction,
                strength=base_signal.strength,
                confidence=base_signal.confidence * regime_confidence_multiplier,
                composite_score=base_signal.composite_score * regime_confidence_multiplier,
                technical_signals=base_signal.technical_signals,
                volume_signals=base_signal.volume_signals,
                regime_info={'regime': regime.value, 'weights_applied': weights}
            )
            
            return adjusted_signal
            
        except Exception as e:
            logger.warning(f"Error applying regime weights: {e}, returning base signal")
            return base_signal
    
    def _apply_macro_overlay(self, signal: EnsembleSignal, symbol: str, macro_result) -> EnsembleSignal:
        """
        Apply macro-economic overlay to signal using advanced macro analysis
        
        Integrates:
        - Fed policy stance and interest rate environment
        - Inflation expectations and commodity signals
        - Global macro factors and currency movements
        - Economic surprise indices
        """
        try:
            # Apply macro adjustments to signal
            equity_boost = macro_result.equity_boost
            risk_adjustment = macro_result.risk_adjustment
            
            # Adjust signal strength based on macro environment
            adjusted_strength = signal.strength * (1 + equity_boost)
            adjusted_strength = max(-1.0, min(1.0, adjusted_strength))
            
            # Adjust confidence based on macro regime clarity
            macro_confidence_boost = macro_result.signal_confidence - 0.5  # Center around 0
            adjusted_confidence = signal.confidence * (1 + 0.2 * macro_confidence_boost)
            adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
            
            # Create adjusted signal
            adjusted_signal = EnsembleSignal(
                symbol=signal.symbol,
                timestamp=signal.timestamp,
                direction=signal.direction,
                strength=adjusted_strength,
                confidence=adjusted_confidence,
                composite_score=signal.composite_score * (1 + equity_boost),
                technical_signals=signal.technical_signals,
                volume_signals=signal.volume_signals,
                regime_info=signal.regime_info,
                signal_weights=signal.signal_weights,
                weight_explanation=f"{signal.weight_explanation} + Macro({macro_result.macro_regime.value})",
                price=signal.price,
                supporting_factors=signal.supporting_factors + [f"Macro: {macro_result.macro_narrative}"],
                risk_factors=signal.risk_factors,
                recommended_action=signal.recommended_action
            )
            
            logger.info(f"Macro overlay applied: {macro_result.macro_regime.value} "
                       f"(boost: {equity_boost:+.2f}, risk adj: {risk_adjustment:.2f})")
            
            return adjusted_signal
            
        except Exception as e:
            logger.warning(f"Error applying macro overlay: {e}")
            return signal
    
    def _apply_factor_timing(self, signal: EnsembleSignal, symbol: str) -> EnsembleSignal:
        """
        Apply factor timing adjustments (placeholder)
        Will be implemented in factor_analysis.py
        """
        # Placeholder - will be enhanced in Phase 1 Day 7-10
        return signal
    
    def _calculate_position_size(self, signal: EnsembleSignal, regime: MarketRegime, data: pd.DataFrame) -> float:
        """
        Calculate optimal position size using Kelly criterion and regime adjustments
        """
        try:
            # Base Kelly sizing (simplified)
            win_prob = min(0.75, signal.confidence)
            avg_win = 0.08  # 8% average winner
            avg_loss = 0.04  # 4% average loser
            
            # Kelly formula: f = (bp - q) / b
            b = avg_win / avg_loss
            kelly_fraction = (b * win_prob - (1 - win_prob)) / b
            kelly_size = max(0, kelly_fraction * 0.25)  # 25% Kelly scaling
            
            # Regime adjustment
            regime_multipliers = {
                MarketRegime.LOW_VOL_BULL: 1.0,
                MarketRegime.HIGH_VOL_BULL: 0.8,
                MarketRegime.BEAR_MARKET: 0.6,
                MarketRegime.CRISIS: 0.4
            }
            
            regime_adjusted_size = kelly_size * regime_multipliers[regime]
            
            # Volatility targeting
            returns = data['close'].pct_change().dropna()
            if len(returns) >= 20:
                realized_vol = returns.tail(20).std() * np.sqrt(252)
                vol_adjustment = self.target_portfolio_volatility / max(0.05, realized_vol)
                vol_adjustment = max(0.5, min(2.0, vol_adjustment))
            else:
                vol_adjustment = 1.0
            
            # Final position size
            final_size = regime_adjusted_size * vol_adjustment
            
            # Apply maximum position size constraint
            return min(final_size, self.max_position_size)
            
        except Exception as e:
            logger.warning(f"Error calculating position size: {e}, using default")
            return 0.01
    
    def _should_execute_trade(self, signal: EnsembleSignal, regime: MarketRegime, position_size: float) -> Tuple[bool, str]:
        """
        Determine if trade should be executed based on enhanced criteria
        """
        try:
            reasons = []
            
            # Confidence threshold check
            min_confidence = self.regime_confidence_thresholds[regime]
            if signal.confidence < min_confidence:
                return False, f"Confidence {signal.confidence:.2f} below regime threshold {min_confidence:.2f}"
            
            # Position size check
            if position_size < 0.005:  # 0.5% minimum
                return False, f"Position size {position_size:.1%} too small to be meaningful"
            
            # Signal strength check
            if signal.strength < 0.3:
                return False, f"Signal strength {signal.strength:.2f} too weak"
            
            # Direction check (avoid neutral signals)
            if signal.direction == SignalDirection.NEUTRAL:
                return False, "Neutral signal direction"
            
            reasons.append(f"Confidence {signal.confidence:.2f} above {min_confidence:.2f} threshold")
            reasons.append(f"Position size {position_size:.1%} acceptable")
            reasons.append(f"Signal strength {signal.strength:.2f} sufficient")
            
            return True, "; ".join(reasons)
            
        except Exception as e:
            logger.warning(f"Error in trade decision logic: {e}")
            return False, f"Error in trade decision: {e}"
    
    def _calculate_enhancement_attribution(self, base_signal: EnsembleSignal, final_signal: EnsembleSignal) -> Dict[str, float]:
        """
        Calculate attribution of enhancements to final signal
        """
        return {
            'base_strength': base_signal.strength,
            'base_confidence': base_signal.confidence,
            'final_strength': final_signal.strength,
            'final_confidence': final_signal.confidence,
            'strength_improvement': final_signal.strength - base_signal.strength,
            'confidence_improvement': final_signal.confidence - base_signal.confidence,
            'regime_contribution': 0.0,  # Will be calculated with actual regime adjustments
            'macro_contribution': 0.0,   # Will be calculated with macro integration
            'factor_contribution': 0.0   # Will be calculated with factor timing
        }
    
    def _extract_technical_indicators(self, data: pd.DataFrame) -> dict:
        """Extract technical indicators from market data"""
        try:
            if len(data) < 20:
                # Not enough data for proper technical analysis
                return {
                    'rsi': 50.0,
                    'macd': 0.0,
                    'macd_signal': 0.0,
                    'bollinger_position': 0.0,
                    'stoch_k': 50.0,
                    'stoch_d': 50.0,
                    'williams_r': -50.0,
                    'cci': 0.0,
                    'momentum': 0.0,
                    'price_oscillator': 0.0
                }
            
            close_prices = data['close'].values
            high_prices = data['high'].values if 'high' in data else close_prices
            low_prices = data['low'].values if 'low' in data else close_prices
            
            # RSI calculation (simple version)
            price_changes = np.diff(close_prices)
            gains = np.where(price_changes > 0, price_changes, 0)
            losses = np.where(price_changes < 0, -price_changes, 0)
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
            rs = avg_gain / (avg_loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            
            # Simple moving averages
            sma_12 = np.mean(close_prices[-12:]) if len(close_prices) >= 12 else close_prices[-1]
            sma_26 = np.mean(close_prices[-26:]) if len(close_prices) >= 26 else close_prices[-1]
            macd = sma_12 - sma_26
            
            # MACD Signal line (9-period EMA of MACD)
            macd_signal = macd * 0.9  # Simplified
            
            # Bollinger Band position
            sma_20 = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else close_prices[-1]
            std_20 = np.std(close_prices[-20:]) if len(close_prices) >= 20 else 0
            bollinger_position = (close_prices[-1] - sma_20) / (2 * std_20 + 1e-8)
            
            # Stochastic oscillator
            recent_high = np.max(high_prices[-14:]) if len(high_prices) >= 14 else high_prices[-1]
            recent_low = np.min(low_prices[-14:]) if len(low_prices) >= 14 else low_prices[-1]
            stoch_k = 100 * (close_prices[-1] - recent_low) / (recent_high - recent_low + 1e-8)
            stoch_d = stoch_k * 0.9  # Simplified
            
            # Williams %R
            williams_r = -100 * (recent_high - close_prices[-1]) / (recent_high - recent_low + 1e-8)
            
            # Commodity Channel Index (simplified)
            typical_price = np.mean([high_prices[-1], low_prices[-1], close_prices[-1]])
            sma_tp = np.mean([(h + l + c) / 3 for h, l, c in zip(high_prices[-20:], low_prices[-20:], close_prices[-20:])])
            mean_deviation = np.mean([abs((h + l + c) / 3 - sma_tp) for h, l, c in zip(high_prices[-20:], low_prices[-20:], close_prices[-20:])])
            cci = (typical_price - sma_tp) / (0.015 * mean_deviation + 1e-8)
            
            # Momentum
            momentum = close_prices[-1] - close_prices[-10] if len(close_prices) >= 10 else 0
            
            # Price oscillator
            sma_10 = np.mean(close_prices[-10:]) if len(close_prices) >= 10 else close_prices[-1]
            price_oscillator = (sma_10 - sma_20) / sma_20 * 100
            
            return {
                'rsi': max(0, min(100, rsi)),
                'macd': macd,
                'macd_signal': macd_signal,
                'bollinger_position': max(-2, min(2, bollinger_position)),
                'stoch_k': max(0, min(100, stoch_k)),
                'stoch_d': max(0, min(100, stoch_d)),
                'williams_r': max(-100, min(0, williams_r)),
                'cci': max(-200, min(200, cci)),
                'momentum': momentum,
                'price_oscillator': price_oscillator
            }
            
        except Exception as e:
            logger.warning(f"Error extracting technical indicators: {e}")
            # Return neutral values
            return {
                'rsi': 50.0,
                'macd': 0.0,
                'macd_signal': 0.0,
                'bollinger_position': 0.0,
                'stoch_k': 50.0,
                'stoch_d': 50.0,
                'williams_r': -50.0,
                'cci': 0.0,
                'momentum': 0.0,
                'price_oscillator': 0.0
            }
    
    def _extract_volume_signals(self, data: pd.DataFrame) -> dict:
        """Extract volume signals from market data"""
        try:
            if len(data) < 10 or 'volume' not in data.columns:
                # Not enough data or no volume data
                return {
                    'obv_trend': 0.0,
                    'cmf': 0.0,
                    'vwap_position': 0.0,
                    'volume_sma_ratio': 1.0,
                    'volume_oscillator': 0.0,
                    'price_volume_trend': 0.0,
                    'accumulation_distribution': 0.0,
                    'money_flow_index': 50.0
                }
            
            close_prices = data['close'].values
            volumes = data['volume'].values
            high_prices = data['high'].values if 'high' in data else close_prices
            low_prices = data['low'].values if 'low' in data else close_prices
            
            # On-Balance Volume trend
            price_changes = np.diff(close_prices)
            obv = np.cumsum(np.where(price_changes > 0, volumes[1:], 
                                   np.where(price_changes < 0, -volumes[1:], 0)))
            obv_trend = (obv[-1] - obv[-10]) / (np.mean(volumes) + 1e-8) if len(obv) >= 10 else 0
            
            # Chaikin Money Flow (simplified)
            money_flow_multiplier = (2 * close_prices - high_prices - low_prices) / (high_prices - low_prices + 1e-8)
            money_flow_volume = money_flow_multiplier * volumes
            cmf = np.sum(money_flow_volume[-20:]) / (np.sum(volumes[-20:]) + 1e-8) if len(volumes) >= 20 else 0
            
            # VWAP position
            typical_price = (high_prices + low_prices + close_prices) / 3
            vwap = np.sum(typical_price[-20:] * volumes[-20:]) / (np.sum(volumes[-20:]) + 1e-8) if len(volumes) >= 20 else close_prices[-1]
            vwap_position = (close_prices[-1] - vwap) / vwap
            
            # Volume SMA ratio
            volume_sma = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
            volume_sma_ratio = volumes[-1] / (volume_sma + 1e-8)
            
            # Volume oscillator
            volume_sma_fast = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
            volume_sma_slow = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
            volume_oscillator = (volume_sma_fast - volume_sma_slow) / (volume_sma_slow + 1e-8)
            
            # Price Volume Trend
            pvt = np.cumsum((close_prices[1:] - close_prices[:-1]) / close_prices[:-1] * volumes[1:])
            price_volume_trend = (pvt[-1] - pvt[-10]) / (np.mean(volumes) + 1e-8) if len(pvt) >= 10 else 0
            
            # Accumulation/Distribution
            ad_line = np.cumsum(money_flow_volume)
            accumulation_distribution = (ad_line[-1] - ad_line[-20]) / (np.mean(volumes) + 1e-8) if len(ad_line) >= 20 else 0
            
            # Money Flow Index (simplified)
            positive_flow = np.sum(typical_price[-14:] * volumes[-14:] * (close_prices[-14:] > close_prices[-15:-1]))
            negative_flow = np.sum(typical_price[-14:] * volumes[-14:] * (close_prices[-14:] < close_prices[-15:-1]))
            money_flow_index = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-8))) if len(close_prices) >= 15 else 50
            
            return {
                'obv_trend': max(-2, min(2, obv_trend)),
                'cmf': max(-1, min(1, cmf)),
                'vwap_position': max(-0.2, min(0.2, vwap_position)),
                'volume_sma_ratio': max(0.1, min(5.0, volume_sma_ratio)),
                'volume_oscillator': max(-2, min(2, volume_oscillator)),
                'price_volume_trend': max(-2, min(2, price_volume_trend)),
                'accumulation_distribution': max(-2, min(2, accumulation_distribution)),
                'money_flow_index': max(0, min(100, money_flow_index))
            }
            
        except Exception as e:
            logger.warning(f"Error extracting volume signals: {e}")
            # Return neutral values
            return {
                'obv_trend': 0.0,
                'cmf': 0.0,
                'vwap_position': 0.0,
                'volume_sma_ratio': 1.0,
                'volume_oscillator': 0.0,
                'price_volume_trend': 0.0,
                'accumulation_distribution': 0.0,
                'money_flow_index': 50.0
            }
    
    def _convert_signal_to_dict(self, signal: EnsembleSignal) -> Dict[str, Any]:
        """Convert EnsembleSignal to dictionary for storage"""
        return {
            'symbol': signal.symbol,
            'timestamp': signal.timestamp.isoformat(),
            'direction': signal.direction.name,
            'strength': signal.strength,
            'confidence': signal.confidence,
            'composite_score': signal.composite_score
        }
    
    def get_enhancement_statistics(self) -> Dict[str, Any]:
        """Get statistics on enhancement performance"""
        if not self.signal_history:
            return {"message": "No signals generated yet"}
        
        recent_signals = self.signal_history[-100:]  # Last 100 signals
        
        return {
            'total_signals': len(self.signal_history),
            'recent_signals': len(recent_signals),
            'regime_distribution': {
                regime.value: len([s for s in recent_signals if s.market_regime == regime])
                for regime in MarketRegime
            },
            'average_confidence': np.mean([s.confidence for s in recent_signals]),
            'average_position_size': np.mean([s.optimal_position_size for s in recent_signals]),
            'trade_rate': len([s for s in recent_signals if s.should_trade]) / len(recent_signals)
        }

if __name__ == "__main__":
    # Test the enhanced signal scoring system
    enhanced_scorer = EnhancedEnsembleSignalScoring()
    logger.info("Enhanced Ensemble Signal Scoring system ready for testing")