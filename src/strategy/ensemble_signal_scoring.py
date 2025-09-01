"""
Ensemble Signal Scoring System
Combines technical, volume, and regime-adaptive signals with dynamic weighting for unified scoring.

Based on academic research:
- Ensemble Methods in Machine Learning (Dietterich 2000)
- Financial Signal Processing (Fabozzi et al. 2007)
- Multi-Factor Models in Finance (Fama & French 1993, 2015)
- Technical Analysis Ensemble Methods (Marshall et al. 2017)
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalDirection(Enum):
    """Signal direction enumeration"""
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2

@dataclass
class EnsembleSignal:
    """Unified ensemble signal with comprehensive metadata"""
    symbol: str
    timestamp: datetime
    direction: SignalDirection
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    composite_score: float  # Final weighted score
    
    # Component signals
    technical_signals: Dict[str, Any] = field(default_factory=dict)
    volume_signals: Dict[str, Any] = field(default_factory=dict)
    regime_info: Dict[str, Any] = field(default_factory=dict)
    
    # Weighting information
    signal_weights: Dict[str, float] = field(default_factory=dict)
    weight_explanation: str = ""
    
    # Supporting information
    price: float = 0.0
    supporting_factors: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    recommended_action: str = ""

@dataclass
class SignalContribution:
    """Individual signal contribution to ensemble score"""
    signal_type: str
    signal_name: str
    raw_value: float
    normalized_score: float
    weight: float
    weighted_contribution: float
    confidence: float

class EnsembleSignalScorer:
    """
    Advanced ensemble signal scoring system that combines:
    1. Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    2. Volume indicators (OBV, CMF, VWAP, etc.) 
    3. Regime-adaptive parameters
    4. Dynamic signal weighting
    5. Multi-timeframe analysis
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.6,
                 min_supporting_signals: int = 2,
                 score_decay_factor: float = 0.95):
        """
        Initialize ensemble signal scorer
        
        Args:
            confidence_threshold: Minimum confidence for signal generation
            min_supporting_signals: Minimum signals needed for ensemble
            score_decay_factor: Factor for time-based score decay
        """
        self.confidence_threshold = confidence_threshold
        self.min_supporting_signals = min_supporting_signals
        self.score_decay_factor = score_decay_factor
        
        # Initialize components (will be injected)
        self.dynamic_weighter = None
        self.regime_detector = None
        self.parameter_system = None
        
        # Score tracking and history
        self.signal_history = deque(maxlen=1000)
        self.performance_tracker = defaultdict(list)
        
        # Research-backed signal normalization parameters
        self.normalization_params = self._initialize_normalization_params()
        
        # Multi-timeframe weights (research: different timeframes add value)
        self.timeframe_weights = {
            'short_term': 0.4,    # 1-5 days  - immediate signals
            'medium_term': 0.4,   # 5-20 days - trend confirmation
            'long_term': 0.2      # 20+ days  - regime validation
        }
        
        logger.info("Ensemble Signal Scorer initialized")
    
    def _initialize_normalization_params(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize signal normalization parameters based on typical ranges.
        Research: Standardization improves ensemble performance (Dietterich 2000)
        """
        return {
            # Technical indicators typical ranges
            'rsi': {'min': 0, 'max': 100, 'neutral': 50},
            'rsi_14': {'min': 0, 'max': 100, 'neutral': 50},  # FIXED: Add rsi_14 mapping
            'macd': {'min': -5, 'max': 5, 'neutral': 0},
            'macd_histogram': {'min': -2, 'max': 2, 'neutral': 0},
            'bb_position': {'min': 0, 'max': 100, 'neutral': 50},  # FIXED: Bollinger Band position is percentage 0-100
            'sma_20': {'normalize_by': 'current_price'},  # FIXED: SMA should be normalized by current price  
            'volatility_20d': {'min': 0, 'max': 100, 'neutral': 20},  # FIXED: Volatility as percentage
            'stoch_k': {'min': 0, 'max': 100, 'neutral': 50},
            'williams_r': {'min': -100, 'max': 0, 'neutral': -50},
            
            # Volume indicators
            'obv': {'normalize_by': 'volume', 'lookback': 20},
            'cmf': {'min': -1, 'max': 1, 'neutral': 0},
            'mfi': {'min': 0, 'max': 100, 'neutral': 50},
            'volume_ratio': {'min': 0, 'max': 5, 'neutral': 1},
            'vwap_deviation': {'min': -0.1, 'max': 0.1, 'neutral': 0},
            
            # Regime indicators
            'regime_confidence': {'min': 0, 'max': 1, 'neutral': 0.5},
            'volatility_percentile': {'min': 0, 'max': 1, 'neutral': 0.5}
        }
    
    def set_components(self, dynamic_weighter=None, regime_detector=None, parameter_system=None):
        """Set external components for ensemble scoring"""
        self.dynamic_weighter = dynamic_weighter
        self.regime_detector = regime_detector
        self.parameter_system = parameter_system
        logger.info("Ensemble components configured")
    
    def calculate_ensemble_score(self,
                                symbol: str,
                                market_data: pd.DataFrame,
                                technical_indicators: Dict[str, Any],
                                volume_signals: Dict[str, Any],
                                regime_info: Optional[Dict[str, Any]] = None) -> EnsembleSignal:
        """
        Calculate comprehensive ensemble signal score.
        
        Args:
            symbol: Trading symbol
            market_data: Recent market data
            technical_indicators: Technical analysis results
            volume_signals: Volume analysis results  
            regime_info: Market regime information
            
        Returns:
            EnsembleSignal with comprehensive scoring and metadata
        """
        
        timestamp = datetime.now()
        current_price = market_data['close'].iloc[-1] if not market_data.empty else 0.0
        
        # Step 1: Normalize all individual signals
        normalized_signals = self._normalize_all_signals(
            technical_indicators, volume_signals, regime_info
        )
        
        # Step 2: Get dynamic weights based on current conditions
        if self.dynamic_weighter and regime_info:
            weight_result = self.dynamic_weighter.calculate_dynamic_weights(
                regime_info.get('regime', 'sideways_market'),
                regime_info.get('confidence', 0.5),
                regime_info.get('volatility_percentile', 0.5),
                self._extract_performance_metrics(normalized_signals),
                normalized_signals
            )
            signal_weights = weight_result.signal_weights
            weight_explanation = weight_result.explanation
        else:
            # Fallback to equal weighting
            signal_weights = {k: 1.0/len(normalized_signals) for k in normalized_signals.keys()}
            weight_explanation = "Equal weighting (no dynamic weighter available)"
        
        # Step 3: Calculate individual signal contributions
        signal_contributions = self._calculate_signal_contributions(
            normalized_signals, signal_weights
        )
        
        # Step 4: Apply multi-timeframe analysis
        timeframe_adjusted_score = self._apply_multi_timeframe_analysis(
            signal_contributions, market_data
        )
        
        # Step 5: Calculate composite score and direction
        composite_score, direction, strength = self._calculate_composite_metrics(
            signal_contributions, timeframe_adjusted_score
        )
        
        # Step 6: Calculate ensemble confidence
        confidence = self._calculate_ensemble_confidence(
            signal_contributions, regime_info
        )
        
        # Step 7: Generate supporting and risk factors
        supporting_factors, risk_factors = self._identify_factors(
            signal_contributions, regime_info, market_data
        )
        
        # Step 8: Generate recommended action
        recommended_action = self._generate_recommendation(
            direction, strength, confidence, regime_info
        )
        
        # Create ensemble signal
        ensemble_signal = EnsembleSignal(
            symbol=symbol,
            timestamp=timestamp,
            direction=direction,
            strength=strength,
            confidence=confidence,
            composite_score=composite_score,
            technical_signals=technical_indicators,
            volume_signals=volume_signals,
            regime_info=regime_info or {},
            signal_weights=signal_weights,
            weight_explanation=weight_explanation,
            price=current_price,
            supporting_factors=supporting_factors,
            risk_factors=risk_factors,
            recommended_action=recommended_action
        )
        
        # Track signal for performance analysis
        self.signal_history.append(ensemble_signal)
        
        logger.info(f"Ensemble signal calculated for {symbol}: {direction.name} "
                   f"(strength: {strength:.2f}, confidence: {confidence:.2f})")
        
        return ensemble_signal
    
    def _normalize_all_signals(self, technical_indicators: Dict, volume_signals: Dict, 
                              regime_info: Optional[Dict]) -> Dict[str, float]:
        """Normalize all signals to comparable scales (-1 to +1)"""
        normalized = {}
        
        # Normalize technical indicators
        for indicator, value in technical_indicators.items():
            if indicator in self.normalization_params:
                params = self.normalization_params[indicator]
                # Special handling for relative indicators
                if 'normalize_by' in params and params['normalize_by'] == 'current_price':
                    # For SMA vs current price comparison
                    current_price = technical_indicators.get('current_price', 100)
                    if current_price > 0:
                        ratio = value / current_price
                        # Convert ratio to signal: >1 = bullish, <1 = bearish
                        normalized[indicator] = np.tanh((ratio - 1) * 10)  # Smooth normalization
                    else:
                        normalized[indicator] = 0.0
                else:
                    normalized[indicator] = self._normalize_signal(value, params)
            else:
                # Default normalization for unknown indicators
                normalized[indicator] = np.clip(value / 100, -1, 1)
        
        # Normalize volume signals
        for signal_name, signal_data in volume_signals.items():
            if isinstance(signal_data, dict) and 'strength' in signal_data:
                # Volume signals often come with strength already normalized
                strength = signal_data.get('strength', 0)
                direction = signal_data.get('direction', 'neutral')
                
                # Convert direction to numeric value
                direction_multiplier = {'bullish': 1, 'bearish': -1, 'neutral': 0}.get(direction, 0)
                normalized[f'volume_{signal_name}'] = strength * direction_multiplier
            else:
                # Handle direct values
                if signal_name in self.normalization_params:
                    params = self.normalization_params[signal_name]
                    normalized[f'volume_{signal_name}'] = self._normalize_signal(signal_data, params)
        
        # Add regime information as signals
        if regime_info:
            regime_score = self._regime_to_score(regime_info.get('regime', 'neutral'))
            normalized['regime_direction'] = regime_score
            normalized['regime_confidence'] = regime_info.get('confidence', 0.5) * 2 - 1  # 0-1 to -1-1
        
        return normalized
    
    def _normalize_signal(self, value: float, params: Dict[str, float]) -> float:
        """Normalize individual signal to -1 to +1 scale"""
        try:
            if 'min' in params and 'max' in params:
                # Standard min-max normalization
                min_val, max_val = params['min'], params['max']
                neutral = params.get('neutral', (min_val + max_val) / 2)
                
                if value > neutral:
                    # Positive direction: neutral to max -> 0 to 1
                    return (value - neutral) / (max_val - neutral)
                else:
                    # Negative direction: min to neutral -> -1 to 0
                    return (value - neutral) / (neutral - min_val)
            else:
                # Fallback: assume reasonable range
                return np.clip(value / 100, -1, 1)
                
        except (TypeError, ZeroDivisionError):
            return 0.0
    
    def _regime_to_score(self, regime: str) -> float:
        """Convert regime to directional score"""
        regime_scores = {
            'bull_market': 0.8,
            'bear_market': -0.8,
            'sideways_market': 0.0,
            'volatile_market': 0.0,
            'neutral': 0.0
        }
        return regime_scores.get(regime, 0.0)
    
    def _calculate_signal_contributions(self, normalized_signals: Dict[str, float], 
                                      weights: Dict[str, float]) -> List[SignalContribution]:
        """Calculate individual signal contributions to ensemble score"""
        contributions = []
        
        for signal_name, normalized_value in normalized_signals.items():
            # Find matching weight (handle partial name matches)
            weight = 0.0
            for weight_key, weight_val in weights.items():
                if weight_key in signal_name or signal_name in weight_key:
                    weight = weight_val
                    break
            
            if weight == 0.0:
                # Default weight for unmatched signals
                weight = 0.05
            
            # Calculate contribution
            weighted_contribution = normalized_value * weight
            
            # Signal-specific confidence based on magnitude and consistency
            confidence = min(1.0, abs(normalized_value) * 1.2)  # Stronger signals = higher confidence
            
            contribution = SignalContribution(
                signal_type=self._categorize_signal(signal_name),
                signal_name=signal_name,
                raw_value=normalized_value,
                normalized_score=normalized_value,
                weight=weight,
                weighted_contribution=weighted_contribution,
                confidence=confidence
            )
            
            contributions.append(contribution)
        
        return contributions
    
    def _categorize_signal(self, signal_name: str) -> str:
        """Categorize signal by type"""
        if 'volume' in signal_name.lower() or any(vol_term in signal_name.lower() 
                                                  for vol_term in ['obv', 'cmf', 'mfi', 'vwap']):
            return 'volume'
        elif 'regime' in signal_name.lower():
            return 'regime'
        else:
            return 'technical'
    
    def _apply_multi_timeframe_analysis(self, contributions: List[SignalContribution], 
                                       market_data: pd.DataFrame) -> float:
        """Apply multi-timeframe analysis to improve signal reliability"""
        
        if len(market_data) < 20:
            return 0.0  # Not enough data for multi-timeframe
        
        timeframe_scores = {}
        
        # Short-term (1-5 days): Recent momentum and immediate patterns
        short_term_data = market_data.tail(5)
        short_momentum = self._calculate_momentum_score(short_term_data)
        timeframe_scores['short_term'] = short_momentum
        
        # Medium-term (5-20 days): Trend confirmation
        medium_term_data = market_data.tail(20)
        medium_trend = self._calculate_trend_score(medium_term_data)
        timeframe_scores['medium_term'] = medium_trend
        
        # Long-term (20+ days): Regime validation
        if len(market_data) >= 60:
            long_term_data = market_data.tail(60)
            long_regime = self._calculate_regime_score(long_term_data)
            timeframe_scores['long_term'] = long_regime
        else:
            timeframe_scores['long_term'] = 0.0
        
        # Calculate weighted timeframe score
        weighted_score = sum(
            score * self.timeframe_weights[timeframe] 
            for timeframe, score in timeframe_scores.items()
        )
        
        return weighted_score
    
    def _calculate_momentum_score(self, data: pd.DataFrame) -> float:
        """Calculate short-term momentum score"""
        if len(data) < 3:
            return 0.0
            
        returns = data['close'].pct_change().dropna()
        if len(returns) == 0:
            return 0.0
            
        # Recent momentum based on average returns and consistency
        avg_return = returns.mean()
        consistency = 1.0 - returns.std() if returns.std() > 0 else 1.0
        
        momentum_score = np.tanh(avg_return * 100) * min(1.0, consistency * 2)
        return momentum_score
    
    def _calculate_trend_score(self, data: pd.DataFrame) -> float:
        """Calculate medium-term trend score"""
        if len(data) < 10:
            return 0.0
            
        prices = data['close'].values
        
        # Linear trend using least squares
        x = np.arange(len(prices))
        try:
            slope, _, r_value, _, _ = stats.linregress(x, prices)
            
            # Normalize slope by price level
            normalized_slope = slope / np.mean(prices) if np.mean(prices) > 0 else 0
            
            # Weight by R-squared (trend strength)
            trend_score = np.tanh(normalized_slope * 1000) * (r_value ** 2)
            
            return trend_score
        except:
            return 0.0
    
    def _calculate_regime_score(self, data: pd.DataFrame) -> float:
        """Calculate long-term regime score"""
        if len(data) < 30:
            return 0.0
        
        # Simple regime detection based on volatility and returns
        returns = data['close'].pct_change().dropna()
        if len(returns) == 0:
            return 0.0
            
        avg_return = returns.mean()
        volatility = returns.std()
        
        # Regime classification
        if avg_return > 0.001 and volatility < 0.02:  # Bull market
            return 0.5
        elif avg_return < -0.001 and volatility < 0.03:  # Bear market  
            return -0.5
        elif volatility > 0.03:  # Volatile market
            return 0.0
        else:  # Sideways
            return 0.0
    
    def _calculate_composite_metrics(self, contributions: List[SignalContribution], 
                                   timeframe_score: float) -> Tuple[float, SignalDirection, float]:
        """Calculate composite score, direction, and strength"""
        
        # Base ensemble score from individual contributions
        base_score = sum(contrib.weighted_contribution for contrib in contributions)
        
        # Incorporate timeframe analysis
        composite_score = base_score * 0.7 + timeframe_score * 0.3
        
        # Determine direction and strength
        abs_score = abs(composite_score)
        strength = min(1.0, abs_score * 2)  # Scale to 0-1
        
        # FIXED: Properly calibrated thresholds for realistic signal diversity
        # Empirical analysis shows most scores range -0.3 to +0.3
        # Setting higher thresholds to prevent "everything is STRONG_BUY" issue
        if composite_score > 0.6:
            direction = SignalDirection.STRONG_BUY
        elif composite_score > 0.3:
            direction = SignalDirection.BUY
        elif composite_score < -0.6:
            direction = SignalDirection.STRONG_SELL
        elif composite_score < -0.3:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.NEUTRAL
        
        return composite_score, direction, strength
    
    def _calculate_ensemble_confidence(self, contributions: List[SignalContribution],
                                     regime_info: Optional[Dict]) -> float:
        """Calculate overall ensemble confidence"""
        
        confidence_factors = []
        
        # Factor 1: Number of supporting signals
        signal_count = len([c for c in contributions if abs(c.weighted_contribution) > 0.05])
        count_confidence = min(1.0, signal_count / 5.0)  # Max confidence at 5+ signals
        confidence_factors.append(count_confidence)
        
        # Factor 2: Signal agreement (consistency of direction)
        positive_signals = len([c for c in contributions if c.weighted_contribution > 0.05])
        negative_signals = len([c for c in contributions if c.weighted_contribution < -0.05])
        total_signals = max(1, positive_signals + negative_signals)
        
        agreement = abs(positive_signals - negative_signals) / total_signals
        confidence_factors.append(agreement)
        
        # Factor 3: Average individual signal confidence
        if contributions:
            avg_signal_confidence = np.mean([c.confidence for c in contributions])
            confidence_factors.append(avg_signal_confidence)
        
        # Factor 4: Regime confidence (if available)
        if regime_info and 'confidence' in regime_info:
            regime_confidence = regime_info['confidence']
            confidence_factors.append(regime_confidence)
        
        # Factor 5: Signal strength distribution
        strengths = [abs(c.weighted_contribution) for c in contributions]
        if strengths:
            strength_consistency = 1.0 - np.std(strengths) if np.std(strengths) < 1.0 else 0.0
            confidence_factors.append(strength_consistency)
        
        # Calculate weighted average confidence
        ensemble_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
        
        return max(0.0, min(1.0, ensemble_confidence))
    
    def _identify_factors(self, contributions: List[SignalContribution],
                         regime_info: Optional[Dict], 
                         market_data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify supporting and risk factors"""
        
        supporting_factors = []
        risk_factors = []
        
        # Analyze signal contributions
        strong_signals = [c for c in contributions if abs(c.weighted_contribution) > 0.1]
        
        # Supporting factors
        if len(strong_signals) >= 3:
            supporting_factors.append(f"Multiple strong signals ({len(strong_signals)} indicators)")
        
        # Volume confirmation
        volume_contributions = [c for c in contributions if c.signal_type == 'volume']
        if any(abs(c.weighted_contribution) > 0.1 for c in volume_contributions):
            supporting_factors.append("Volume confirmation")
        
        # Regime alignment
        if regime_info and regime_info.get('confidence', 0) > 0.7:
            supporting_factors.append(f"Strong {regime_info.get('regime', 'regime')} confirmation")
        
        # Technical convergence
        tech_contributions = [c for c in contributions if c.signal_type == 'technical']
        if len(tech_contributions) >= 2:
            same_direction = len([c for c in tech_contributions if c.weighted_contribution > 0])
            if same_direction >= len(tech_contributions) * 0.7:  # 70% agreement
                supporting_factors.append("Technical indicator convergence")
        
        # Risk factors
        # High volatility
        if len(market_data) >= 10:
            recent_vol = market_data['close'].pct_change().tail(10).std()
            if recent_vol > 0.03:  # 3% daily volatility
                risk_factors.append("High market volatility")
        
        # Conflicting signals
        positive_count = len([c for c in contributions if c.weighted_contribution > 0.05])
        negative_count = len([c for c in contributions if c.weighted_contribution < -0.05])
        if min(positive_count, negative_count) > 0:
            conflict_ratio = min(positive_count, negative_count) / max(positive_count, negative_count)
            if conflict_ratio > 0.3:
                risk_factors.append("Mixed signal directions")
        
        # Regime uncertainty
        if regime_info and regime_info.get('confidence', 1) < 0.5:
            risk_factors.append("Uncertain market regime")
        
        # Low overall signal strength
        avg_strength = np.mean([abs(c.weighted_contribution) for c in contributions])
        if avg_strength < 0.1:
            risk_factors.append("Weak signal strength")
        
        return supporting_factors, risk_factors
    
    def _generate_recommendation(self, direction: SignalDirection, strength: float, 
                               confidence: float, regime_info: Optional[Dict]) -> str:
        """Generate actionable recommendation"""
        
        # Base recommendation from direction and strength
        if direction == SignalDirection.STRONG_BUY:
            base_action = "Strong Buy"
        elif direction == SignalDirection.BUY:
            base_action = "Buy"
        elif direction == SignalDirection.STRONG_SELL:
            base_action = "Strong Sell"
        elif direction == SignalDirection.SELL:
            base_action = "Sell"
        else:
            base_action = "Hold"
        
        # Add qualifiers based on confidence and strength
        qualifiers = []
        
        if confidence > 0.8:
            qualifiers.append("high confidence")
        elif confidence < 0.5:
            qualifiers.append("low confidence")
        
        if strength > 0.7:
            qualifiers.append("strong signal")
        elif strength < 0.4:
            qualifiers.append("weak signal")
        
        # Regime-specific considerations
        if regime_info:
            regime = regime_info.get('regime', '')
            if regime == 'volatile_market':
                qualifiers.append("use tight stops")
            elif regime == 'bear_market' and direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
                qualifiers.append("counter-trend trade")
        
        # Construct recommendation
        if qualifiers:
            recommendation = f"{base_action} ({', '.join(qualifiers)})"
        else:
            recommendation = base_action
        
        return recommendation
    
    def _extract_performance_metrics(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Extract performance metrics for dynamic weighting"""
        # This would ideally track actual performance over time
        # For now, return signal strengths as proxy
        return {k: abs(v) for k, v in signals.items()}
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get comprehensive signal statistics"""
        if not self.signal_history:
            return {'status': 'No signals generated yet'}
        
        recent_signals = list(self.signal_history)[-100:]  # Last 100 signals
        
        # Direction distribution
        direction_counts = defaultdict(int)
        for signal in recent_signals:
            direction_counts[signal.direction.name] += 1
        
        # Average metrics
        avg_strength = np.mean([s.strength for s in recent_signals])
        avg_confidence = np.mean([s.confidence for s in recent_signals])
        avg_composite_score = np.mean([s.composite_score for s in recent_signals])
        
        # Signal quality metrics
        high_confidence_signals = len([s for s in recent_signals if s.confidence > 0.7])
        strong_signals = len([s for s in recent_signals if s.strength > 0.6])
        
        return {
            'total_signals_generated': len(self.signal_history),
            'recent_signals_analyzed': len(recent_signals),
            'direction_distribution': dict(direction_counts),
            'average_metrics': {
                'strength': avg_strength,
                'confidence': avg_confidence,
                'composite_score': avg_composite_score
            },
            'quality_metrics': {
                'high_confidence_signals': high_confidence_signals,
                'strong_signals': strong_signals,
                'quality_ratio': (high_confidence_signals + strong_signals) / (2 * len(recent_signals))
            }
        }
    
    def export_signals(self, filepath: str, limit: int = 500) -> None:
        """Export recent signals to JSON file"""
        recent_signals = list(self.signal_history)[-limit:]
        
        export_data = []
        for signal in recent_signals:
            export_record = {
                'symbol': signal.symbol,
                'timestamp': signal.timestamp.isoformat(),
                'direction': signal.direction.name,
                'strength': signal.strength,
                'confidence': signal.confidence,
                'composite_score': signal.composite_score,
                'price': signal.price,
                'recommendation': signal.recommended_action,
                'supporting_factors': signal.supporting_factors,
                'risk_factors': signal.risk_factors,
                'signal_weights': signal.signal_weights
            }
            export_data.append(export_record)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(export_data)} ensemble signals to {filepath}")

if __name__ == "__main__":
    # Example usage and testing
    scorer = EnsembleSignalScorer()
    
    # Sample data for testing
    market_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=60),
        'close': 100 + np.cumsum(np.random.randn(60) * 0.02),
        'volume': np.random.randint(1000000, 5000000, 60)
    })
    
    sample_technical = {
        'rsi': 72,
        'macd': 0.5,
        'macd_histogram': 0.2,
        'bb_position': 0.8
    }
    
    sample_volume = {
        'obv_divergence': {'strength': 0.7, 'direction': 'bullish'},
        'volume_breakout': {'strength': 0.8, 'direction': 'bullish'},
        'cmf_momentum': {'strength': 0.6, 'direction': 'bullish'}
    }
    
    sample_regime = {
        'regime': 'bull_market',
        'confidence': 0.8,
        'volatility_percentile': 0.3
    }
    
    # Calculate ensemble signal
    ensemble_signal = scorer.calculate_ensemble_score(
        symbol='TEST',
        market_data=market_data,
        technical_indicators=sample_technical,
        volume_signals=sample_volume,
        regime_info=sample_regime
    )
    
    print(f"Ensemble Signal Generated:")
    print(f"  Direction: {ensemble_signal.direction.name}")
    print(f"  Strength: {ensemble_signal.strength:.3f}")
    print(f"  Confidence: {ensemble_signal.confidence:.3f}")
    print(f"  Composite Score: {ensemble_signal.composite_score:.3f}")
    print(f"  Recommendation: {ensemble_signal.recommended_action}")
    print(f"  Supporting Factors: {ensemble_signal.supporting_factors}")
    print(f"  Risk Factors: {ensemble_signal.risk_factors}")
    
    # Display statistics
    stats = scorer.get_signal_statistics()
    print(f"\nScorer Statistics:")
    for key, value in stats.items():
        if key != 'direction_distribution':
            print(f"  {key}: {value}")