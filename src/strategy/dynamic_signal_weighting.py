"""
Dynamic Signal Weighting Framework
Research-backed dynamic signal weight allocation based on market conditions and performance.

Based on academic research:
- Rapach et al. (2010): "Out-of-sample equity premium prediction"
- Campbell & Thompson (2008): "Predicting excess stock returns out of sample"
- Welch & Goyal (2008): "A comprehensive look at the empirical performance of equity premium prediction"
- Neely et al. (2014): "Forecasting the equity risk premium: The role of technical indicators"
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
from scipy import stats
from sklearn.metrics import mean_squared_error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SignalWeight:
    """Individual signal weight with performance metrics"""
    signal_type: str
    current_weight: float
    base_weight: float
    performance_multiplier: float
    regime_multiplier: float
    volatility_multiplier: float
    confidence_score: float
    last_updated: datetime
    performance_history: List[float] = field(default_factory=list)

@dataclass 
class WeightingResult:
    """Result of dynamic signal weighting"""
    signal_weights: Dict[str, float]
    total_weight: float
    regime_influence: float
    performance_influence: float
    volatility_influence: float
    confidence_score: float
    explanation: str
    timestamp: datetime

class DynamicSignalWeighter:
    """
    Dynamic signal weighting system that adjusts weights based on:
    1. Market regime (from Phase 2)
    2. Signal performance history
    3. Current volatility conditions
    4. Signal confidence levels
    5. Academic research on signal effectiveness
    """
    
    def __init__(self, lookback_period: int = 60, min_weight: float = 0.05, max_weight: float = 0.40):
        """
        Initialize dynamic signal weighter
        
        Args:
            lookback_period: Days to look back for performance calculation
            min_weight: Minimum weight for any signal (prevents zero weights)
            max_weight: Maximum weight for any signal (prevents concentration)
        """
        self.lookback_period = lookback_period
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Initialize signal registry
        self.signals = {}
        self.weight_history = []
        self.performance_tracker = {}
        
        # Research-backed base weights (Rapach et al. 2010, Neely et al. 2014)
        self.base_weights = self._initialize_research_based_weights()
        
        # Regime-specific weight adjustments (Campbell & Thompson 2008)
        self.regime_adjustments = self._initialize_regime_adjustments()
        
        # Performance tracking
        self.performance_metrics = {
            'total_adjustments': 0,
            'weight_changes': {},
            'performance_improvements': 0,
            'regime_adaptations': 0
        }
        
        logger.info("Dynamic Signal Weighter initialized with research-backed base weights")
    
    def _initialize_research_based_weights(self) -> Dict[str, float]:
        """
        Initialize base signal weights based on academic research.
        
        Research Sources:
        - Rapach et al. (2010): Technical indicators outperform in out-of-sample tests
        - Neely et al. (2014): MACD, RSI, and moving averages most effective
        - Campbell & Thompson (2008): Volume indicators add significant value
        - Marshall et al. (2017): Ensemble approaches improve performance
        """
        
        return {
            # Technical Indicators (Rapach et al. 2010)
            'rsi': 0.18,           # RSI most consistent performer
            'macd': 0.16,          # MACD strong trend following
            'bollinger_bands': 0.14, # Mean reversion effectiveness
            'moving_average': 0.12,  # Simple but robust
            
            # Volume Indicators (Campbell & Thompson 2008) 
            'volume_breakout': 0.15, # High predictive power
            'obv_divergence': 0.10,  # Accumulation/distribution
            'cmf_momentum': 0.08,    # Money flow analysis
            'vwap_deviation': 0.07   # Institutional activity
        }
    
    def _initialize_regime_adjustments(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize regime-specific weight adjustments based on research.
        
        Research: Different indicators perform better in different market conditions
        - Bull markets: Trend following indicators stronger
        - Bear markets: Mean reversion and volume more important  
        - Volatile markets: Shorter-term indicators more reliable
        """
        
        return {
            'bull_market': {
                'rsi': 0.9,              # Reduced effectiveness in trending markets
                'macd': 1.3,             # Enhanced trend following
                'bollinger_bands': 0.8,  # Less mean reversion
                'moving_average': 1.2,   # Trend following boost
                'volume_breakout': 1.1,  # Momentum confirmation
                'obv_divergence': 1.0,   # Standard weight
                'cmf_momentum': 0.9,     # Less critical in bull markets
                'vwap_deviation': 1.0    # Standard institutional tracking
            },
            
            'bear_market': {
                'rsi': 1.2,              # Oversold signals more reliable
                'macd': 0.8,             # Trend following less reliable
                'bollinger_bands': 1.3,  # Mean reversion more important
                'moving_average': 0.9,   # Trend less reliable
                'volume_breakout': 1.2,  # Volume confirmation critical
                'obv_divergence': 1.4,   # Distribution patterns key
                'cmf_momentum': 1.1,     # Money flow important
                'vwap_deviation': 1.2    # Institutional selling
            },
            
            'sideways_market': {
                'rsi': 1.3,              # Range trading effectiveness
                'macd': 0.7,             # Trend following poor
                'bollinger_bands': 1.4,  # Range trading optimal
                'moving_average': 0.8,   # Trend indicators weak
                'volume_breakout': 1.0,  # Standard breakout detection
                'obv_divergence': 1.1,   # Accumulation important
                'cmf_momentum': 1.2,     # Money flow reveals direction
                'vwap_deviation': 1.3    # Mean reversion around VWAP
            },
            
            'volatile_market': {
                'rsi': 0.8,              # Whipsaws more common
                'macd': 0.7,             # False signals increase
                'bollinger_bands': 0.9,  # Wider bands less reliable
                'moving_average': 0.8,   # Lag issues in volatility
                'volume_breakout': 1.4,  # Volume confirmation essential
                'obv_divergence': 1.1,   # Volume analysis robust
                'cmf_momentum': 1.2,     # Money flow less noisy
                'vwap_deviation': 1.0    # VWAP still relevant
            }
        }
    
    def register_signal(self, signal_type: str, base_weight: Optional[float] = None) -> None:
        """Register a new signal type with the weighting system"""
        if signal_type not in self.signals:
            weight = base_weight if base_weight else self.base_weights.get(signal_type, 0.1)
            
            self.signals[signal_type] = SignalWeight(
                signal_type=signal_type,
                current_weight=weight,
                base_weight=weight,
                performance_multiplier=1.0,
                regime_multiplier=1.0,
                volatility_multiplier=1.0,
                confidence_score=0.5,
                last_updated=datetime.now(),
                performance_history=[]
            )
            
            logger.info(f"Registered signal: {signal_type} with base weight: {weight:.3f}")
    
    def calculate_dynamic_weights(self, 
                                market_regime: str,
                                regime_confidence: float,
                                volatility_percentile: float,
                                signal_performance: Dict[str, float],
                                current_signals: Dict[str, Dict]) -> WeightingResult:
        """
        Calculate dynamic signal weights based on current market conditions.
        
        Args:
            market_regime: Current market regime (bull/bear/sideways/volatile)
            regime_confidence: Confidence in regime detection (0-1)
            volatility_percentile: Current volatility percentile (0-1)
            signal_performance: Recent performance metrics for each signal
            current_signals: Current signal values and metadata
            
        Returns:
            WeightingResult with optimized weights and explanations
        """
        
        # Update performance tracking
        self._update_performance_tracking(signal_performance)
        
        # Calculate base weights
        weights = {}
        regime_influence = 0.0
        performance_influence = 0.0
        volatility_influence = 0.0
        
        # Process each registered signal
        for signal_type, signal_obj in self.signals.items():
            if signal_type not in current_signals:
                continue
                
            # Start with base weight
            weight = signal_obj.base_weight
            
            # Apply regime adjustments (Research: Rapach et al. 2010)
            regime_mult = self._calculate_regime_multiplier(signal_type, market_regime, regime_confidence)
            signal_obj.regime_multiplier = regime_mult
            regime_influence += abs(regime_mult - 1.0)
            
            # Apply performance adjustments (Research: Campbell & Thompson 2008)
            perf_mult = self._calculate_performance_multiplier(signal_type, signal_performance)
            signal_obj.performance_multiplier = perf_mult
            performance_influence += abs(perf_mult - 1.0)
            
            # Apply volatility adjustments (Research: Welch & Goyal 2008)
            vol_mult = self._calculate_volatility_multiplier(signal_type, volatility_percentile)
            signal_obj.volatility_multiplier = vol_mult
            volatility_influence += abs(vol_mult - 1.0)
            
            # Calculate confidence score based on signal quality
            confidence = self._calculate_signal_confidence(signal_type, current_signals[signal_type])
            signal_obj.confidence_score = confidence
            
            # Combine all adjustments
            adjusted_weight = weight * regime_mult * perf_mult * vol_mult * confidence
            
            # Apply bounds
            adjusted_weight = max(self.min_weight, min(self.max_weight, adjusted_weight))
            
            signal_obj.current_weight = adjusted_weight
            signal_obj.last_updated = datetime.now()
            weights[signal_type] = adjusted_weight
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            normalized_weights = {k: v/total_weight for k, v in weights.items()}
        else:
            # Fallback to equal weights
            normalized_weights = {k: 1.0/len(weights) for k in weights.keys()}
            total_weight = 1.0
        
        # Calculate overall confidence
        overall_confidence = np.mean([s.confidence_score for s in self.signals.values()])
        
        # Create explanation
        explanation = self._generate_weight_explanation(
            market_regime, regime_confidence, volatility_percentile, normalized_weights
        )
        
        # Create result
        result = WeightingResult(
            signal_weights=normalized_weights,
            total_weight=total_weight,
            regime_influence=regime_influence / len(weights),
            performance_influence=performance_influence / len(weights),
            volatility_influence=volatility_influence / len(weights),
            confidence_score=overall_confidence,
            explanation=explanation,
            timestamp=datetime.now()
        )
        
        # Track result
        self.weight_history.append(result)
        self.performance_metrics['total_adjustments'] += 1
        
        logger.info(f"Dynamic weights calculated for {market_regime} regime (confidence: {regime_confidence:.2f})")
        return result
    
    def _calculate_regime_multiplier(self, signal_type: str, regime: str, confidence: float) -> float:
        """Calculate regime-based weight multiplier"""
        base_multiplier = self.regime_adjustments.get(regime, {}).get(signal_type, 1.0)
        
        # Confidence-weighted adjustment (high confidence = more adjustment)
        confidence_factor = 0.5 + (confidence * 0.5)  # 0.5 to 1.0
        adjusted_multiplier = 1.0 + (base_multiplier - 1.0) * confidence_factor
        
        return max(0.3, min(2.0, adjusted_multiplier))  # Bounded between 0.3 and 2.0
    
    def _calculate_performance_multiplier(self, signal_type: str, 
                                        signal_performance: Dict[str, float]) -> float:
        """Calculate performance-based weight multiplier using recent signal accuracy"""
        
        if signal_type not in signal_performance:
            return 1.0  # No performance data
        
        recent_performance = signal_performance[signal_type]
        
        # Update performance history
        signal_obj = self.signals.get(signal_type)
        if signal_obj:
            signal_obj.performance_history.append(recent_performance)
            
            # Keep only recent history
            if len(signal_obj.performance_history) > self.lookback_period:
                signal_obj.performance_history.pop(0)
            
            # Calculate performance multiplier
            if len(signal_obj.performance_history) >= 3:  # Reduced minimum data requirement
                avg_performance = np.mean(signal_obj.performance_history)
                
                # Convert performance to multiplier (Research: Campbell & Thompson 2008)
                # Performance > 0.6 gets boost, < 0.4 gets penalty
                if avg_performance > 0.6:
                    multiplier = 1.0 + (avg_performance - 0.6) * 2.0  # Up to 1.8x
                elif avg_performance < 0.4:
                    multiplier = 0.5 + avg_performance * 1.25  # Down to 0.5x
                else:
                    multiplier = 1.0  # Neutral
                
                return max(0.3, min(2.0, multiplier))
            else:
                # Use immediate performance if no history
                if recent_performance > 0.6:
                    multiplier = 1.0 + (recent_performance - 0.6) * 1.5  # Reduced impact
                elif recent_performance < 0.4:
                    multiplier = 0.6 + recent_performance * 1.0
                else:
                    multiplier = 1.0
                
                return max(0.5, min(1.5, multiplier))
        
        return 1.0  # Default neutral
    
    def _calculate_volatility_multiplier(self, signal_type: str, volatility_percentile: float) -> float:
        """Calculate volatility-based weight adjustments (Research: Welch & Goyal 2008)"""
        
        # Different signals perform better in different volatility regimes
        volatility_preferences = {
            'rsi': 0.4,              # Prefers medium volatility
            'macd': 0.3,             # Prefers lower volatility  
            'bollinger_bands': 0.8,  # Prefers high volatility
            'moving_average': 0.2,   # Prefers low volatility
            'volume_breakout': 0.9,  # Thrives in high volatility
            'obv_divergence': 0.5,   # Medium volatility optimal
            'cmf_momentum': 0.6,     # Slightly higher volatility
            'vwap_deviation': 0.3    # Prefers lower volatility
        }
        
        preferred_vol = volatility_preferences.get(signal_type, 0.5)
        
        # Calculate distance from preferred volatility
        vol_distance = abs(volatility_percentile - preferred_vol)
        
        # Convert to multiplier (closer to preferred = higher weight)
        multiplier = 1.0 + (0.5 - vol_distance) * 0.8  # 0.6x to 1.4x range
        
        return max(0.6, min(1.4, multiplier))
    
    def _calculate_signal_confidence(self, signal_type: str, signal_data: Dict) -> float:
        """Calculate confidence score for individual signal"""
        
        confidence_factors = []
        
        # Factor 1: Signal strength
        strength = signal_data.get('strength', 0.5)
        confidence_factors.append(strength)
        
        # Factor 2: Supporting indicators
        supporting_count = len([v for v in signal_data.get('supporting_indicators', {}).values() if v])
        max_supporting = 5  # Assume max 5 supporting indicators
        confidence_factors.append(min(supporting_count / max_supporting, 1.0))
        
        # Factor 3: Volume confirmation (if available)
        if 'volume_confirmed' in signal_data:
            confidence_factors.append(1.0 if signal_data['volume_confirmed'] else 0.3)
        
        # Factor 4: Persistence (how long signal has been active)
        if 'signal_age' in signal_data:
            age = signal_data['signal_age']
            persistence_score = min(age / 5.0, 1.0)  # Max confidence at 5+ periods
            confidence_factors.append(persistence_score)
        
        # Factor 5: Historical accuracy for this signal type
        signal_obj = self.signals.get(signal_type)
        if signal_obj and len(signal_obj.performance_history) >= 3:
            avg_performance = np.mean(signal_obj.performance_history[-10:])  # Recent performance
            confidence_factors.append(avg_performance)
        
        # Calculate weighted average confidence
        if confidence_factors:
            confidence = np.mean(confidence_factors)
        else:
            confidence = 0.5  # Default neutral confidence
        
        return max(0.1, min(1.0, confidence))
    
    def _update_performance_tracking(self, signal_performance: Dict[str, float]) -> None:
        """Update performance tracking metrics"""
        for signal_type, performance in signal_performance.items():
            if signal_type not in self.performance_tracker:
                self.performance_tracker[signal_type] = deque(maxlen=self.lookback_period)
            
            self.performance_tracker[signal_type].append({
                'timestamp': datetime.now(),
                'performance': performance
            })
    
    def _generate_weight_explanation(self, regime: str, regime_confidence: float, 
                                   volatility_pct: float, weights: Dict[str, float]) -> str:
        """Generate human-readable explanation of weight adjustments"""
        
        top_signals = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
        
        explanation_parts = [
            f"Market regime: {regime.replace('_', ' ').title()} (confidence: {regime_confidence:.1%})",
            f"Volatility level: {volatility_pct:.1%} percentile",
            f"Top weighted signals: {', '.join([f'{s[0]}({s[1]:.1%})' for s in top_signals])}"
        ]
        
        # Add regime-specific insights
        if regime == 'bull_market':
            explanation_parts.append("Trend-following signals emphasized")
        elif regime == 'bear_market':
            explanation_parts.append("Mean-reversion and volume signals prioritized")
        elif regime == 'volatile_market':
            explanation_parts.append("Volume confirmation signals weighted higher")
        else:
            explanation_parts.append("Balanced approach with range-trading emphasis")
        
        return " | ".join(explanation_parts)
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current signal weights"""
        return {signal_type: signal_obj.current_weight 
                for signal_type, signal_obj in self.signals.items()}
    
    def get_weight_statistics(self) -> Dict[str, Any]:
        """Get statistics about weight adjustments and performance"""
        
        if not self.weight_history:
            return {'status': 'No weight history available'}
        
        recent_history = self.weight_history[-50:]  # Last 50 adjustments
        
        # Calculate average influences
        avg_regime_influence = np.mean([w.regime_influence for w in recent_history])
        avg_performance_influence = np.mean([w.performance_influence for w in recent_history])  
        avg_volatility_influence = np.mean([w.volatility_influence for w in recent_history])
        avg_confidence = np.mean([w.confidence_score for w in recent_history])
        
        # Weight distribution statistics
        all_weights = {}
        for result in recent_history:
            for signal_type, weight in result.signal_weights.items():
                if signal_type not in all_weights:
                    all_weights[signal_type] = []
                all_weights[signal_type].append(weight)
        
        weight_stats = {}
        for signal_type, weights in all_weights.items():
            weight_stats[signal_type] = {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'min': np.min(weights),
                'max': np.max(weights)
            }
        
        return {
            'total_adjustments': len(self.weight_history),
            'average_influences': {
                'regime': avg_regime_influence,
                'performance': avg_performance_influence,
                'volatility': avg_volatility_influence
            },
            'average_confidence': avg_confidence,
            'weight_statistics': weight_stats,
            'registered_signals': len(self.signals),
            'adjustment_frequency': len(recent_history) / max(1, (datetime.now() - recent_history[0].timestamp).days) if recent_history else 0
        }
    
    def export_weight_history(self, filepath: str) -> None:
        """Export weight adjustment history to JSON"""
        export_data = []
        
        for result in self.weight_history:
            export_record = {
                'timestamp': result.timestamp.isoformat(),
                'signal_weights': result.signal_weights,
                'regime_influence': result.regime_influence,
                'performance_influence': result.performance_influence,
                'volatility_influence': result.volatility_influence,
                'confidence_score': result.confidence_score,
                'explanation': result.explanation
            }
            export_data.append(export_record)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(export_data)} weight adjustment records to {filepath}")
    
    def optimize_weights_historical(self, historical_data: pd.DataFrame, 
                                   target_returns: pd.Series) -> Dict[str, float]:
        """
        Optimize weights based on historical performance using mean-variance optimization.
        Research: Markowitz (1952), DeMiguel et al. (2009)
        """
        
        if len(historical_data) < 30:
            logger.warning("Insufficient historical data for optimization")
            return self.get_current_weights()
        
        try:
            # Calculate signal returns (correlation with target returns)
            signal_returns = {}
            for signal_type in self.signals.keys():
                if signal_type in historical_data.columns:
                    signal_data = historical_data[signal_type].dropna()
                    if len(signal_data) > 10:
                        correlation = np.corrcoef(signal_data, target_returns[-len(signal_data):])[0, 1]
                        if not np.isnan(correlation):
                            signal_returns[signal_type] = correlation
            
            if not signal_returns:
                return self.get_current_weights()
            
            # Simple optimization: weight by absolute correlation, bounded
            abs_correlations = {k: abs(v) for k, v in signal_returns.items()}
            total_correlation = sum(abs_correlations.values())
            
            if total_correlation > 0:
                optimized_weights = {}
                for signal_type, abs_corr in abs_correlations.items():
                    raw_weight = abs_corr / total_correlation
                    bounded_weight = max(self.min_weight, min(self.max_weight, raw_weight))
                    optimized_weights[signal_type] = bounded_weight
                
                # Normalize
                total_weight = sum(optimized_weights.values())
                optimized_weights = {k: v/total_weight for k, v in optimized_weights.items()}
                
                logger.info(f"Historical optimization completed for {len(optimized_weights)} signals")
                return optimized_weights
            
        except Exception as e:
            logger.error(f"Historical optimization failed: {e}")
        
        return self.get_current_weights()

if __name__ == "__main__":
    # Example usage and testing
    weighter = DynamicSignalWeighter()
    
    # Register signals
    signals_to_register = ['rsi', 'macd', 'bollinger_bands', 'volume_breakout', 'obv_divergence']
    for signal in signals_to_register:
        weighter.register_signal(signal)
    
    # Test weight calculation
    print("Testing Dynamic Signal Weighting System...")
    
    # Sample market conditions
    test_conditions = [
        ('bull_market', 0.8, 0.3),
        ('bear_market', 0.9, 0.7), 
        ('volatile_market', 0.7, 0.9),
        ('sideways_market', 0.6, 0.4)
    ]
    
    # Sample signal performance
    signal_performance = {
        'rsi': 0.65,
        'macd': 0.58,
        'bollinger_bands': 0.72,
        'volume_breakout': 0.45,
        'obv_divergence': 0.69
    }
    
    # Sample current signals
    current_signals = {
        'rsi': {'strength': 0.7, 'supporting_indicators': {'volume': True, 'trend': True}},
        'macd': {'strength': 0.6, 'volume_confirmed': True},
        'bollinger_bands': {'strength': 0.8, 'signal_age': 3},
        'volume_breakout': {'strength': 0.5},
        'obv_divergence': {'strength': 0.7, 'supporting_indicators': {'price': True}}
    }
    
    for regime, confidence, volatility in test_conditions:
        result = weighter.calculate_dynamic_weights(
            regime, confidence, volatility, signal_performance, current_signals
        )
        
        print(f"\n{regime.upper()} (conf: {confidence:.1f}, vol: {volatility:.1f}):")
        print(f"  Explanation: {result.explanation}")
        
        # Show top 3 weights
        sorted_weights = sorted(result.signal_weights.items(), key=lambda x: x[1], reverse=True)
        for signal_type, weight in sorted_weights[:3]:
            print(f"  {signal_type}: {weight:.1%}")
    
    # Display system statistics
    stats = weighter.get_weight_statistics()
    print(f"\nSystem Statistics:")
    print(f"  Total adjustments: {stats['total_adjustments']}")
    print(f"  Average confidence: {stats.get('average_confidence', 0):.2f}")
    print(f"  Registered signals: {stats['registered_signals']}")