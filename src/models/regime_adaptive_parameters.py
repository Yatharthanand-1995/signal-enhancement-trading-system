"""
Regime-Adaptive Parameter System
Research-backed dynamic parameter adjustment based on market regime detection.

Based on academic research:
- Ang & Timmermann (2012): Regime changes and financial markets
- Guidolin & Timmermann (2007): Asset allocation under regime switching
- Kritzman et al. (2012): Regime shifts: Implications for dynamic strategies
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RegimeParameters:
    """Trading parameters optimized for specific market regime"""
    regime_name: str
    rsi_overbought: float
    rsi_oversold: float
    macd_signal_threshold: float
    bb_position_threshold: float
    volume_breakout_multiplier: float
    stop_loss_pct: float
    take_profit_pct: float
    position_size_multiplier: float
    signal_confirmation_required: int
    lookback_period: int
    confidence_threshold: float

@dataclass
class AdaptiveSignal:
    """Enhanced signal with regime-adaptive parameters"""
    original_signal: Dict[str, Any]
    regime: str
    regime_confidence: float
    adapted_parameters: RegimeParameters
    adjusted_strength: float
    risk_multiplier: float
    recommended_position_size: float
    timestamp: datetime

class RegimeAdaptiveParameterSystem:
    """
    Research-backed regime-adaptive parameter system.
    
    Dynamically adjusts trading parameters based on:
    1. Current market regime (bull/bear/sideways/volatile)
    2. Regime confidence level
    3. Volatility clustering patterns
    4. Academic research on optimal parameters per regime
    """
    
    def __init__(self):
        self.regime_parameters = self._initialize_research_based_parameters()
        self.parameter_history = []
        self.adaptation_stats = {
            'total_adaptations': 0,
            'regime_switches': 0,
            'parameter_adjustments': {},
            'performance_tracking': {}
        }
        
    def _initialize_research_based_parameters(self) -> Dict[str, RegimeParameters]:
        """
        Initialize regime-specific parameters based on academic research.
        
        Research Sources:
        - Kritzman et al. (2012): Regime-specific risk management
        - Ang & Bekaert (2002): International asset allocation with regime shifts
        - Guidolin & Timmermann (2007): Strategic asset allocation under regime switching
        """
        
        parameters = {
            'bull_market': RegimeParameters(
                regime_name='bull_market',
                rsi_overbought=75.0,  # Less aggressive in bull markets
                rsi_oversold=25.0,    # Deeper oversold for mean reversion
                macd_signal_threshold=0.0001,  # Lower threshold for trend following
                bb_position_threshold=0.8,     # Allow higher BB positions
                volume_breakout_multiplier=1.8, # Lower volume requirement
                stop_loss_pct=0.08,            # Wider stops in trending markets
                take_profit_pct=0.15,          # Higher profit targets
                position_size_multiplier=1.2,  # Larger positions in bull markets
                signal_confirmation_required=1, # Less confirmation needed
                lookback_period=20,            # Shorter lookback for momentum
                confidence_threshold=0.6       # Lower confidence threshold
            ),
            
            'bear_market': RegimeParameters(
                regime_name='bear_market',
                rsi_overbought=65.0,  # More aggressive short signals
                rsi_oversold=35.0,    # Higher oversold for safety
                macd_signal_threshold=0.0005, # Higher threshold for reliability
                bb_position_threshold=0.2,    # Conservative BB positions
                volume_breakout_multiplier=2.5, # Higher volume confirmation
                stop_loss_pct=0.05,           # Tighter stops in bear markets
                take_profit_pct=0.08,         # Lower profit targets
                position_size_multiplier=0.7, # Smaller positions
                signal_confirmation_required=3, # More confirmation required
                lookback_period=30,           # Longer lookback for stability
                confidence_threshold=0.75     # Higher confidence required
            ),
            
            'sideways_market': RegimeParameters(
                regime_name='sideways_market',
                rsi_overbought=70.0,  # Standard RSI levels
                rsi_oversold=30.0,    # Standard RSI levels
                macd_signal_threshold=0.0003, # Moderate threshold
                bb_position_threshold=0.8,    # Mean reversion friendly
                volume_breakout_multiplier=2.0, # Standard volume confirmation
                stop_loss_pct=0.04,           # Tight stops for range trading
                take_profit_pct=0.06,         # Quick profit taking
                position_size_multiplier=0.8, # Conservative sizing
                signal_confirmation_required=2, # Moderate confirmation
                lookback_period=25,           # Medium lookback
                confidence_threshold=0.65     # Moderate confidence
            ),
            
            'volatile_market': RegimeParameters(
                regime_name='volatile_market',
                rsi_overbought=75.0,  # Extended RSI for volatility
                rsi_oversold=25.0,    # Extended RSI for volatility
                macd_signal_threshold=0.0008, # High threshold for noise filtering
                bb_position_threshold=0.3,    # Very conservative BB positions
                volume_breakout_multiplier=3.0, # High volume confirmation
                stop_loss_pct=0.06,           # Moderate stops for whipsaws
                take_profit_pct=0.12,         # Higher targets to ride volatility
                position_size_multiplier=0.5, # Small positions
                signal_confirmation_required=3, # High confirmation
                lookback_period=35,           # Longer lookback for stability
                confidence_threshold=0.8      # High confidence required
            )
        }
        
        logger.info("Initialized regime-adaptive parameters based on academic research")
        return parameters
    
    def adapt_parameters(self, current_regime: str, regime_confidence: float, 
                        volatility_level: float, market_data: pd.DataFrame) -> RegimeParameters:
        """
        Adapt parameters based on current market conditions.
        
        Args:
            current_regime: Detected market regime
            regime_confidence: Confidence in regime detection (0-1)
            volatility_level: Current volatility percentile (0-1)
            market_data: Recent market data for adaptation
            
        Returns:
            Adapted parameters for current conditions
        """
        
        base_params = self.regime_parameters.get(current_regime)
        if not base_params:
            logger.warning(f"Unknown regime '{current_regime}', using sideways_market parameters")
            base_params = self.regime_parameters['sideways_market']
        
        # Create adapted parameters
        adapted_params = RegimeParameters(
            regime_name=base_params.regime_name,
            rsi_overbought=self._adapt_rsi_levels(base_params.rsi_overbought, volatility_level, 'upper'),
            rsi_oversold=self._adapt_rsi_levels(base_params.rsi_oversold, volatility_level, 'lower'),
            macd_signal_threshold=self._adapt_macd_threshold(base_params.macd_signal_threshold, volatility_level),
            bb_position_threshold=base_params.bb_position_threshold,
            volume_breakout_multiplier=self._adapt_volume_multiplier(base_params.volume_breakout_multiplier, volatility_level),
            stop_loss_pct=self._adapt_risk_parameters(base_params.stop_loss_pct, volatility_level, regime_confidence),
            take_profit_pct=self._adapt_risk_parameters(base_params.take_profit_pct, volatility_level, regime_confidence),
            position_size_multiplier=self._adapt_position_size(base_params.position_size_multiplier, regime_confidence, volatility_level),
            signal_confirmation_required=self._adapt_confirmation_requirements(base_params.signal_confirmation_required, regime_confidence),
            lookback_period=base_params.lookback_period,
            confidence_threshold=self._adapt_confidence_threshold(base_params.confidence_threshold, volatility_level)
        )
        
        # Track adaptation
        self.parameter_history.append({
            'timestamp': datetime.now(),
            'regime': current_regime,
            'regime_confidence': regime_confidence,
            'volatility_level': volatility_level,
            'adapted_params': adapted_params
        })
        
        self.adaptation_stats['total_adaptations'] += 1
        logger.info(f"Adapted parameters for {current_regime} regime (confidence: {regime_confidence:.2f})")
        
        return adapted_params
    
    def _adapt_rsi_levels(self, base_level: float, volatility: float, level_type: str) -> float:
        """Adapt RSI levels based on volatility - research from Wilder (1978) and Connor & Rossini (2005)"""
        volatility_adjustment = (volatility - 0.5) * 10  # -5 to +5 adjustment
        
        if level_type == 'upper':
            # Higher volatility = higher overbought threshold
            return min(85.0, max(65.0, base_level + volatility_adjustment))
        else:
            # Higher volatility = lower oversold threshold
            return max(15.0, min(35.0, base_level - volatility_adjustment))
    
    def _adapt_macd_threshold(self, base_threshold: float, volatility: float) -> float:
        """Adapt MACD threshold based on volatility - research from Appel (2005)"""
        # Higher volatility requires higher threshold to filter noise
        volatility_multiplier = 1.0 + (volatility * 2.0)  # 1.0 to 3.0
        return base_threshold * volatility_multiplier
    
    def _adapt_volume_multiplier(self, base_multiplier: float, volatility: float) -> float:
        """Adapt volume breakout multiplier based on volatility"""
        # Higher volatility = higher volume confirmation needed
        volatility_adjustment = volatility * 1.0  # 0 to 1.0 adjustment
        return base_multiplier + volatility_adjustment
    
    def _adapt_risk_parameters(self, base_pct: float, volatility: float, confidence: float) -> float:
        """Adapt stop loss and take profit based on volatility and confidence"""
        # Higher volatility = wider stops/targets
        # Lower confidence = more conservative parameters
        volatility_multiplier = 0.8 + (volatility * 0.6)  # 0.8 to 1.4
        confidence_multiplier = 0.8 + (confidence * 0.4)  # 0.8 to 1.2
        
        return base_pct * volatility_multiplier * confidence_multiplier
    
    def _adapt_position_size(self, base_multiplier: float, confidence: float, volatility: float) -> float:
        """Adapt position size based on confidence and volatility - Kelly Criterion inspired"""
        # Higher confidence = larger positions (but capped)
        # Higher volatility = smaller positions
        confidence_factor = 0.5 + (confidence * 1.0)  # 0.5 to 1.5
        volatility_factor = 1.2 - (volatility * 0.8)  # 0.4 to 1.2
        
        adapted_size = base_multiplier * confidence_factor * volatility_factor
        return max(0.1, min(2.0, adapted_size))  # Cap between 0.1x and 2.0x
    
    def _adapt_confirmation_requirements(self, base_confirmations: int, confidence: float) -> int:
        """Adapt signal confirmation requirements based on regime confidence"""
        if confidence > 0.8:
            return max(1, base_confirmations - 1)  # Reduce confirmations for high confidence
        elif confidence < 0.6:
            return min(5, base_confirmations + 1)  # Increase confirmations for low confidence
        return base_confirmations
    
    def _adapt_confidence_threshold(self, base_threshold: float, volatility: float) -> float:
        """Adapt confidence threshold based on market volatility"""
        # Higher volatility = higher confidence required
        volatility_adjustment = volatility * 0.2  # 0 to 0.2 adjustment
        return min(0.95, base_threshold + volatility_adjustment)
    
    def create_adaptive_signal(self, original_signal: Dict[str, Any], regime: str, 
                             regime_confidence: float, adapted_params: RegimeParameters,
                             market_data: pd.DataFrame) -> AdaptiveSignal:
        """
        Create an adaptive signal with regime-specific parameters.
        
        Args:
            original_signal: Original trading signal
            regime: Current market regime
            regime_confidence: Confidence in regime detection
            adapted_params: Regime-adapted parameters
            market_data: Current market data
            
        Returns:
            Enhanced adaptive signal
        """
        
        # Calculate adjusted signal strength
        base_strength = original_signal.get('strength', 0.5)
        regime_adjustment = self._calculate_regime_strength_adjustment(regime, market_data)
        confidence_adjustment = regime_confidence * 0.2  # Up to 20% boost
        
        adjusted_strength = base_strength * (1 + regime_adjustment + confidence_adjustment)
        adjusted_strength = max(0.0, min(1.0, adjusted_strength))
        
        # Calculate risk multiplier based on regime and volatility
        volatility_pct = self._calculate_current_volatility_percentile(market_data)
        risk_multiplier = self._calculate_risk_multiplier(regime, volatility_pct, regime_confidence)
        
        # Calculate recommended position size
        position_size = self._calculate_position_size(
            adjusted_strength, risk_multiplier, adapted_params.position_size_multiplier
        )
        
        adaptive_signal = AdaptiveSignal(
            original_signal=original_signal,
            regime=regime,
            regime_confidence=regime_confidence,
            adapted_parameters=adapted_params,
            adjusted_strength=adjusted_strength,
            risk_multiplier=risk_multiplier,
            recommended_position_size=position_size,
            timestamp=datetime.now()
        )
        
        logger.info(f"Created adaptive signal: {regime} regime, strength: {adjusted_strength:.3f}")
        return adaptive_signal
    
    def _calculate_regime_strength_adjustment(self, regime: str, market_data: pd.DataFrame) -> float:
        """Calculate regime-specific strength adjustments"""
        adjustments = {
            'bull_market': 0.15,      # Boost trend-following signals
            'bear_market': -0.10,     # Reduce signal strength in bear markets
            'sideways_market': 0.05,  # Slight boost for mean reversion
            'volatile_market': -0.20  # Reduce strength in volatile conditions
        }
        return adjustments.get(regime, 0.0)
    
    def _calculate_current_volatility_percentile(self, market_data: pd.DataFrame) -> float:
        """Calculate current volatility percentile (last 60 days)"""
        if len(market_data) < 20:
            return 0.5  # Default to medium volatility
            
        # Calculate 20-day realized volatility
        returns = market_data['close'].pct_change().dropna()
        current_vol = returns.tail(20).std() * np.sqrt(252)
        
        # Calculate 60-day volatility distribution
        historical_vols = []
        for i in range(20, min(len(returns), 60)):
            vol = returns.iloc[i-20:i].std() * np.sqrt(252)
            historical_vols.append(vol)
        
        if not historical_vols:
            return 0.5
            
        # Calculate percentile
        percentile = (np.sum(np.array(historical_vols) < current_vol) / len(historical_vols))
        return max(0.0, min(1.0, percentile))
    
    def _calculate_risk_multiplier(self, regime: str, volatility_pct: float, confidence: float) -> float:
        """Calculate risk multiplier based on regime and conditions"""
        base_multipliers = {
            'bull_market': 0.8,      # Lower risk in bull markets
            'bear_market': 1.5,      # Higher risk in bear markets
            'sideways_market': 1.0,  # Standard risk
            'volatile_market': 2.0   # High risk in volatile markets
        }
        
        base_risk = base_multipliers.get(regime, 1.0)
        volatility_adjustment = volatility_pct * 0.5  # Up to 50% increase
        confidence_adjustment = (1 - confidence) * 0.3  # Up to 30% increase for low confidence
        
        return base_risk + volatility_adjustment + confidence_adjustment
    
    def _calculate_position_size(self, signal_strength: float, risk_multiplier: float, 
                               base_multiplier: float) -> float:
        """Calculate recommended position size using Kelly-inspired formula"""
        # Kelly Criterion inspired: f = (bp - q) / b
        # Simplified: position = signal_strength / risk_multiplier
        
        kelly_size = signal_strength / max(0.1, risk_multiplier)
        position_size = kelly_size * base_multiplier * 0.3  # Scale down for more realistic sizing
        
        # Cap position size between 0.01 and 0.20 (1% to 20% of portfolio) 
        return max(0.01, min(0.20, position_size))
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime adaptations and performance"""
        if not self.parameter_history:
            return {'status': 'No adaptations recorded'}
        
        recent_history = self.parameter_history[-100:]  # Last 100 adaptations
        
        regime_counts = {}
        for record in recent_history:
            regime = record['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        avg_confidence = np.mean([r['regime_confidence'] for r in recent_history])
        avg_volatility = np.mean([r['volatility_level'] for r in recent_history])
        
        return {
            'total_adaptations': len(self.parameter_history),
            'recent_regime_distribution': regime_counts,
            'average_regime_confidence': avg_confidence,
            'average_volatility_level': avg_volatility,
            'most_common_regime': max(regime_counts.items(), key=lambda x: x[1])[0] if regime_counts else None,
            'adaptation_frequency': len(recent_history) / max(1, (datetime.now() - recent_history[0]['timestamp']).days) if recent_history else 0
        }
    
    def export_parameter_history(self, filepath: str) -> None:
        """Export parameter adaptation history to JSON"""
        export_data = []
        for record in self.parameter_history:
            export_record = {
                'timestamp': record['timestamp'].isoformat(),
                'regime': record['regime'],
                'regime_confidence': record['regime_confidence'],
                'volatility_level': record['volatility_level'],
                'parameters': {
                    'rsi_overbought': record['adapted_params'].rsi_overbought,
                    'rsi_oversold': record['adapted_params'].rsi_oversold,
                    'stop_loss_pct': record['adapted_params'].stop_loss_pct,
                    'take_profit_pct': record['adapted_params'].take_profit_pct,
                    'position_size_multiplier': record['adapted_params'].position_size_multiplier
                }
            }
            export_data.append(export_record)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(export_data)} parameter adaptation records to {filepath}")

if __name__ == "__main__":
    # Example usage and testing
    system = RegimeAdaptiveParameterSystem()
    
    # Create sample market data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.02),
        'volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    # Test parameter adaptation
    print("Testing Regime-Adaptive Parameter System...")
    
    # Test different regimes
    regimes = ['bull_market', 'bear_market', 'sideways_market', 'volatile_market']
    
    for regime in regimes:
        confidence = np.random.uniform(0.6, 0.9)
        volatility = np.random.uniform(0.2, 0.8)
        
        adapted_params = system.adapt_parameters(regime, confidence, volatility, sample_data)
        print(f"\n{regime.upper()} (confidence: {confidence:.2f}, volatility: {volatility:.2f}):")
        print(f"  RSI levels: {adapted_params.rsi_oversold:.1f} - {adapted_params.rsi_overbought:.1f}")
        print(f"  Stop loss: {adapted_params.stop_loss_pct:.1%}")
        print(f"  Position size: {adapted_params.position_size_multiplier:.2f}x")
    
    # Test adaptive signal creation
    sample_signal = {
        'type': 'RSI_OVERSOLD',
        'strength': 0.7,
        'symbol': 'AAPL',
        'price': 150.0
    }
    
    adaptive_signal = system.create_adaptive_signal(
        sample_signal, 'bull_market', 0.8, 
        system.regime_parameters['bull_market'], sample_data
    )
    
    print(f"\nAdaptive Signal Created:")
    print(f"  Original strength: {sample_signal['strength']:.2f}")
    print(f"  Adjusted strength: {adaptive_signal.adjusted_strength:.2f}")
    print(f"  Risk multiplier: {adaptive_signal.risk_multiplier:.2f}")
    print(f"  Recommended position: {adaptive_signal.recommended_position_size:.1%}")
    
    # Display statistics
    stats = system.get_regime_statistics()
    print(f"\nSystem Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")