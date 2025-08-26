"""
Volume-Based Signal Generation
Research-backed volume signal analysis for enhanced trading decisions

Based on academic research findings:
- OBV divergence signals (Granville 1963)
- CMF momentum signals (Chaikin)
- MFI overbought/oversold levels
- Volume breakout confirmation
- VWAP institutional trading signals
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

@dataclass
class VolumeSignal:
    """Volume signal with research-backed scoring"""
    signal_type: str
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    direction: str  # 'BUY', 'SELL', 'HOLD'
    explanation: str
    supporting_indicators: List[str]
    volume_value: Optional[float] = None
    price_at_signal: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert signal to dictionary for database storage"""
        return {
            'signal_type': self.signal_type,
            'strength': self.strength,
            'confidence': self.confidence,
            'direction': self.direction,
            'explanation': self.explanation,
            'supporting_indicators': json.dumps(self.supporting_indicators),
            'volume_value': self.volume_value,
            'price_at_signal': self.price_at_signal,
            'timestamp': self.timestamp or datetime.now()
        }

class VolumeSignalGenerator:
    """Generate volume-based trading signals using academic research findings"""
    
    def __init__(self):
        self.logger = logger
        
        # Research-backed thresholds from academic studies
        self.thresholds = {
            # CMF thresholds (Chaikin research)
            'cmf_strong_bullish': 0.2,      # Strong buying pressure
            'cmf_bullish': 0.1,             # CMF > 0.1 indicates buying pressure
            'cmf_bearish': -0.1,            # CMF < -0.1 indicates selling pressure
            'cmf_strong_bearish': -0.2,     # Strong selling pressure
            
            # MFI thresholds (research-optimized)
            'mfi_overbought': 80,           # MFI overbought level
            'mfi_oversold': 20,             # MFI oversold level
            'mfi_extreme_overbought': 90,   # Extreme overbought
            'mfi_extreme_oversold': 10,     # Extreme oversold
            
            # Volume spike thresholds
            'volume_spike': 2.0,            # 2x average volume indicates unusual activity
            'volume_surge': 3.0,            # 3x average volume indicates strong interest
            
            # OBV divergence parameters
            'obv_divergence_periods': 5,    # Look back for divergence analysis
            'obv_trend_periods': 10,        # Trend analysis period
            
            # VWAP deviation thresholds
            'vwap_deviation_threshold': 0.02,  # 2% deviation from VWAP
            'vwap_strong_deviation': 0.05,     # 5% strong deviation
            
            # Price movement confirmation thresholds
            'price_movement_threshold': 0.015,  # 1.5% price movement
            'strong_price_movement': 0.03       # 3% strong price movement
        }
        
        # Signal confidence modifiers based on research
        self.confidence_modifiers = {
            'volume_confirmation': 0.15,     # +15% confidence with volume
            'multiple_indicators': 0.10,     # +10% with multiple indicators
            'trend_alignment': 0.05,         # +5% with trend alignment
            'extreme_reading': 0.20          # +20% with extreme readings
        }
    
    def analyze_obv_signals(self, df: pd.DataFrame) -> List[VolumeSignal]:
        """
        Analyze OBV for accumulation/distribution signals
        Research: Granville (1963) - OBV precedes price movements
        
        Args:
            df: DataFrame with OHLCV data and OBV indicator
            
        Returns:
            List of OBV-based signals
        """
        signals = []
        
        if len(df) < self.thresholds['obv_divergence_periods'] or 'obv' not in df.columns:
            return signals
        
        try:
            latest = df.iloc[-1]
            recent = df.tail(self.thresholds['obv_divergence_periods'])
            trend_data = df.tail(self.thresholds['obv_trend_periods'])
            
            # OBV and price trend analysis
            obv_slope = self._calculate_slope(recent['obv'])
            price_slope = self._calculate_slope(recent['close'])
            
            # OBV momentum
            obv_momentum = self._calculate_slope(trend_data['obv'])
            
            # Bullish divergence: Price falling, OBV rising
            if price_slope < -0.001 and obv_slope > 0.001:  # Meaningful slopes
                strength = min(abs(obv_slope) / max(abs(price_slope), 0.001), 1.0)
                confidence = 0.75
                
                # Enhance confidence if volume is above average
                if 'volume_ratio_20' in df.columns and latest['volume_ratio_20'] > 1.2:
                    confidence += self.confidence_modifiers['volume_confirmation']
                
                signals.append(VolumeSignal(
                    signal_type='OBV_BULLISH_DIVERGENCE',
                    strength=strength,
                    confidence=min(confidence, 0.95),
                    direction='BUY',
                    explanation=f'Bullish divergence: Price declining ({price_slope:.4f}) while OBV rising ({obv_slope:.4f})',
                    supporting_indicators=['obv', 'price_trend'],
                    volume_value=float(latest['obv']),
                    price_at_signal=float(latest['close'])
                ))
            
            # Bearish divergence: Price rising, OBV falling
            elif price_slope > 0.001 and obv_slope < -0.001:
                strength = min(abs(obv_slope) / max(abs(price_slope), 0.001), 1.0)
                confidence = 0.75
                
                # Enhance confidence with volume confirmation
                if 'volume_ratio_20' in df.columns and latest['volume_ratio_20'] > 1.2:
                    confidence += self.confidence_modifiers['volume_confirmation']
                
                signals.append(VolumeSignal(
                    signal_type='OBV_BEARISH_DIVERGENCE',
                    strength=strength,
                    confidence=min(confidence, 0.95),
                    direction='SELL',
                    explanation=f'Bearish divergence: Price rising ({price_slope:.4f}) while OBV falling ({obv_slope:.4f})',
                    supporting_indicators=['obv', 'price_trend'],
                    volume_value=float(latest['obv']),
                    price_at_signal=float(latest['close'])
                ))
            
            # OBV momentum signals
            if abs(obv_momentum) > 0.002:  # Significant OBV momentum
                direction = 'BUY' if obv_momentum > 0 else 'SELL'
                signal_type = 'OBV_MOMENTUM_BULLISH' if obv_momentum > 0 else 'OBV_MOMENTUM_BEARISH'
                
                signals.append(VolumeSignal(
                    signal_type=signal_type,
                    strength=min(abs(obv_momentum) * 100, 1.0),
                    confidence=0.60,
                    direction=direction,
                    explanation=f'Strong OBV momentum: {obv_momentum:.4f}',
                    supporting_indicators=['obv'],
                    volume_value=float(latest['obv']),
                    price_at_signal=float(latest['close'])
                ))
                
        except Exception as e:
            self.logger.warning(f"Error in OBV analysis: {str(e)}")
        
        return signals
    
    def analyze_cmf_signals(self, df: pd.DataFrame) -> List[VolumeSignal]:
        """
        Analyze Chaikin Money Flow signals
        Research: Better oscillator than OBV for 20-day periods
        
        Args:
            df: DataFrame with OHLCV data and CMF indicator
            
        Returns:
            List of CMF-based signals
        """
        signals = []
        
        if len(df) < 2 or 'cmf' not in df.columns:
            return signals
        
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            cmf_current = latest['cmf']
            cmf_prev = prev['cmf']
            
            if pd.isna(cmf_current) or pd.isna(cmf_prev):
                return signals
            
            # CMF crossing above bullish threshold
            if (cmf_prev <= self.thresholds['cmf_bullish'] and 
                cmf_current > self.thresholds['cmf_bullish']):
                
                strength = min(cmf_current / self.thresholds['cmf_bullish'], 1.0)
                confidence = 0.70
                
                # Strong bullish signal
                if cmf_current > self.thresholds['cmf_strong_bullish']:
                    confidence += self.confidence_modifiers['extreme_reading']
                    strength = min(strength * 1.5, 1.0)
                
                signals.append(VolumeSignal(
                    signal_type='CMF_BULLISH_CROSS',
                    strength=strength,
                    confidence=min(confidence, 0.95),
                    direction='BUY',
                    explanation=f'CMF bullish cross: {cmf_current:.3f} (threshold: {self.thresholds["cmf_bullish"]:.3f})',
                    supporting_indicators=['cmf'],
                    volume_value=float(cmf_current),
                    price_at_signal=float(latest['close'])
                ))
            
            # CMF crossing below bearish threshold
            elif (cmf_prev >= self.thresholds['cmf_bearish'] and 
                  cmf_current < self.thresholds['cmf_bearish']):
                
                strength = min(abs(cmf_current) / abs(self.thresholds['cmf_bearish']), 1.0)
                confidence = 0.70
                
                # Strong bearish signal
                if cmf_current < self.thresholds['cmf_strong_bearish']:
                    confidence += self.confidence_modifiers['extreme_reading']
                    strength = min(strength * 1.5, 1.0)
                
                signals.append(VolumeSignal(
                    signal_type='CMF_BEARISH_CROSS',
                    strength=strength,
                    confidence=min(confidence, 0.95),
                    direction='SELL',
                    explanation=f'CMF bearish cross: {cmf_current:.3f} (threshold: {self.thresholds["cmf_bearish"]:.3f})',
                    supporting_indicators=['cmf'],
                    volume_value=float(cmf_current),
                    price_at_signal=float(latest['close'])
                ))
            
            # CMF momentum signals
            cmf_change = cmf_current - cmf_prev
            if abs(cmf_change) > 0.05:  # Significant CMF change
                direction = 'BUY' if cmf_change > 0 else 'SELL'
                signal_type = 'CMF_MOMENTUM_BULLISH' if cmf_change > 0 else 'CMF_MOMENTUM_BEARISH'
                
                signals.append(VolumeSignal(
                    signal_type=signal_type,
                    strength=min(abs(cmf_change) * 10, 1.0),
                    confidence=0.55,
                    direction=direction,
                    explanation=f'CMF momentum: {cmf_change:+.3f}',
                    supporting_indicators=['cmf'],
                    volume_value=float(cmf_current),
                    price_at_signal=float(latest['close'])
                ))
                
        except Exception as e:
            self.logger.warning(f"Error in CMF analysis: {str(e)}")
        
        return signals
    
    def analyze_mfi_signals(self, df: pd.DataFrame) -> List[VolumeSignal]:
        """
        Analyze Money Flow Index signals
        Research: Volume-weighted RSI, more reliable in volatile markets
        
        Args:
            df: DataFrame with OHLCV data and MFI indicator
            
        Returns:
            List of MFI-based signals
        """
        signals = []
        
        if len(df) < 2 or 'mfi' not in df.columns:
            return signals
        
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            mfi_current = latest['mfi']
            mfi_prev = prev['mfi']
            
            if pd.isna(mfi_current) or pd.isna(mfi_prev):
                return signals
            
            # MFI oversold bounce
            if (mfi_prev <= self.thresholds['mfi_oversold'] and 
                mfi_current > self.thresholds['mfi_oversold']):
                
                oversold_depth = self.thresholds['mfi_oversold'] - min(mfi_prev, mfi_current)
                strength = min(oversold_depth / self.thresholds['mfi_oversold'], 1.0)
                confidence = 0.65
                
                # Extreme oversold bounce
                if min(mfi_prev, mfi_current) <= self.thresholds['mfi_extreme_oversold']:
                    confidence += self.confidence_modifiers['extreme_reading']
                    strength = min(strength * 1.3, 1.0)
                
                signals.append(VolumeSignal(
                    signal_type='MFI_OVERSOLD_BOUNCE',
                    strength=strength,
                    confidence=min(confidence, 0.95),
                    direction='BUY',
                    explanation=f'MFI oversold bounce: {mfi_prev:.1f} → {mfi_current:.1f}',
                    supporting_indicators=['mfi', 'volume'],
                    volume_value=float(mfi_current),
                    price_at_signal=float(latest['close'])
                ))
            
            # MFI overbought reversal
            elif (mfi_prev >= self.thresholds['mfi_overbought'] and 
                  mfi_current < self.thresholds['mfi_overbought']):
                
                overbought_level = max(mfi_prev, mfi_current) - self.thresholds['mfi_overbought']
                strength = min(overbought_level / (100 - self.thresholds['mfi_overbought']), 1.0)
                confidence = 0.65
                
                # Extreme overbought reversal
                if max(mfi_prev, mfi_current) >= self.thresholds['mfi_extreme_overbought']:
                    confidence += self.confidence_modifiers['extreme_reading']
                    strength = min(strength * 1.3, 1.0)
                
                signals.append(VolumeSignal(
                    signal_type='MFI_OVERBOUGHT_REVERSAL',
                    strength=strength,
                    confidence=min(confidence, 0.95),
                    direction='SELL',
                    explanation=f'MFI overbought reversal: {mfi_prev:.1f} → {mfi_current:.1f}',
                    supporting_indicators=['mfi', 'volume'],
                    volume_value=float(mfi_current),
                    price_at_signal=float(latest['close'])
                ))
                
        except Exception as e:
            self.logger.warning(f"Error in MFI analysis: {str(e)}")
        
        return signals
    
    def analyze_volume_breakout(self, df: pd.DataFrame) -> List[VolumeSignal]:
        """
        Analyze volume breakouts
        Research: Volume spikes precede significant price movements
        
        Args:
            df: DataFrame with OHLCV data and volume ratios
            
        Returns:
            List of volume breakout signals
        """
        signals = []
        
        if len(df) < 20:  # Need enough data for volume average
            return signals
        
        try:
            latest = df.iloc[-1]
            
            # Use pre-calculated volume ratio if available, otherwise calculate
            if 'volume_ratio_20' in df.columns:
                volume_ratio = latest['volume_ratio_20']
            else:
                volume_avg = df['volume'].tail(20).mean()
                volume_ratio = latest['volume'] / volume_avg
            
            if pd.isna(volume_ratio) or volume_ratio <= 0:
                return signals
            
            # Volume spike detection
            if volume_ratio >= self.thresholds['volume_spike']:
                # Determine direction based on price action
                price_change = (latest['close'] - latest['open']) / latest['open']
                
                # Bullish volume breakout
                if price_change > self.thresholds['price_movement_threshold']:
                    strength = min(volume_ratio / self.thresholds['volume_spike'], 1.0)
                    confidence = 0.80
                    
                    # Strong price movement with volume
                    if price_change > self.thresholds['strong_price_movement']:
                        confidence += self.confidence_modifiers['extreme_reading']
                        strength = min(strength * 1.2, 1.0)
                    
                    # Volume surge (3x+ volume)
                    if volume_ratio >= self.thresholds['volume_surge']:
                        strength = min(strength * 1.5, 1.0)
                        confidence += 0.1
                    
                    signals.append(VolumeSignal(
                        signal_type='VOLUME_BREAKOUT_BULLISH',
                        strength=strength,
                        confidence=min(confidence, 0.95),
                        direction='BUY',
                        explanation=f'Bullish volume breakout: {volume_ratio:.1f}x volume with {price_change:+.2%} price move',
                        supporting_indicators=['volume', 'price_action'],
                        volume_value=float(latest['volume']),
                        price_at_signal=float(latest['close'])
                    ))
                
                # Bearish volume breakout
                elif price_change < -self.thresholds['price_movement_threshold']:
                    strength = min(volume_ratio / self.thresholds['volume_spike'], 1.0)
                    confidence = 0.80
                    
                    # Strong price decline with volume
                    if price_change < -self.thresholds['strong_price_movement']:
                        confidence += self.confidence_modifiers['extreme_reading']
                        strength = min(strength * 1.2, 1.0)
                    
                    # Volume surge
                    if volume_ratio >= self.thresholds['volume_surge']:
                        strength = min(strength * 1.5, 1.0)
                        confidence += 0.1
                    
                    signals.append(VolumeSignal(
                        signal_type='VOLUME_BREAKOUT_BEARISH',
                        strength=strength,
                        confidence=min(confidence, 0.95),
                        direction='SELL',
                        explanation=f'Bearish volume breakout: {volume_ratio:.1f}x volume with {price_change:+.2%} price move',
                        supporting_indicators=['volume', 'price_action'],
                        volume_value=float(latest['volume']),
                        price_at_signal=float(latest['close'])
                    ))
                
                # High volume but unclear direction
                elif volume_ratio >= self.thresholds['volume_surge']:
                    signals.append(VolumeSignal(
                        signal_type='VOLUME_SURGE_NEUTRAL',
                        strength=min(volume_ratio / self.thresholds['volume_surge'], 1.0),
                        confidence=0.50,
                        direction='HOLD',
                        explanation=f'Volume surge without clear direction: {volume_ratio:.1f}x volume',
                        supporting_indicators=['volume'],
                        volume_value=float(latest['volume']),
                        price_at_signal=float(latest['close'])
                    ))
                    
        except Exception as e:
            self.logger.warning(f"Error in volume breakout analysis: {str(e)}")
        
        return signals
    
    def analyze_vwap_signals(self, df: pd.DataFrame) -> List[VolumeSignal]:
        """
        Analyze VWAP deviation signals
        Research: VWAP acts as dynamic support/resistance
        
        Args:
            df: DataFrame with OHLCV data and VWAP
            
        Returns:
            List of VWAP-based signals
        """
        signals = []
        
        if len(df) < 2 or 'vwap' not in df.columns:
            return signals
        
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            current_price = latest['close']
            current_vwap = latest['vwap']
            prev_price = prev['close']
            prev_vwap = prev['vwap']
            
            if pd.isna(current_vwap) or pd.isna(prev_vwap):
                return signals
            
            # VWAP deviation analysis
            current_deviation = (current_price - current_vwap) / current_vwap
            prev_deviation = (prev_price - prev_vwap) / prev_vwap
            
            # VWAP cross signals
            if prev_deviation <= 0 and current_deviation > 0:  # Price crosses above VWAP
                strength = min(abs(current_deviation) / self.thresholds['vwap_deviation_threshold'], 1.0)
                confidence = 0.60
                
                # Strong deviation
                if abs(current_deviation) > self.thresholds['vwap_strong_deviation']:
                    confidence += self.confidence_modifiers['extreme_reading']
                
                signals.append(VolumeSignal(
                    signal_type='VWAP_BULLISH_CROSS',
                    strength=strength,
                    confidence=min(confidence, 0.90),
                    direction='BUY',
                    explanation=f'Price crossed above VWAP: {current_deviation:+.2%} deviation',
                    supporting_indicators=['vwap', 'price'],
                    volume_value=float(current_vwap),
                    price_at_signal=float(current_price)
                ))
            
            elif prev_deviation >= 0 and current_deviation < 0:  # Price crosses below VWAP
                strength = min(abs(current_deviation) / self.thresholds['vwap_deviation_threshold'], 1.0)
                confidence = 0.60
                
                # Strong deviation
                if abs(current_deviation) > self.thresholds['vwap_strong_deviation']:
                    confidence += self.confidence_modifiers['extreme_reading']
                
                signals.append(VolumeSignal(
                    signal_type='VWAP_BEARISH_CROSS',
                    strength=strength,
                    confidence=min(confidence, 0.90),
                    direction='SELL',
                    explanation=f'Price crossed below VWAP: {current_deviation:+.2%} deviation',
                    supporting_indicators=['vwap', 'price'],
                    volume_value=float(current_vwap),
                    price_at_signal=float(current_price)
                ))
            
            # Mean reversion signals (extreme deviations)
            if abs(current_deviation) > self.thresholds['vwap_strong_deviation']:
                direction = 'SELL' if current_deviation > 0 else 'BUY'  # Mean reversion
                signal_type = 'VWAP_MEAN_REVERSION_SELL' if current_deviation > 0 else 'VWAP_MEAN_REVERSION_BUY'
                
                signals.append(VolumeSignal(
                    signal_type=signal_type,
                    strength=min(abs(current_deviation) / self.thresholds['vwap_strong_deviation'], 1.0),
                    confidence=0.70,
                    direction=direction,
                    explanation=f'Extreme VWAP deviation ({current_deviation:+.2%}) suggests mean reversion',
                    supporting_indicators=['vwap', 'mean_reversion'],
                    volume_value=float(current_vwap),
                    price_at_signal=float(current_price)
                ))
                
        except Exception as e:
            self.logger.warning(f"Error in VWAP analysis: {str(e)}")
        
        return signals
    
    def generate_volume_signals(self, df: pd.DataFrame) -> Dict[str, List[VolumeSignal]]:
        """
        Generate comprehensive volume-based signals
        
        Args:
            df: DataFrame with OHLCV data and volume indicators
            
        Returns:
            Dictionary of signal categories with lists of signals
        """
        all_signals = {
            'obv_signals': [],
            'cmf_signals': [],
            'mfi_signals': [],
            'volume_breakout_signals': [],
            'vwap_signals': []
        }
        
        try:
            # Generate all signal types
            all_signals['obv_signals'] = self.analyze_obv_signals(df)
            all_signals['cmf_signals'] = self.analyze_cmf_signals(df)
            all_signals['mfi_signals'] = self.analyze_mfi_signals(df)
            all_signals['volume_breakout_signals'] = self.analyze_volume_breakout(df)
            all_signals['vwap_signals'] = self.analyze_vwap_signals(df)
            
            # Log signal summary
            total_signals = sum(len(signals) for signals in all_signals.values())
            if total_signals > 0:
                self.logger.info(f"Generated {total_signals} volume signals across {len(all_signals)} categories")
            
        except Exception as e:
            self.logger.error(f"Error generating volume signals: {str(e)}")
        
        return all_signals
    
    def get_strongest_signals(self, signals_dict: Dict[str, List[VolumeSignal]], 
                            min_strength: float = 0.6, 
                            min_confidence: float = 0.6) -> List[VolumeSignal]:
        """
        Filter and return strongest volume signals
        
        Args:
            signals_dict: Dictionary of signal categories
            min_strength: Minimum signal strength threshold
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of filtered signals sorted by strength
        """
        strong_signals = []
        
        for category, signals in signals_dict.items():
            for signal in signals:
                if signal.strength >= min_strength and signal.confidence >= min_confidence:
                    strong_signals.append(signal)
        
        # Sort by combined score (strength * confidence)
        strong_signals.sort(key=lambda s: s.strength * s.confidence, reverse=True)
        
        return strong_signals
    
    def _calculate_slope(self, series: pd.Series) -> float:
        """Calculate slope of a series using linear regression"""
        if len(series) < 2:
            return 0.0
        
        try:
            # Remove NaN values
            clean_series = series.dropna()
            if len(clean_series) < 2:
                return 0.0
            
            x = np.arange(len(clean_series))
            y = clean_series.values
            
            # Linear regression slope
            n = len(x)
            if n < 2:
                return 0.0
                
            denominator = n * np.sum(x**2) - np.sum(x)**2
            if denominator == 0:
                return 0.0
                
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / denominator
            
            return slope
            
        except Exception as e:
            self.logger.warning(f"Error calculating slope: {str(e)}")
            return 0.0

# Example usage and testing
if __name__ == "__main__":
    # Test the volume signal generator
    generator = VolumeSignalGenerator()
    
    # Create sample data with volume indicators
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'trade_date': dates,
        'open': np.random.normal(100, 1, 30),
        'high': np.random.normal(102, 1, 30),
        'low': np.random.normal(98, 1, 30),
        'close': np.random.normal(100, 1, 30),
        'volume': np.random.randint(1000000, 3000000, 30),
        'obv': np.cumsum(np.random.normal(0, 1000000, 30)),
        'cmf': np.random.normal(0, 0.1, 30),
        'mfi': np.random.normal(50, 20, 30),
        'vwap': np.random.normal(100, 1, 30),
        'volume_ratio_20': np.random.uniform(0.5, 3.0, 30)
    })
    
    # Ensure valid ranges
    sample_data['high'] = np.maximum(sample_data['high'], sample_data[['open', 'close']].max(axis=1))
    sample_data['low'] = np.minimum(sample_data['low'], sample_data[['open', 'close']].min(axis=1))
    sample_data['mfi'] = np.clip(sample_data['mfi'], 0, 100)
    sample_data['cmf'] = np.clip(sample_data['cmf'], -1, 1)
    
    try:
        # Generate volume signals
        signals = generator.generate_volume_signals(sample_data)
        
        print("Volume signals generated successfully!")
        for category, signal_list in signals.items():
            print(f"{category}: {len(signal_list)} signals")
            for signal in signal_list:
                print(f"  - {signal.signal_type}: {signal.direction} (strength: {signal.strength:.2f}, confidence: {signal.confidence:.2f})")
        
        # Get strongest signals
        strong_signals = generator.get_strongest_signals(signals, min_strength=0.5, min_confidence=0.5)
        print(f"\nStrongest signals: {len(strong_signals)}")
        
        for signal in strong_signals[:5]:  # Top 5 signals
            print(f"  {signal.signal_type}: {signal.direction} - {signal.explanation}")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()