"""
Volume Indicators Implementation
Research-backed volume analysis for enhanced signal generation

Based on academic research:
- Granville (1963) - On-Balance Volume
- Chaikin Money Flow - Superior oscillator for 20-day periods
- Money Flow Index - Volume-weighted RSI equivalent
- VWAP - Critical for institutional trading detection
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class VolumeIndicatorCalculator:
    """Calculate advanced volume indicators based on academic research"""
    
    def __init__(self):
        self.logger = logger
        
        # Research-backed parameters
        self.cmf_period = 20      # Chaikin Money Flow optimal period
        self.mfi_period = 14      # Money Flow Index standard period
        self.obv_smoothing = 5    # OBV smoothing period for noise reduction
        self.volume_sma_periods = [10, 20]  # Volume moving averages
        self.unusual_volume_threshold = 2.0  # 2x average volume threshold
        
    def calculate_obv(self, prices: pd.Series, volumes: pd.Series) -> pd.Series:
        """
        On-Balance Volume (OBV) - Granville (1963)
        Research: Critical for accumulation/distribution analysis
        
        Args:
            prices: Close prices series
            volumes: Volume series
            
        Returns:
            OBV series
        """
        if len(prices) != len(volumes):
            raise ValueError("Prices and volumes must have same length")
        
        price_changes = prices.diff()
        
        # OBV calculation: add volume on up days, subtract on down days
        volume_direction = np.where(
            price_changes > 0, volumes,
            np.where(price_changes < 0, -volumes, 0)
        )
        
        obv = pd.Series(volume_direction, index=prices.index).cumsum()
        
        # Apply smoothing to reduce noise (research enhancement)
        obv_smoothed = obv.rolling(window=self.obv_smoothing, min_periods=1).mean()
        
        return obv_smoothed.fillna(0)
    
    def calculate_cmf(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     volume: pd.Series, period: int = None) -> pd.Series:
        """
        Chaikin Money Flow (CMF)
        Research: Superior to OBV for 20-day oscillations
        
        Args:
            high, low, close: Price series
            volume: Volume series
            period: Lookback period (default: self.cmf_period)
            
        Returns:
            CMF series (-1 to +1)
        """
        if period is None:
            period = self.cmf_period
        
        # Money Flow Multiplier
        mf_multiplier = ((close - low) - (high - close)) / (high - low)
        
        # Handle division by zero (when high == low)
        mf_multiplier = mf_multiplier.fillna(0)
        
        # Money Flow Volume
        mf_volume = mf_multiplier * volume
        
        # Chaikin Money Flow
        cmf = (mf_volume.rolling(window=period).sum() / 
               volume.rolling(window=period).sum())
        
        return cmf.fillna(0)
    
    def calculate_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     volume: pd.Series, period: int = None) -> pd.Series:
        """
        Money Flow Index (MFI) - Volume-weighted RSI equivalent
        Research: More reliable than RSI in volatile markets
        
        Args:
            high, low, close: Price series
            volume: Volume series  
            period: Lookback period (default: self.mfi_period)
            
        Returns:
            MFI series (0 to 100)
        """
        if period is None:
            period = self.mfi_period
        
        # Typical Price
        typical_price = (high + low + close) / 3
        
        # Raw Money Flow
        money_flow = typical_price * volume
        
        # Positive and Negative Money Flow
        price_change = typical_price.diff()
        
        positive_flow = pd.Series(
            np.where(price_change > 0, money_flow, 0), 
            index=money_flow.index
        )
        negative_flow = pd.Series(
            np.where(price_change < 0, money_flow, 0), 
            index=money_flow.index
        )
        
        # Money Flow Ratio
        positive_flow_sum = positive_flow.rolling(window=period).sum()
        negative_flow_sum = negative_flow.rolling(window=period).sum()
        
        # Avoid division by zero
        money_ratio = positive_flow_sum / negative_flow_sum.replace(0, np.nan)
        
        # Money Flow Index
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi.fillna(50)  # Neutral value when no data
    
    def calculate_vwap(self, high: pd.Series, low: pd.Series, close: pd.Series,
                      volume: pd.Series) -> pd.Series:
        """
        Volume-Weighted Average Price (VWAP)
        Research: Critical for institutional trading detection
        
        Args:
            high, low, close: Price series
            volume: Volume series
            
        Returns:
            VWAP series
        """
        # Typical Price
        typical_price = (high + low + close) / 3
        
        # Cumulative values for VWAP calculation
        cumulative_pv = (typical_price * volume).cumsum()
        cumulative_volume = volume.cumsum()
        
        # VWAP
        vwap = cumulative_pv / cumulative_volume
        
        return vwap.ffill()
    
    def calculate_accumulation_distribution(self, high: pd.Series, low: pd.Series,
                                          close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Accumulation/Distribution Line - Williams (1973)
        Research: Volume-price relationship indicator
        
        Args:
            high, low, close: Price series
            volume: Volume series
            
        Returns:
            A/D Line series
        """
        # Money Flow Multiplier (same as CMF)
        mf_multiplier = ((close - low) - (high - close)) / (high - low)
        mf_multiplier = mf_multiplier.fillna(0)
        
        # Money Flow Volume
        mf_volume = mf_multiplier * volume
        
        # Accumulation/Distribution Line (cumulative)
        ad_line = mf_volume.cumsum()
        
        return ad_line
    
    def calculate_price_volume_trend(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Price Volume Trend (PVT)
        Research: Better than OBV for trend confirmation
        
        Args:
            close: Close price series
            volume: Volume series
            
        Returns:
            PVT series
        """
        # Price change percentage
        price_change_pct = close.pct_change()
        
        # Price Volume Trend
        pvt_change = price_change_pct * volume
        pvt = pvt_change.cumsum()
        
        return pvt.fillna(0)
    
    def calculate_volume_profile(self, prices: pd.Series, volumes: pd.Series,
                               bins: int = 20) -> Dict[str, Any]:
        """
        Volume Profile Analysis
        Research: Critical for support/resistance identification
        
        Args:
            prices: Price series (typically close prices)
            volumes: Volume series
            bins: Number of price levels for volume distribution
            
        Returns:
            Dictionary with volume profile data
        """
        if len(prices) == 0 or len(volumes) == 0:
            return {}
        
        try:
            price_min, price_max = prices.min(), prices.max()
            
            if price_min == price_max:
                # Handle case where all prices are the same
                return {
                    'poc_price': float(price_min),
                    'total_volume': int(volumes.sum()),
                    'price_levels': {f'{price_min:.2f}': int(volumes.sum())},
                    'value_area_high': float(price_min),
                    'value_area_low': float(price_min)
                }
            
            # Create price bins
            price_bins = np.linspace(price_min, price_max, bins + 1)
            
            # Calculate volume at each price level
            volume_profile = {}
            total_volume = 0
            
            for i in range(bins):
                bin_min = price_bins[i]
                bin_max = price_bins[i + 1]
                
                # Find prices within this bin
                mask = (prices >= bin_min) & (prices <= bin_max)
                volume_at_level = volumes[mask].sum()
                
                avg_price = (bin_min + bin_max) / 2
                volume_profile[f'{avg_price:.2f}'] = int(volume_at_level)
                total_volume += volume_at_level
            
            # Find Point of Control (POC) - price level with highest volume
            poc_price = float(max(volume_profile.items(), key=lambda x: x[1])[0])
            
            # Calculate Value Area (70% of volume around POC)
            sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
            value_area_volume = 0
            target_volume = total_volume * 0.7
            value_area_prices = []
            
            for price_str, vol in sorted_levels:
                value_area_volume += vol
                value_area_prices.append(float(price_str))
                if value_area_volume >= target_volume:
                    break
            
            value_area_high = max(value_area_prices) if value_area_prices else poc_price
            value_area_low = min(value_area_prices) if value_area_prices else poc_price
            
            return {
                'poc_price': poc_price,
                'total_volume': int(total_volume),
                'price_levels': volume_profile,
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'value_area_volume_pct': float(value_area_volume / total_volume) if total_volume > 0 else 0
            }
            
        except Exception as e:
            self.logger.warning(f"Volume profile calculation failed: {str(e)}")
            return {}
    
    def calculate_volume_ratios(self, volume: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate volume ratio indicators
        Research: Volume spikes indicate significant price movements
        
        Args:
            volume: Volume series
            
        Returns:
            Dictionary of volume ratio series
        """
        ratios = {}
        
        # Volume moving averages
        for period in self.volume_sma_periods:
            sma_key = f'volume_sma_{period}'
            ratio_key = f'volume_ratio_{period}'
            
            volume_sma = volume.rolling(window=period).mean()
            volume_ratio = volume / volume_sma
            
            ratios[sma_key] = volume_sma
            ratios[ratio_key] = volume_ratio
        
        # Volume EMA (exponential moving average)
        volume_ema_20 = volume.ewm(span=20).mean()
        ratios['volume_ema_20'] = volume_ema_20
        ratios['volume_ratio_ema_20'] = volume / volume_ema_20
        
        # Unusual volume detection
        primary_ratio = ratios.get('volume_ratio_20', volume / volume.rolling(20).mean())
        ratios['unusual_volume'] = primary_ratio >= self.unusual_volume_threshold
        
        return ratios
    
    def calculate_all_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive volume indicator set
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all volume indicators added
        """
        if len(df) < 20:
            self.logger.warning("Insufficient data for volume indicator calculations")
            return df.copy()
        
        result_df = df.copy()
        
        try:
            # Core volume indicators
            self.logger.info("Calculating OBV...")
            result_df['obv'] = self.calculate_obv(df['close'], df['volume'])
            
            if all(col in df.columns for col in ['high', 'low']):
                self.logger.info("Calculating CMF...")
                result_df['cmf'] = self.calculate_cmf(
                    df['high'], df['low'], df['close'], df['volume']
                )
                
                self.logger.info("Calculating MFI...")
                result_df['mfi'] = self.calculate_mfi(
                    df['high'], df['low'], df['close'], df['volume']
                )
                
                self.logger.info("Calculating VWAP...")
                result_df['vwap'] = self.calculate_vwap(
                    df['high'], df['low'], df['close'], df['volume']
                )
                
                self.logger.info("Calculating A/D Line...")
                result_df['accumulation_distribution'] = self.calculate_accumulation_distribution(
                    df['high'], df['low'], df['close'], df['volume']
                )
            
            self.logger.info("Calculating PVT...")
            result_df['price_volume_trend'] = self.calculate_price_volume_trend(
                df['close'], df['volume']
            )
            
            # Volume ratios and unusual volume detection
            self.logger.info("Calculating volume ratios...")
            volume_ratios = self.calculate_volume_ratios(df['volume'])
            for key, series in volume_ratios.items():
                result_df[key] = series
            
            # Volume profiles (calculated for rolling windows)
            self.logger.info("Calculating volume profiles...")
            volume_profiles = []
            profile_window = 20  # 20-day volume profile
            
            for i in range(len(df)):
                if i >= profile_window - 1:
                    # Get window data
                    window_data = df.iloc[max(0, i - profile_window + 1):i + 1]
                    profile = self.calculate_volume_profile(
                        window_data['close'], window_data['volume']
                    )
                    volume_profiles.append(profile)
                else:
                    volume_profiles.append({})
            
            result_df['volume_profile'] = volume_profiles
            
            self.logger.info(f"Volume indicators calculated successfully for {len(df)} records")
            
        except Exception as e:
            self.logger.error(f"Error calculating volume indicators: {str(e)}")
            # Return original dataframe if calculation fails
            return df.copy()
        
        return result_df
    
    def get_volume_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for volume indicators
        
        Args:
            df: DataFrame with volume indicators
            
        Returns:
            Dictionary of summary statistics
        """
        if df.empty:
            return {}
        
        stats = {}
        
        try:
            volume_cols = [
                'obv', 'cmf', 'mfi', 'vwap', 'accumulation_distribution', 
                'price_volume_trend', 'volume_ratio_20'
            ]
            
            for col in volume_cols:
                if col in df.columns:
                    series = df[col].dropna()
                    if not series.empty:
                        stats[col] = {
                            'mean': float(series.mean()),
                            'std': float(series.std()),
                            'min': float(series.min()),
                            'max': float(series.max()),
                            'current': float(series.iloc[-1]) if len(series) > 0 else None
                        }
            
            # Special statistics
            if 'unusual_volume' in df.columns:
                unusual_days = df['unusual_volume'].sum()
                stats['unusual_volume_days'] = int(unusual_days)
                stats['unusual_volume_pct'] = float(unusual_days / len(df) * 100)
            
            if 'cmf' in df.columns:
                cmf_series = df['cmf'].dropna()
                if not cmf_series.empty:
                    stats['cmf_bullish_signals'] = int((cmf_series > 0.1).sum())
                    stats['cmf_bearish_signals'] = int((cmf_series < -0.1).sum())
            
            if 'mfi' in df.columns:
                mfi_series = df['mfi'].dropna()
                if not mfi_series.empty:
                    stats['mfi_overbought_signals'] = int((mfi_series > 80).sum())
                    stats['mfi_oversold_signals'] = int((mfi_series < 20).sum())
            
        except Exception as e:
            self.logger.error(f"Error calculating volume summary stats: {str(e)}")
        
        return stats

# Example usage and testing
if __name__ == "__main__":
    # Test the volume indicator calculator
    calculator = VolumeIndicatorCalculator()
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=60, freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'trade_date': dates,
        'open': np.random.normal(100, 2, 60),
        'high': np.random.normal(102, 2, 60),
        'low': np.random.normal(98, 2, 60),
        'close': np.random.normal(100, 2, 60),
        'volume': np.random.randint(1000000, 5000000, 60)
    })
    
    # Ensure high >= low and other constraints
    sample_data['high'] = np.maximum(sample_data['high'], sample_data[['open', 'close']].max(axis=1))
    sample_data['low'] = np.minimum(sample_data['low'], sample_data[['open', 'close']].min(axis=1))
    
    try:
        # Calculate volume indicators
        result = calculator.calculate_all_volume_indicators(sample_data)
        
        print("Volume indicators calculated successfully!")
        print(f"Original columns: {len(sample_data.columns)}")
        print(f"Result columns: {len(result.columns)}")
        print(f"New indicators: {set(result.columns) - set(sample_data.columns)}")
        
        # Get summary statistics
        stats = calculator.get_volume_summary_stats(result)
        print(f"\nSummary statistics calculated for {len(stats)} indicators")
        
        # Display sample values
        if 'obv' in result.columns:
            print(f"OBV range: {result['obv'].min():.0f} to {result['obv'].max():.0f}")
        if 'cmf' in result.columns:
            print(f"CMF range: {result['cmf'].min():.3f} to {result['cmf'].max():.3f}")
        if 'mfi' in result.columns:
            print(f"MFI range: {result['mfi'].min():.1f} to {result['mfi'].max():.1f}")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()