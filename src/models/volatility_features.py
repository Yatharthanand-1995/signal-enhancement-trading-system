"""
Advanced Volatility Feature Engineering
Research-backed volatility features for enhanced regime detection and signal generation

Based on academic research:
- Garman-Klass volatility estimator (more efficient than close-to-close)
- Yang-Zhang volatility estimator (most efficient, handles gaps)
- Parkinson volatility estimator (uses high-low range)
- Rogers-Satchell volatility estimator (handles overnight gaps)
- Volatility clustering and regime detection features
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class VolatilityFeatureEngineer:
    """
    Advanced volatility feature engineering based on academic research
    """
    
    def __init__(self):
        self.logger = logger
        
        # Research-backed parameters
        self.short_window = 5
        self.medium_window = 20
        self.long_window = 60
        self.annual_factor = 252  # Trading days per year
        
        # Volatility estimator parameters
        self.estimator_windows = [10, 20, 30, 60]
        self.regime_windows = [60, 120, 252]  # For regime classification
        
        logger.info("Initialized volatility feature engineer with research-backed parameters")
        
    def calculate_realized_volatility(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate realized volatility (standard approach)
        
        Args:
            returns: Daily returns series
            window: Rolling window for calculation
            
        Returns:
            Annualized realized volatility series
        """
        return returns.rolling(window=window).std() * np.sqrt(self.annual_factor)
    
    def calculate_garman_klass_volatility(self, high: pd.Series, low: pd.Series, 
                                        open_: pd.Series, close: pd.Series,
                                        window: int = 20) -> pd.Series:
        """
        Garman-Klass volatility estimator
        Research: More efficient than close-to-close volatility
        Formula: 0.5 * ln(H/L)^2 - (2*ln(2)-1) * ln(C/O)^2
        
        Args:
            high, low, open_, close: OHLC price series
            window: Rolling window for calculation
            
        Returns:
            Annualized Garman-Klass volatility series
        """
        try:
            # Garman-Klass formula
            hl_component = 0.5 * (np.log(high / low)) ** 2
            oc_component = (2 * np.log(2) - 1) * (np.log(close / open_)) ** 2
            
            gk = hl_component - oc_component
            
            # Rolling average and annualization
            gk_vol = np.sqrt(gk.rolling(window=window).mean() * self.annual_factor)
            
            return gk_vol.ffill()
            
        except Exception as e:
            self.logger.warning(f"Error calculating Garman-Klass volatility: {str(e)}")
            return self.calculate_realized_volatility(close.pct_change(), window)
    
    def calculate_parkinson_volatility(self, high: pd.Series, low: pd.Series,
                                     window: int = 20) -> pd.Series:
        """
        Parkinson volatility estimator
        Research: Uses high-low range, more efficient than close-to-close
        Formula: ln(H/L)^2 / (4*ln(2))
        
        Args:
            high, low: High and low price series
            window: Rolling window for calculation
            
        Returns:
            Annualized Parkinson volatility series
        """
        try:
            hl_ratio = np.log(high / low)
            parkinson = (hl_ratio ** 2) / (4 * np.log(2))
            
            # Rolling average and annualization
            park_vol = np.sqrt(parkinson.rolling(window=window).mean() * self.annual_factor)
            
            return park_vol.ffill()
            
        except Exception as e:
            self.logger.warning(f"Error calculating Parkinson volatility: {str(e)}")
            return self.calculate_realized_volatility(high.pct_change(), window)
    
    def calculate_rogers_satchell_volatility(self, high: pd.Series, low: pd.Series,
                                           open_: pd.Series, close: pd.Series,
                                           window: int = 20) -> pd.Series:
        """
        Rogers-Satchell volatility estimator
        Research: Handles overnight gaps better than Garman-Klass
        Formula: ln(H/C) * ln(H/O) + ln(L/C) * ln(L/O)
        
        Args:
            high, low, open_, close: OHLC price series
            window: Rolling window for calculation
            
        Returns:
            Annualized Rogers-Satchell volatility series
        """
        try:
            rs = (np.log(high / close) * np.log(high / open_) + 
                  np.log(low / close) * np.log(low / open_))
            
            # Rolling average and annualization
            rs_vol = np.sqrt(rs.rolling(window=window).mean() * self.annual_factor)
            
            return rs_vol.fillna(method='ffill')
            
        except Exception as e:
            self.logger.warning(f"Error calculating Rogers-Satchell volatility: {str(e)}")
            return self.calculate_realized_volatility(close.pct_change(), window)
    
    def calculate_yang_zhang_volatility(self, high: pd.Series, low: pd.Series,
                                      open_: pd.Series, close: pd.Series,
                                      window: int = 20) -> pd.Series:
        """
        Yang-Zhang volatility estimator
        Research: Most efficient estimator, combines overnight and intraday volatility
        
        Args:
            high, low, open_, close: OHLC price series
            window: Rolling window for calculation
            
        Returns:
            Annualized Yang-Zhang volatility series
        """
        try:
            # Overnight returns (close to open)
            prev_close = close.shift(1)
            overnight = np.log(open_ / prev_close)
            
            # Open-to-close returns  
            oc = np.log(close / open_)
            
            # Rogers-Satchell component
            rs = (np.log(high / close) * np.log(high / open_) + 
                  np.log(low / close) * np.log(low / open_))
            
            # Yang-Zhang combination with research-backed k factor
            k = 0.34 / (1.34 + (window + 1) / (window - 1))
            
            overnight_var = overnight.rolling(window=window).var()
            oc_var = oc.rolling(window=window).var()
            rs_mean = rs.rolling(window=window).mean()
            
            yz = overnight_var + k * oc_var + (1 - k) * rs_mean
            
            # Annualized volatility
            yz_vol = np.sqrt(yz * self.annual_factor)
            
            return yz_vol.fillna(method='ffill')
            
        except Exception as e:
            self.logger.warning(f"Error calculating Yang-Zhang volatility: {str(e)}")
            return self.calculate_realized_volatility(close.pct_change(), window)
    
    def calculate_volatility_regime_indicators(self, volatilities: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Calculate volatility regime indicators for enhanced regime detection
        
        Args:
            volatilities: Dictionary of volatility estimator series
            
        Returns:
            DataFrame with regime indicators
        """
        result = pd.DataFrame()
        
        # Use the most sophisticated estimator as primary, fallback to available
        primary_vol = (volatilities.get('yang_zhang') or 
                      volatilities.get('garman_klass') or 
                      volatilities.get('realized') or 
                      list(volatilities.values())[0])
        
        try:
            # Volatility percentiles (key regime indicators)
            for window in self.regime_windows:
                result[f'vol_percentile_{window}'] = primary_vol.rolling(window).rank(pct=True)
            
            # Volatility of volatility (VoV) - critical for regime detection
            result['vol_of_vol'] = primary_vol.rolling(self.medium_window).std()
            
            # Volatility mean reversion signals
            vol_ma_short = primary_vol.rolling(self.medium_window).mean()
            vol_ma_long = primary_vol.rolling(self.long_window).mean()
            
            result['vol_mean_reversion'] = (primary_vol - vol_ma_long) / vol_ma_long
            result['vol_trend'] = (vol_ma_short - vol_ma_long) / vol_ma_long
            
            # Volatility momentum (research-backed feature)
            result['vol_momentum_5'] = primary_vol.pct_change(5)
            result['vol_momentum_10'] = primary_vol.pct_change(10)
            result['vol_momentum_20'] = primary_vol.pct_change(20)
            
            # Volatility regime classification (research thresholds)
            vol_25th = primary_vol.rolling(self.annual_factor).quantile(0.25)
            vol_75th = primary_vol.rolling(self.annual_factor).quantile(0.75)
            
            result['vol_regime'] = np.where(
                primary_vol < vol_25th, 0,  # Low volatility
                np.where(primary_vol > vol_75th, 2, 1)  # High vs Medium volatility
            )
            
            # Volatility clustering (GARCH-like behavior)
            vol_ma = primary_vol.rolling(self.short_window).mean()
            vol_ma_prev = vol_ma.shift(1)
            
            result['vol_clustering'] = (
                (vol_ma > vol_ma.rolling(self.medium_window).mean()) & 
                (vol_ma_prev > vol_ma_prev.rolling(self.medium_window).mean())
            ).astype(int)
            
            # Volatility breakouts (regime transition indicators)
            vol_threshold = primary_vol.rolling(self.long_window).quantile(0.90)
            result['vol_breakout'] = (primary_vol > vol_threshold).astype(int)
            
            # Volatility efficiency (consistency across estimators)
            if len(volatilities) > 1:
                vol_df = pd.DataFrame(volatilities)
                result['vol_estimator_consistency'] = 1 - (vol_df.std(axis=1) / vol_df.mean(axis=1))
                result['vol_estimator_range'] = vol_df.max(axis=1) - vol_df.min(axis=1)
            
            # Volatility skewness and kurtosis (regime characteristics)
            result['vol_skew'] = primary_vol.rolling(self.long_window).apply(
                lambda x: stats.skew(x.dropna()) if len(x.dropna()) > 10 else 0
            )
            result['vol_kurtosis'] = primary_vol.rolling(self.long_window).apply(
                lambda x: stats.kurtosis(x.dropna()) if len(x.dropna()) > 10 else 0
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility regime indicators: {str(e)}")
        
        return result.ffill().fillna(0)
    
    def calculate_volume_volatility_features(self, volume: pd.Series, 
                                           returns: pd.Series) -> pd.DataFrame:
        """
        Calculate volume-volatility relationship features
        Research: Volume-volatility correlation indicates market efficiency and regime changes
        
        Args:
            volume: Volume series
            returns: Returns series
            
        Returns:
            DataFrame with volume-volatility features
        """
        result = pd.DataFrame()
        
        try:
            # Volume volatility (normalized by mean)
            vol_vol = volume.rolling(self.medium_window).std() / volume.rolling(self.medium_window).mean()
            result['volume_volatility'] = vol_vol
            
            # Volume-price volatility correlations (multiple windows)
            abs_returns = returns.abs()
            
            for window in [self.short_window, self.medium_window, self.long_window]:
                correlation = abs_returns.rolling(window).corr(volume)
                result[f'vol_price_corr_{window}'] = correlation
            
            # Volume-weighted volatility
            vol_weights = volume / volume.rolling(self.medium_window).sum()
            result['volume_weighted_volatility'] = (abs_returns * vol_weights).rolling(self.medium_window).sum()
            
            # Abnormal volume-volatility relationship
            expected_vol = abs_returns.rolling(self.long_window).mean()
            actual_vol = abs_returns
            vol_ratio = volume / volume.rolling(self.long_window).mean()
            
            result['abnormal_vol_vol_ratio'] = (actual_vol / expected_vol) * vol_ratio
            
            # Volume volatility regimes
            vol_vol_25th = vol_vol.rolling(self.annual_factor).quantile(0.25)
            vol_vol_75th = vol_vol.rolling(self.annual_factor).quantile(0.75)
            
            result['volume_vol_regime'] = np.where(
                vol_vol < vol_vol_25th, 0,  # Low volume volatility
                np.where(vol_vol > vol_vol_75th, 2, 1)  # High vs Medium
            )
            
            # Volume efficiency (how well volume predicts price volatility)
            lagged_volume = volume.shift(1)
            vol_efficiency = abs_returns.rolling(self.medium_window).corr(lagged_volume)
            result['volume_efficiency'] = vol_efficiency
            
        except Exception as e:
            self.logger.error(f"Error calculating volume-volatility features: {str(e)}")
        
        return result.ffill().fillna(0)
    
    def calculate_intraday_volatility_features(self, high: pd.Series, low: pd.Series,
                                             open_: pd.Series, close: pd.Series) -> pd.DataFrame:
        """
        Calculate intraday volatility and gap features for regime detection
        
        Args:
            high, low, open_, close: OHLC price series
            
        Returns:
            DataFrame with intraday volatility features
        """
        result = pd.DataFrame()
        
        try:
            # Intraday range volatility
            daily_range = (high - low) / open_  # Normalized by opening price
            result['intraday_range'] = daily_range
            result['intraday_range_ma'] = daily_range.rolling(self.medium_window).mean()
            result['intraday_range_volatility'] = daily_range.rolling(self.medium_window).std()
            
            # Gap analysis (overnight volatility)
            prev_close = close.shift(1)
            gaps = (open_ - prev_close) / prev_close
            
            result['gap_returns'] = gaps
            result['gap_volatility'] = gaps.abs().rolling(self.medium_window).mean()
            result['gap_frequency'] = (gaps.abs() > gaps.abs().rolling(self.long_window).quantile(0.75)).rolling(self.medium_window).mean()
            
            # Intraday vs overnight volatility ratio
            intraday_returns = (close - open_) / open_
            intraday_vol = intraday_returns.abs().rolling(self.medium_window).mean()
            overnight_vol = gaps.abs().rolling(self.medium_window).mean()
            
            result['intraday_overnight_ratio'] = intraday_vol / (overnight_vol + 1e-8)  # Avoid division by zero
            
            # Opening vs closing strength
            result['open_close_ratio'] = open_ / close.shift(1)  # Opening strength
            result['close_strength'] = close / high  # How close to high did it close
            result['open_strength'] = (open_ - low) / (high - low)  # Where did it open in the range
            
            # Regime indicators based on intraday patterns
            range_percentile = daily_range.rolling(self.long_window).rank(pct=True)
            result['range_regime'] = np.where(
                range_percentile < 0.33, 0,  # Low volatility
                np.where(range_percentile > 0.67, 2, 1)  # High vs Medium
            )
            
            # Gap regime indicators
            gap_percentile = gaps.abs().rolling(self.long_window).rank(pct=True)
            result['gap_regime'] = np.where(
                gap_percentile < 0.33, 0,  # Low gap volatility
                np.where(gap_percentile > 0.67, 2, 1)  # High vs Medium
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating intraday volatility features: {str(e)}")
        
        return result.ffill().fillna(0)
    
    def calculate_all_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive volatility feature set
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all volatility features added
        """
        if len(df) < max(self.estimator_windows):
            self.logger.warning(f"Insufficient data for volatility features: {len(df)} < {max(self.estimator_windows)}")
            return df.copy()
        
        result_df = df.copy()
        
        try:
            self.logger.info("Calculating comprehensive volatility features...")
            
            # Multiple volatility estimators
            volatilities = {}
            
            # Realized volatility (baseline)
            returns = df['close'].pct_change()
            volatilities['realized'] = self.calculate_realized_volatility(returns)
            result_df['realized_volatility'] = volatilities['realized']
            
            # Advanced volatility estimators (if OHLC data available)
            if all(col in df.columns for col in ['high', 'low', 'open']):
                self.logger.info("Calculating advanced volatility estimators...")
                
                # Garman-Klass volatility
                volatilities['garman_klass'] = self.calculate_garman_klass_volatility(
                    df['high'], df['low'], df['open'], df['close']
                )
                result_df['garman_klass_volatility'] = volatilities['garman_klass']
                
                # Parkinson volatility
                volatilities['parkinson'] = self.calculate_parkinson_volatility(
                    df['high'], df['low']
                )
                result_df['parkinson_volatility'] = volatilities['parkinson']
                
                # Rogers-Satchell volatility
                volatilities['rogers_satchell'] = self.calculate_rogers_satchell_volatility(
                    df['high'], df['low'], df['open'], df['close']
                )
                result_df['rogers_satchell_volatility'] = volatilities['rogers_satchell']
                
                # Yang-Zhang volatility (most sophisticated)
                volatilities['yang_zhang'] = self.calculate_yang_zhang_volatility(
                    df['high'], df['low'], df['open'], df['close']
                )
                result_df['yang_zhang_volatility'] = volatilities['yang_zhang']
                
                # Intraday volatility features
                self.logger.info("Calculating intraday volatility features...")
                intraday_features = self.calculate_intraday_volatility_features(
                    df['high'], df['low'], df['open'], df['close']
                )
                for col in intraday_features.columns:
                    result_df[col] = intraday_features[col]
            
            # Volatility regime indicators
            self.logger.info("Calculating volatility regime indicators...")
            regime_features = self.calculate_volatility_regime_indicators(volatilities)
            for col in regime_features.columns:
                result_df[col] = regime_features[col]
            
            # Volume-volatility features (if volume data available)
            if 'volume' in df.columns:
                self.logger.info("Calculating volume-volatility features...")
                vol_vol_features = self.calculate_volume_volatility_features(
                    df['volume'], returns
                )
                for col in vol_vol_features.columns:
                    result_df[col] = vol_vol_features[col]
            
            # Cross-volatility features (if multiple estimators available)
            if len(volatilities) > 1:
                self.logger.info("Calculating cross-volatility features...")
                
                # Volatility spread (difference between estimators)
                if 'yang_zhang' in volatilities and 'realized' in volatilities:
                    result_df['vol_spread_yz_realized'] = (
                        volatilities['yang_zhang'] - volatilities['realized']
                    )
                
                # Volatility consistency across estimators
                vol_values = pd.DataFrame(volatilities)
                result_df['vol_estimator_std'] = vol_values.std(axis=1)
                result_df['vol_estimator_mean'] = vol_values.mean(axis=1)
                result_df['vol_estimator_cv'] = result_df['vol_estimator_std'] / result_df['vol_estimator_mean']
            
            feature_count = len(result_df.columns) - len(df.columns)
            self.logger.info(f"✅ Added {feature_count} volatility features successfully")
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility features: {str(e)}")
            # Return original dataframe if calculation fails
            return df.copy()
        
        return result_df
    
    def get_volatility_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for volatility features
        
        Args:
            df: DataFrame with volatility features
            
        Returns:
            Dictionary of summary statistics
        """
        if df.empty:
            return {}
        
        stats_dict = {}
        
        try:
            # Core volatility measures
            vol_cols = [col for col in df.columns if 'volatility' in col.lower()]
            regime_cols = [col for col in df.columns if 'regime' in col.lower()]
            momentum_cols = [col for col in df.columns if 'momentum' in col.lower()]
            
            for col_group, cols in [('volatility', vol_cols), ('regime', regime_cols), ('momentum', momentum_cols)]:
                if cols:
                    group_stats = {}
                    for col in cols:
                        if col in df.columns:
                            series = df[col].dropna()
                            if not series.empty:
                                group_stats[col] = {
                                    'mean': float(series.mean()),
                                    'std': float(series.std()),
                                    'min': float(series.min()),
                                    'max': float(series.max()),
                                    'current': float(series.iloc[-1]) if len(series) > 0 else None
                                }
                    
                    if group_stats:
                        stats_dict[f'{col_group}_features'] = group_stats
            
            # Special regime statistics
            if 'vol_regime' in df.columns:
                regime_counts = df['vol_regime'].value_counts().to_dict()
                stats_dict['vol_regime_distribution'] = regime_counts
            
            # Volatility clustering statistics
            if 'vol_clustering' in df.columns:
                clustering_periods = df['vol_clustering'].sum()
                stats_dict['vol_clustering_periods'] = int(clustering_periods)
                stats_dict['vol_clustering_pct'] = float(clustering_periods / len(df) * 100)
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility summary stats: {str(e)}")
        
        return stats_dict

# Example usage and testing
if __name__ == "__main__":
    # Test volatility feature engineer
    engineer = VolatilityFeatureEngineer()
    
    # Create sample OHLCV data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Create realistic price data with volatility clustering
    returns = []
    volatility_regime = 0.02  # Base volatility
    
    for i in range(100):
        # Create volatility clustering
        if i > 30 and i < 60:  # High volatility period
            volatility_regime = 0.05
        elif i > 70:  # Low volatility period  
            volatility_regime = 0.01
        else:
            volatility_regime = 0.02
        
        daily_return = np.random.normal(0.001, volatility_regime)
        returns.append(daily_return)
    
    # Convert to price series
    prices = [100]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = prices[1:]  # Remove initial price
    
    # Create OHLCV data
    sample_data = pd.DataFrame({
        'trade_date': dates,
        'open': [p * np.random.uniform(0.999, 1.001) for p in prices],
        'high': [p * np.random.uniform(1.000, 1.020) for p in prices],
        'low': [p * np.random.uniform(0.980, 1.000) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    
    # Ensure OHLC constraints
    sample_data['high'] = np.maximum(sample_data['high'], sample_data[['open', 'close']].max(axis=1))
    sample_data['low'] = np.minimum(sample_data['low'], sample_data[['open', 'close']].min(axis=1))
    
    try:
        print("Testing Volatility Feature Engineer")
        print("=" * 40)
        
        # Calculate all volatility features
        result = engineer.calculate_all_volatility_features(sample_data)
        
        print(f"✅ Volatility features calculated successfully")
        print(f"Original columns: {len(sample_data.columns)}")
        print(f"Enhanced columns: {len(result.columns)}")
        print(f"New features: {len(result.columns) - len(sample_data.columns)}")
        
        # Get summary statistics
        stats = engineer.get_volatility_summary_stats(result)
        
        print(f"\n📊 Feature Summary:")
        for category, features in stats.items():
            if isinstance(features, dict) and any('mean' in str(v) for v in features.values()):
                print(f"  {category}: {len(features)} features")
                # Show a sample feature
                sample_feature = list(features.keys())[0]
                sample_stats = features[sample_feature]
                if isinstance(sample_stats, dict) and 'current' in sample_stats:
                    print(f"    {sample_feature}: current={sample_stats['current']:.4f}")
        
        # Test specific volatility estimators
        if 'yang_zhang_volatility' in result.columns:
            yz_vol = result['yang_zhang_volatility'].dropna()
            print(f"\n📈 Yang-Zhang Volatility: {yz_vol.iloc[-1]:.4f} (annualized)")
        
        if 'vol_regime' in result.columns:
            regime_counts = result['vol_regime'].value_counts()
            print(f"\n🎯 Volatility Regimes: {regime_counts.to_dict()}")
        
        print(f"\n✅ Volatility feature engineering test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()