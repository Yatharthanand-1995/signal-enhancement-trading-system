#!/usr/bin/env python3
"""
Validate Real Market Features - Phase 1, Day 2-3
Test which technical indicators actually correlate with forward returns
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class RealFeatureValidator:
    """Validate technical features on real market data"""
    
    def __init__(self):
        self.data_dir = 'data/full_market'
        self.results_dir = 'results/feature_validation'
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_real_data(self):
        """Load real market data"""
        
        print("📊 LOADING REAL MARKET DATA")
        print("=" * 40)
        
        train_path = os.path.join(self.data_dir, 'train_data.csv')
        if not os.path.exists(train_path):
            raise FileNotFoundError("Run real_data_pipeline.py first")
        
        train_data = pd.read_csv(train_path)
        train_data['date'] = pd.to_datetime(train_data['date'])
        
        print(f"Training data loaded: {len(train_data):,} records")
        print(f"Date range: {train_data['date'].min().date()} to {train_data['date'].max().date()}")
        print(f"Symbols: {train_data['symbol'].nunique()} ({', '.join(train_data['symbol'].unique()[:5])}...)")
        print(f"Features: {len(train_data.columns)} columns")
        
        return train_data
    
    def calculate_forward_returns(self, data, periods=[1, 2, 5, 10, 20]):
        """Calculate forward returns for different holding periods"""
        
        print(f"\n🔄 CALCULATING FORWARD RETURNS")
        print("-" * 35)
        print(f"Periods: {periods} days")
        
        enhanced_data = []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy().sort_values('date')
            
            # Calculate forward returns for each period
            for period in periods:
                col_name = f'forward_return_{period}d'
                symbol_data[col_name] = symbol_data['close'].shift(-period) / symbol_data['close'] - 1
            
            # Calculate forward volatility (useful for risk prediction)
            symbol_data['forward_volatility_5d'] = symbol_data['price_change'].rolling(5).std().shift(-5)
            
            enhanced_data.append(symbol_data)
        
        result = pd.concat(enhanced_data, ignore_index=True)
        
        # Remove rows without forward returns (end of dataset)
        max_period = max(periods)
        result = result.dropna(subset=[f'forward_return_{max_period}d'])
        
        print(f"Enhanced dataset: {len(result):,} records with forward returns")
        
        return result
    
    def analyze_feature_correlations(self, data):
        """Analyze correlations between features and forward returns"""
        
        print(f"\n🔍 FEATURE CORRELATION ANALYSIS")
        print("=" * 45)
        
        # Select features for analysis (exclude non-numeric and target columns)
        feature_columns = [
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_lower', 'bb_position', 'atr_14',
            'volume_ratio', 'price_change', 'price_change_5d', 'price_change_20d'
        ]
        
        # Create relative features (more predictive than absolute values)
        data = self._create_relative_features(data)
        
        relative_features = [
            'price_vs_sma5', 'price_vs_sma10', 'price_vs_sma20', 'price_vs_sma50',
            'sma5_vs_sma20', 'sma10_vs_sma20', 'rsi_14', 'bb_position',
            'volume_ratio', 'price_change', 'price_change_5d', 'price_change_20d',
            'macd_normalized', 'atr_normalized'
        ]
        
        target_columns = ['forward_return_1d', 'forward_return_5d', 'forward_return_20d']
        
        print("Testing correlations with forward returns...")
        
        correlation_results = {}
        
        for target in target_columns:
            print(f"\n📈 {target.upper()} CORRELATIONS:")
            print("-" * 30)
            
            correlations = []
            
            for feature in relative_features:
                if feature in data.columns:
                    # Remove outliers for cleaner correlation
                    clean_data = self._remove_outliers(data[[feature, target]]).dropna()
                    
                    if len(clean_data) > 1000:  # Minimum sample size
                        corr = clean_data[feature].corr(clean_data[target])
                        correlations.append({
                            'feature': feature,
                            'correlation': corr,
                            'abs_correlation': abs(corr),
                            'sample_size': len(clean_data)
                        })
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
            
            correlation_results[target] = correlations
            
            # Display top correlations
            for corr_info in correlations[:10]:
                direction = "📈" if corr_info['correlation'] > 0 else "📉" 
                strength = "STRONG" if corr_info['abs_correlation'] > 0.1 else "MODERATE" if corr_info['abs_correlation'] > 0.05 else "WEAK"
                
                print(f"{corr_info['feature']:<20}: {corr_info['correlation']:+.4f} {direction} {strength}")
        
        return correlation_results
    
    def _create_relative_features(self, data):
        """Create relative features that are more predictive"""
        
        data = data.copy()
        
        # Price relative to moving averages (momentum indicators)
        data['price_vs_sma5'] = (data['close'] - data['sma_5']) / data['sma_5']
        data['price_vs_sma10'] = (data['close'] - data['sma_10']) / data['sma_10']
        data['price_vs_sma20'] = (data['close'] - data['sma_20']) / data['sma_20']
        data['price_vs_sma50'] = (data['close'] - data['sma_50']) / data['sma_50']
        
        # SMA crossovers (trend indicators)
        data['sma5_vs_sma20'] = (data['sma_5'] - data['sma_20']) / data['sma_20']
        data['sma10_vs_sma20'] = (data['sma_10'] - data['sma_20']) / data['sma_20']
        
        # Normalized MACD
        data['macd_normalized'] = data['macd'] / data['close']
        
        # Normalized ATR (volatility)
        data['atr_normalized'] = data['atr_14'] / data['close']
        
        return data
    
    def _remove_outliers(self, data, method='iqr'):
        """Remove outliers to get cleaner correlations"""
        
        if method == 'iqr':
            for col in data.select_dtypes(include=[np.number]).columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
        return data
    
    def identify_promising_features(self, correlation_results, min_correlation=0.05):
        """Identify features with consistent predictive power"""
        
        print(f"\n🎯 PROMISING FEATURE IDENTIFICATION")
        print("=" * 45)
        print(f"Minimum correlation threshold: {min_correlation}")
        
        # Score features across all time horizons
        feature_scores = {}
        
        for feature in correlation_results['forward_return_1d'][0]['feature']:
            scores = []
            for target, correlations in correlation_results.items():
                for corr_info in correlations:
                    if corr_info['feature'] == feature:
                        scores.append(corr_info['abs_correlation'])
                        break
            
            if scores:
                feature_scores[feature] = {
                    'avg_correlation': np.mean(scores),
                    'max_correlation': max(scores),
                    'consistency': np.std(scores),  # Lower is more consistent
                    'predictive_periods': len([s for s in scores if s > min_correlation])
                }
        
        # Rank features
        promising_features = []
        
        for feature, metrics in feature_scores.items():
            if metrics['avg_correlation'] > min_correlation:
                promising_features.append({
                    'feature': feature,
                    'avg_correlation': metrics['avg_correlation'],
                    'max_correlation': metrics['max_correlation'],
                    'consistency_score': 1 / (1 + metrics['consistency']),  # Higher is better
                    'predictive_periods': metrics['predictive_periods']
                })
        
        # Sort by average correlation
        promising_features.sort(key=lambda x: x['avg_correlation'], reverse=True)
        
        print(f"\n🌟 TOP PROMISING FEATURES:")
        print("-" * 35)
        
        for i, feature_info in enumerate(promising_features[:15], 1):
            consistency = "HIGH" if feature_info['consistency_score'] > 0.7 else "MED" if feature_info['consistency_score'] > 0.5 else "LOW"
            
            print(f"{i:2d}. {feature_info['feature']:<20}: "
                  f"Avg {feature_info['avg_correlation']:.3f}, "
                  f"Max {feature_info['max_correlation']:.3f}, "
                  f"Consistent: {consistency}")
        
        return promising_features
    
    def test_regime_dependence(self, data, promising_features):
        """Test if features work differently in different market regimes"""
        
        print(f"\n📊 REGIME-DEPENDENT ANALYSIS")
        print("=" * 40)
        
        # Define market regimes based on SPY returns
        spy_data = data[data['symbol'] == 'SPY'].copy() if 'SPY' in data['symbol'].values else data.copy()
        
        # Volatility regimes
        vol_threshold_low = spy_data['price_change'].rolling(20).std().quantile(0.33)
        vol_threshold_high = spy_data['price_change'].rolling(20).std().quantile(0.67)
        
        spy_data['volatility_regime'] = 'medium'
        spy_data.loc[spy_data['price_change'].rolling(20).std() <= vol_threshold_low, 'volatility_regime'] = 'low'
        spy_data.loc[spy_data['price_change'].rolling(20).std() >= vol_threshold_high, 'volatility_regime'] = 'high'
        
        # Trend regimes
        spy_data['trend_regime'] = 'sideways'
        spy_data.loc[spy_data['sma5_vs_sma20'] > 0.02, 'trend_regime'] = 'bull'
        spy_data.loc[spy_data['sma5_vs_sma20'] < -0.02, 'trend_regime'] = 'bear'
        
        regime_results = {}
        
        top_features = [f['feature'] for f in promising_features[:5]]
        
        for feature in top_features:
            if feature in data.columns:
                feature_data = data[[feature, 'forward_return_5d', 'symbol']].dropna()
                
                # Add regime info (use SPY as market proxy)
                feature_data = feature_data.merge(
                    spy_data[['date', 'volatility_regime', 'trend_regime']], 
                    left_index=True, right_index=True, how='left'
                )
                
                regime_correlations = {}
                
                for regime_type in ['volatility_regime', 'trend_regime']:
                    regime_correlations[regime_type] = {}
                    
                    for regime in feature_data[regime_type].unique():
                        if pd.notna(regime):
                            regime_data = feature_data[feature_data[regime_type] == regime]
                            if len(regime_data) > 100:  # Minimum sample
                                corr = regime_data[feature].corr(regime_data['forward_return_5d'])
                                regime_correlations[regime_type][regime] = {
                                    'correlation': corr,
                                    'sample_size': len(regime_data)
                                }
                
                regime_results[feature] = regime_correlations
        
        # Display regime analysis
        for feature, regimes in regime_results.items():
            print(f"\n📈 {feature.upper()}:")
            
            if 'volatility_regime' in regimes:
                print("  Volatility Regimes:")
                for regime, stats in regimes['volatility_regime'].items():
                    print(f"    {regime.capitalize():<8}: {stats['correlation']:+.3f} (n={stats['sample_size']})")
            
            if 'trend_regime' in regimes:
                print("  Trend Regimes:")
                for regime, stats in regimes['trend_regime'].items():
                    print(f"    {regime.capitalize():<8}: {stats['correlation']:+.3f} (n={stats['sample_size']})")
        
        return regime_results
    
    def create_feature_recommendation(self, promising_features, regime_results):
        """Create final feature recommendations"""
        
        print(f"\n🎯 FEATURE RECOMMENDATIONS")
        print("=" * 50)
        
        # Categorize features by type and strength
        momentum_features = []
        trend_features = []
        volatility_features = []
        volume_features = []
        
        for feature_info in promising_features[:10]:
            feature = feature_info['feature']
            
            if 'price_vs_sma' in feature or 'price_change' in feature:
                momentum_features.append(feature_info)
            elif 'sma' in feature and 'vs' in feature:
                trend_features.append(feature_info)
            elif 'atr' in feature or 'bb_position' in feature:
                volatility_features.append(feature_info)
            elif 'volume' in feature:
                volume_features.append(feature_info)
        
        print("📈 MOMENTUM FEATURES (Price vs Moving Averages):")
        for feature in momentum_features[:3]:
            print(f"  ✅ {feature['feature']} (correlation: {feature['avg_correlation']:.3f})")
        
        print(f"\n📊 TREND FEATURES (Moving Average Relationships):")
        for feature in trend_features[:3]:
            print(f"  ✅ {feature['feature']} (correlation: {feature['avg_correlation']:.3f})")
        
        print(f"\n🌊 VOLATILITY FEATURES (Risk Indicators):")
        for feature in volatility_features[:2]:
            print(f"  ✅ {feature['feature']} (correlation: {feature['avg_correlation']:.3f})")
        
        print(f"\n🔊 VOLUME FEATURES (Market Participation):")
        for feature in volume_features[:2]:
            print(f"  ✅ {feature['feature']} (correlation: {feature['avg_correlation']:.3f})")
        
        # Final recommendation
        top_10_features = [f['feature'] for f in promising_features[:10]]
        
        print(f"\n🚀 RECOMMENDED FEATURE SET:")
        print("=" * 30)
        print("Use these features for ML model development:")
        
        for i, feature in enumerate(top_10_features, 1):
            corr = next(f['avg_correlation'] for f in promising_features if f['feature'] == feature)
            print(f"{i:2d}. {feature} ({corr:.3f})")
        
        return {
            'top_features': top_10_features,
            'momentum_features': [f['feature'] for f in momentum_features],
            'trend_features': [f['feature'] for f in trend_features],
            'volatility_features': [f['feature'] for f in volatility_features],
            'volume_features': [f['feature'] for f in volume_features]
        }

def main():
    """Main execution function"""
    
    print("🔍 REAL MARKET FEATURE VALIDATION - PHASE 1")
    print("=" * 60)
    print("Goal: Validate technical features on real market data")
    print("Approach: Correlation analysis with forward returns")
    print()
    
    try:
        validator = RealFeatureValidator()
        
        # Step 1: Load real market data
        data = validator.load_real_data()
        
        # Step 2: Calculate forward returns
        enhanced_data = validator.calculate_forward_returns(data)
        
        # Step 3: Analyze feature correlations
        correlation_results = validator.analyze_feature_correlations(enhanced_data)
        
        # Step 4: Identify promising features
        promising_features = validator.identify_promising_features(correlation_results)
        
        # Step 5: Test regime dependence
        regime_results = validator.test_regime_dependence(enhanced_data, promising_features)
        
        # Step 6: Create recommendations
        recommendations = validator.create_feature_recommendation(promising_features, regime_results)
        
        print(f"\n🎉 FEATURE VALIDATION COMPLETE")
        print("=" * 40)
        
        # Success criteria
        strong_features = len([f for f in promising_features if f['avg_correlation'] > 0.1])
        moderate_features = len([f for f in promising_features if f['avg_correlation'] > 0.05])
        
        print(f"Strong features (>10% correlation): {strong_features}")
        print(f"Moderate features (>5% correlation): {moderate_features}")
        print(f"Total promising features: {len(promising_features)}")
        
        if strong_features >= 2 and moderate_features >= 5:
            print(f"\n✅ FEATURE VALIDATION SUCCESS!")
            print("Sufficient predictive features identified on real data")
            print("Ready to proceed to regime detection development")
            return True
        else:
            print(f"\n⚠️ LIMITED PREDICTIVE POWER")
            print("Features show weak correlation - need alternative approach")
            return False
        
    except Exception as e:
        print(f"\n❌ FEATURE VALIDATION FAILED")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🚀 NEXT STEP: Regime detection development")
        print("Features validated - proceed to Phase 1 Day 4-5")
    else:
        print(f"\n🔧 REQUIRED: Investigate alternative feature engineering approaches")