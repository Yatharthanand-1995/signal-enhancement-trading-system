"""
Data Processing Utilities
Centralized data processing functions for dashboard components
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Centralized data processing utilities for dashboard
    Handles data transformation, filtering, and aggregation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.cache_enabled = self.config.get('cache_enabled', True)
        
    @lru_cache(maxsize=128)
    def process_signals_data(self, data_hash: str, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process signals data for dashboard display
        Cached for performance
        """
        try:
            processed_df = signals_df.copy()
            
            # Ensure required columns exist
            required_columns = [
                'symbol', 'signal_direction', 'signal_strength', 'confidence', 'current_price'
            ]
            
            missing_columns = set(required_columns) - set(processed_df.columns)
            if missing_columns:
                logger.warning(f"Missing columns in signals data: {missing_columns}")
                
                # Add default values for missing columns
                for col in missing_columns:
                    if col == 'signal_strength':
                        processed_df[col] = 0.5
                    elif col == 'confidence':
                        processed_df[col] = 0.5
                    elif col == 'current_price':
                        processed_df[col] = 0.0
                    else:
                        processed_df[col] = 'N/A'
            
            # Clean and standardize data
            processed_df = self._clean_signals_data(processed_df)
            
            # Add derived columns
            processed_df = self._add_derived_columns(processed_df)
            
            # Sort by signal strength and confidence
            processed_df = processed_df.sort_values(
                ['signal_strength', 'confidence'], 
                ascending=[False, False]
            )
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Error processing signals data: {str(e)}")
            return signals_df
    
    def _clean_signals_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize signals data"""
        cleaned_df = df.copy()
        
        # Standardize signal direction values
        signal_mapping = {
            'STRONG BUY': 'STRONG_BUY',
            'Strong Buy': 'STRONG_BUY',
            'BUY': 'BUY',
            'Buy': 'BUY',
            'NEUTRAL': 'NEUTRAL',
            'Neutral': 'NEUTRAL',
            'Hold': 'NEUTRAL',
            'SELL': 'SELL',
            'Sell': 'SELL',
            'STRONG SELL': 'STRONG_SELL',
            'Strong Sell': 'STRONG_SELL'
        }
        
        if 'signal_direction' in cleaned_df.columns:
            cleaned_df['signal_direction'] = cleaned_df['signal_direction'].map(
                lambda x: signal_mapping.get(str(x), str(x))
            )
        
        # Clean numeric columns
        numeric_columns = ['signal_strength', 'confidence', 'current_price', 'rsi_14', 'volume']
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        
        # Remove rows with invalid signal directions
        valid_signals = ['STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG_SELL']
        if 'signal_direction' in cleaned_df.columns:
            cleaned_df = cleaned_df[cleaned_df['signal_direction'].isin(valid_signals)]
        
        return cleaned_df
    
    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns for enhanced analysis"""
        enhanced_df = df.copy()
        
        # Signal score (combination of strength and confidence)
        if 'signal_strength' in enhanced_df.columns and 'confidence' in enhanced_df.columns:
            enhanced_df['signal_score'] = (
                enhanced_df['signal_strength'] * enhanced_df['confidence']
            ).round(3)
        
        # Risk level categorization
        if 'signal_strength' in enhanced_df.columns:
            enhanced_df['risk_level'] = enhanced_df['signal_strength'].apply(self._categorize_risk_level)
        
        # Price change percentage
        if 'current_price' in enhanced_df.columns and 'previous_close' in enhanced_df.columns:
            enhanced_df['price_change_pct'] = (
                (enhanced_df['current_price'] - enhanced_df['previous_close']) / 
                enhanced_df['previous_close'] * 100
            ).round(2)
        
        # Volume category
        if 'volume' in enhanced_df.columns:
            enhanced_df['volume_category'] = enhanced_df['volume'].apply(self._categorize_volume)
        
        # RSI category
        if 'rsi_14' in enhanced_df.columns:
            enhanced_df['rsi_category'] = enhanced_df['rsi_14'].apply(self._categorize_rsi)
        
        return enhanced_df
    
    @staticmethod
    def _categorize_risk_level(strength: float) -> str:
        """Categorize risk level based on signal strength"""
        if pd.isna(strength):
            return 'Unknown'
        elif strength >= 0.8:
            return 'High'
        elif strength >= 0.6:
            return 'Medium'
        elif strength >= 0.4:
            return 'Low'
        else:
            return 'Very Low'
    
    @staticmethod
    def _categorize_volume(volume: float) -> str:
        """Categorize volume levels"""
        if pd.isna(volume):
            return 'Unknown'
        elif volume >= 10_000_000:
            return 'Very High'
        elif volume >= 5_000_000:
            return 'High'
        elif volume >= 1_000_000:
            return 'Medium'
        else:
            return 'Low'
    
    @staticmethod
    def _categorize_rsi(rsi: float) -> str:
        """Categorize RSI levels"""
        if pd.isna(rsi):
            return 'Unknown'
        elif rsi >= 80:
            return 'Extremely Overbought'
        elif rsi >= 70:
            return 'Overbought'
        elif rsi >= 30:
            return 'Neutral'
        elif rsi >= 20:
            return 'Oversold'
        else:
            return 'Extremely Oversold'
    
    def aggregate_signals_by_sector(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate signals by sector for sector analysis"""
        if 'sector' not in signals_df.columns:
            logger.warning("Sector column not found in signals data")
            return pd.DataFrame()
        
        try:
            sector_agg = signals_df.groupby('sector').agg({
                'signal_strength': ['mean', 'std', 'count'],
                'confidence': 'mean',
                'current_price': 'mean',
                'signal_direction': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'NEUTRAL'
            }).round(3)
            
            # Flatten column names
            sector_agg.columns = ['_'.join(col).strip() for col in sector_agg.columns]
            
            # Rename columns for clarity
            sector_agg = sector_agg.rename(columns={
                'signal_strength_mean': 'avg_signal_strength',
                'signal_strength_std': 'signal_volatility',
                'signal_strength_count': 'stock_count',
                'confidence_mean': 'avg_confidence',
                'current_price_mean': 'avg_price',
                'signal_direction_<lambda>': 'dominant_signal'
            })
            
            return sector_agg.reset_index()
            
        except Exception as e:
            logger.error(f"Error aggregating signals by sector: {str(e)}")
            return pd.DataFrame()
    
    def calculate_portfolio_metrics(self, signals_df: pd.DataFrame, 
                                   weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Calculate portfolio-level metrics"""
        try:
            metrics = {}
            
            # Basic counts
            total_stocks = len(signals_df)
            metrics['total_stocks'] = total_stocks
            
            # Signal distribution
            signal_counts = signals_df['signal_direction'].value_counts()
            metrics['signal_distribution'] = signal_counts.to_dict()
            
            # Calculate percentages
            for signal, count in signal_counts.items():
                metrics[f'{signal.lower()}_pct'] = (count / total_stocks * 100) if total_stocks > 0 else 0
            
            # Average metrics
            numeric_columns = ['signal_strength', 'confidence', 'rsi_14']
            for col in numeric_columns:
                if col in signals_df.columns:
                    metrics[f'avg_{col}'] = signals_df[col].mean()
                    metrics[f'std_{col}'] = signals_df[col].std()
            
            # Portfolio-weighted metrics if weights provided
            if weights:
                weighted_metrics = self._calculate_weighted_metrics(signals_df, weights)
                metrics.update(weighted_metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {}
    
    def _calculate_weighted_metrics(self, signals_df: pd.DataFrame, 
                                   weights: Dict[str, float]) -> Dict[str, Any]:
        """Calculate portfolio metrics weighted by position sizes"""
        try:
            weighted_metrics = {}
            
            # Create weights series aligned with signals dataframe
            signals_df['weight'] = signals_df['symbol'].map(weights).fillna(0)
            total_weight = signals_df['weight'].sum()
            
            if total_weight > 0:
                # Normalize weights
                signals_df['normalized_weight'] = signals_df['weight'] / total_weight
                
                # Weighted averages
                weighted_metrics['weighted_avg_signal_strength'] = (
                    signals_df['signal_strength'] * signals_df['normalized_weight']
                ).sum()
                
                weighted_metrics['weighted_avg_confidence'] = (
                    signals_df['confidence'] * signals_df['normalized_weight']
                ).sum()
                
                if 'rsi_14' in signals_df.columns:
                    weighted_metrics['weighted_avg_rsi'] = (
                        signals_df['rsi_14'] * signals_df['normalized_weight']
                    ).sum()
            
            return weighted_metrics
            
        except Exception as e:
            logger.error(f"Error calculating weighted metrics: {str(e)}")
            return {}
    
    def filter_signals(self, signals_df: pd.DataFrame, 
                      filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply multiple filters to signals data"""
        try:
            filtered_df = signals_df.copy()
            
            # Signal direction filter
            if 'signal_direction' in filters:
                signal_filter = filters['signal_direction']
                if signal_filter != 'All':
                    if isinstance(signal_filter, list):
                        filtered_df = filtered_df[filtered_df['signal_direction'].isin(signal_filter)]
                    else:
                        filtered_df = filtered_df[filtered_df['signal_direction'] == signal_filter]
            
            # Strength range filter
            if 'min_strength' in filters:
                min_strength = filters['min_strength']
                filtered_df = filtered_df[filtered_df['signal_strength'] >= min_strength]
            
            if 'max_strength' in filters:
                max_strength = filters['max_strength']
                filtered_df = filtered_df[filtered_df['signal_strength'] <= max_strength]
            
            # Confidence filter
            if 'min_confidence' in filters:
                min_confidence = filters['min_confidence']
                filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
            
            # Price range filter
            if 'price_range' in filters:
                price_range = filters['price_range']
                if len(price_range) == 2:
                    min_price, max_price = price_range
                    if min_price is not None:
                        filtered_df = filtered_df[filtered_df['current_price'] >= min_price]
                    if max_price is not None:
                        filtered_df = filtered_df[filtered_df['current_price'] <= max_price]
            
            # Sector filter
            if 'sector' in filters and 'sector' in filtered_df.columns:
                sector_filter = filters['sector']
                if sector_filter != 'All':
                    if isinstance(sector_filter, list):
                        filtered_df = filtered_df[filtered_df['sector'].isin(sector_filter)]
                    else:
                        filtered_df = filtered_df[filtered_df['sector'] == sector_filter]
            
            # RSI filter
            if 'rsi_range' in filters and 'rsi_14' in filtered_df.columns:
                rsi_range = filters['rsi_range']
                if len(rsi_range) == 2:
                    min_rsi, max_rsi = rsi_range
                    if min_rsi is not None:
                        filtered_df = filtered_df[filtered_df['rsi_14'] >= min_rsi]
                    if max_rsi is not None:
                        filtered_df = filtered_df[filtered_df['rsi_14'] <= max_rsi]
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error filtering signals: {str(e)}")
            return signals_df
    
    def prepare_chart_data(self, signals_df: pd.DataFrame, chart_type: str) -> Dict[str, Any]:
        """Prepare data specifically formatted for different chart types"""
        try:
            chart_data = {}
            
            if chart_type == 'heatmap':
                # Prepare data for signals heatmap
                if not signals_df.empty:
                    chart_data['heatmap_data'] = signals_df.pivot_table(
                        values='signal_strength',
                        index='symbol', 
                        columns=signals_df.get('date', 'current'),
                        fill_value=0
                    )
            
            elif chart_type == 'distribution':
                # Prepare data for signal distribution charts
                chart_data['signal_counts'] = signals_df['signal_direction'].value_counts()
                chart_data['strength_distribution'] = signals_df['signal_strength'].values
                chart_data['confidence_distribution'] = signals_df['confidence'].values
            
            elif chart_type == 'sector_analysis':
                # Prepare sector-level aggregated data
                chart_data['sector_data'] = self.aggregate_signals_by_sector(signals_df)
            
            elif chart_type == 'correlation':
                # Prepare correlation matrix data
                numeric_columns = ['signal_strength', 'confidence', 'rsi_14', 'current_price']
                available_columns = [col for col in numeric_columns if col in signals_df.columns]
                if available_columns:
                    chart_data['correlation_matrix'] = signals_df[available_columns].corr()
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error preparing chart data for {chart_type}: {str(e)}")
            return {}
    
    def validate_data_quality(self, signals_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and return quality metrics"""
        try:
            quality_metrics = {
                'total_rows': len(signals_df),
                'total_columns': len(signals_df.columns),
                'missing_data': {},
                'data_types': {},
                'quality_score': 100,
                'issues': []
            }
            
            # Check for missing data
            for column in signals_df.columns:
                missing_count = signals_df[column].isna().sum()
                missing_pct = (missing_count / len(signals_df)) * 100
                quality_metrics['missing_data'][column] = {
                    'count': missing_count,
                    'percentage': round(missing_pct, 2)
                }
                
                if missing_pct > 50:
                    quality_metrics['quality_score'] -= 10
                    quality_metrics['issues'].append(f"High missing data in {column}: {missing_pct:.1f}%")
            
            # Check data types
            for column in signals_df.columns:
                quality_metrics['data_types'][column] = str(signals_df[column].dtype)
            
            # Check for data consistency
            if 'signal_strength' in signals_df.columns:
                invalid_strength = ((signals_df['signal_strength'] < 0) | 
                                  (signals_df['signal_strength'] > 1)).sum()
                if invalid_strength > 0:
                    quality_metrics['quality_score'] -= 15
                    quality_metrics['issues'].append(f"Invalid signal strength values: {invalid_strength}")
            
            if 'confidence' in signals_df.columns:
                invalid_confidence = ((signals_df['confidence'] < 0) | 
                                    (signals_df['confidence'] > 1)).sum()
                if invalid_confidence > 0:
                    quality_metrics['quality_score'] -= 15
                    quality_metrics['issues'].append(f"Invalid confidence values: {invalid_confidence}")
            
            # Ensure quality score doesn't go below 0
            quality_metrics['quality_score'] = max(0, quality_metrics['quality_score'])
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error validating data quality: {str(e)}")
            return {
                'total_rows': 0,
                'total_columns': 0,
                'quality_score': 0,
                'issues': [f"Data validation error: {str(e)}"]
            }