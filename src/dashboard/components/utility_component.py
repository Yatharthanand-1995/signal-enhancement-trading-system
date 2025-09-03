"""
Utility Component
Contains commonly used utility functions extracted from main dashboard
Provides styling, formatting, and calculation utilities
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from functools import lru_cache
import logging
from .base_component import BaseComponent

logger = logging.getLogger(__name__)

class UtilityComponent(BaseComponent):
    """
    Utility component containing common styling and calculation functions
    Extracted from main dashboard for better modularity and performance
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("UtilityComponent", config)
        
        # Color scheme configuration
        self.colors = {
            'strong_buy': '#10B981',      # Green
            'buy': '#34D399',             # Light green
            'neutral': '#6B7280',         # Gray
            'sell': '#F87171',            # Light red
            'strong_sell': '#EF4444',     # Red
            'warning': '#F59E0B',         # Amber
            'high_confidence': '#10B981',  # Green
            'medium_confidence': '#F59E0B', # Amber
            'low_confidence': '#EF4444',   # Red
        }
    
    def render(self, **kwargs) -> Any:
        """Utility component doesn't render directly"""
        pass
    
    # Styling Functions (Extracted from main.py lines 29-106)
    @staticmethod
    @lru_cache(maxsize=1000)
    def style_signals(val: str) -> str:
        """
        Style signal values with colors and badges
        Cached for performance improvement
        """
        try:
            val_str = str(val)
            if 'STRONG_BUY' in val_str:
                return 'background: var(--success-green, #10B981); color: white; padding: 4px 8px; border-radius: 6px; font-weight: 600;'
            elif 'BUY' in val_str:
                return 'background: var(--success-green-light, #34D399); color: white; padding: 4px 8px; border-radius: 6px; font-weight: 600;'
            elif 'STRONG_SELL' in val_str:
                return 'background: var(--danger-red, #EF4444); color: white; padding: 4px 8px; border-radius: 6px; font-weight: 600;'
            elif 'SELL' in val_str:
                return 'background: var(--danger-red-light, #F87171); color: white; padding: 4px 8px; border-radius: 6px; font-weight: 600;'
            elif 'Strong' in val_str:
                return 'background: var(--warning-amber, #F59E0B); color: white; padding: 4px 8px; border-radius: 6px; font-weight: 600;'
            else:
                return 'background: var(--neutral-gray, #6B7280); color: white; padding: 4px 8px; border-radius: 6px; font-weight: 600;'
        except Exception:
            return 'color: var(--text-primary, #111827);'
    
    @staticmethod
    @lru_cache(maxsize=500)
    def style_risk_reward(val: Union[str, float]) -> str:
        """Style risk/reward ratios with colors"""
        try:
            if isinstance(val, str) and ('N/A' in val or val == ''):
                return 'color: var(--text-secondary, #6B7280);'
            
            ratio = float(val) if not isinstance(val, (int, float)) else val
            
            if ratio >= 3.0:
                return 'background: var(--success-green, #10B981); color: white; padding: 2px 6px; border-radius: 4px; font-weight: 600;'
            elif ratio >= 2.0:
                return 'background: var(--success-green-light, #34D399); color: white; padding: 2px 6px; border-radius: 4px; font-weight: 600;'
            elif ratio >= 1.5:
                return 'background: var(--warning-amber, #F59E0B); color: white; padding: 2px 6px; border-radius: 4px; font-weight: 600;'
            else:
                return 'background: var(--danger-red, #EF4444); color: white; padding: 2px 6px; border-radius: 4px; font-weight: 600;'
        except (ValueError, TypeError):
            return 'color: var(--text-secondary, #6B7280);'
    
    @staticmethod
    @lru_cache(maxsize=500)
    def style_confidence(val: Union[str, float]) -> str:
        """Style confidence values with appropriate colors"""
        try:
            if isinstance(val, str) and ('N/A' in val or val == ''):
                return 'color: var(--text-secondary, #6B7280);'
            
            confidence = float(val) if not isinstance(val, (int, float)) else val
            
            if confidence >= 0.8:
                return 'background: var(--success-green, #10B981); color: white; padding: 2px 6px; border-radius: 4px; font-weight: 600;'
            elif confidence >= 0.6:
                return 'background: var(--warning-amber, #F59E0B); color: white; padding: 2px 6px; border-radius: 4px; font-weight: 600;'
            else:
                return 'background: var(--danger-red, #EF4444); color: white; padding: 2px 6px; border-radius: 4px; font-weight: 600;'
        except (ValueError, TypeError):
            return 'color: var(--text-secondary, #6B7280);'
    
    @staticmethod
    @lru_cache(maxsize=500)
    def style_rsi(val: Union[str, float]) -> str:
        """Style RSI values based on overbought/oversold levels"""
        try:
            if isinstance(val, str) and ('N/A' in val or val == ''):
                return 'color: var(--text-secondary, #6B7280);'
            
            rsi = float(val) if not isinstance(val, (int, float)) else val
            
            if rsi >= 80:
                return 'background: var(--danger-red, #EF4444); color: white; padding: 2px 6px; border-radius: 4px; font-weight: 600;'  # Overbought
            elif rsi >= 70:
                return 'background: var(--warning-amber, #F59E0B); color: white; padding: 2px 6px; border-radius: 4px; font-weight: 600;'
            elif rsi <= 20:
                return 'background: var(--success-green, #10B981); color: white; padding: 2px 6px; border-radius: 4px; font-weight: 600;'  # Oversold
            elif rsi <= 30:
                return 'background: var(--success-green-light, #34D399); color: white; padding: 2px 6px; border-radius: 4px; font-weight: 600;'
            else:
                return 'color: var(--text-primary, #111827);'
        except (ValueError, TypeError):
            return 'color: var(--text-secondary, #6B7280);'
    
    @staticmethod
    @lru_cache(maxsize=500) 
    def style_volume(val: Union[str, float]) -> str:
        """Style volume values with appropriate scaling"""
        try:
            if isinstance(val, str) and ('N/A' in val or val == ''):
                return 'color: var(--text-secondary, #6B7280);'
            
            volume = float(val) if not isinstance(val, (int, float)) else val
            
            # High volume (above average) - important for signal confirmation
            if volume >= 2000000:
                return 'background: var(--success-green, #10B981); color: white; padding: 2px 6px; border-radius: 4px; font-weight: 600;'
            elif volume >= 1000000:
                return 'background: var(--warning-amber, #F59E0B); color: white; padding: 2px 6px; border-radius: 4px; font-weight: 600;'
            else:
                return 'color: var(--text-primary, #111827);'
        except (ValueError, TypeError):
            return 'color: var(--text-secondary, #6B7280);'
    
    @staticmethod
    @lru_cache(maxsize=500)
    def style_bollinger(val: Union[str, float]) -> str:
        """Style Bollinger Band position values"""
        try:
            if isinstance(val, str) and ('N/A' in val or val == ''):
                return 'color: var(--text-secondary, #6B7280);'
            
            bb_pos = float(val) if not isinstance(val, (int, float)) else val
            
            if bb_pos >= 0.8:
                return 'background: var(--danger-red, #EF4444); color: white; padding: 2px 6px; border-radius: 4px; font-weight: 600;'  # Near upper band
            elif bb_pos <= 0.2:
                return 'background: var(--success-green, #10B981); color: white; padding: 2px 6px; border-radius: 4px; font-weight: 600;'  # Near lower band
            else:
                return 'color: var(--text-primary, #111827);'
        except (ValueError, TypeError):
            return 'color: var(--text-secondary, #6B7280);'
    
    # Calculation Functions (Extracted from main.py)
    @staticmethod
    @lru_cache(maxsize=100)
    def calculate_regime_confidence(vix: float, fear_greed: float, breadth_ratio: float) -> float:
        """
        Calculate market regime confidence based on multiple indicators
        Cached for performance improvement
        """
        try:
            # Normalize indicators to 0-1 scale
            vix_norm = min(max((vix - 15) / 35, 0), 1)  # VIX 15-50 range
            fg_norm = fear_greed / 100  # Fear & Greed 0-100 range
            br_norm = min(max(breadth_ratio, 0), 1)  # Breadth ratio 0-1 range
            
            # Weight the indicators (VIX and F&G are more important)
            confidence = (0.4 * (1 - vix_norm) + 0.4 * fg_norm + 0.2 * br_norm)
            return round(confidence, 3)
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.5  # Neutral confidence
    
    @staticmethod
    @lru_cache(maxsize=200)
    def calculate_rsi(prices_tuple: tuple, period: int = 14) -> float:
        """
        Calculate RSI (Relative Strength Index)
        Optimized with caching and tuple input for hashability
        """
        try:
            prices = np.array(prices_tuple)
            if len(prices) < period + 1:
                return 50.0  # Neutral RSI if not enough data
            
            delta = np.diff(prices)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            avg_gain = np.mean(gain[-period:])
            avg_loss = np.mean(loss[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return round(rsi, 2)
        except (TypeError, ValueError, IndexError):
            return 50.0
    
    @staticmethod
    @lru_cache(maxsize=150)
    def calculate_macd(prices_tuple: tuple, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        Returns dict with macd_line, signal_line, and histogram
        """
        try:
            prices = np.array(prices_tuple)
            if len(prices) < slow + signal:
                return {'macd_line': 0.0, 'signal_line': 0.0, 'histogram': 0.0}
            
            # Calculate EMAs
            exp1 = prices.ewm(span=fast).mean()
            exp2 = prices.ewm(span=slow).mean()
            
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd_line': round(float(macd_line.iloc[-1]), 4),
                'signal_line': round(float(signal_line.iloc[-1]), 4), 
                'histogram': round(float(histogram.iloc[-1]), 4)
            }
        except (TypeError, ValueError, IndexError):
            return {'macd_line': 0.0, 'signal_line': 0.0, 'histogram': 0.0}
    
    @staticmethod
    @lru_cache(maxsize=150)
    def calculate_bollinger_bands(prices_tuple: tuple, window: int = 20, num_std: int = 2) -> Dict[str, float]:
        """
        Calculate Bollinger Bands
        Returns dict with upper, middle, lower bands and current position
        """
        try:
            prices = pd.Series(prices_tuple)
            if len(prices) < window:
                current_price = prices.iloc[-1] if len(prices) > 0 else 0
                return {
                    'upper': current_price,
                    'middle': current_price,
                    'lower': current_price,
                    'position': 0.5
                }
            
            rolling_mean = prices.rolling(window=window).mean()
            rolling_std = prices.rolling(window=window).std()
            
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            
            current_price = prices.iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            current_middle = rolling_mean.iloc[-1]
            
            # Calculate position within bands (0 = lower band, 1 = upper band)
            if current_upper > current_lower:
                position = (current_price - current_lower) / (current_upper - current_lower)
            else:
                position = 0.5
            
            return {
                'upper': round(float(current_upper), 2),
                'middle': round(float(current_middle), 2),
                'lower': round(float(current_lower), 2),
                'position': round(float(position), 3)
            }
        except (TypeError, ValueError, IndexError):
            return {'upper': 0.0, 'middle': 0.0, 'lower': 0.0, 'position': 0.5}
    
    @staticmethod
    def format_signal_strength(strength: float) -> str:
        """Format signal strength as readable text"""
        if strength >= 0.8:
            return "Very Strong"
        elif strength >= 0.6:
            return "Strong"
        elif strength >= 0.4:
            return "Moderate"
        elif strength >= 0.2:
            return "Weak"
        else:
            return "Very Weak"
    
    @staticmethod
    def get_signal_emoji(signal_direction: str) -> str:
        """Get emoji for signal direction"""
        signal_emojis = {
            'STRONG_BUY': '🚀',
            'BUY': '⬆️',
            'NEUTRAL': '➡️',
            'SELL': '⬇️',
            'STRONG_SELL': '💥'
        }
        return signal_emojis.get(signal_direction, '❓')
    
    def create_styled_metric_card(self, title: str, value: str, delta: str = None,
                                 signal_type: str = None, help_text: str = None) -> None:
        """
        Create a styled metric card with signal-appropriate coloring
        """
        if signal_type:
            # Apply signal-based styling
            if 'BUY' in signal_type:
                delta_color = "normal"
            elif 'SELL' in signal_type:
                delta_color = "inverse"
            else:
                delta_color = None
        else:
            delta_color = None
        
        self.create_metric_card(
            title=title,
            value=value,
            delta=delta,
            delta_color=delta_color,
            help_text=help_text
        )
    
    def batch_style_dataframe(self, df: pd.DataFrame, style_config: Dict[str, str]) -> pd.DataFrame:
        """
        Apply multiple styling functions to a dataframe efficiently
        
        Args:
            df: DataFrame to style
            style_config: Dict mapping column names to styling function names
        """
        styled_df = df.copy()
        
        style_functions = {
            'signals': self.style_signals,
            'risk_reward': self.style_risk_reward,
            'confidence': self.style_confidence,
            'rsi': self.style_rsi,
            'volume': self.style_volume,
            'bollinger': self.style_bollinger
        }
        
        for column, style_func_name in style_config.items():
            if column in styled_df.columns and style_func_name in style_functions:
                style_func = style_functions[style_func_name]
                styled_df = styled_df.style.applymap(style_func, subset=[column])
        
        return styled_df