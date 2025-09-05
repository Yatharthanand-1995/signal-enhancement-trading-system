"""
Base Component Class for Dashboard Components
Provides common functionality and interface for all dashboard components
"""
import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from functools import lru_cache
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseComponent(ABC):
    """
    Abstract base class for all dashboard components
    Provides common functionality like caching, error handling, and state management
    """
    
    def __init__(self, component_name: str, config: Optional[Dict[str, Any]] = None):
        self.component_name = component_name
        self.config = config or {}
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes default
        self._state = {}
        
    @abstractmethod
    def render(self, **kwargs) -> Any:
        """
        Abstract method that each component must implement
        This is the main rendering method for the component
        """
        pass
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get component state value"""
        return self._state.get(key, default)
    
    def set_state(self, key: str, value: Any) -> None:
        """Set component state value"""
        self._state[key] = value
        
    def clear_state(self) -> None:
        """Clear all component state"""
        self._state.clear()
        
    def log_info(self, message: str) -> None:
        """Log info message with component context"""
        logger.info(f"[{self.component_name}] {message}")
        
    def log_error(self, message: str, error: Exception = None) -> None:
        """Log error message with component context"""
        if error:
            logger.error(f"[{self.component_name}] {message}: {str(error)}")
        else:
            logger.error(f"[{self.component_name}] {message}")
    
    def handle_error(self, error: Exception, fallback_message: str = "Component error occurred") -> None:
        """
        Handle component errors gracefully
        Shows error in Streamlit UI and logs the error
        """
        self.log_error("Component error", error)
        st.error(f"⚠️ {fallback_message}")
        
        if st.session_state.get('debug_mode', False):
            st.exception(error)
    
    def create_cached_data_loader(self, loader_func, cache_key: str):
        """
        Create a cached version of a data loading function
        """
        @lru_cache(maxsize=128)
        def cached_loader(*args, **kwargs):
            try:
                return loader_func(*args, **kwargs)
            except Exception as e:
                self.log_error(f"Error in cached loader for {cache_key}", e)
                return None
        
        return cached_loader
    
    def format_currency(self, value: float, precision: int = 2) -> str:
        """Format currency values consistently"""
        if pd.isna(value) or value is None:
            return "N/A"
        
        if abs(value) >= 1e9:
            return f"${value/1e9:.1f}B"
        elif abs(value) >= 1e6:
            return f"${value/1e6:.1f}M"
        elif abs(value) >= 1e3:
            return f"${value/1e3:.1f}K"
        else:
            return f"${value:.{precision}f}"
    
    def format_percentage(self, value: float, precision: int = 2) -> str:
        """Format percentage values consistently"""
        if pd.isna(value) or value is None:
            return "N/A"
        return f"{value:.{precision}f}%"
    
    def format_large_number(self, value: float, precision: int = 1) -> str:
        """Format large numbers with appropriate suffixes"""
        if pd.isna(value) or value is None:
            return "N/A"
        
        if abs(value) >= 1e12:
            return f"{value/1e12:.{precision}f}T"
        elif abs(value) >= 1e9:
            return f"{value/1e9:.{precision}f}B"
        elif abs(value) >= 1e6:
            return f"{value/1e6:.{precision}f}M"
        elif abs(value) >= 1e3:
            return f"{value/1e3:.{precision}f}K"
        else:
            return f"{value:.{precision}f}"
    
    def get_color_for_value(self, value: float, positive_color: str = "#10B981", 
                           negative_color: str = "#EF4444", neutral_color: str = "#6B7280") -> str:
        """Get appropriate color for a numeric value"""
        if pd.isna(value) or value is None:
            return neutral_color
        elif value > 0:
            return positive_color
        elif value < 0:
            return negative_color
        else:
            return neutral_color
    
    def create_metric_card(self, title: str, value: str, delta: str = None, 
                          delta_color: str = None, help_text: str = None) -> None:
        """Create a styled metric card"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.metric(
                label=title,
                value=value,
                delta=delta,
                delta_color=delta_color,
                help=help_text
            )
    
    def validate_data(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate that data contains required columns"""
        if data is None or data.empty:
            self.log_error("Data validation failed: Empty or None data")
            return False
        
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            self.log_error(f"Data validation failed: Missing columns: {missing_columns}")
            return False
        
        return True
    
    def safe_render(self, render_func, *args, **kwargs):
        """
        Safely execute a render function with error handling
        """
        try:
            return render_func(*args, **kwargs)
        except Exception as e:
            self.handle_error(e, f"Error rendering {self.component_name}")
            return None