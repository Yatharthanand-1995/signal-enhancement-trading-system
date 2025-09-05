"""
Chart Helper Utilities
Common chart styling and helper functions
"""
from typing import Dict, List, Any
import plotly.graph_objects as go

class ChartStyler:
    """Chart styling utilities"""
    
    def __init__(self):
        self.default_colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728'
        }
    
    def get_color_palette(self) -> List[str]:
        """Get default color palette"""
        return list(self.default_colors.values())
    
    def style_chart(self, fig: go.Figure, title: str = None) -> go.Figure:
        """Apply default styling to chart"""
        if title:
            fig.update_layout(title=title)
        
        fig.update_layout(template="plotly_white")
        return fig