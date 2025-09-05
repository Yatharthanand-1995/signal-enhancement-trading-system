"""
Performance Charts Component
Handles all chart generation and performance visualization
Extracted from main dashboard for better modularity and performance
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging

from .base_component import BaseComponent
from .utility_component import UtilityComponent

logger = logging.getLogger(__name__)

class PerformanceChartsComponent(BaseComponent):
    """
    Component responsible for creating all performance charts and visualizations
    Includes price charts, technical indicators, performance metrics, and portfolio analytics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("PerformanceChartsComponent", config)
        self.utility = UtilityComponent(config)
        
        # Chart configuration
        self.default_height = config.get('chart_height', 500) if config else 500
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17becf',
            'background': '#ffffff',
            'grid': '#e0e0e0'
        }
        
        # Technical indicator colors
        self.indicator_colors = {
            'rsi': '#9467bd',
            'macd_line': '#17becf',
            'macd_signal': '#ff7f0e',
            'macd_histogram': '#7f7f7f',
            'bb_upper': '#d62728',
            'bb_middle': '#2ca02c',
            'bb_lower': '#d62728',
            'volume': '#1f77b4'
        }
    
    def render(self, chart_type: str, data: Union[pd.DataFrame, Dict], **kwargs) -> go.Figure:
        """
        Main render method for different chart types
        """
        chart_methods = {
            'price_chart': self.create_price_chart,
            'technical_indicators': self.create_technical_indicators_chart,
            'performance_overview': self.create_performance_overview,
            'portfolio_allocation': self.create_portfolio_allocation_chart,
            'signals_heatmap': self.create_signals_heatmap,
            'returns_distribution': self.create_returns_distribution,
            'correlation_matrix': self.create_correlation_matrix,
            'risk_metrics': self.create_risk_metrics_chart,
            'sector_performance': self.create_sector_performance_chart,
            'signal_accuracy': self.create_signal_accuracy_chart
        }
        
        if chart_type in chart_methods:
            return chart_methods[chart_type](data, **kwargs)
        else:
            self.log_error(f"Unknown chart type: {chart_type}")
            return self._create_error_chart(f"Chart type '{chart_type}' not supported")
    
    def create_price_chart(self, stock_data: Dict, **kwargs) -> go.Figure:
        """
        Create comprehensive price chart with technical indicators
        """
        try:
            symbol = stock_data.get('symbol', 'Unknown')
            hist_data = stock_data.get('historical_data')
            
            if hist_data is None or hist_data.empty:
                return self._create_error_chart("No historical data available")
            
            # Create subplot with secondary y-axis for volume
            fig = make_subplots(
                rows=4, cols=1,
                row_heights=[0.5, 0.2, 0.15, 0.15],
                shared_xaxis=True,
                vertical_spacing=0.02,
                subplot_titles=[
                    f'{symbol} - Price & Indicators',
                    'Volume',
                    'RSI',
                    'MACD'
                ]
            )
            
            # Main price chart (candlestick)
            fig.add_trace(
                go.Candlestick(
                    x=hist_data.index,
                    open=hist_data['Open'],
                    high=hist_data['High'], 
                    low=hist_data['Low'],
                    close=hist_data['Close'],
                    name='Price',
                    increasing_line_color=self.color_scheme['success'],
                    decreasing_line_color=self.color_scheme['danger']
                ),
                row=1, col=1
            )
            
            # Add Bollinger Bands if available
            if all(col in hist_data.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
                fig.add_trace(
                    go.Scatter(
                        x=hist_data.index,
                        y=hist_data['BB_Upper'],
                        line=dict(color=self.indicator_colors['bb_upper'], width=1, dash='dash'),
                        name='BB Upper',
                        opacity=0.7
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=hist_data.index,
                        y=hist_data['BB_Middle'],
                        line=dict(color=self.indicator_colors['bb_middle'], width=1),
                        name='BB Middle'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=hist_data.index,
                        y=hist_data['BB_Lower'],
                        line=dict(color=self.indicator_colors['bb_lower'], width=1, dash='dash'),
                        name='BB Lower',
                        fill='tonexty',
                        fillcolor='rgba(214, 39, 40, 0.1)',
                        opacity=0.7
                    ),
                    row=1, col=1
                )
            
            # Volume chart
            if 'Volume' in hist_data.columns:
                fig.add_trace(
                    go.Bar(
                        x=hist_data.index,
                        y=hist_data['Volume'],
                        name='Volume',
                        marker_color=self.indicator_colors['volume'],
                        opacity=0.7
                    ),
                    row=2, col=1
                )
            
            # RSI chart
            if 'RSI' in hist_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=hist_data.index,
                        y=hist_data['RSI'],
                        line=dict(color=self.indicator_colors['rsi'], width=2),
                        name='RSI'
                    ),
                    row=3, col=1
                )
                
                # Add RSI reference lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
            
            # MACD chart
            if all(col in hist_data.columns for col in ['MACD_Line', 'MACD_Signal', 'MACD_Histogram']):
                fig.add_trace(
                    go.Scatter(
                        x=hist_data.index,
                        y=hist_data['MACD_Line'],
                        line=dict(color=self.indicator_colors['macd_line'], width=2),
                        name='MACD Line'
                    ),
                    row=4, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=hist_data.index,
                        y=hist_data['MACD_Signal'],
                        line=dict(color=self.indicator_colors['macd_signal'], width=2),
                        name='MACD Signal'
                    ),
                    row=4, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=hist_data.index,
                        y=hist_data['MACD_Histogram'],
                        name='MACD Histogram',
                        marker_color=self.indicator_colors['macd_histogram'],
                        opacity=0.7
                    ),
                    row=4, col=1
                )
            
            # Update layout
            fig.update_layout(
                height=self.default_height * 1.5,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                title=f"{symbol} - Technical Analysis",
                template="plotly_white"
            )
            
            # Update y-axes
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
            fig.update_yaxes(title_text="MACD", row=4, col=1)
            
            return fig
            
        except Exception as e:
            self.handle_error(e, "Error creating price chart")
            return self._create_error_chart("Failed to create price chart")
    
    def create_technical_indicators_chart(self, indicators_data: Dict, **kwargs) -> go.Figure:
        """
        Create standalone technical indicators chart
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['RSI Distribution', 'MACD Signals', 'BB Position', 'Volume Analysis'],
                specs=[[{"type": "histogram"}, {"type": "scatter"}],
                       [{"type": "histogram"}, {"type": "bar"}]]
            )
            
            # RSI Distribution
            if 'rsi_values' in indicators_data:
                fig.add_trace(
                    go.Histogram(
                        x=indicators_data['rsi_values'],
                        nbinsx=20,
                        name='RSI Distribution',
                        marker_color=self.indicator_colors['rsi']
                    ),
                    row=1, col=1
                )
            
            # MACD Signals
            if 'macd_data' in indicators_data:
                macd_data = indicators_data['macd_data']
                fig.add_trace(
                    go.Scatter(
                        x=macd_data.index,
                        y=macd_data['MACD_Line'],
                        mode='lines',
                        name='MACD Line',
                        line=dict(color=self.indicator_colors['macd_line'])
                    ),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=macd_data.index,
                        y=macd_data['MACD_Signal'],
                        mode='lines',
                        name='Signal Line',
                        line=dict(color=self.indicator_colors['macd_signal'])
                    ),
                    row=1, col=2
                )
            
            # Bollinger Bands Position Distribution
            if 'bb_positions' in indicators_data:
                fig.add_trace(
                    go.Histogram(
                        x=indicators_data['bb_positions'],
                        nbinsx=20,
                        name='BB Position',
                        marker_color=self.indicator_colors['bb_middle']
                    ),
                    row=2, col=1
                )
            
            # Volume Analysis
            if 'volume_data' in indicators_data:
                volume_data = indicators_data['volume_data']
                fig.add_trace(
                    go.Bar(
                        x=volume_data.index,
                        y=volume_data['Volume'],
                        name='Volume',
                        marker_color=self.indicator_colors['volume']
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=600,
                showlegend=True,
                title="Technical Indicators Analysis",
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            self.handle_error(e, "Error creating technical indicators chart")
            return self._create_error_chart("Failed to create technical indicators chart")
    
    def create_performance_overview(self, performance_data: Dict, **kwargs) -> go.Figure:
        """
        Create comprehensive performance overview chart
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Portfolio Value Over Time',
                    'Monthly Returns',
                    'Drawdown Analysis', 
                    'Risk-Return Scatter'
                ],
                specs=[[{"type": "scatter"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # Portfolio value over time
            if 'portfolio_values' in performance_data:
                portfolio_data = performance_data['portfolio_values']
                fig.add_trace(
                    go.Scatter(
                        x=portfolio_data.index,
                        y=portfolio_data['Total_Value'],
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color=self.color_scheme['primary'], width=3),
                        fill='tonexty'
                    ),
                    row=1, col=1
                )
            
            # Monthly returns
            if 'monthly_returns' in performance_data:
                returns = performance_data['monthly_returns']
                colors = [self.color_scheme['success'] if r > 0 else self.color_scheme['danger'] for r in returns.values]
                
                fig.add_trace(
                    go.Bar(
                        x=returns.index,
                        y=returns.values,
                        name='Monthly Returns',
                        marker_color=colors
                    ),
                    row=1, col=2
                )
            
            # Drawdown analysis
            if 'drawdown' in performance_data:
                drawdown_data = performance_data['drawdown']
                fig.add_trace(
                    go.Scatter(
                        x=drawdown_data.index,
                        y=drawdown_data['Drawdown'],
                        mode='lines',
                        name='Drawdown',
                        line=dict(color=self.color_scheme['danger']),
                        fill='tozeroy',
                        fillcolor='rgba(214, 39, 40, 0.3)'
                    ),
                    row=2, col=1
                )
            
            # Risk-Return scatter
            if 'risk_return' in performance_data:
                risk_return = performance_data['risk_return']
                fig.add_trace(
                    go.Scatter(
                        x=risk_return['Risk'],
                        y=risk_return['Return'],
                        mode='markers',
                        name='Stocks',
                        marker=dict(
                            size=10,
                            color=risk_return['Sharpe'],
                            colorscale='RdYlGn',
                            showscale=True,
                            colorbar=dict(title="Sharpe Ratio")
                        ),
                        text=risk_return.index,
                        textposition="top center"
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=700,
                showlegend=True,
                title="Portfolio Performance Overview",
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            self.handle_error(e, "Error creating performance overview")
            return self._create_error_chart("Failed to create performance overview")
    
    def create_signals_heatmap(self, signals_df: pd.DataFrame, **kwargs) -> go.Figure:
        """
        Create signals heatmap showing signal distribution across stocks and time
        """
        try:
            # Prepare data for heatmap
            if 'date' not in signals_df.columns:
                signals_df['date'] = datetime.now().strftime('%Y-%m-%d')
            
            # Create pivot table for heatmap
            heatmap_data = signals_df.pivot_table(
                values='signal_strength',
                index='symbol',
                columns='date',
                fill_value=0
            )
            
            # Create custom colorscale for signals
            colorscale = [
                [0, '#d62728'],      # Strong sell - red
                [0.2, '#ff7f0e'],    # Sell - orange
                [0.4, '#7f7f7f'],    # Neutral - gray
                [0.6, '#2ca02c'],    # Buy - green
                [0.8, '#17becf'],    # Strong buy - cyan
                [1.0, '#1f77b4']     # Very strong buy - blue
            ]
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale=colorscale,
                zmid=0.5,
                colorbar=dict(title="Signal Strength"),
                hovertemplate='<b>%{y}</b><br>Date: %{x}<br>Signal Strength: %{z:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Signals Heatmap - Signal Strength Across Stocks",
                xaxis_title="Date",
                yaxis_title="Symbol",
                height=max(400, len(heatmap_data.index) * 20),
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            self.handle_error(e, "Error creating signals heatmap")
            return self._create_error_chart("Failed to create signals heatmap")
    
    def create_correlation_matrix(self, returns_data: pd.DataFrame, **kwargs) -> go.Figure:
        """
        Create correlation matrix heatmap for stock returns
        """
        try:
            correlation_matrix = returns_data.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="Correlation"),
                hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Stock Returns Correlation Matrix",
                height=600,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            self.handle_error(e, "Error creating correlation matrix")
            return self._create_error_chart("Failed to create correlation matrix")
    
    def create_sector_performance_chart(self, sector_data: Dict, **kwargs) -> go.Figure:
        """
        Create sector performance comparison chart
        """
        try:
            sectors = list(sector_data.keys())
            performance = [sector_data[sector].get('return', 0) for sector in sectors]
            colors = [self.color_scheme['success'] if p > 0 else self.color_scheme['danger'] for p in performance]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=sectors,
                    y=performance,
                    marker_color=colors,
                    text=[f"{p:.1f}%" for p in performance],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Sector Performance Comparison",
                xaxis_title="Sector",
                yaxis_title="Return (%)",
                height=400,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            self.handle_error(e, "Error creating sector performance chart")
            return self._create_error_chart("Failed to create sector performance chart")
    
    def create_risk_metrics_chart(self, risk_data: Dict, **kwargs) -> go.Figure:
        """
        Create risk metrics visualization
        """
        try:
            metrics = ['VaR_95', 'Expected_Shortfall', 'Max_Drawdown', 'Volatility']
            values = [risk_data.get(metric, 0) for metric in metrics]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name='Risk Metrics',
                line_color=self.color_scheme['danger']
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(values) * 1.2] if values else [0, 1]
                    )),
                showlegend=False,
                title="Portfolio Risk Metrics",
                height=400
            )
            
            return fig
            
        except Exception as e:
            self.handle_error(e, "Error creating risk metrics chart")
            return self._create_error_chart("Failed to create risk metrics chart")
    
    def create_signal_accuracy_chart(self, accuracy_data: Dict, **kwargs) -> go.Figure:
        """
        Create signal accuracy tracking chart
        """
        try:
            signal_types = list(accuracy_data.keys())
            accuracy_rates = [accuracy_data[signal_type].get('accuracy', 0) for signal_type in signal_types]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=signal_types,
                    y=accuracy_rates,
                    marker_color=self.color_scheme['info'],
                    text=[f"{acc:.1f}%" for acc in accuracy_rates],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Signal Accuracy by Type",
                xaxis_title="Signal Type",
                yaxis_title="Accuracy (%)",
                height=400,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            self.handle_error(e, "Error creating signal accuracy chart")
            return self._create_error_chart("Failed to create signal accuracy chart")
    
    def create_portfolio_allocation_chart(self, allocation_data: Dict, **kwargs) -> go.Figure:
        """
        Create portfolio allocation pie chart
        """
        try:
            labels = list(allocation_data.keys())
            values = list(allocation_data.values())
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.3,
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig.update_layout(
                title="Portfolio Allocation",
                height=400,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            self.handle_error(e, "Error creating portfolio allocation chart")
            return self._create_error_chart("Failed to create portfolio allocation chart")
    
    def create_returns_distribution(self, returns: pd.Series, **kwargs) -> go.Figure:
        """
        Create returns distribution histogram with statistics
        """
        try:
            fig = go.Figure()
            
            # Histogram
            fig.add_trace(go.Histogram(
                x=returns,
                nbinsx=30,
                name='Returns Distribution',
                marker_color=self.color_scheme['primary'],
                opacity=0.7
            ))
            
            # Add mean line
            mean_return = returns.mean()
            fig.add_vline(
                x=mean_return,
                line_dash="dash",
                line_color=self.color_scheme['success'],
                annotation_text=f"Mean: {mean_return:.2%}"
            )
            
            # Add standard deviation lines
            std_return = returns.std()
            fig.add_vline(
                x=mean_return + std_return,
                line_dash="dot",
                line_color=self.color_scheme['warning'],
                opacity=0.7
            )
            fig.add_vline(
                x=mean_return - std_return,
                line_dash="dot",
                line_color=self.color_scheme['warning'],
                opacity=0.7
            )
            
            fig.update_layout(
                title="Returns Distribution",
                xaxis_title="Return (%)",
                yaxis_title="Frequency",
                height=400,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            self.handle_error(e, "Error creating returns distribution")
            return self._create_error_chart("Failed to create returns distribution")
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create a simple error chart when main chart fails"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=f"Chart Error: {error_message}",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color="red")
        )
        
        fig.update_layout(
            title="Chart Error",
            height=300,
            template="plotly_white"
        )
        
        return fig
    
    def create_interactive_dashboard(self, data: Dict, **kwargs) -> None:
        """
        Create interactive dashboard with multiple charts
        """
        try:
            st.subheader("Performance Dashboard")
            
            # Create tabs for different chart categories
            tab1, tab2, tab3, tab4 = st.tabs([
                "Price Analysis", "Performance Metrics", "Risk Analysis", "Portfolio View"
            ])
            
            with tab1:
                if 'price_data' in data:
                    price_chart = self.create_price_chart(data['price_data'])
                    st.plotly_chart(price_chart, use_container_width=True)
                
                if 'technical_indicators' in data:
                    tech_chart = self.create_technical_indicators_chart(data['technical_indicators'])
                    st.plotly_chart(tech_chart, use_container_width=True)
            
            with tab2:
                if 'performance_data' in data:
                    perf_chart = self.create_performance_overview(data['performance_data'])
                    st.plotly_chart(perf_chart, use_container_width=True)
                
                if 'returns_data' in data:
                    returns_chart = self.create_returns_distribution(data['returns_data'])
                    st.plotly_chart(returns_chart, use_container_width=True)
            
            with tab3:
                if 'risk_data' in data:
                    risk_chart = self.create_risk_metrics_chart(data['risk_data'])
                    st.plotly_chart(risk_chart, use_container_width=True)
                
                if 'correlation_data' in data:
                    corr_chart = self.create_correlation_matrix(data['correlation_data'])
                    st.plotly_chart(corr_chart, use_container_width=True)
            
            with tab4:
                if 'allocation_data' in data:
                    alloc_chart = self.create_portfolio_allocation_chart(data['allocation_data'])
                    st.plotly_chart(alloc_chart, use_container_width=True)
                
                if 'sector_data' in data:
                    sector_chart = self.create_sector_performance_chart(data['sector_data'])
                    st.plotly_chart(sector_chart, use_container_width=True)
                    
        except Exception as e:
            self.handle_error(e, "Error creating interactive dashboard")