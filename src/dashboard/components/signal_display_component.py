"""
Signal Display Component
Handles all signal visualization and display logic
Extracted from main dashboard for better modularity
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging

from .base_component import BaseComponent
from .utility_component import UtilityComponent

logger = logging.getLogger(__name__)

class SignalDisplayComponent(BaseComponent):
    """
    Component responsible for displaying trading signals in various formats
    Includes signal tables, breakdowns, and detailed analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("SignalDisplayComponent", config)
        self.utility = UtilityComponent(config)
        
        # Signal display configuration
        self.default_columns = [
            'symbol', 'company_name', 'current_price', 'signal_direction',
            'signal_strength', 'confidence', 'rsi_14', 'volume', 'change_1d'
        ]
        
        self.signal_colors = {
            'STRONG_BUY': '#10B981',
            'BUY': '#34D399',
            'NEUTRAL': '#6B7280',
            'SELL': '#F87171',
            'STRONG_SELL': '#EF4444'
        }
    
    def render(self, signals_data: pd.DataFrame, **kwargs) -> None:
        """
        Main render method for signal display
        """
        display_mode = kwargs.get('display_mode', 'table')
        
        if display_mode == 'table':
            self.render_signal_table(signals_data, **kwargs)
        elif display_mode == 'breakdown':
            self.render_signal_breakdown(signals_data, **kwargs)
        elif display_mode == 'detailed':
            self.render_detailed_analysis(signals_data, **kwargs)
        else:
            self.log_error(f"Unknown display mode: {display_mode}")
    
    def render_signal_table(self, signals_data: pd.DataFrame, **kwargs) -> None:
        """
        Render the main signals table with styling and filtering
        """
        try:
            if signals_data.empty:
                st.warning("No signal data available")
                return
            
            # Apply filters
            filtered_data = self._apply_signal_filters(signals_data, **kwargs)
            
            if filtered_data.empty:
                st.info("No signals match the current filters")
                return
            
            # Create display columns configuration
            display_columns = kwargs.get('columns', self.default_columns)
            display_data = self._prepare_table_data(filtered_data, display_columns)
            
            # Create styled table
            styled_table = self._create_styled_table(display_data)
            
            # Display the table
            st.dataframe(
                styled_table,
                use_container_width=True,
                height=kwargs.get('table_height', 600)
            )
            
            # Add summary metrics
            self._display_table_summary(filtered_data)
            
        except Exception as e:
            self.handle_error(e, "Error rendering signal table")
    
    def render_signal_breakdown(self, signals_data: pd.DataFrame, symbol: str = None) -> None:
        """
        Render detailed signal breakdown for specific stock or summary
        """
        try:
            if symbol:
                # Single stock breakdown
                stock_data = signals_data[signals_data['symbol'] == symbol]
                if not stock_data.empty:
                    self._render_single_stock_breakdown(stock_data.iloc[0])
                else:
                    st.error(f"No data found for symbol: {symbol}")
            else:
                # Portfolio-wide signal breakdown
                self._render_portfolio_breakdown(signals_data)
                
        except Exception as e:
            self.handle_error(e, "Error rendering signal breakdown")
    
    def render_detailed_analysis(self, signals_data: pd.DataFrame, symbol: str) -> None:
        """
        Render detailed technical analysis for a specific stock
        """
        try:
            stock_data = signals_data[signals_data['symbol'] == symbol]
            
            if stock_data.empty:
                st.error(f"No data found for symbol: {symbol}")
                return
            
            stock_row = stock_data.iloc[0]
            
            # Create tabs for different analysis views
            tab1, tab2, tab3, tab4 = st.tabs([
                "Signal Analysis", "Technical Indicators", "Risk Metrics", "Trading Setup"
            ])
            
            with tab1:
                self._render_signal_analysis(stock_row)
            
            with tab2:
                self._render_technical_indicators(stock_row)
            
            with tab3:
                self._render_risk_metrics(stock_row)
            
            with tab4:
                self._render_trading_setup(stock_row)
                
        except Exception as e:
            self.handle_error(e, "Error rendering detailed analysis")
    
    def _apply_signal_filters(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Apply various filters to the signals data"""
        filtered_data = data.copy()
        
        # Signal direction filter
        signal_filter = kwargs.get('signal_filter', 'All')
        if signal_filter != 'All':
            filtered_data = filtered_data[filtered_data['signal_direction'] == signal_filter]
        
        # Minimum confidence filter
        min_confidence = kwargs.get('min_confidence', 0.0)
        if min_confidence > 0:
            filtered_data = filtered_data[filtered_data['confidence'] >= min_confidence]
        
        # Minimum strength filter
        min_strength = kwargs.get('min_strength', 0.0)
        if min_strength > 0:
            filtered_data = filtered_data[filtered_data['signal_strength'] >= min_strength]
        
        # Sector filter
        sector_filter = kwargs.get('sector_filter')
        if sector_filter and sector_filter != 'All':
            if 'sector' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['sector'] == sector_filter]
        
        # Price range filter
        min_price = kwargs.get('min_price')
        max_price = kwargs.get('max_price')
        if min_price:
            filtered_data = filtered_data[filtered_data['current_price'] >= min_price]
        if max_price:
            filtered_data = filtered_data[filtered_data['current_price'] <= max_price]
        
        return filtered_data
    
    def _prepare_table_data(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Prepare data for table display with proper formatting"""
        display_data = data[columns].copy()
        
        # Format numeric columns
        if 'current_price' in display_data.columns:
            display_data['current_price'] = display_data['current_price'].apply(
                lambda x: self.utility.format_currency(x)
            )
        
        if 'signal_strength' in display_data.columns:
            display_data['signal_strength'] = display_data['signal_strength'].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
            )
        
        if 'confidence' in display_data.columns:
            display_data['confidence'] = display_data['confidence'].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
            )
        
        if 'rsi_14' in display_data.columns:
            display_data['rsi_14'] = display_data['rsi_14'].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
            )
        
        if 'volume' in display_data.columns:
            display_data['volume'] = display_data['volume'].apply(
                lambda x: self.utility.format_large_number(x)
            )
        
        if 'change_1d' in display_data.columns:
            display_data['change_1d'] = display_data['change_1d'].apply(
                lambda x: self.utility.format_percentage(x) if pd.notna(x) else "N/A"
            )
        
        return display_data
    
    def _create_styled_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply styling to the table data"""
        # Define styling configuration
        style_config = {
            'signal_direction': 'signals',
            'confidence': 'confidence',
            'rsi_14': 'rsi'
        }
        
        # Apply batch styling
        styled_data = self.utility.batch_style_dataframe(data, style_config)
        
        return styled_data
    
    def _display_table_summary(self, data: pd.DataFrame) -> None:
        """Display summary metrics for the signals table"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_signals = len(data)
            self.utility.create_metric_card(
                "Total Signals",
                str(total_signals),
                help_text="Total number of signals in the current view"
            )
        
        with col2:
            buy_signals = len(data[data['signal_direction'].isin(['BUY', 'STRONG_BUY'])])
            buy_pct = (buy_signals / total_signals * 100) if total_signals > 0 else 0
            self.utility.create_metric_card(
                "Buy Signals",
                f"{buy_signals} ({buy_pct:.1f}%)",
                help_text="Number and percentage of buy signals"
            )
        
        with col3:
            sell_signals = len(data[data['signal_direction'].isin(['SELL', 'STRONG_SELL'])])
            sell_pct = (sell_signals / total_signals * 100) if total_signals > 0 else 0
            self.utility.create_metric_card(
                "Sell Signals", 
                f"{sell_signals} ({sell_pct:.1f}%)",
                help_text="Number and percentage of sell signals"
            )
        
        with col4:
            avg_confidence = data['confidence'].mean() if 'confidence' in data.columns else 0
            self.utility.create_metric_card(
                "Avg Confidence",
                f"{avg_confidence:.2f}",
                help_text="Average confidence across all signals"
            )
    
    def _render_single_stock_breakdown(self, stock_data: pd.Series) -> None:
        """Render detailed breakdown for a single stock"""
        symbol = stock_data['symbol']
        company_name = stock_data.get('company_name', symbol)
        
        st.header(f"{symbol} - {company_name}")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            price = stock_data.get('current_price', 0)
            change = stock_data.get('change_1d', 0)
            self.utility.create_styled_metric_card(
                "Current Price",
                self.utility.format_currency(price),
                f"{change:+.2f}%" if change else None,
                help_text="Current stock price and daily change"
            )
        
        with col2:
            signal = stock_data.get('signal_direction', 'NEUTRAL')
            strength = stock_data.get('signal_strength', 0)
            self.utility.create_styled_metric_card(
                "Signal",
                f"{signal} {self.utility.get_signal_emoji(signal)}",
                f"Strength: {strength:.2f}",
                signal_type=signal,
                help_text="Current trading signal and strength"
            )
        
        with col3:
            confidence = stock_data.get('confidence', 0)
            self.utility.create_styled_metric_card(
                "Confidence",
                f"{confidence:.2f}",
                help_text="Signal confidence level (0-1)"
            )
        
        with col4:
            rsi = stock_data.get('rsi_14', 50)
            self.utility.create_styled_metric_card(
                "RSI (14)",
                f"{rsi:.1f}",
                help_text="14-period Relative Strength Index"
            )
        
        # Signal components breakdown
        self._render_signal_components(stock_data)
    
    def _render_signal_components(self, stock_data: pd.Series) -> None:
        """Render breakdown of signal components"""
        st.subheader("Signal Components")
        
        # Create component breakdown chart
        components = {}
        
        # Extract technical indicator signals
        if 'rsi_signal' in stock_data:
            components['RSI'] = stock_data['rsi_signal']
        if 'macd_signal' in stock_data:
            components['MACD'] = stock_data['macd_signal']
        if 'bb_signal' in stock_data:
            components['Bollinger Bands'] = stock_data['bb_signal']
        if 'volume_signal' in stock_data:
            components['Volume'] = stock_data['volume_signal']
        
        if components:
            # Create radar chart for component signals
            fig = self._create_component_radar_chart(components)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No component signal data available")
    
    def _create_component_radar_chart(self, components: Dict[str, float]) -> go.Figure:
        """Create radar chart for signal components"""
        categories = list(components.keys())
        values = [abs(v) for v in components.values()]  # Use absolute values for radar
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Signal Strength',
            line_color='rgb(16, 185, 129)',
            fillcolor='rgba(16, 185, 129, 0.25)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="Signal Component Breakdown",
            height=400
        )
        
        return fig
    
    def _render_portfolio_breakdown(self, data: pd.DataFrame) -> None:
        """Render portfolio-wide signal breakdown"""
        st.subheader("Portfolio Signal Distribution")
        
        # Signal distribution pie chart
        signal_counts = data['signal_direction'].value_counts()
        
        fig = px.pie(
            values=signal_counts.values,
            names=signal_counts.index,
            title="Signal Direction Distribution",
            color_discrete_map=self.signal_colors
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence distribution histogram
        if 'confidence' in data.columns:
            fig_hist = px.histogram(
                data,
                x='confidence',
                nbins=20,
                title="Signal Confidence Distribution",
                labels={'confidence': 'Confidence Level', 'count': 'Number of Signals'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    def _render_signal_analysis(self, stock_data: pd.Series) -> None:
        """Render signal analysis tab"""
        st.subheader("Signal Analysis")
        
        # Signal strength gauge
        strength = stock_data.get('signal_strength', 0)
        confidence = stock_data.get('confidence', 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_strength = self._create_gauge_chart(strength, "Signal Strength", "strength")
            st.plotly_chart(fig_strength, use_container_width=True)
        
        with col2:
            fig_confidence = self._create_gauge_chart(confidence, "Confidence", "confidence")
            st.plotly_chart(fig_confidence, use_container_width=True)
        
        # Signal explanation
        signal_direction = stock_data.get('signal_direction', 'NEUTRAL')
        explanation = self._generate_signal_explanation(stock_data)
        
        st.subheader("Signal Explanation")
        st.write(explanation)
    
    def _create_gauge_chart(self, value: float, title: str, chart_type: str) -> go.Figure:
        """Create gauge chart for metrics"""
        if chart_type == "strength":
            color_ranges = [
                [0, 0.2, "red"],
                [0.2, 0.4, "orange"], 
                [0.4, 0.6, "yellow"],
                [0.6, 0.8, "lightgreen"],
                [0.8, 1.0, "green"]
            ]
        else:  # confidence
            color_ranges = [
                [0, 0.3, "red"],
                [0.3, 0.6, "orange"],
                [0.6, 0.8, "lightgreen"],
                [0.8, 1.0, "green"]
            ]
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            gauge={
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [{'range': [r[0], r[1]], 'color': r[2]} for r in color_ranges],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ))
        
        fig.update_layout(height=250)
        return fig
    
    def _generate_signal_explanation(self, stock_data: pd.Series) -> str:
        """Generate human-readable explanation of the signal"""
        signal = stock_data.get('signal_direction', 'NEUTRAL')
        strength = stock_data.get('signal_strength', 0)
        confidence = stock_data.get('confidence', 0)
        
        explanation = f"**{signal}** signal with {self.utility.format_signal_strength(strength)} strength "
        explanation += f"and {confidence:.1%} confidence.\n\n"
        
        # Add technical analysis context
        rsi = stock_data.get('rsi_14', 50)
        if rsi > 70:
            explanation += f"• RSI at {rsi:.1f} suggests overbought conditions\n"
        elif rsi < 30:
            explanation += f"• RSI at {rsi:.1f} suggests oversold conditions\n"
        
        # Add more context based on available data
        if 'macd_histogram' in stock_data and stock_data['macd_histogram'] != 0:
            macd_hist = stock_data['macd_histogram']
            if macd_hist > 0:
                explanation += "• MACD histogram is positive, indicating bullish momentum\n"
            else:
                explanation += "• MACD histogram is negative, indicating bearish momentum\n"
        
        return explanation
    
    def _render_technical_indicators(self, stock_data: pd.Series) -> None:
        """Render technical indicators tab"""
        st.subheader("Technical Indicators")
        
        # Create metrics grid
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rsi = stock_data.get('rsi_14', 50)
            st.metric("RSI (14)", f"{rsi:.1f}", help="Relative Strength Index")
            
            volume = stock_data.get('volume', 0)
            st.metric("Volume", self.utility.format_large_number(volume))
        
        with col2:
            macd_line = stock_data.get('macd_line', 0)
            macd_signal = stock_data.get('macd_signal', 0)
            st.metric("MACD Line", f"{macd_line:.4f}")
            st.metric("MACD Signal", f"{macd_signal:.4f}")
        
        with col3:
            bb_upper = stock_data.get('bb_upper', 0)
            bb_lower = stock_data.get('bb_lower', 0)
            bb_position = stock_data.get('bb_position', 0.5)
            st.metric("BB Upper", f"{bb_upper:.2f}")
            st.metric("BB Lower", f"{bb_lower:.2f}")
            st.metric("BB Position", f"{bb_position:.2f}")
    
    def _render_risk_metrics(self, stock_data: pd.Series) -> None:
        """Render risk metrics tab"""
        st.subheader("Risk Metrics")
        
        # Risk metrics would come from risk management component
        volatility = stock_data.get('volatility_20d', 0)
        atr = stock_data.get('atr_14', 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("20-Day Volatility", f"{volatility:.1%}" if volatility else "N/A")
        
        with col2:
            st.metric("ATR (14)", f"{atr:.2f}" if atr else "N/A")
    
    def _render_trading_setup(self, stock_data: pd.Series) -> None:
        """Render trading setup tab"""
        st.subheader("Trading Setup")
        
        # Entry/exit levels
        entry_price = stock_data.get('entry_price', stock_data.get('current_price', 0))
        stop_loss = stock_data.get('stop_loss', 0)
        take_profit = stock_data.get('take_profit', 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Entry Price", self.utility.format_currency(entry_price))
        
        with col2:
            st.metric("Stop Loss", self.utility.format_currency(stop_loss) if stop_loss else "N/A")
        
        with col3:
            st.metric("Take Profit", self.utility.format_currency(take_profit) if take_profit else "N/A")
        
        # Risk/reward calculation
        if stop_loss and take_profit and entry_price:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            st.metric("Risk/Reward Ratio", f"{rr_ratio:.2f}" if rr_ratio else "N/A")