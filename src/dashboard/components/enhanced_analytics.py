"""
Enhanced Analytics for Paper Trading Dashboard
Advanced performance tracking, risk analysis, and portfolio insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class EnhancedAnalytics:
    """Enhanced analytics for paper trading performance"""
    
    def __init__(self, paper_engine):
        self.paper_engine = paper_engine
    
    def calculate_advanced_metrics(self) -> Dict:
        """Calculate advanced performance metrics"""
        
        trade_history = self.paper_engine.trade_history
        positions = self.paper_engine.positions
        
        if not trade_history:
            return self._empty_metrics()
        
        # Get sell trades for P&L analysis
        sell_trades = [t for t in trade_history if t.action == 'SELL']
        
        if not sell_trades:
            return self._basic_metrics()
        
        # Calculate returns
        returns = [t.pnl / self.paper_engine.initial_capital for t in sell_trades]
        
        # Advanced metrics
        metrics = {
            'total_trades': len(trade_history),
            'completed_trades': len(sell_trades),
            'win_rate': len([r for r in returns if r > 0]) / len(returns) if returns else 0,
            'avg_return': np.mean(returns) if returns else 0,
            'volatility': np.std(returns) if len(returns) > 1 else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_win': max(returns) if returns else 0,
            'max_loss': min(returns) if returns else 0,
            'profit_factor': self._calculate_profit_factor(returns),
            'expectancy': np.mean(returns) if returns else 0,
            'current_drawdown': self._calculate_current_drawdown(),
            'max_consecutive_wins': self._calculate_consecutive_wins(returns),
            'max_consecutive_losses': self._calculate_consecutive_losses(returns),
            'avg_hold_days': self._calculate_avg_holding_period(),
            'position_concentration': self._calculate_position_concentration(),
            'sector_diversification': self._calculate_sector_diversification()
        }
        
        return metrics
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {key: 0 for key in [
            'total_trades', 'completed_trades', 'win_rate', 'avg_return', 
            'volatility', 'sharpe_ratio', 'max_win', 'max_loss', 'profit_factor',
            'expectancy', 'current_drawdown', 'max_consecutive_wins', 
            'max_consecutive_losses', 'avg_hold_days', 'position_concentration',
            'sector_diversification'
        ]}
    
    def _basic_metrics(self) -> Dict:
        """Return basic metrics when no completed trades"""
        basic = self._empty_metrics()
        basic['total_trades'] = len(self.paper_engine.trade_history)
        return basic
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio (simplified, assuming risk-free rate = 0)"""
        if not returns or len(returns) < 2:
            return 0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # Annualized Sharpe (simplified)
        return (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    
    def _calculate_profit_factor(self, returns: List[float]) -> float:
        """Calculate profit factor (gross profits / gross losses)"""
        if not returns:
            return 0
        
        profits = [r for r in returns if r > 0]
        losses = [abs(r) for r in returns if r < 0]
        
        total_profit = sum(profits) if profits else 0
        total_loss = sum(losses) if losses else 0
        
        return total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        metrics = self.paper_engine.get_performance_metrics()
        return metrics.get('max_drawdown', 0)
    
    def _calculate_consecutive_wins(self, returns: List[float]) -> int:
        """Calculate maximum consecutive wins"""
        if not returns:
            return 0
        
        max_wins = current_wins = 0
        
        for return_val in returns:
            if return_val > 0:
                current_wins += 1
                max_wins = max(max_wins, current_wins)
            else:
                current_wins = 0
        
        return max_wins
    
    def _calculate_consecutive_losses(self, returns: List[float]) -> int:
        """Calculate maximum consecutive losses"""
        if not returns:
            return 0
        
        max_losses = current_losses = 0
        
        for return_val in returns:
            if return_val < 0:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0
        
        return max_losses
    
    def _calculate_avg_holding_period(self) -> float:
        """Calculate average holding period in days"""
        positions_df = self.paper_engine.get_positions_summary()
        
        if positions_df.empty:
            return 0
        
        return positions_df['Days Held'].mean()
    
    def _calculate_position_concentration(self) -> float:
        """Calculate position concentration (Herfindahl index)"""
        if not self.paper_engine.positions:
            return 0
        
        portfolio_value = self.paper_engine.get_current_portfolio_value()
        position_weights = []
        
        for position in self.paper_engine.positions.values():
            weight = position.get_position_value() / portfolio_value
            position_weights.append(weight ** 2)
        
        return sum(position_weights)  # Herfindahl-Hirschman Index
    
    def _calculate_sector_diversification(self) -> float:
        """Simplified sector diversification score"""
        # This is a simplified version - would need sector mapping in real implementation
        num_positions = len(self.paper_engine.positions)
        if num_positions == 0:
            return 0
        
        # Simple diversification score based on number of positions
        return min(1.0, num_positions / 10.0)  # Max score at 10+ positions
    
    def render_advanced_metrics_dashboard(self):
        """Render advanced metrics dashboard"""
        
        st.subheader("📊 Advanced Performance Analytics")
        
        metrics = self.calculate_advanced_metrics()
        
        # Key Performance Indicators
        st.markdown("**🎯 Key Performance Indicators**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Sharpe Ratio",
                f"{metrics['sharpe_ratio']:.2f}",
                help="Risk-adjusted return measure"
            )
        
        with col2:
            st.metric(
                "Win Rate",
                f"{metrics['win_rate']:.1%}",
                help="Percentage of profitable trades"
            )
        
        with col3:
            st.metric(
                "Profit Factor",
                f"{metrics['profit_factor']:.2f}",
                help="Gross profits / Gross losses"
            )
        
        with col4:
            st.metric(
                "Avg Return/Trade",
                f"{metrics['avg_return']:.2%}",
                help="Average return per completed trade"
            )
        
        # Risk Analysis
        st.markdown("**⚠️ Risk Analysis**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Volatility",
                f"{metrics['volatility']:.2%}",
                help="Standard deviation of returns"
            )
        
        with col2:
            st.metric(
                "Max Drawdown",
                f"-{metrics['current_drawdown']:.2f}%",
                help="Maximum portfolio decline"
            )
        
        with col3:
            concentration_score = metrics['position_concentration']
            concentration_level = "Low" if concentration_score < 0.2 else "Medium" if concentration_score < 0.5 else "High"
            st.metric(
                "Concentration Risk",
                concentration_level,
                help="Portfolio concentration level"
            )
        
        # Trading Patterns
        st.markdown("**📈 Trading Patterns**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Max Consecutive Wins",
                f"{metrics['max_consecutive_wins']}",
                help="Longest winning streak"
            )
        
        with col2:
            st.metric(
                "Max Consecutive Losses",
                f"{metrics['max_consecutive_losses']}",
                help="Longest losing streak"
            )
        
        with col3:
            st.metric(
                "Avg Holding Period",
                f"{metrics['avg_hold_days']:.1f} days",
                help="Average days per position"
            )
        
        with col4:
            diversification = metrics['sector_diversification']
            div_level = "Low" if diversification < 0.3 else "Medium" if diversification < 0.7 else "High"
            st.metric(
                "Diversification",
                div_level,
                help="Portfolio diversification level"
            )
    
    def render_performance_charts(self):
        """Render advanced performance charts"""
        
        st.markdown("**📊 Performance Charts**")
        
        # P&L Distribution
        sell_trades = [t for t in self.paper_engine.trade_history if t.action == 'SELL']
        
        if sell_trades:
            tab1, tab2, tab3 = st.tabs(["P&L Distribution", "Monthly Performance", "Risk Metrics"])
            
            with tab1:
                self._render_pnl_distribution(sell_trades)
            
            with tab2:
                self._render_monthly_performance(sell_trades)
            
            with tab3:
                self._render_risk_metrics_chart()
        else:
            st.info("📊 Performance charts will appear after completing some trades")
    
    def _render_pnl_distribution(self, sell_trades):
        """Render P&L distribution histogram"""
        
        pnl_values = [trade.pnl for trade in sell_trades]
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=pnl_values,
            nbinsx=min(20, len(pnl_values)),
            name="P&L Distribution",
            marker_color='rgba(59, 130, 246, 0.7)',
            marker_line=dict(width=1, color='rgb(59, 130, 246)')
        ))
        
        # Add vertical line at break-even
        fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.7, 
                      annotation_text="Break Even")
        
        fig.update_layout(
            title="P&L Distribution per Trade",
            xaxis_title="P&L ($)",
            yaxis_title="Number of Trades",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Best Trade", f"${max(pnl_values):+,.2f}")
        
        with col2:
            st.metric("Worst Trade", f"${min(pnl_values):+,.2f}")
        
        with col3:
            st.metric("Median P&L", f"${np.median(pnl_values):+,.2f}")
    
    def _render_monthly_performance(self, sell_trades):
        """Render monthly performance analysis"""
        
        # Group trades by month
        monthly_data = {}
        
        for trade in sell_trades:
            month_key = trade.timestamp.strftime('%Y-%m')
            if month_key not in monthly_data:
                monthly_data[month_key] = {'pnl': 0, 'trades': 0, 'wins': 0}
            
            monthly_data[month_key]['pnl'] += trade.pnl
            monthly_data[month_key]['trades'] += 1
            if trade.pnl > 0:
                monthly_data[month_key]['wins'] += 1
        
        if monthly_data:
            months = list(monthly_data.keys())
            pnl_values = [data['pnl'] for data in monthly_data.values()]
            win_rates = [data['wins'] / data['trades'] * 100 for data in monthly_data.values()]
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Monthly P&L', 'Monthly Win Rate'),
                specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
            )
            
            # Monthly P&L
            colors = ['green' if pnl > 0 else 'red' for pnl in pnl_values]
            fig.add_trace(
                go.Bar(x=months, y=pnl_values, marker_color=colors, name="Monthly P&L"),
                row=1, col=1
            )
            
            # Win rate
            fig.add_trace(
                go.Scatter(x=months, y=win_rates, mode='lines+markers', name="Win Rate %",
                          line=dict(color='blue')),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=False)
            fig.update_xaxes(title_text="Month", row=2, col=1)
            fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
            fig.update_yaxes(title_text="Win Rate (%)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need more trade history for monthly analysis")
    
    def _render_risk_metrics_chart(self):
        """Render risk metrics visualization"""
        
        metrics = self.calculate_advanced_metrics()
        
        # Risk metrics radar chart
        categories = ['Win Rate', 'Profit Factor', 'Sharpe Ratio', 'Diversification', 
                     'Low Concentration', 'Low Volatility']
        
        # Normalize metrics to 0-1 scale for radar chart
        values = [
            metrics['win_rate'],
            min(1.0, metrics['profit_factor'] / 3.0),  # Cap at 3.0 for visualization
            min(1.0, max(0, (metrics['sharpe_ratio'] + 1) / 3)),  # Normalize Sharpe
            metrics['sector_diversification'],
            1 - metrics['position_concentration'],  # Invert concentration (lower is better)
            max(0, 1 - metrics['volatility'] * 10)  # Invert volatility
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Risk Profile',
            line_color='rgba(59, 130, 246, 0.8)',
            fillcolor='rgba(59, 130, 246, 0.2)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="Risk Profile Analysis",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk interpretation
        avg_score = np.mean(values)
        if avg_score > 0.7:
            risk_level = "🟢 Low Risk"
        elif avg_score > 0.4:
            risk_level = "🟡 Medium Risk"
        else:
            risk_level = "🔴 High Risk"
        
        st.metric("Overall Risk Assessment", risk_level)
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report"""
        
        metrics = self.calculate_advanced_metrics()
        
        report = f"""
# Paper Trading Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Portfolio Summary
- Current Value: ${self.paper_engine.get_current_portfolio_value():,.2f}
- Total Return: {self.paper_engine.get_performance_metrics()['total_return']:+.2f}%
- Active Positions: {len(self.paper_engine.positions)}

## Trading Statistics
- Total Trades: {metrics['total_trades']}
- Completed Trades: {metrics['completed_trades']}
- Win Rate: {metrics['win_rate']:.1%}
- Average Return per Trade: {metrics['avg_return']:+.2%}

## Risk Metrics
- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
- Volatility: {metrics['volatility']:.2%}
- Maximum Drawdown: {metrics['current_drawdown']:.2f}%
- Profit Factor: {metrics['profit_factor']:.2f}

## Trading Patterns
- Max Consecutive Wins: {metrics['max_consecutive_wins']}
- Max Consecutive Losses: {metrics['max_consecutive_losses']}
- Average Holding Period: {metrics['avg_hold_days']:.1f} days

## Risk Assessment
- Position Concentration: {metrics['position_concentration']:.2f}
- Diversification Score: {metrics['sector_diversification']:.2f}
        """
        
        return report.strip()