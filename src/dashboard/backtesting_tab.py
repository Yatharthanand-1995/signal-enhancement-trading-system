"""
Backtesting Tab for Streamlit Dashboard
Comprehensive backtesting analysis with market regime comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sqlite3
import json
from typing import Dict, List, Optional, Tuple

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backtesting.enhanced_backtest_engine import run_backtest_analysis
from utils.backtesting_schema_sqlite import sqlite_backtesting_schema
from utils.logging_setup import get_logger

logger = get_logger(__name__)

class BacktestingDashboard:
    """Backtesting dashboard component for Streamlit"""
    
    def __init__(self, db_path: str = "data/historical_stocks.db"):
        self.db_path = db_path
        
    def get_connection(self):
        """Get database connection"""
        return sqlite_backtesting_schema.get_connection()
        
    def get_available_results(self) -> pd.DataFrame:
        """Get list of available backtest results"""
        
        try:
            with self.get_connection() as conn:
                query = """
                SELECT 
                    br.result_id,
                    bc.config_name,
                    bc.start_date,
                    bc.end_date,
                    bc.strategy_type,
                    bc.initial_capital,
                    br.total_return,
                    br.annualized_return,
                    br.sharpe_ratio,
                    br.max_drawdown,
                    br.total_trades,
                    br.win_rate,
                    br.excess_return,
                    br.created_at
                FROM backtest_results br
                JOIN backtest_configs bc ON br.config_id = bc.config_id
                ORDER BY br.created_at DESC
                """
                
                df = pd.read_sql_query(query, conn)
                return df
                
        except Exception as e:
            logger.error(f"Failed to load backtest results: {str(e)}")
            return pd.DataFrame()
    
    def get_backtest_details(self, result_id: int) -> Dict:
        """Get detailed backtest results for a specific result ID"""
        
        try:
            with self.get_connection() as conn:
                # Main results
                results_query = """
                SELECT br.*, bc.config_name, bc.strategy_type, bc.parameters
                FROM backtest_results br
                JOIN backtest_configs bc ON br.config_id = bc.config_id
                WHERE br.result_id = ?
                """
                
                main_results = pd.read_sql_query(results_query, conn, params=[result_id])
                
                if main_results.empty:
                    return {}
                
                result = main_results.iloc[0].to_dict()
                
                # Get daily portfolio values
                portfolio_query = """
                SELECT * FROM backtest_portfolio_values 
                WHERE result_id = ? 
                ORDER BY date
                """
                
                portfolio_values = pd.read_sql_query(portfolio_query, conn, params=[result_id])
                portfolio_values['date'] = pd.to_datetime(portfolio_values['date'])
                
                # Get individual trades
                trades_query = """
                SELECT * FROM backtest_trades 
                WHERE result_id = ? 
                ORDER BY entry_date DESC
                """
                
                trades = pd.read_sql_query(trades_query, conn, params=[result_id])
                
                return {
                    'main_results': result,
                    'portfolio_values': portfolio_values,
                    'trades': trades
                }
                
        except Exception as e:
            logger.error(f"Failed to load backtest details: {str(e)}")
            return {}
    
    def get_benchmark_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get benchmark performance data"""
        
        try:
            with self.get_connection() as conn:
                query = """
                SELECT date, adj_close_price, daily_return, cumulative_return
                FROM benchmark_performance 
                WHERE symbol = ? AND date BETWEEN ? AND ?
                ORDER BY date
                """
                
                df = pd.read_sql_query(query, conn, params=[symbol, start_date, end_date])
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                
                return df
                
        except Exception as e:
            logger.error(f"Failed to load benchmark data: {str(e)}")
            return pd.DataFrame()
    
    def get_market_regimes(self) -> pd.DataFrame:
        """Get market regime data"""
        
        try:
            with self.get_connection() as conn:
                query = """
                SELECT * FROM market_regimes 
                ORDER BY start_date
                """
                
                df = pd.read_sql_query(query, conn)
                df['start_date'] = pd.to_datetime(df['start_date'])
                df['end_date'] = pd.to_datetime(df['end_date'])
                
                return df
                
        except Exception as e:
            logger.error(f"Failed to load market regimes: {str(e)}")
            return pd.DataFrame()
    
    def render_performance_summary(self, results: Dict):
        """Render performance summary cards"""
        
        if not results:
            st.warning("No backtest results selected")
            return
        
        main = results['main_results']
        
        # Performance metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📈 Total Return", 
                value=f"{main['total_return']:.2%}",
                delta=f"vs Benchmark: {main.get('excess_return', 0):.2%}"
            )
            st.metric(
                label="📊 Sharpe Ratio",
                value=f"{main['sharpe_ratio']:.2f}",
                delta="Risk-Adjusted" if main['sharpe_ratio'] > 1.0 else None
            )
        
        with col2:
            st.metric(
                label="💰 Annualized Return",
                value=f"{main['annualized_return']:.2%}",
                delta=f"Volatility: {main['volatility']:.1%}"
            )
            st.metric(
                label="⚠️ Max Drawdown",
                value=f"{main['max_drawdown']:.2%}",
                delta=f"{main['max_drawdown_duration']} days",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                label="🎯 Win Rate",
                value=f"{main['win_rate']:.1%}",
                delta=f"{main['profitable_trades']}/{main['total_trades']} trades"
            )
            st.metric(
                label="💡 Profit Factor",
                value=f"{main['profit_factor']:.2f}",
                delta="Good" if main['profit_factor'] > 1.5 else "Acceptable" if main['profit_factor'] > 1.0 else "Poor"
            )
        
        with col4:
            st.metric(
                label="🏦 Final Value",
                value=f"${main['final_portfolio_value']:,.0f}",
                delta=f"${main['final_portfolio_value'] - 100000:,.0f}"  # Assuming $100k initial
            )
            st.metric(
                label="📅 Avg Hold Days",
                value=f"{main['avg_holding_days']:.1f}",
                delta=f"Total Fees: ${main['total_fees']:.0f}"
            )
    
    def render_equity_curve(self, results: Dict):
        """Render equity curve with benchmark comparison"""
        
        if not results or results['portfolio_values'].empty:
            st.warning("No portfolio data available")
            return
        
        portfolio_df = results['portfolio_values']
        main = results['main_results']
        
        # Get benchmark data for comparison
        start_date = portfolio_df['date'].min().strftime('%Y-%m-%d')
        end_date = portfolio_df['date'].max().strftime('%Y-%m-%d')
        spy_data = self.get_benchmark_data('SPY', start_date, end_date)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio Value vs SPY Benchmark', 'Drawdown Analysis'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Portfolio equity curve
        fig.add_trace(
            go.Scatter(
                x=portfolio_df['date'],
                y=portfolio_df['portfolio_value'],
                name='Strategy Portfolio',
                line=dict(color='#2E86C1', width=2),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                            'Date: %{x}<br>' +
                            'Value: $%{y:,.0f}<br>' +
                            '<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Benchmark comparison
        if not spy_data.empty:
            # Normalize SPY to same starting value
            initial_value = portfolio_df['portfolio_value'].iloc[0]
            spy_normalized = initial_value * (1 + spy_data['cumulative_return'])
            
            fig.add_trace(
                go.Scatter(
                    x=spy_data['date'],
                    y=spy_normalized,
                    name='SPY Benchmark',
                    line=dict(color='#E74C3C', width=2, dash='dash'),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Date: %{x}<br>' +
                                'Value: $%{y:,.0f}<br>' +
                                '<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=portfolio_df['date'],
                y=portfolio_df['drawdown'] * 100,  # Convert to percentage
                name='Drawdown %',
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.3)',
                line=dict(color='#E74C3C', width=1),
                hovertemplate='<b>Drawdown</b><br>' +
                            'Date: %{x}<br>' +
                            'Drawdown: %{y:.2f}%<br>' +
                            '<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add market regime shading
        regimes = self.get_market_regimes()
        if not regimes.empty:
            for _, regime in regimes.iterrows():
                regime_start = max(regime['start_date'], portfolio_df['date'].min())
                regime_end = min(regime['end_date'] if pd.notna(regime['end_date']) else portfolio_df['date'].max(), 
                               portfolio_df['date'].max())
                
                if regime_start <= regime_end:
                    fig.add_vrect(
                        x0=regime_start, x1=regime_end,
                        fillcolor="rgba(128, 128, 128, 0.1)",
                        layer="below", line_width=0,
                        annotation_text=regime['regime_name'],
                        annotation_position="top left",
                        row=1, col=1
                    )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            title=f"Backtest Performance: {main['config_name']}",
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_trades_analysis(self, results: Dict):
        """Render detailed trades analysis"""
        
        if not results or results['trades'].empty:
            st.warning("No trades data available")
            return
        
        trades_df = results['trades']
        
        # Trade distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Trade P&L Distribution")
            
            fig = px.histogram(
                trades_df,
                x='pnl_percent',
                nbins=30,
                title="Trade Returns Distribution",
                labels={'pnl_percent': 'Return %', 'count': 'Number of Trades'},
                color_discrete_sequence=['#3498DB']
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Breakeven")
            fig.add_vline(x=trades_df['pnl_percent'].mean(), line_dash="dash", line_color="green", 
                         annotation_text=f"Avg: {trades_df['pnl_percent'].mean():.2%}")
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Holding Period Analysis")
            
            fig = px.scatter(
                trades_df,
                x='holding_days',
                y='pnl_percent',
                size='position_value',
                color='entry_regime',
                title="Return vs Holding Period",
                labels={'holding_days': 'Days Held', 'pnl_percent': 'Return %'},
                hover_data=['symbol', 'entry_date', 'exit_reason']
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Top winners and losers
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("🏆 Top 10 Winning Trades")
            winners = trades_df.nlargest(10, 'net_pnl')[['symbol', 'entry_date', 'exit_date', 'net_pnl', 'pnl_percent', 'holding_days']]
            winners['net_pnl'] = winners['net_pnl'].apply(lambda x: f"${x:.2f}")
            winners['pnl_percent'] = winners['pnl_percent'].apply(lambda x: f"{x:.2%}")
            st.dataframe(winners, use_container_width=True)
        
        with col4:
            st.subheader("📉 Top 10 Losing Trades")
            losers = trades_df.nsmallest(10, 'net_pnl')[['symbol', 'entry_date', 'exit_date', 'net_pnl', 'pnl_percent', 'holding_days']]
            losers['net_pnl'] = losers['net_pnl'].apply(lambda x: f"${x:.2f}")
            losers['pnl_percent'] = losers['pnl_percent'].apply(lambda x: f"{x:.2%}")
            st.dataframe(losers, use_container_width=True)
    
    def render_regime_analysis(self, results: Dict):
        """Render market regime performance analysis"""
        
        if not results or results['trades'].empty:
            st.warning("No trades data available for regime analysis")
            return
        
        trades_df = results['trades']
        
        # Group by regime
        regime_stats = trades_df.groupby('entry_regime').agg({
            'net_pnl': ['count', 'sum', 'mean'],
            'pnl_percent': ['mean', 'std'],
            'holding_days': 'mean'
        }).round(4)
        
        regime_stats.columns = ['Total_Trades', 'Total_PnL', 'Avg_PnL', 'Avg_Return_Pct', 'Return_Volatility', 'Avg_Holding_Days']
        regime_stats['Win_Rate'] = trades_df.groupby('entry_regime').apply(lambda x: (x['net_pnl'] > 0).mean())
        
        st.subheader("Performance by Market Regime")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Regime performance table
            display_stats = regime_stats.copy()
            display_stats['Total_PnL'] = display_stats['Total_PnL'].apply(lambda x: f"${x:.0f}")
            display_stats['Avg_PnL'] = display_stats['Avg_PnL'].apply(lambda x: f"${x:.2f}")
            display_stats['Avg_Return_Pct'] = display_stats['Avg_Return_Pct'].apply(lambda x: f"{x:.2%}")
            display_stats['Win_Rate'] = display_stats['Win_Rate'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(display_stats, use_container_width=True)
        
        with col2:
            # Regime performance chart
            fig = px.bar(
                x=regime_stats.index,
                y=regime_stats['Avg_Return_Pct'],
                title="Average Return by Market Regime",
                labels={'y': 'Average Return %', 'x': 'Market Regime'},
                color=regime_stats['Avg_Return_Pct'],
                color_continuous_scale=['red', 'yellow', 'green']
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_run_new_backtest(self):
        """Render interface to run new backtest"""
        
        st.subheader("🚀 Run New Backtest")
        
        with st.form("new_backtest_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                config_name = st.text_input("Strategy Name", value="Custom Strategy Test")
                start_date = st.date_input("Start Date", value=datetime(2022, 1, 1))
                end_date = st.date_input("End Date", value=datetime(2024, 6, 30))
                initial_capital = st.number_input("Initial Capital ($)", value=100000, min_value=10000, step=10000)
                
            with col2:
                strategy_type = st.selectbox("Strategy Type", ["signal_based", "momentum", "mean_reversion"])
                max_position_size = st.slider("Max Position Size (%)", min_value=1, max_value=20, value=8, step=1) / 100
                max_positions = st.number_input("Max Positions", value=12, min_value=1, max_value=50, step=1)
                transaction_costs = st.slider("Transaction Costs (%)", min_value=0.01, max_value=1.0, value=0.08, step=0.01) / 100
            
            # Stock selection
            available_stocks = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V',
                'PG', 'UNH', 'HD', 'MA', 'DIS', 'ADBE', 'CRM', 'NFLX', 'KO', 'PEP'
            ]
            
            selected_symbols = st.multiselect(
                "Select Stocks for Backtesting",
                available_stocks,
                default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
            )
            
            benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "IWM", "RSP", "VTI"], index=0)
            
            submitted = st.form_submit_button("🚀 Run Backtest")
            
            if submitted and selected_symbols:
                config_data = {
                    'config_name': config_name,
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'initial_capital': initial_capital,
                    'strategy_type': strategy_type,
                    'max_position_size': max_position_size,
                    'max_positions': max_positions,
                    'transaction_costs': transaction_costs,
                    'slippage_rate': 0.0005,
                    'commission_per_trade': 1.0,
                    'signal_threshold': 0.3
                }
                
                with st.spinner("Running backtest... This may take a few minutes."):
                    try:
                        results = run_backtest_analysis(config_data, selected_symbols, benchmark)
                        
                        st.success(f"✅ Backtest completed successfully!")
                        st.info(f"📊 Results: {results.total_return:.2%} return, {results.sharpe_ratio:.2f} Sharpe ratio")
                        st.info(f"💾 Saved as Result ID: {results.result_id}")
                        
                        # Refresh the page to show new results
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Backtest failed: {str(e)}")
                        logger.error(f"Backtest failed: {str(e)}")
    
    def render_backtesting_tab(self):
        """Main render function for the backtesting tab"""
        
        st.header("🔬 Backtesting Analysis")
        st.markdown("Comprehensive backtesting analysis with market regime comparison and benchmark performance")
        
        # Get available results
        available_results = self.get_available_results()
        
        if available_results.empty:
            st.warning("No backtest results found. Run a new backtest to get started!")
            self.render_run_new_backtest()
            return
        
        # Sidebar for result selection
        st.sidebar.markdown("### 📊 Backtest Results")
        
        # Format results for display
        result_options = {}
        for _, row in available_results.iterrows():
            display_name = f"{row['config_name']} ({row['start_date']} to {row['end_date']}) - {row['total_return']:.2%}"
            result_options[display_name] = row['result_id']
        
        selected_display = st.sidebar.selectbox(
            "Select Backtest Result:",
            list(result_options.keys())
        )
        
        if selected_display:
            selected_result_id = result_options[selected_display]
            
            # Load detailed results
            with st.spinner("Loading backtest details..."):
                detailed_results = self.get_backtest_details(selected_result_id)
            
            if detailed_results:
                # Create tabs for different analysis views
                analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4, analysis_tab5 = st.tabs([
                    "📈 Performance", "💰 Trades", "🌍 Regimes", "📊 Comparison", "🚀 New Backtest"
                ])
                
                with analysis_tab1:
                    self.render_performance_summary(detailed_results)
                    st.markdown("---")
                    self.render_equity_curve(detailed_results)
                
                with analysis_tab2:
                    self.render_trades_analysis(detailed_results)
                
                with analysis_tab3:
                    self.render_regime_analysis(detailed_results)
                
                with analysis_tab4:
                    st.subheader("📊 Multi-Benchmark Comparison")
                    # Placeholder for multi-benchmark comparison
                    st.info("Multi-benchmark comparison coming soon!")
                
                with analysis_tab5:
                    self.render_run_new_backtest()
            
            else:
                st.error("Failed to load backtest details")
        
        else:
            st.info("Please select a backtest result to analyze")

# Global instance
backtesting_dashboard = BacktestingDashboard()