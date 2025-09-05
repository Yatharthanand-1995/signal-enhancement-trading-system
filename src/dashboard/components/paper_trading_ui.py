"""
Paper Trading Dashboard UI Components
Streamlit UI for paper trading functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from .paper_trading import PaperTradingEngine
from .signal_integration import SignalTradingIntegrator
from .historical_signal_trader import HistoricalSignalTrader
from .enhanced_analytics import EnhancedAnalytics
from .target_price_manager import TargetPriceManager
from .target_price_ui import render_target_price_monitor
import logging

logger = logging.getLogger(__name__)

class PaperTradingUI:
    """Paper Trading Dashboard UI"""
    
    def __init__(self):
        # Initialize paper trading engine in session state
        if 'paper_trading_engine' not in st.session_state:
            st.session_state.paper_trading_engine = PaperTradingEngine()
        
        self.engine = st.session_state.paper_trading_engine
        
        # Initialize signal integrator
        if 'signal_integrator' not in st.session_state:
            st.session_state.signal_integrator = SignalTradingIntegrator(self.engine)
        
        self.integrator = st.session_state.signal_integrator
        
        # Initialize historical signal trader
        if 'historical_trader' not in st.session_state:
            st.session_state.historical_trader = HistoricalSignalTrader(self.engine)
        
        self.historical_trader = st.session_state.historical_trader
        
        # Initialize enhanced analytics
        if 'enhanced_analytics' not in st.session_state:
            st.session_state.enhanced_analytics = EnhancedAnalytics(self.engine)
        
        self.analytics = st.session_state.enhanced_analytics
        
        # Initialize target price manager
        if 'target_price_manager' not in st.session_state:
            st.session_state.target_price_manager = TargetPriceManager(self.engine)
        
        self.target_manager = st.session_state.target_price_manager
    
    def render_sidebar_summary(self):
        """Render paper trading summary in sidebar"""
        with st.sidebar:
            st.markdown("---")
            st.subheader("📈 Paper Trading")
            
            # Get performance metrics
            metrics = self.engine.get_performance_metrics()
            
            # Portfolio value with color coding
            current_value = metrics['current_value']
            total_return = metrics['total_return']
            
            if total_return >= 0:
                color = "🟢"
                value_color = "#22C55E"
            else:
                color = "🔴"
                value_color = "#EF4444"
            
            # Display key metrics
            st.markdown(f"""
            <div style="background-color: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                <div style="font-size: 14px; color: #94A3B8; margin-bottom: 4px;">Portfolio Value</div>
                <div style="font-size: 18px; font-weight: 600; color: {value_color};">
                    ${current_value:,.0f} {color}
                </div>
                <div style="font-size: 12px; color: #64748B;">
                    Return: {total_return:+.1f}% | P&L: ${metrics['total_pnl']:+,.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Positions", metrics['positions_count'], help="Active positions")
                st.metric("Win Rate", f"{metrics['win_rate']:.1f}%", help="Winning trades percentage")
            
            with col2:
                st.metric("Cash", f"${metrics['cash']:,.0f}", help="Available cash")
                st.metric("Max DD", f"-{metrics['max_drawdown']:.1f}%", help="Maximum drawdown")
            
            # Controls
            st.markdown("**Controls**")
            if st.button("🔄 Update Positions", help="Update all positions with current prices"):
                with st.spinner("Updating positions..."):
                    self.engine.update_positions()
                st.success("Positions updated!")
                st.rerun()
            
            if st.button("⚠️ Reset Portfolio", help="Reset paper trading portfolio"):
                if st.confirm("Reset paper trading portfolio? This cannot be undone."):
                    self.engine.reset_portfolio()
                    st.success("Portfolio reset!")
                    st.rerun()
    
    def render_main_dashboard(self):
        """Render main paper trading dashboard"""
        st.title("📈 Paper Trading Dashboard")
        st.markdown("*Validate signals with risk-free trading simulation*")
        
        # Performance overview
        metrics = self.engine.get_performance_metrics()
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_value = metrics['current_value']
            delta_value = current_value - self.engine.initial_capital
            st.metric(
                "Portfolio Value", 
                f"${current_value:,.0f}", 
                delta=f"${delta_value:+,.0f}",
                help="Current total portfolio value"
            )
        
        with col2:
            total_return = metrics['total_return']
            st.metric(
                "Total Return", 
                f"{total_return:+.2f}%", 
                delta=None,
                help="Total return since inception"
            )
        
        with col3:
            st.metric(
                "Win Rate", 
                f"{metrics['win_rate']:.1f}%", 
                delta=f"{metrics['winning_trades']}/{metrics['total_trades']} trades",
                help="Percentage of profitable trades"
            )
        
        with col4:
            st.metric(
                "Max Drawdown", 
                f"-{metrics['max_drawdown']:.2f}%", 
                delta=None,
                help="Maximum portfolio decline from peak"
            )
        
        # Portfolio composition chart
        if len(self.engine.positions) > 0:
            st.subheader("📊 Portfolio Composition")
            
            positions_df = self.engine.get_positions_summary()
            
            # Create pie chart of position values
            fig = px.pie(
                positions_df,
                values='Value',
                names='Symbol',
                title="Position Allocation",
                hover_data=['Return %', 'P&L']
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tabs for detailed views
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Current Positions", 
            "🎯 Target Price Monitor",
            "📅 Yesterday's Signals", 
            "🤖 Live Signal Trading", 
            "Trade History", 
            "Performance Analytics",
            "📊 Advanced Analytics"
        ])
        
        with tab1:
            self.render_positions_table()
        
        with tab2:
            render_target_price_monitor(self.target_manager)
        
        with tab3:
            self.render_historical_signal_trading()
        
        with tab4:
            self.render_signal_trading()
        
        with tab5:
            self.render_trade_history()
        
        with tab6:
            self.render_performance_analytics()
        
        with tab7:
            self.render_advanced_analytics()
    
    def render_positions_table(self):
        """Render current positions table"""
        positions_df = self.engine.get_positions_summary()
        
        if positions_df.empty:
            st.info("No current positions. Positions will appear here when signals trigger trades.")
            return
        
        st.subheader("📋 Current Positions")
        
        # Format the dataframe for better display
        display_df = positions_df.copy()
        display_df['Entry Price'] = display_df['Entry Price'].apply(lambda x: f"${x:.2f}")
        display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"${x:.2f}")
        display_df['P&L'] = display_df['P&L'].apply(lambda x: f"${x:+,.2f}")
        display_df['Value'] = display_df['Value'].apply(lambda x: f"${x:,.2f}")
        display_df['Return %'] = display_df['Return %'].apply(lambda x: f"{x:+.2f}%")
        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.2f}")
        
        # Color code the return column
        def color_returns(val):
            if val.startswith('+'):
                return 'color: #22C55E'  # Green for positive
            elif val.startswith('-'):
                return 'color: #EF4444'  # Red for negative
            return ''
        
        styled_df = display_df.style.applymap(color_returns, subset=['Return %', 'P&L'])
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Shares": st.column_config.NumberColumn("Shares", format="%.0f"),
                "Entry Price": st.column_config.TextColumn("Entry Price"),
                "Current Price": st.column_config.TextColumn("Current Price"),
                "Entry Date": st.column_config.TextColumn("Entry Date"),
                "Days Held": st.column_config.NumberColumn("Days", format="%.0f"),
                "Return %": st.column_config.TextColumn("Return %"),
                "P&L": st.column_config.TextColumn("P&L"),
                "Value": st.column_config.TextColumn("Value"),
                "Signal": st.column_config.TextColumn("Signal"),
                "Confidence": st.column_config.TextColumn("Confidence")
            }
        )
        
        # Position actions
        if not positions_df.empty:
            st.markdown("**Actions**")
            symbol_options = positions_df['Symbol'].tolist()
            if symbol_options:  # Additional safety check
                selected_symbol = st.selectbox(
                    "Select position to manage:",
                    options=symbol_options,
                    index=0,  # Explicit index
                    help="Select a position to view actions"
                )
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button(f"🔄 Sell {selected_symbol}", type="secondary"):
                        if self.engine.execute_sell_order(selected_symbol, "MANUAL_SELL"):
                            st.success(f"Successfully sold {selected_symbol}")
                            st.rerun()
                        else:
                            st.error(f"Failed to sell {selected_symbol}")
                
                with col2:
                    position_info = positions_df[positions_df['Symbol'] == selected_symbol].iloc[0]
                    st.info(f"Entry: {position_info['Entry Date']} | Signal: {position_info['Signal']} | Confidence: {position_info['Confidence']}")
            else:
                st.info("No positions available for management")
    
    def render_historical_signal_trading(self):
        """Render historical signal trading interface"""
        st.subheader("📅 Yesterday's Signal Trading")
        
        st.info("""
        **Realistic Paper Trading:**
        - Trade based on yesterday's signals (like real trading)
        - Store daily signals automatically for next-day execution
        - Simulate how you would actually trade in practice
        """)
        
        # Signal history summary
        history_summary = self.historical_trader.get_signal_history_summary()
        
        st.markdown("**📊 Signal History Status**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Days Stored", history_summary['total_days'])
        
        with col2:
            st.metric("Total Signals", history_summary['total_signals'])
        
        with col3:
            avg_signals = history_summary.get('avg_signals_per_day', 0)
            st.metric("Avg/Day", f"{avg_signals:.1f}")
        
        with col4:
            symbols_count = len(history_summary.get('symbols_tracked', []))
            st.metric("Symbols", symbols_count)
        
        if history_summary['date_range']:
            st.info(f"📅 Signal history available: {history_summary['date_range']}")
        
        st.divider()
        
        # Yesterday's signals execution
        st.markdown("**🚀 Execute Yesterday's Signals**")
        
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        yesterdays_signals = self.historical_trader.get_yesterdays_signals()
        
        if yesterdays_signals:
            st.success(f"✅ Found {len(yesterdays_signals)} signals from {yesterday}")
            
            # Show yesterday's signals preview
            with st.expander("📋 Preview Yesterday's Signals"):
                signal_preview = []
                for symbol, data in list(yesterdays_signals.items())[:10]:  # Show first 10
                    signal_preview.append({
                        'Symbol': symbol,
                        'Signal': data.get('signal', 'N/A'),
                        'Confidence': f"{data.get('confidence', 0):.1%}",
                        'Close Price': f"${data.get('close_price', 0):.2f}"
                    })
                
                if signal_preview:
                    st.dataframe(pd.DataFrame(signal_preview), hide_index=True)
                
                if len(yesterdays_signals) > 10:
                    st.info(f"... and {len(yesterdays_signals) - 10} more signals")
            
            # Execution controls
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🧪 Dry Run", help="Preview what trades would be executed"):
                    with st.spinner("Running dry execution..."):
                        result = self.historical_trader.execute_yesterdays_signals(dry_run=True)
                    
                    if result['status'] == 'success':
                        st.success(f"✅ Dry run complete: {result['trades_executed']} trades would be executed")
                        
                        if result['results']:
                            st.markdown("**Proposed Trades:**")
                            for trade in result['results']:
                                action = trade.get('action_type', 'N/A')
                                symbol = trade.get('symbol', 'N/A')
                                conf = trade.get('confidence', 0)
                                st.write(f"• {action} {symbol} (confidence: {conf:.1%})")
                    else:
                        st.error(f"❌ Dry run failed: {result.get('error', 'Unknown error')}")
            
            with col2:
                if st.button("💰 Execute Trades", type="primary", help="Execute actual trades based on yesterday's signals"):
                    with st.spinner("Executing yesterday's signals..."):
                        result = self.historical_trader.execute_yesterdays_signals(dry_run=False)
                    
                    if result['status'] == 'success':
                        st.success(f"✅ Execution complete: {result['trades_executed']} trades executed!")
                        
                        if result['results']:
                            st.markdown("**Executed Trades:**")
                            for trade in result['results']:
                                action = trade.get('action_type', 'N/A')
                                symbol = trade.get('symbol', 'N/A')
                                price = trade.get('current_price', 0)
                                st.write(f"• {action} {symbol} at ${price:.2f}")
                        
                        if result['errors']:
                            st.warning("⚠️ Some errors occurred:")
                            for error in result['errors']:
                                st.write(f"• {error}")
                        
                        # Refresh the page to show updated positions
                        st.rerun()
                    
                    else:
                        st.error(f"❌ Execution failed: {result.get('error', 'Unknown error')}")
        
        else:
            st.warning(f"⚠️ No signals found for {yesterday}")
            st.info("💡 Signals are stored when you visit the 'Market Signals' dashboard section.")
        
        st.divider()
        
        # Historical simulation
        st.markdown("**📈 Historical Simulation**")
        
        available_dates = self.historical_trader.get_available_dates()
        
        if len(available_dates) == 0:
            st.info("📅 **No historical signals available yet**")
            st.markdown("""
            Historical signals are automatically stored when you visit the **📊 Market Signals** section.
            
            **To get started:**
            1. Visit the Market Signals dashboard to generate and store today's signals
            2. Return here tomorrow to see yesterday's signals for trading
            3. Or check back later if signals are being generated now
            """)
            return
        
        elif len(available_dates) == 1:
            st.info(f"📅 **Only one day of signals available:** {available_dates[0]}")
            st.markdown("Visit the Market Signals section daily to build up historical data for simulation.")
            return
        
        elif len(available_dates) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                if available_dates and len(available_dates) > 0:  # Additional safety check
                    start_date = st.selectbox(
                        "Start Date",
                        options=available_dates,
                        index=0,
                        help="First date to simulate from"
                    )
                else:
                    st.warning("No start date options available")
                    return
            
            with col2:
                if available_dates and len(available_dates) > 0:  # Additional safety check
                    # Safe index calculation with double bounds checking
                    end_index = max(0, min(len(available_dates) - 1, len(available_dates) - 1))
                    if end_index >= 0 and end_index < len(available_dates):
                        end_date = st.selectbox(
                            "End Date", 
                            options=available_dates,
                            index=end_index,
                            help="Last date to simulate to"
                        )
                    else:
                        st.warning("Unable to determine valid end date index")
                        return
                else:
                    st.warning("No end date options available")
                    return
            
            if st.button("🎮 Run Historical Simulation", help="Simulate trading over the selected date range"):
                if start_date <= end_date:
                    with st.spinner(f"Simulating trading from {start_date} to {end_date}..."):
                        # Save current state warning
                        st.warning("⚠️ This will modify your current portfolio state. Consider backing up your data.")
                        
                        result = self.historical_trader.simulate_historical_trading(start_date, end_date)
                    
                    if result['status'] == 'success':
                        st.success(f"✅ Simulation complete!")
                        
                        # Show results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Trades", result['total_trades'])
                        
                        with col2:
                            st.metric("Total Return", f"{result['total_return']:+.2f}%")
                        
                        with col3:
                            st.metric("P&L", f"${result['total_pnl']:+,.2f}")
                        
                        # Daily results chart
                        if result['daily_results']:
                            daily_df = pd.DataFrame(result['daily_results'])
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=daily_df['date'],
                                y=daily_df['portfolio_value'],
                                mode='lines+markers',
                                name='Portfolio Value',
                                line=dict(color='#3B82F6', width=2)
                            ))
                            
                            fig.update_layout(
                                title="Historical Simulation - Portfolio Value",
                                xaxis_title="Date",
                                yaxis_title="Portfolio Value ($)",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        st.error(f"❌ Simulation failed: {result.get('error', 'Unknown error')}")
                
                else:
                    st.error("❌ Start date must be before or equal to end date")
        
        else:
            st.info("📅 Need at least 2 days of signal history for simulation")
        
        st.divider()
        
        # Store current signals for tomorrow
        st.markdown("**💾 Store Today's Signals**")
        st.info("Visit the 'Market Signals' dashboard to automatically store today's signals for tomorrow's trading.")
        
        if st.button("🔄 Manual Signal Storage", help="Manually trigger signal storage (for testing)"):
            # This would typically be called from the main dashboard
            # For now, show a placeholder
            st.success("✅ Signal storage would be triggered from the main dashboard")
            st.info("Switch to 'Market Signals' tab to generate and store signals automatically")
    
    def render_signal_trading(self):
        """Render signal-based trading interface"""
        st.subheader("🤖 Automated Signal Trading")
        
        st.info("""
        **How it works:**
        - Paper trading automatically processes signals from the main dashboard
        - Trades are executed when signals meet confidence thresholds
        - You can adjust settings and manually process signals here
        """)
        
        # Trading settings
        st.markdown("**⚙️ Trading Settings**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence threshold
            new_confidence = st.slider(
                "Minimum Confidence for Trades",
                min_value=0.1,
                max_value=1.0,
                value=self.engine.min_confidence,
                step=0.05,
                help="Minimum signal confidence required to execute trades"
            )
            
            if new_confidence != self.engine.min_confidence:
                self.engine.min_confidence = new_confidence
                self.engine.save_state()
                st.success(f"Updated minimum confidence to {new_confidence:.1%}")
        
        with col2:
            # Max position percentage
            new_max_pos = st.slider(
                "Max Position Size %",
                min_value=0.01,
                max_value=0.25,
                value=self.engine.max_position_pct,
                step=0.01,
                help="Maximum percentage of portfolio per position"
            )
            
            if new_max_pos != self.engine.max_position_pct:
                self.engine.max_position_pct = new_max_pos
                self.engine.save_state()
                st.success(f"Updated max position size to {new_max_pos:.1%}")
        
        st.divider()
        
        # Manual signal processing
        st.markdown("**🔄 Manual Signal Processing**")
        st.info("To connect with live signals, switch to the 'Market Signals' tab and return here to see automatic trading.")
        
        # Signal integration status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Signals Tracked", 
                len(self.integrator.last_processed_signals),
                help="Number of signals currently being tracked"
            )
        
        with col2:
            st.metric(
                "Min Confidence", 
                f"{self.engine.min_confidence:.1%}",
                help="Current minimum confidence threshold"
            )
        
        with col3:
            st.metric(
                "Auto Trading", 
                "✅ Active" if len(self.integrator.last_processed_signals) > 0 else "⏸️ Standby",
                help="Status of automatic trading based on signals"
            )
        
        # Recent signal processing results
        if hasattr(self.integrator, 'last_processing_results'):
            st.markdown("**📊 Recent Signal Processing**")
            
            results = self.integrator.last_processing_results
            
            if results.get('status') == 'success':
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"✅ Processed {results.get('signals_processed', 0)} signals")
                    st.info(f"🔄 Executed {results.get('trades_executed', 0)} trades")
                
                with col2:
                    if results.get('results'):
                        st.markdown("**Trade Actions:**")
                        for result in results['results'][-5:]:  # Show last 5
                            action_type = result.get('action_type', 'NO_ACTION')
                            symbol = result.get('symbol', 'N/A')
                            if action_type in ['BUY', 'SELL']:
                                st.write(f"• {action_type} {symbol}")
            
            elif results.get('status') == 'error':
                st.error(f"❌ Error: {results.get('error', 'Unknown error')}")
        
        # Manual controls
        st.markdown("**🎮 Manual Controls**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Update All Positions", help="Update all positions with current market prices"):
                with st.spinner("Updating positions..."):
                    self.engine.update_positions()
                st.success("Positions updated with current prices!")
                st.rerun()
        
        with col2:
            if st.button("💰 Process Demo Signals", help="Process demo signals for testing"):
                with st.spinner("Processing demo signals..."):
                    # Create sample signals for testing
                    demo_signals = pd.DataFrame({
                        'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
                        'Signal': ['BUY', 'SELL', 'NEUTRAL'],
                        'Confidence': [0.75, 0.65, 0.45],
                        'Strength': [0.8, 0.7, 0.3],
                        'Close': [150.0, 300.0, 2500.0]
                    })
                    
                    results = self.integrator.process_signal_dataframe(demo_signals)
                    self.integrator.last_processing_results = results
                    
                    if results['status'] == 'success':
                        st.success(f"✅ Demo processing complete! Executed {results['trades_executed']} trades")
                    else:
                        st.error(f"❌ Demo processing failed: {results.get('error', 'Unknown error')}")
                
                st.rerun()
    
    def render_trade_history(self):
        """Render trade history"""
        st.subheader("📜 Trade History")
        
        trade_df = self.engine.get_trade_history_df(limit=100)
        
        if trade_df.empty:
            st.info("No trades yet. Trade history will appear here after signal-based trades are executed.")
            return
        
        # Format for display
        display_df = trade_df.copy()
        display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
        display_df['Shares'] = display_df['Shares'].apply(lambda x: f"{x:.0f}")
        
        # Format P&L column
        if 'P&L' in display_df.columns:
            display_df['P&L'] = display_df['P&L'].apply(
                lambda x: f"${x:+,.2f}" if pd.notna(x) else "—"
            )
        
        # Color code actions
        def color_actions(val):
            if val == 'BUY':
                return 'color: #22C55E'  # Green for buy
            elif val == 'SELL':
                return 'color: #EF4444'  # Red for sell
            return ''
        
        styled_df = display_df.style.applymap(color_actions, subset=['Action'])
        if 'P&L' in display_df.columns:
            def color_pnl(val):
                if val.startswith('$+'):
                    return 'color: #22C55E'  # Green for profit
                elif val.startswith('$-'):
                    return 'color: #EF4444'  # Red for loss
                return ''
            styled_df = styled_df.applymap(color_pnl, subset=['P&L'])
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Timestamp": st.column_config.TextColumn("Date", width="medium"),
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Action": st.column_config.TextColumn("Action", width="small"),
                "Shares": st.column_config.TextColumn("Shares"),
                "Price": st.column_config.TextColumn("Price"),
                "P&L": st.column_config.TextColumn("P&L"),
                "Signal Info": st.column_config.TextColumn("Signal")
            }
        )
        
        # Trade statistics
        if len(self.engine.trade_history) > 0:
            st.markdown("**Trade Statistics**")
            
            buy_trades = [t for t in self.engine.trade_history if t.action == 'BUY']
            sell_trades = [t for t in self.engine.trade_history if t.action == 'SELL']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Trades", len(self.engine.trade_history))
            
            with col2:
                if sell_trades:
                    avg_pnl = np.mean([t.pnl for t in sell_trades])
                    st.metric("Avg P&L per Trade", f"${avg_pnl:+,.2f}")
                else:
                    st.metric("Avg P&L per Trade", "$0.00")
            
            with col3:
                if sell_trades:
                    profitable_trades = len([t for t in sell_trades if t.pnl > 0])
                    win_rate = (profitable_trades / len(sell_trades)) * 100
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                else:
                    st.metric("Win Rate", "—")
    
    def render_performance_analytics(self):
        """Render performance analytics"""
        st.subheader("📈 Performance Analytics")
        
        if len(self.engine.trade_history) < 2:
            st.info("More trade data needed for performance analytics.")
            return
        
        # P&L over time chart
        sell_trades = [t for t in self.engine.trade_history if t.action == 'SELL']
        
        if sell_trades:
            # Create cumulative P&L chart
            pnl_data = []
            cumulative_pnl = 0
            
            for trade in sell_trades:
                cumulative_pnl += trade.pnl
                pnl_data.append({
                    'Date': trade.timestamp,
                    'Trade P&L': trade.pnl,
                    'Cumulative P&L': cumulative_pnl,
                    'Symbol': trade.symbol
                })
            
            pnl_df = pd.DataFrame(pnl_data)
            
            # Cumulative P&L line chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=pnl_df['Date'],
                y=pnl_df['Cumulative P&L'],
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='#3B82F6', width=2),
                marker=dict(size=6)
            ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            fig.update_layout(
                title="Cumulative P&L Over Time",
                xaxis_title="Date",
                yaxis_title="P&L ($)",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade performance distribution
            col1, col2 = st.columns(2)
            
            with col1:
                # P&L distribution histogram
                fig_hist = px.histogram(
                    pnl_df,
                    x='Trade P&L',
                    nbins=20,
                    title="P&L Distribution per Trade",
                    color_discrete_sequence=['#3B82F6']
                )
                fig_hist.add_vline(x=0, line_dash="dash", line_color="gray")
                fig_hist.update_layout(height=350)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Performance by symbol
                symbol_pnl = pnl_df.groupby('Symbol')['Trade P&L'].agg(['sum', 'count']).reset_index()
                symbol_pnl.columns = ['Symbol', 'Total P&L', 'Trades']
                
                fig_symbol = px.bar(
                    symbol_pnl.head(10),
                    x='Symbol',
                    y='Total P&L',
                    title="P&L by Symbol (Top 10)",
                    color='Total P&L',
                    color_continuous_scale='RdYlGn'
                )
                fig_symbol.update_layout(height=350)
                st.plotly_chart(fig_symbol, use_container_width=True)
        
        # Risk metrics
        st.markdown("**Risk Metrics**")
        metrics = self.engine.get_performance_metrics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Max Drawdown",
                f"-{metrics['max_drawdown']:.2f}%",
                help="Maximum decline from peak portfolio value"
            )
        
        with col2:
            # Calculate Sharpe ratio approximation (simplified)
            if len(sell_trades) > 1:
                returns = [t.pnl / self.engine.initial_capital for t in sell_trades]
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_approx = (avg_return / std_return) if std_return > 0 else 0
                st.metric(
                    "Return/Risk Ratio",
                    f"{sharpe_approx:.2f}",
                    help="Simplified risk-adjusted return measure"
                )
            else:
                st.metric("Return/Risk Ratio", "—")
        
        with col3:
            # Position concentration
            if self.engine.positions:
                position_values = [pos.get_position_value() for pos in self.engine.positions.values()]
                max_position_pct = (max(position_values) / metrics['current_value']) * 100
                st.metric(
                    "Max Position %",
                    f"{max_position_pct:.1f}%",
                    help="Largest single position as % of portfolio"
                )
            else:
                st.metric("Max Position %", "—")
    
    def render_advanced_analytics(self):
        """Render advanced analytics dashboard"""
        st.subheader("📊 Advanced Analytics & Insights")
        
        # Enhanced analytics
        self.analytics.render_advanced_metrics_dashboard()
        
        st.divider()
        
        # Performance charts
        self.analytics.render_performance_charts()
        
        st.divider()
        
        # Generate and download report
        st.markdown("**📄 Performance Report**")
        
        if st.button("📊 Generate Detailed Report", help="Generate comprehensive performance report"):
            report = self.analytics.generate_performance_report()
            
            # Display report
            st.markdown("**Generated Report:**")
            st.text_area("Performance Report", report, height=300)
            
            # Download button
            st.download_button(
                label="💾 Download Report",
                data=report,
                file_name=f"paper_trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

def render_paper_trading_dashboard():
    """Main function to render paper trading dashboard"""
    ui = PaperTradingUI()
    ui.render_main_dashboard()

def render_paper_trading_sidebar():
    """Function to render paper trading sidebar"""
    ui = PaperTradingUI()
    ui.render_sidebar_summary()