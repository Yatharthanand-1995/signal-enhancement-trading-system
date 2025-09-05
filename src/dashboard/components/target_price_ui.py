"""
Target Price Monitor UI
Enhanced UI for target price tracking and live price comparison
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf

def render_target_price_monitor(target_manager):
    """Render enhanced target price monitoring interface"""
    
    st.subheader("🎯 Target Price Monitor & Live Price Comparison")
    
    # Get positions with target price info
    positions_df = target_manager.get_target_price_summary()
    
    if positions_df.empty:
        st.info("No current positions. Positions will appear here when signals trigger trades.")
        return
    
    # Control panel
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("**📊 Live Price Monitoring**")
    
    with col2:
        if st.button("🔄 Update All Prices", help="Fetch latest market prices"):
            with st.spinner("Updating prices..."):
                target_manager.paper_engine.update_positions()
            st.success("✅ Prices updated!")
            st.rerun()
    
    with col3:
        if st.button("🔍 Check Exit Conditions", help="Check all positions for exit signals"):
            results = target_manager.execute_automated_exits(dry_run=True)
            if results['positions_exited'] > 0:
                st.warning(f"⚠️ {results['positions_exited']} positions ready to exit!")
                for exit_info in results['exits_executed']:
                    st.write(f"• **{exit_info['symbol']}**: {exit_info['reason']} at ${exit_info['price']:.2f}")
            else:
                st.success("✅ No positions need to exit")
    
    # Portfolio overview
    risk_metrics = target_manager.calculate_portfolio_risk_metrics()
    
    if risk_metrics:
        st.markdown("**📈 Portfolio Overview**")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Positions", risk_metrics['total_positions'])
        
        with col2:
            st.metric("Position Value", f"${risk_metrics['total_position_value']:,.0f}")
        
        with col3:
            st.metric("Cash %", f"{risk_metrics['cash_percentage']:.1%}")
        
        with col4:
            st.metric("Largest Position", f"{risk_metrics['largest_position_pct']:.1%}")
        
        with col5:
            unrealized_pnl = risk_metrics['portfolio_unrealized_pnl']
            pnl_color = "🟢" if unrealized_pnl >= 0 else "🔴"
            st.metric("Unrealized P&L", f"${unrealized_pnl:+,.0f} {pnl_color}")
    
    st.markdown("---")
    
    # Individual position analysis
    for idx, (_, position) in enumerate(positions_df.iterrows()):
        symbol = position['Symbol']
        
        # Expandable position card
        with st.expander(
            f"**{symbol}** • {position['Return %']:+.1f}% • Target: {position['Target Progress']:.1%}", 
            expanded=(idx < 2)  # Expand first 2 positions
        ):
            
            # Main price comparison
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**💰 Price Levels**")
                entry_price = position['Entry Price']
                current_price = position['Current Price']
                target_price = position['Target Price']
                stop_loss = position['Stop Loss']
                
                # Visual price ladder
                target_reached = current_price >= target_price
                st.markdown(f"🎯 **Target**: ${target_price:.2f}" + 
                           (" ✅" if target_reached else f" (+${target_price - current_price:.2f})"))
                
                price_color = "🟢" if current_price >= entry_price else "🔴"
                st.markdown(f"📊 **Current**: ${current_price:.2f} {price_color}")
                
                st.markdown(f"📍 **Entry**: ${entry_price:.2f}")
                
                stop_distance = (current_price - stop_loss) / current_price
                stop_color = "🛑" if stop_distance <= 0.1 else "🟡" if stop_distance <= 0.2 else "🟢"
                st.markdown(f"🛡️ **Stop**: ${stop_loss:.2f} {stop_color}")
            
            with col2:
                st.markdown("**📊 Progress & Performance**")
                
                # Target progress
                progress = position['Target Progress']
                if progress >= 1.0:
                    st.success(f"🎯 Target Reached! {progress:.1%}")
                elif progress >= 0.8:
                    st.warning(f"🟡 Near Target: {progress:.1%}")
                else:
                    st.info(f"⚪ Progress: {progress:.1%}")
                
                # Progress bar
                progress_val = min(progress, 1.0)  # Cap at 100% for display
                st.progress(progress_val)
                
                # Key metrics
                st.metric("Days Held", f"{position['Days Held']} days")
                st.metric("Return %", f"{position['Return %']:+.1f}%")
                st.metric("Confidence", f"{position['Confidence']:.2f}")
            
            with col3:
                st.markdown("**💸 P&L & Actions**")
                
                # Current P&L
                return_pct = position['Return %']
                pnl_value = (current_price - entry_price) * position.get('Shares', 0)
                
                if return_pct >= 0:
                    st.success(f"💚 P&L: ${pnl_value:+,.0f}")
                else:
                    st.error(f"💔 P&L: ${pnl_value:+,.0f}")
                
                # Potential targets
                shares = position.get('Shares', 0)
                to_target = (target_price - current_price) * shares
                if to_target > 0:
                    st.metric("Potential Gain", f"${to_target:+,.0f}")
                
                # Action buttons
                if st.button(f"🚪 Sell {symbol}", key=f"sell_{symbol}", type="primary"):
                    success = target_manager.paper_engine.execute_sell_order(symbol, "MANUAL_EXIT")
                    if success:
                        st.success(f"✅ Successfully sold {symbol}!")
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to sell {symbol}")
            
            # Intraday range (if available)
            intraday_high = position.get('Intraday High', current_price)
            intraday_low = position.get('Intraday Low', current_price)
            
            if intraday_high != intraday_low:
                st.markdown("**📈 Intraday Range**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    high_gain = (intraday_high - entry_price) * shares if shares > 0 else 0
                    st.metric("High", f"${intraday_high:.2f}", 
                             delta=f"${high_gain:+,.0f}" if high_gain != 0 else None)
                
                with col2:
                    st.metric("Current", f"${current_price:.2f}", 
                             delta=f"{return_pct:+.1f}%")
                
                with col3:
                    low_loss = (intraday_low - entry_price) * shares if shares > 0 else 0
                    st.metric("Low", f"${intraday_low:.2f}", 
                             delta=f"${low_loss:+,.0f}" if low_loss != 0 else None)
    
    # Automated exit toggle
    st.markdown("---")
    st.markdown("**🤖 Automated Exit Control**")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        auto_execute = st.toggle("🤖 Enable Auto-Exit", 
                               help="Automatically execute exits when conditions are met")
    
    with col2:
        if auto_execute:
            st.warning("⚠️ Auto-exit is ENABLED - positions will be sold automatically when exit conditions are met!")
        else:
            st.info("ℹ️ Auto-exit is disabled - you must manually sell positions")
    
    # Execute automated exits if enabled
    if auto_execute:
        with st.spinner("Checking for automated exits..."):
            results = target_manager.execute_automated_exits(dry_run=False)
        
        if results['positions_exited'] > 0:
            st.success(f"🎉 Executed {results['positions_exited']} automated exits!")
            total_pnl = 0
            for exit_info in results['exits_executed']:
                pnl = exit_info.get('pnl', 0)
                total_pnl += pnl
                st.write(f"• **{exit_info['symbol']}**: {exit_info['reason']} at ${exit_info['price']:.2f} (P&L: ${pnl:+,.2f})")
            
            if total_pnl > 0:
                st.success(f"💰 Total realized P&L: ${total_pnl:+,.2f}")
                st.balloons()  # Celebration for profitable trades!
            
            st.rerun()  # Refresh to show updated positions
        
        elif results['errors']:
            st.error("⚠️ Errors during automated execution:")
            for error in results['errors']:
                st.write(f"• {error}")
        
        else:
            st.info("✅ No positions ready for automated exit")


def render_price_comparison_chart(positions_df):
    """Render price comparison chart for all positions"""
    
    if positions_df.empty:
        return
    
    st.markdown("**📊 Price Comparison Chart**")
    
    # Prepare data for chart
    chart_data = []
    
    for _, position in positions_df.iterrows():
        symbol = position['Symbol']
        entry_price = position['Entry Price']
        current_price = position['Current Price']
        target_price = position['Target Price']
        
        chart_data.extend([
            {'Symbol': symbol, 'Price Type': 'Entry', 'Price': entry_price, 'Color': 'blue'},
            {'Symbol': symbol, 'Price Type': 'Current', 'Price': current_price, 
             'Color': 'green' if current_price >= entry_price else 'red'},
            {'Symbol': symbol, 'Price Type': 'Target', 'Price': target_price, 'Color': 'purple'}
        ])
    
    chart_df = pd.DataFrame(chart_data)
    
    if not chart_df.empty:
        fig = px.bar(chart_df, x='Symbol', y='Price', color='Price Type',
                     title="Entry vs Current vs Target Prices",
                     color_discrete_map={
                         'Entry': '#3B82F6',    # Blue
                         'Current': '#10B981',   # Green  
                         'Target': '#8B5CF6'     # Purple
                     })
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)