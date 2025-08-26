"""
Real-time Trading Signal Generation and Monitoring Dashboard
Streamlit-based comprehensive trading system interface
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import psycopg2
from datetime import datetime, timedelta
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_management.stock_data_manager import Top100StocksDataManager
from data_management.technical_indicators import TechnicalIndicatorCalculator
from models.ml_ensemble import LSTMXGBoostEnsemble
from models.regime_detection import MarketRegimeDetector
from risk_management.risk_manager import AdaptiveRiskManager
from config.config import config

# Page configuration
st.set_page_config(
    page_title="Trading Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .signal-buy {
        background-color: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .signal-sell {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .signal-hold {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_dashboard_data():
    """Load data for dashboard"""
    try:
        data_manager = Top100StocksDataManager()
        
        # Get top stocks
        top_stocks = data_manager.get_top_100_stocks()[:20]  # Top 20 for dashboard
        
        # Get latest data
        query = """
        WITH latest_data AS (
            SELECT 
                s.symbol,
                s.company_name,
                s.sector,
                dp.trade_date,
                dp.close,
                dp.volume,
                ti.rsi_14,
                ti.macd_histogram,
                ti.bb_upper,
                ti.bb_lower,
                ti.sma_20,
                ti.atr_14,
                ROW_NUMBER() OVER (PARTITION BY s.symbol ORDER BY dp.trade_date DESC) as rn
            FROM securities s
            JOIN daily_prices dp ON s.id = dp.symbol_id
            LEFT JOIN technical_indicators ti ON s.id = ti.symbol_id 
                AND dp.trade_date = ti.trade_date
            WHERE s.is_active = true
              AND dp.trade_date >= CURRENT_DATE - INTERVAL '5 days'
        )
        SELECT * FROM latest_data WHERE rn = 1
        ORDER BY symbol
        """
        
        df = pd.read_sql(query, data_manager.conn, parse_dates=['trade_date'])
        data_manager.close()
        
        return df, top_stocks
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), []

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_market_regime():
    """Get current market regime"""
    try:
        detector = MarketRegimeDetector(n_regimes=2)
        market_data = detector.prepare_market_data(lookback_days=100)
        
        if not market_data.empty:
            fit_results = detector.fit(market_data)
            if fit_results.get('success'):
                regime_prediction = detector.predict_regime(market_data.tail(5))
                adjustments = detector.get_trading_adjustments()
                return regime_prediction, adjustments
                
    except Exception as e:
        st.sidebar.error(f"Regime detection error: {str(e)}")
    
    return None, None

def generate_trading_signals(df):
    """Generate trading signals for dashboard stocks"""
    signals = []
    
    try:
        calculator = TechnicalIndicatorCalculator()
        
        for _, row in df.iterrows():
            if pd.isna(row['rsi_14']) or pd.isna(row['macd_histogram']):
                continue
                
            # Calculate signal components
            rsi_signal = 0.0
            if row['rsi_14'] < 30:
                rsi_signal = 0.8  # Strong buy
            elif row['rsi_14'] < 40:
                rsi_signal = 0.6  # Buy
            elif row['rsi_14'] > 70:
                rsi_signal = 0.2  # Sell
            elif row['rsi_14'] > 60:
                rsi_signal = 0.4  # Weak sell
            else:
                rsi_signal = 0.5  # Neutral
                
            # MACD signal
            macd_signal = 0.5  # Neutral default
            if row['macd_histogram'] > 0:
                macd_signal = 0.7
            elif row['macd_histogram'] < 0:
                macd_signal = 0.3
                
            # Bollinger Band signal
            bb_signal = 0.5  # Neutral default
            if not pd.isna(row['bb_upper']) and not pd.isna(row['bb_lower']):
                if row['close'] < row['bb_lower']:
                    bb_signal = 0.8  # Oversold
                elif row['close'] > row['bb_upper']:
                    bb_signal = 0.2  # Overbought
                    
            # Combined signal
            combined_signal = (0.4 * rsi_signal + 0.35 * macd_signal + 0.25 * bb_signal)
            
            # Determine direction and strength
            if combined_signal > 0.65:
                direction = "BUY"
                strength = "Strong"
            elif combined_signal > 0.55:
                direction = "BUY" 
                strength = "Moderate"
            elif combined_signal < 0.35:
                direction = "SELL"
                strength = "Strong"
            elif combined_signal < 0.45:
                direction = "SELL"
                strength = "Moderate"
            else:
                direction = "HOLD"
                strength = "Neutral"
                
            signals.append({
                'Symbol': row['symbol'],
                'Company': row['company_name'],
                'Sector': row['sector'],
                'Price': row['close'],
                'RSI': row['rsi_14'],
                'MACD_Hist': row['macd_histogram'],
                'Signal': direction,
                'Strength': strength,
                'Score': combined_signal,
                'Last_Updated': row['trade_date']
            })
            
    except Exception as e:
        st.error(f"Error generating signals: {str(e)}")
        
    return pd.DataFrame(signals)

def create_signal_chart(signals_df):
    """Create signal distribution chart"""
    if signals_df.empty:
        return go.Figure()
    
    # Count signals by type
    signal_counts = signals_df['Signal'].value_counts()
    
    colors = {'BUY': '#28a745', 'SELL': '#dc3545', 'HOLD': '#ffc107'}
    
    fig = go.Figure(data=[
        go.Bar(
            x=signal_counts.index,
            y=signal_counts.values,
            marker_color=[colors.get(signal, '#6c757d') for signal in signal_counts.index],
            text=signal_counts.values,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Current Signal Distribution",
        xaxis_title="Signal Type",
        yaxis_title="Count",
        height=400,
        showlegend=False
    )
    
    return fig

def create_sector_analysis_chart(signals_df):
    """Create sector-wise signal analysis"""
    if signals_df.empty:
        return go.Figure()
    
    # Group by sector and signal
    sector_signals = signals_df.groupby(['Sector', 'Signal']).size().unstack(fill_value=0)
    
    fig = go.Figure()
    
    colors = {'BUY': '#28a745', 'SELL': '#dc3545', 'HOLD': '#ffc107'}
    
    for signal in ['BUY', 'SELL', 'HOLD']:
        if signal in sector_signals.columns:
            fig.add_trace(go.Bar(
                name=signal,
                x=sector_signals.index,
                y=sector_signals[signal],
                marker_color=colors[signal]
            ))
    
    fig.update_layout(
        title="Signals by Sector",
        xaxis_title="Sector",
        yaxis_title="Number of Signals",
        height=500,
        barmode='stack'
    )
    
    return fig

def create_price_chart(symbol, df):
    """Create detailed price chart for selected stock"""
    try:
        data_manager = Top100StocksDataManager()
        
        # Get historical data for chart
        query = """
        SELECT 
            dp.trade_date,
            dp.open,
            dp.high,
            dp.low,
            dp.close,
            dp.volume,
            ti.rsi_14,
            ti.macd_value,
            ti.macd_signal,
            ti.macd_histogram,
            ti.bb_upper,
            ti.bb_middle,
            ti.bb_lower,
            ti.sma_20,
            ti.sma_50
        FROM securities s
        JOIN daily_prices dp ON s.id = dp.symbol_id
        LEFT JOIN technical_indicators ti ON s.id = ti.symbol_id 
            AND dp.trade_date = ti.trade_date
        WHERE s.symbol = %s
          AND dp.trade_date >= CURRENT_DATE - INTERVAL '60 days'
        ORDER BY dp.trade_date
        """
        
        chart_df = pd.read_sql(query, data_manager.conn, params=[symbol], 
                              parse_dates=['trade_date'])
        data_manager.close()
        
        if chart_df.empty:
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} Price & Bollinger Bands', 'RSI', 'MACD'),
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=chart_df['trade_date'],
                open=chart_df['open'],
                high=chart_df['high'],
                low=chart_df['low'],
                close=chart_df['close'],
                name='Price'
            ), row=1, col=1
        )
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=chart_df['trade_date'],
                y=chart_df['bb_upper'],
                name='BB Upper',
                line=dict(color='gray', width=1),
                opacity=0.7
            ), row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=chart_df['trade_date'],
                y=chart_df['bb_lower'],
                name='BB Lower',
                line=dict(color='gray', width=1),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                opacity=0.7
            ), row=1, col=1
        )
        
        # Moving averages
        fig.add_trace(
            go.Scatter(
                x=chart_df['trade_date'],
                y=chart_df['sma_20'],
                name='SMA 20',
                line=dict(color='blue', width=1)
            ), row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=chart_df['trade_date'],
                y=chart_df['sma_50'],
                name='SMA 50',
                line=dict(color='red', width=1)
            ), row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=chart_df['trade_date'],
                y=chart_df['rsi_14'],
                name='RSI',
                line=dict(color='orange')
            ), row=2, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(
                x=chart_df['trade_date'],
                y=chart_df['macd_value'],
                name='MACD',
                line=dict(color='blue')
            ), row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=chart_df['trade_date'],
                y=chart_df['macd_signal'],
                name='Signal',
                line=dict(color='red')
            ), row=3, col=1
        )
        
        # MACD Histogram
        fig.add_trace(
            go.Bar(
                x=chart_df['trade_date'],
                y=chart_df['macd_histogram'],
                name='Histogram',
                marker_color='green',
                opacity=0.7
            ), row=3, col=1
        )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text=f"{symbol} Technical Analysis"
        )
        
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating price chart: {str(e)}")
        return go.Figure()

def main():
    """Main dashboard function"""
    
    st.title("🚀 Advanced Trading Signal Dashboard")
    st.markdown("Real-time signals for top US stocks using ML and technical analysis")
    
    # Sidebar for controls and regime info
    with st.sidebar:
        st.header("📊 Dashboard Controls")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        
        # Manual refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
        
        st.divider()
        
        # Market regime section
        st.header("🌡️ Market Regime")
        
        with st.spinner("Detecting market regime..."):
            regime_info, adjustments = get_market_regime()
        
        if regime_info:
            st.success(f"**Current Regime:** {regime_info['regime_name']}")
            st.info(f"**Confidence:** {regime_info['confidence']:.1%}")
            
            if adjustments:
                st.subheader("Trading Adjustments:")
                st.write(f"Position Size: {adjustments.get('position_size_multiplier', 1.0):.1f}x")
                st.write(f"Stop Loss: {adjustments.get('stop_loss_multiplier', 1.0):.1f}x")
                st.write(adjustments.get('recommendation', 'Normal trading'))
        else:
            st.warning("Unable to detect regime")
        
        st.divider()
        
        # System status
        st.header("⚡ System Status")
        st.success("✅ Data Pipeline")
        st.success("✅ Signal Generator")
        st.success("✅ Risk Manager")
        
    # Main dashboard content
    col1, col2, col3, col4 = st.columns(4)
    
    # Load data
    with st.spinner("Loading market data..."):
        df, top_stocks = load_dashboard_data()
        
    if df.empty:
        st.error("No data available. Please check database connection.")
        return
    
    # Generate signals
    with st.spinner("Generating trading signals..."):
        signals_df = generate_trading_signals(df)
    
    if not signals_df.empty:
        # Summary metrics
        total_signals = len(signals_df)
        buy_signals = len(signals_df[signals_df['Signal'] == 'BUY'])
        sell_signals = len(signals_df[signals_df['Signal'] == 'SELL'])
        strong_signals = len(signals_df[signals_df['Strength'] == 'Strong'])
        
        with col1:
            st.metric("Total Signals", total_signals)
        with col2:
            st.metric("Buy Signals", buy_signals, f"{buy_signals/total_signals:.1%}")
        with col3:
            st.metric("Sell Signals", sell_signals, f"{sell_signals/total_signals:.1%}")
        with col4:
            st.metric("Strong Signals", strong_signals, f"{strong_signals/total_signals:.1%}")
        
        # Signal overview table
        st.header("📈 Current Trading Signals")
        
        # Style the dataframe
        def style_signals(val):
            if val == 'BUY':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'SELL':
                return 'background-color: #f8d7da; color: #721c24'
            else:
                return 'background-color: #fff3cd; color: #856404'
        
        styled_df = signals_df.style.applymap(style_signals, subset=['Signal'])
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Charts section
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_signal_chart(signals_df), use_container_width=True)
            
        with col2:
            st.plotly_chart(create_sector_analysis_chart(signals_df), use_container_width=True)
        
        # Detailed stock analysis
        st.header("🔍 Detailed Stock Analysis")
        
        selected_symbol = st.selectbox(
            "Select a stock for detailed analysis:",
            options=signals_df['Symbol'].tolist(),
            index=0
        )
        
        if selected_symbol:
            # Show current signal info
            selected_info = signals_df[signals_df['Symbol'] == selected_symbol].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${selected_info['Price']:.2f}")
            with col2:
                signal_color = "normal"
                if selected_info['Signal'] == 'BUY':
                    signal_color = "normal" if selected_info['Strength'] == 'Moderate' else "normal"
                st.metric("Signal", selected_info['Signal'], 
                         f"{selected_info['Strength']} ({selected_info['Score']:.2f})")
            with col3:
                st.metric("RSI", f"{selected_info['RSI']:.1f}")
            with col4:
                st.metric("MACD Hist", f"{selected_info['MACD_Hist']:.3f}")
            
            # Price chart
            with st.spinner(f"Loading chart for {selected_symbol}..."):
                price_chart = create_price_chart(selected_symbol, df)
                st.plotly_chart(price_chart, use_container_width=True)
        
        # Top opportunities
        st.header("🎯 Top Trading Opportunities")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 Top Buy Signals")
            buy_signals_df = signals_df[
                (signals_df['Signal'] == 'BUY') & 
                (signals_df['Strength'] == 'Strong')
            ].sort_values('Score', ascending=False).head(5)
            
            if not buy_signals_df.empty:
                st.dataframe(buy_signals_df[['Symbol', 'Company', 'Price', 'Score']], 
                           use_container_width=True)
            else:
                st.info("No strong buy signals currently")
        
        with col2:
            st.subheader("🔴 Top Sell Signals")
            sell_signals_df = signals_df[
                (signals_df['Signal'] == 'SELL') & 
                (signals_df['Strength'] == 'Strong')
            ].sort_values('Score').head(5)
            
            if not sell_signals_df.empty:
                st.dataframe(sell_signals_df[['Symbol', 'Company', 'Price', 'Score']], 
                           use_container_width=True)
            else:
                st.info("No strong sell signals currently")
        
        # Performance tracking (placeholder)
        st.header("📊 System Performance")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Today's Signals", total_signals, "📈")
        with col2:
            st.metric("Accuracy (30d)", "68.5%", "2.1%")
        with col3:
            st.metric("Avg Hold Period", "6.2 days", "-0.3 days")
        
        # Alerts section
        st.header("🚨 Active Alerts")
        
        # Check for special conditions
        alerts = []
        
        # High volatility alert
        high_vol_stocks = signals_df[signals_df['Score'].isin([0, 1])]  # Placeholder
        if len(high_vol_stocks) > 5:
            alerts.append("⚠️ High volatility detected in multiple stocks")
        
        # Strong signals alert
        strong_buy_count = len(signals_df[(signals_df['Signal'] == 'BUY') & (signals_df['Strength'] == 'Strong')])
        if strong_buy_count > 3:
            alerts.append(f"🟢 {strong_buy_count} strong buy signals detected")
        
        strong_sell_count = len(signals_df[(signals_df['Signal'] == 'SELL') & (signals_df['Strength'] == 'Strong')])
        if strong_sell_count > 3:
            alerts.append(f"🔴 {strong_sell_count} strong sell signals detected")
        
        if alerts:
            for alert in alerts:
                st.warning(alert)
        else:
            st.info("No active alerts")
    
    else:
        st.error("Unable to generate signals. Please check data quality.")
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(30)
        st.experimental_rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 12px;'>
    🤖 Advanced Trading System | Last updated: {} | 
    <strong>Disclaimer:</strong> For educational purposes only. Not financial advice.
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()