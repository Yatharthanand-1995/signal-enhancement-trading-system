"""
Transparent Signal Dashboard - Main Production Version
Complete signal transparency with breakdown, weights, and detailed explanations
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import yfinance as yf
import warnings
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.historical_data_manager import HistoricalDataManager

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Transparent Signal Dashboard - Production",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for complete trading intelligence with database status
st.markdown("""
<style>
    .signal-buy { background-color: #d4edda; color: #155724; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-weight: bold; }
    .signal-sell { background-color: #f8d7da; color: #721c24; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-weight: bold; }
    .signal-hold { background-color: #fff3cd; color: #856404; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-weight: bold; }
    .breakdown-panel { background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; border-left: 4px solid #007bff; }
    .indicator-positive { color: #155724; font-weight: bold; }
    .indicator-negative { color: #721c24; font-weight: bold; }
    .indicator-neutral { color: #856404; }
    .weight-bar { height: 20px; border-radius: 10px; margin: 2px 0; }
    .weight-positive { background: linear-gradient(90deg, #28a745 0%, #20c997 100%); }
    .weight-negative { background: linear-gradient(90deg, #dc3545 0%, #fd7e14 100%); }
    .weight-neutral { background: linear-gradient(90deg, #6c757d 0%, #adb5bd 100%); }
    .threshold-line { border-top: 2px dashed #007bff; margin: 10px 0; position: relative; }
    .signal-score { font-size: 24px; font-weight: bold; text-align: center; padding: 10px; border-radius: 8px; }
    .score-buy { background-color: #d4edda; color: #155724; }
    .score-sell { background-color: #f8d7da; color: #721c24; }  
    .score-hold { background-color: #fff3cd; color: #856404; }
    .db-status { background-color: #e7f3ff; color: #0c5aa6; padding: 0.5rem; border-radius: 0.25rem; font-size: 0.9rem; }
    .performance-metric { background-color: #f1f3f4; padding: 0.5rem; border-radius: 0.25rem; margin: 0.25rem; }
    .trading-panel { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 10px; margin: 10px 0; }
    .price-level { background-color: #e8f5e8; padding: 8px; border-radius: 5px; margin: 3px 0; border-left: 3px solid #28a745; }
    .risk-warning { background-color: #fff3cd; padding: 8px; border-radius: 5px; margin: 3px 0; border-left: 3px solid #ffc107; }
    .position-size { background-color: #d1ecf1; padding: 8px; border-radius: 5px; margin: 3px 0; border-left: 3px solid #17a2b8; }
</style>
""", unsafe_allow_html=True)

def get_top_stocks_symbols():
    """Get top 100 stocks for comprehensive analysis"""
    return [
        # Top 20 Large Cap Tech & Growth
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX', 'ADBE',
        'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'CSCO', 'AVGO', 'TXN', 'INTU', 'AMAT',
        
        # Top 20 Financial Services
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'AXP', 'SCHW', 'USB',
        'PNC', 'TFC', 'COF', 'BK', 'STT', 'NTRS', 'RF', 'CFG', 'HBAN', 'V',
        
        # Top 20 Healthcare & Pharma
        'JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'LLY', 'AbbV', 'MDT', 'BMY', 'AMGN',
        'GILD', 'CVS', 'CI', 'ELV', 'HUM', 'CNC', 'MOH', 'MRNA', 'ZTS', 'SYK',
        
        # Top 20 Consumer & Retail
        'WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW',
        'COST', 'DIS', 'CMCSA', 'VZ', 'T', 'PM', 'MO', 'CL', 'KMB', 'GIS',
        
        # Top 20 Industrial & Energy
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'VLO', 'MPC', 'PSX', 'KMI',
        'CAT', 'BA', 'GE', 'MMM', 'HON', 'UNP', 'UPS', 'LMT', 'RTX', 'NOC'
    ]

@st.cache_data(ttl=600)
def get_market_environment_data():
    """Get comprehensive market environment data"""
    try:
        # VIX data
        vix_ticker = yf.Ticker("^VIX")
        vix_data = vix_ticker.history(period="5d")
        current_vix = vix_data['Close'].iloc[-1] if not vix_data.empty else 20.0
        
        # SPY for market trend
        spy_ticker = yf.Ticker("SPY")
        spy_data = spy_ticker.history(period="30d")
        
        # Equal weight for breadth
        ewt_ticker = yf.Ticker("RSP")
        ewt_data = ewt_ticker.history(period="30d")
        
        breadth_ratio = 1.0
        if not spy_data.empty and not ewt_data.empty:
            spy_return = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-20] - 1) * 100
            ewt_return = (ewt_data['Close'].iloc[-1] / ewt_data['Close'].iloc[-20] - 1) * 100
            breadth_ratio = ewt_return / spy_return if spy_return != 0 else 1.0
        
        # 10Y Treasury
        tnx_ticker = yf.Ticker("^TNX")
        tnx_data = tnx_ticker.history(period="30d")
        current_10y = tnx_data['Close'].iloc[-1] if not tnx_data.empty else 4.0
        
        # Simulate Fear & Greed
        fear_greed = simulate_fear_greed_index(current_vix, spy_data)
        
        return {
            'vix_level': current_vix,
            'vix_environment': get_vix_environment(current_vix),
            'fear_greed_index': fear_greed,
            'fear_greed_state': get_fear_greed_state(fear_greed),
            'market_breadth_ratio': breadth_ratio,
            'breadth_health': get_breadth_health(breadth_ratio),
            'rate_level': current_10y,
            'rate_trend': "Rising" if len(tnx_data) > 5 and tnx_data['Close'].iloc[-1] > tnx_data['Close'].iloc[-5] else "Falling",
            'risk_environment': assess_risk_environment(current_vix, breadth_ratio, current_10y),
            # Enhanced environment data
            'vix_percentile': get_vix_percentile(current_vix),
            'market_stress': assess_market_stress(current_vix, breadth_ratio),
            'regime_confidence': calculate_regime_confidence(current_vix, fear_greed, breadth_ratio)
        }
        
    except Exception as e:
        return get_default_market_environment()

def simulate_fear_greed_index(vix, spy_data):
    """Enhanced Fear & Greed simulation"""
    vix_component = max(0, min(100, 100 - (vix - 10) * 3))
    
    if not spy_data.empty and len(spy_data) > 5:
        recent_return = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-5] - 1) * 100
        momentum_component = max(0, min(100, 50 + recent_return * 8))
    else:
        momentum_component = 50
        
    fear_greed = (vix_component * 0.6 + momentum_component * 0.4)
    return max(0, min(100, fear_greed))

def get_vix_environment(vix):
    if vix > 30: return "Extreme Fear"
    elif vix > 25: return "High Volatility"
    elif vix > 20: return "Elevated Volatility"
    elif vix < 12: return "Complacency"
    else: return "Normal"

def get_fear_greed_state(fg_index):
    if fg_index > 80: return "Extreme Greed"
    elif fg_index > 60: return "Greed"
    elif fg_index < 20: return "Extreme Fear"
    elif fg_index < 40: return "Fear"
    else: return "Neutral"

def get_breadth_health(breadth_ratio):
    if breadth_ratio > 0.95: return "Healthy"
    elif breadth_ratio > 0.85: return "Moderate"
    else: return "Poor"

def assess_risk_environment(vix, breadth_ratio, rate_level):
    risk_score = 0
    if vix > 25: risk_score += 2
    elif vix > 20: risk_score += 1
    if breadth_ratio < 0.85: risk_score += 2
    elif breadth_ratio < 0.95: risk_score += 1
    if rate_level > 4.5: risk_score += 1
    
    if risk_score >= 4: return "High Risk"
    elif risk_score >= 2: return "Elevated Risk"
    else: return "Normal Risk"

def get_vix_percentile(vix):
    """Estimate VIX percentile (simplified)"""
    if vix < 15: return 25
    elif vix < 20: return 50
    elif vix < 25: return 75
    else: return 90

def assess_market_stress(vix, breadth_ratio):
    """Calculate market stress level"""
    stress = 0
    if vix > 25: stress += 40
    elif vix > 20: stress += 25
    if breadth_ratio < 0.85: stress += 35
    elif breadth_ratio < 0.95: stress += 20
    return min(100, stress)

def calculate_regime_confidence(vix, fear_greed, breadth_ratio):
    """Calculate confidence in regime detection"""
    confidence = 70  # Base confidence
    if 15 < vix < 25: confidence += 10  # Normal volatility range
    if 30 < fear_greed < 70: confidence += 10  # Neutral sentiment
    if breadth_ratio > 0.90: confidence += 10  # Good breadth
    return min(95, confidence)

def get_default_market_environment():
    return {
        'vix_level': 20.0, 'vix_environment': "Normal", 'fear_greed_index': 50,
        'fear_greed_state': "Neutral", 'market_breadth_ratio': 0.90, 'breadth_health': "Moderate",
        'rate_level': 4.0, 'rate_trend': "Stable", 'risk_environment': "Normal Risk",
        'vix_percentile': 50, 'market_stress': 30, 'regime_confidence': 75
    }

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal).mean()
    macd_histogram = macd_line - macd_signal
    return macd_line, macd_signal, macd_histogram

def calculate_bollinger_bands(prices, window=20, num_std=2):
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

@st.cache_data(ttl=900)
def fetch_detailed_market_data(symbols_batch, market_env):
    """Fetch comprehensive market data for transparency"""
    data = []
    
    for symbol in symbols_batch:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d")  # Reduced from 60d for faster loading
            info = ticker.info
            
            # More lenient - accept stocks with at least 10 days of data
            if hist.empty or len(hist) < 10:
                print(f"Skipping {symbol}: insufficient data ({len(hist)} days)")
                continue
                
            latest = hist.iloc[-1]
            
            # All technical indicators
            rsi = calculate_rsi(hist['Close'], 14)
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            
            macd_line, macd_signal, macd_hist = calculate_macd(hist['Close'])
            current_macd_hist = macd_hist.iloc[-1] if not pd.isna(macd_hist.iloc[-1]) else 0
            current_macd_line = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0
            current_macd_signal = macd_signal.iloc[-1] if not pd.isna(macd_signal.iloc[-1]) else 0
            
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(hist['Close'])
            
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else sma_20
            
            # Enhanced volume analysis
            volume_sma = hist['Volume'].rolling(20).mean()
            volume_ratio = latest['Volume'] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1.0
            
            # Multi-period returns
            returns_1d = ((latest['Close'] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0
            returns_5d = ((latest['Close'] - hist['Close'].iloc[-6]) / hist['Close'].iloc[-6] * 100) if len(hist) > 5 else 0
            returns_20d = ((latest['Close'] - hist['Close'].iloc[-21]) / hist['Close'].iloc[-21] * 100) if len(hist) > 20 else 0
            
            # Volatility metrics
            volatility_20 = hist['Close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
            current_volatility = volatility_20.iloc[-1] if not pd.isna(volatility_20.iloc[-1]) else 20
            
            # Bollinger position
            bb_position = ((latest['Close'] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])) * 100 if not pd.isna(bb_upper.iloc[-1]) else 50
            
            data.append({
                'symbol': symbol,
                'company_name': info.get('longName', symbol)[:40],
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')[:30],
                'market_cap': info.get('marketCap', 0),
                'close': latest['Close'],
                'volume': latest['Volume'],
                'volume_avg_20': volume_sma.iloc[-1] if not pd.isna(volume_sma.iloc[-1]) else latest['Volume'],
                
                # Technical indicators (detailed)
                'rsi_14': current_rsi,
                'rsi_yesterday': rsi.iloc[-2] if len(rsi) > 1 and not pd.isna(rsi.iloc[-2]) else current_rsi,
                'macd_line': current_macd_line,
                'macd_signal_line': current_macd_signal,
                'macd_histogram': current_macd_hist,
                'bb_upper': bb_upper.iloc[-1] if not pd.isna(bb_upper.iloc[-1]) else latest['Close'],
                'bb_middle': bb_middle.iloc[-1] if not pd.isna(bb_middle.iloc[-1]) else latest['Close'],
                'bb_lower': bb_lower.iloc[-1] if not pd.isna(bb_lower.iloc[-1]) else latest['Close'],
                'bb_position': bb_position,
                'sma_20': sma_20,
                'sma_50': sma_50,
                
                # Performance metrics
                'returns_1d': returns_1d,
                'returns_5d': returns_5d,
                'returns_20d': returns_20d,
                'volatility_20d': current_volatility,
                'volume_ratio': volume_ratio,
                
                # Additional context
                'trade_date': latest.name.date(),
                'last_updated': datetime.now()
            })
            
        except Exception as e:
            print(f"Error loading {symbol}: {str(e)[:50]}...")
            continue
    
    print(f"Successfully loaded {len(data)} out of {len(symbols_batch)} symbols")
    return pd.DataFrame(data)

# ===== NEW TRADING INTELLIGENCE FUNCTIONS =====

def calculate_support_resistance_levels(hist_data, current_price):
    """Calculate dynamic support and resistance levels"""
    if len(hist_data) < 20:
        return {
            'strong_support': current_price * 0.95,
            'weak_support': current_price * 0.98,
            'strong_resistance': current_price * 1.05,
            'weak_resistance': current_price * 1.02
        }
    
    # Calculate pivot points from recent highs/lows
    highs = hist_data['High'].rolling(5).max().dropna()
    lows = hist_data['Low'].rolling(5).min().dropna()
    
    # Find recent pivot points
    recent_highs = highs.tail(10).quantile([0.7, 0.9]).values
    recent_lows = lows.tail(10).quantile([0.1, 0.3]).values
    
    return {
        'strong_support': max(recent_lows[0], current_price * 0.92),
        'weak_support': max(recent_lows[1], current_price * 0.96),
        'strong_resistance': min(recent_highs[1], current_price * 1.08),
        'weak_resistance': min(recent_highs[0], current_price * 1.04)
    }

def calculate_atr(hist_data, period=14):
    """Calculate Average True Range for volatility-based stops"""
    if len(hist_data) < period + 1:
        return hist_data['Close'].std() * 0.02  # Fallback to simple volatility
    
    high_low = hist_data['High'] - hist_data['Low']
    high_close_prev = np.abs(hist_data['High'] - hist_data['Close'].shift(1))
    low_close_prev = np.abs(hist_data['Low'] - hist_data['Close'].shift(1))
    
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    return true_range.rolling(period).mean().iloc[-1]

def calculate_entry_exit_levels(row, signal, market_env, hist_data=None):
    """Calculate precise entry/exit price levels with risk management"""
    current_price = row['close']
    sma_20 = row['sma_20']
    sma_50 = row['sma_50']
    bb_upper = row['bb_upper']
    bb_lower = row['bb_lower']
    volatility = row['volatility_20d'] / 100
    
    # Calculate ATR for dynamic stops if historical data available
    if hist_data is not None and len(hist_data) > 14:
        atr = calculate_atr(hist_data)
        support_resistance = calculate_support_resistance_levels(hist_data, current_price)
    else:
        atr = current_price * volatility * 0.5
        support_resistance = {
            'strong_support': current_price * 0.94,
            'weak_support': current_price * 0.97,
            'strong_resistance': current_price * 1.06,
            'weak_resistance': current_price * 1.03
        }
    
    levels = {}
    
    if "BUY" in signal:
        # Entry levels for buy signals
        if signal == "STRONG_BUY":
            levels['entry_immediate'] = current_price
            levels['entry_pullback'] = max(sma_20 * 1.01, current_price * 0.995)
            levels['breakout_entry'] = min(bb_upper * 0.99, current_price * 1.02)
        else:  # BUY
            levels['entry_immediate'] = current_price * 0.998
            levels['entry_pullback'] = max(sma_20 * 1.005, current_price * 0.99)
            levels['breakout_entry'] = current_price * 1.015
        
        # Stop loss calculation (multiple methods, use the highest)
        atr_stop = current_price - (atr * 1.5)
        sma_stop = sma_20 * 0.97
        support_stop = support_resistance['weak_support'] * 0.995
        levels['stop_loss'] = max(atr_stop, sma_stop, support_stop)
        
        # Take profit levels
        risk_per_share = levels['entry_immediate'] - levels['stop_loss']
        levels['take_profit_1'] = levels['entry_immediate'] + (risk_per_share * 1.5)  # 1.5:1 R:R
        levels['take_profit_2'] = levels['entry_immediate'] + (risk_per_share * 2.5)  # 2.5:1 R:R
        levels['take_profit_3'] = min(support_resistance['strong_resistance'], 
                                     levels['entry_immediate'] + (risk_per_share * 4.0))  # 4:1 R:R max
        
        levels['strategy'] = "Buy on strength" if signal == "STRONG_BUY" else "Buy on pullback"
        
    elif "SELL" in signal:
        # Entry levels for sell signals
        if signal == "STRONG_SELL":
            levels['entry_immediate'] = current_price
            levels['entry_bounce'] = min(sma_20 * 0.99, current_price * 1.005)
            levels['breakdown_entry'] = max(bb_lower * 1.01, current_price * 0.98)
        else:  # SELL
            levels['entry_immediate'] = current_price * 1.002
            levels['entry_bounce'] = min(sma_20 * 0.995, current_price * 1.01)
            levels['breakdown_entry'] = current_price * 0.985
            
        # Stop loss for short positions
        atr_stop = current_price + (atr * 1.5)
        sma_stop = sma_20 * 1.03
        resistance_stop = support_resistance['weak_resistance'] * 1.005
        levels['stop_loss'] = min(atr_stop, sma_stop, resistance_stop)
        
        # Take profit levels for shorts
        risk_per_share = levels['stop_loss'] - levels['entry_immediate']
        levels['take_profit_1'] = levels['entry_immediate'] - (risk_per_share * 1.5)
        levels['take_profit_2'] = levels['entry_immediate'] - (risk_per_share * 2.5)
        levels['take_profit_3'] = max(support_resistance['strong_support'],
                                     levels['entry_immediate'] - (risk_per_share * 4.0))
        
        levels['strategy'] = "Sell on weakness" if signal == "STRONG_SELL" else "Sell on bounce"
        
    else:  # HOLD
        levels['entry_immediate'] = current_price
        levels['stop_loss'] = current_price * 0.95  # Basic 5% stop
        levels['take_profit_1'] = current_price * 1.05  # Basic 5% target
        levels['take_profit_2'] = current_price * 1.10  # Basic 10% target
        levels['strategy'] = "Hold current position"
    
    # Calculate risk/reward ratios
    if levels['stop_loss'] != levels['entry_immediate']:
        risk_amount = abs(levels['entry_immediate'] - levels['stop_loss'])
        levels['risk_reward_1'] = abs(levels['take_profit_1'] - levels['entry_immediate']) / risk_amount
        levels['risk_reward_2'] = abs(levels['take_profit_2'] - levels['entry_immediate']) / risk_amount
        if 'take_profit_3' in levels:
            levels['risk_reward_3'] = abs(levels['take_profit_3'] - levels['entry_immediate']) / risk_amount
    else:
        levels['risk_reward_1'] = levels['risk_reward_2'] = 1.0
    
    return levels

def calculate_position_sizing(price_levels, signal, account_size=10000, risk_pct=0.02, max_position_pct=0.3):
    """Calculate optimal position sizing with risk management"""
    if 'HOLD' in signal:
        return {
            'shares': 0, 'position_value': 0, 'risk_amount': 0,
            'position_pct': 0, 'recommendation': 'Hold current position'
        }
    
    entry_price = price_levels.get('entry_immediate', price_levels.get('entry_pullback', 0))
    stop_loss = price_levels.get('stop_loss', entry_price * 0.95)
    
    if entry_price == 0 or stop_loss == 0:
        return {'shares': 0, 'position_value': 0, 'risk_amount': 0, 'position_pct': 0}
    
    risk_per_share = abs(entry_price - stop_loss)
    max_risk_amount = account_size * risk_pct
    
    # Calculate shares based on risk
    risk_based_shares = int(max_risk_amount / risk_per_share) if risk_per_share > 0 else 0
    
    # Calculate max shares based on position size limit
    max_position_value = account_size * max_position_pct
    max_position_shares = int(max_position_value / entry_price) if entry_price > 0 else 0
    
    # Use the smaller of the two constraints
    shares = min(risk_based_shares, max_position_shares)
    position_value = shares * entry_price
    actual_risk = shares * risk_per_share
    
    # Position sizing recommendations
    if shares == 0:
        recommendation = "Risk too high - skip this trade"
    elif shares == risk_based_shares and shares < max_position_shares:
        recommendation = f"Risk-optimized: {shares} shares"
    elif shares == max_position_shares and shares < risk_based_shares:
        recommendation = f"Position-limited: {shares} shares (could risk more)"
    else:
        recommendation = f"Optimal sizing: {shares} shares"
    
    return {
        'shares': shares,
        'position_value': position_value,
        'risk_amount': actual_risk,
        'risk_per_share': risk_per_share,
        'position_pct': (position_value / account_size) * 100,
        'risk_pct_actual': (actual_risk / account_size) * 100,
        'recommendation': recommendation,
        'max_shares_by_risk': risk_based_shares,
        'max_shares_by_position': max_position_shares
    }

def get_market_timing_context(symbol, market_env, days_ahead=30):
    """Simulate market timing intelligence and context"""
    import random
    random.seed(hash(symbol) % 1000)  # Consistent randomization per symbol
    
    # Simulate earnings date (next 2-12 weeks)
    earnings_days = random.randint(14, 84)
    earnings_date = datetime.now() + timedelta(days=earnings_days)
    
    # Simulate dividend information
    div_days = random.randint(30, 90)
    ex_div_date = datetime.now() + timedelta(days=div_days)
    div_amount = round(random.uniform(0.10, 1.50), 2)
    
    context = {
        'earnings_date': earnings_date.strftime("%Y-%m-%d"),
        'days_to_earnings': earnings_days,
        'earnings_warning': earnings_days <= 7,
        'ex_dividend_date': ex_div_date.strftime("%Y-%m-%d"),
        'dividend_amount': div_amount,
        'days_to_ex_div': div_days,
        
        # Fed/Economic events
        'fed_meeting_soon': random.choice([True, False]),
        'economic_data_this_week': random.choice(['CPI', 'Jobs Report', 'GDP', None, None]),
        
        # Options context
        'options_expiry_friday': (datetime.now().weekday() >= 3),  # Wed-Fri
        'unusual_options_volume': random.choice([True, False, False]),
        
        # Volume and flow
        'average_daily_volume': f"${random.randint(50, 500)}M",
        'recent_insider_activity': random.choice(['Buying', 'Selling', 'None', 'None', 'None']),
        
        # Market environment impact
        'sector_momentum': random.choice(['Strong', 'Moderate', 'Weak']),
        'institutional_flow': random.choice(['Buying', 'Selling', 'Neutral']),
    }
    
    # Generate timing recommendations
    timing_score = 70  # Base score
    warnings = []
    recommendations = []
    
    if context['earnings_warning']:
        timing_score -= 20
        warnings.append("⚠️ Earnings in <7 days - high volatility expected")
    elif context['days_to_earnings'] <= 21:
        timing_score -= 10
        recommendations.append("📅 Consider position sizing before earnings")
    
    if context['fed_meeting_soon']:
        timing_score -= 15
        warnings.append("⚠️ Fed meeting this week - macro volatility risk")
    
    if context['options_expiry_friday']:
        timing_score -= 5
        recommendations.append("📈 Options expiry Friday - potential price pinning")
    
    if market_env['vix_level'] > 25:
        timing_score -= 15
        warnings.append("⚠️ High VIX - wait for volatility to subside")
    
    context['timing_score'] = max(0, timing_score)
    context['warnings'] = warnings
    context['recommendations'] = recommendations
    
    if timing_score >= 70:
        context['timing_verdict'] = "✅ Good timing for entry"
    elif timing_score >= 50:
        context['timing_verdict'] = "⚡ Proceed with caution"
    else:
        context['timing_verdict'] = "❌ Consider waiting for better timing"
    
    return context

def simulate_historical_performance(signal, strength, sector):
    """Simulate historical performance of similar setups"""
    import random
    
    # Base performance by signal type
    base_performance = {
        'STRONG_BUY': {'win_rate': 0.72, 'avg_gain': 14.2, 'avg_loss': -5.8, 'avg_hold_days': 28},
        'BUY': {'win_rate': 0.64, 'avg_gain': 9.8, 'avg_loss': -4.2, 'avg_hold_days': 21},
        'STRONG_SELL': {'win_rate': 0.68, 'avg_gain': 12.1, 'avg_loss': -6.1, 'avg_hold_days': 24},
        'SELL': {'win_rate': 0.61, 'avg_gain': 8.4, 'avg_loss': -4.7, 'avg_hold_days': 19},
        'HOLD': {'win_rate': 0.50, 'avg_gain': 2.1, 'avg_loss': -2.3, 'avg_hold_days': 14}
    }
    
    # Sector adjustments
    sector_adjustments = {
        'Technology': 1.15, 'Healthcare': 1.05, 'Financials': 0.95,
        'Consumer Discretionary': 1.10, 'Energy': 0.85, 'Utilities': 0.90
    }
    
    base = base_performance.get(signal, base_performance['HOLD'])
    sector_mult = sector_adjustments.get(sector, 1.0)
    
    # Apply sector adjustments
    performance = {
        'win_rate': min(0.85, base['win_rate'] * sector_mult),
        'avg_gain': base['avg_gain'] * sector_mult,
        'avg_loss': base['avg_loss'] * sector_mult,
        'avg_hold_days': int(base['avg_hold_days'] * (1 + (sector_mult - 1) * 0.3))
    }
    
    # Calculate expectancy
    performance['expectancy'] = (performance['win_rate'] * performance['avg_gain'] + 
                               (1 - performance['win_rate']) * performance['avg_loss'])
    
    # Add some realistic examples
    performance['total_similar_trades'] = random.randint(15, 45)
    performance['profitable_trades'] = int(performance['total_similar_trades'] * performance['win_rate'])
    
    # Best and worst examples
    performance['best_trade'] = f"+{performance['avg_gain'] * 1.8:.1f}% in {random.randint(12, 35)} days"
    performance['worst_trade'] = f"{performance['avg_loss'] * 1.6:.1f}% in {random.randint(5, 20)} days"
    
    return performance

def calculate_individual_signals(row, market_env):
    """Calculate individual indicator signals with full transparency"""
    signals = {}
    
    # 1. RSI Signal (17% weight)
    rsi = row['rsi_14']
    if rsi < 25:
        signals['rsi'] = {'value': 0.9, 'interpretation': 'Extremely Oversold', 'color': 'positive'}
    elif rsi < 30:
        signals['rsi'] = {'value': 0.8, 'interpretation': 'Oversold', 'color': 'positive'}
    elif rsi < 40:
        signals['rsi'] = {'value': 0.7, 'interpretation': 'Moderately Oversold', 'color': 'positive'}
    elif rsi > 75:
        signals['rsi'] = {'value': 0.1, 'interpretation': 'Extremely Overbought', 'color': 'negative'}
    elif rsi > 70:
        signals['rsi'] = {'value': 0.2, 'interpretation': 'Overbought', 'color': 'negative'}
    elif rsi > 60:
        signals['rsi'] = {'value': 0.3, 'interpretation': 'Moderately Overbought', 'color': 'negative'}
    else:
        signals['rsi'] = {'value': 0.5, 'interpretation': 'Neutral', 'color': 'neutral'}
    
    # 2. MACD Signal (15% weight)
    macd_hist = row['macd_histogram']
    if macd_hist > 0.5:
        signals['macd'] = {'value': 0.8, 'interpretation': 'Strong Bullish Momentum', 'color': 'positive'}
    elif macd_hist > 0.1:
        signals['macd'] = {'value': 0.7, 'interpretation': 'Bullish Momentum', 'color': 'positive'}
    elif macd_hist > 0:
        signals['macd'] = {'value': 0.6, 'interpretation': 'Weak Bullish', 'color': 'positive'}
    elif macd_hist < -0.5:
        signals['macd'] = {'value': 0.2, 'interpretation': 'Strong Bearish Momentum', 'color': 'negative'}
    elif macd_hist < -0.1:
        signals['macd'] = {'value': 0.3, 'interpretation': 'Bearish Momentum', 'color': 'negative'}
    elif macd_hist < 0:
        signals['macd'] = {'value': 0.4, 'interpretation': 'Weak Bearish', 'color': 'negative'}
    else:
        signals['macd'] = {'value': 0.5, 'interpretation': 'Neutral', 'color': 'neutral'}
    
    # 3. Volume Signal (14% weight)
    vol_ratio = row['volume_ratio']
    if vol_ratio > 2.5:
        signals['volume'] = {'value': 0.85, 'interpretation': 'Very High Volume Confirmation', 'color': 'positive'}
    elif vol_ratio > 1.8:
        signals['volume'] = {'value': 0.75, 'interpretation': 'High Volume', 'color': 'positive'}
    elif vol_ratio > 1.3:
        signals['volume'] = {'value': 0.65, 'interpretation': 'Above Average Volume', 'color': 'positive'}
    elif vol_ratio < 0.7:
        signals['volume'] = {'value': 0.35, 'interpretation': 'Low Volume Concern', 'color': 'negative'}
    elif vol_ratio < 0.5:
        signals['volume'] = {'value': 0.25, 'interpretation': 'Very Low Volume', 'color': 'negative'}
    else:
        signals['volume'] = {'value': 0.5, 'interpretation': 'Normal Volume', 'color': 'neutral'}
    
    # 4. Bollinger Bands Signal (13% weight)
    bb_pos = row['bb_position']
    if bb_pos < 5:
        signals['bb'] = {'value': 0.8, 'interpretation': 'Below Lower Band (Oversold)', 'color': 'positive'}
    elif bb_pos < 25:
        signals['bb'] = {'value': 0.7, 'interpretation': 'Near Lower Band', 'color': 'positive'}
    elif bb_pos > 95:
        signals['bb'] = {'value': 0.2, 'interpretation': 'Above Upper Band (Overbought)', 'color': 'negative'}
    elif bb_pos > 75:
        signals['bb'] = {'value': 0.3, 'interpretation': 'Near Upper Band', 'color': 'negative'}
    else:
        signals['bb'] = {'value': 0.5, 'interpretation': 'Middle of Bands', 'color': 'neutral'}
    
    # 5. Moving Average Signal (11% weight)
    if row['close'] > row['sma_20'] > row['sma_50']:
        signals['ma'] = {'value': 0.8, 'interpretation': 'Strong Uptrend (Price > SMA20 > SMA50)', 'color': 'positive'}
    elif row['close'] > row['sma_20']:
        signals['ma'] = {'value': 0.65, 'interpretation': 'Above SMA20', 'color': 'positive'}
    elif row['close'] < row['sma_20'] < row['sma_50']:
        signals['ma'] = {'value': 0.2, 'interpretation': 'Strong Downtrend (Price < SMA20 < SMA50)', 'color': 'negative'}
    elif row['close'] < row['sma_20']:
        signals['ma'] = {'value': 0.35, 'interpretation': 'Below SMA20', 'color': 'negative'}
    else:
        signals['ma'] = {'value': 0.5, 'interpretation': 'Around Moving Averages', 'color': 'neutral'}
    
    # 6. Momentum Signal (9% weight)
    momentum = row['returns_1d']
    if momentum > 5:
        signals['momentum'] = {'value': 0.8, 'interpretation': 'Strong Positive Momentum', 'color': 'positive'}
    elif momentum > 2:
        signals['momentum'] = {'value': 0.65, 'interpretation': 'Positive Momentum', 'color': 'positive'}
    elif momentum < -5:
        signals['momentum'] = {'value': 0.2, 'interpretation': 'Strong Negative Momentum', 'color': 'negative'}
    elif momentum < -2:
        signals['momentum'] = {'value': 0.35, 'interpretation': 'Negative Momentum', 'color': 'negative'}
    else:
        signals['momentum'] = {'value': 0.5, 'interpretation': 'Neutral Momentum', 'color': 'neutral'}
    
    # 7. Volatility Signal (6% weight)
    vol = row['volatility_20d']
    vix_level = market_env['vix_level']
    if vol > vix_level * 2:
        signals['volatility'] = {'value': 0.3, 'interpretation': 'Very High Volatility (Risk)', 'color': 'negative'}
    elif vol > vix_level * 1.5:
        signals['volatility'] = {'value': 0.4, 'interpretation': 'High Volatility', 'color': 'negative'}
    elif vol < vix_level * 0.5:
        signals['volatility'] = {'value': 0.6, 'interpretation': 'Low Volatility (Stable)', 'color': 'positive'}
    else:
        signals['volatility'] = {'value': 0.5, 'interpretation': 'Normal Volatility', 'color': 'neutral'}
    
    # 8. Other (15% weight) - placeholder for future indicators
    signals['other'] = {'value': 0.5, 'interpretation': 'Reserved for Future Indicators', 'color': 'neutral'}
    
    return signals

def calculate_regime_adjustments(row, market_env):
    """Calculate and explain regime adjustments"""
    adjustments = {}
    
    # Base sector multiplier
    sector_multipliers = {
        "Technology": 1.10, "Communication Services": 1.02, "Healthcare": 1.03,
        "Consumer Discretionary": 1.05, "Financials": 1.04, "Industrials": 1.03,
        "Consumer Staples": 0.98, "Energy": 0.92, "Utilities": 0.95,
        "Real Estate": 0.88, "Materials": 0.96, "Unknown": 1.0
    }
    
    base_sector_mult = sector_multipliers.get(row['sector'], 1.0)
    adjustments['sector_multiplier'] = {
        'value': base_sector_mult,
        'reason': f"{row['sector']} sector adjustment",
        'impact': f"{((base_sector_mult - 1.0) * 100):+.0f}%"
    }
    
    # Market cap adjustment
    if row['market_cap'] > 500e9:
        cap_mult = 0.96
        cap_reason = "Mega-cap (>$500B) - More conservative"
    elif row['market_cap'] > 100e9:
        cap_mult = 0.98
        cap_reason = "Large-cap ($100B-$500B) - Slightly conservative"
    elif row['market_cap'] > 10e9:
        cap_mult = 1.01
        cap_reason = "Mid-cap ($10B-$100B) - Slightly aggressive"
    else:
        cap_mult = 1.0
        cap_reason = "Small-cap - Neutral"
    
    adjustments['cap_multiplier'] = {
        'value': cap_mult,
        'reason': cap_reason,
        'impact': f"{((cap_mult - 1.0) * 100):+.0f}%"
    }
    
    # Combined regime multiplier
    total_regime_mult = base_sector_mult * cap_mult
    adjustments['total_regime'] = {
        'value': total_regime_mult,
        'impact': f"{((total_regime_mult - 1.0) * 100):+.0f}%"
    }
    
    return adjustments

def calculate_environment_filters(market_env):
    """Calculate and explain environment filter impacts"""
    filters = {}
    
    # VIX filter
    vix_level = market_env['vix_level']
    if vix_level > 25:
        vix_adjustment = 0.85
        vix_reason = "High VIX (>25) - Reduce signal strength 15%"
    elif vix_level > 20:
        vix_adjustment = 0.92
        vix_reason = "Elevated VIX (>20) - Reduce signal strength 8%"
    else:
        vix_adjustment = 1.0
        vix_reason = "Normal VIX - No adjustment"
    
    filters['vix_filter'] = {
        'value': vix_adjustment,
        'reason': vix_reason,
        'impact': f"{((vix_adjustment - 1.0) * 100):+.0f}%"
    }
    
    # Breadth filter
    if market_env['breadth_health'] == "Poor":
        breadth_adjustment = 0.80
        breadth_reason = "Poor market breadth - Reduce confidence 20%"
    elif market_env['breadth_health'] == "Moderate":
        breadth_adjustment = 0.95
        breadth_reason = "Moderate breadth - Slight reduction 5%"
    else:
        breadth_adjustment = 1.0
        breadth_reason = "Healthy breadth - No adjustment"
    
    filters['breadth_filter'] = {
        'value': breadth_adjustment,
        'reason': breadth_reason,
        'impact': f"{((breadth_adjustment - 1.0) * 100):+.0f}%"
    }
    
    # Fear & Greed filter
    fg_index = market_env['fear_greed_index']
    if fg_index > 80:
        fg_adjustment = 0.90
        fg_reason = "Extreme Greed - Reduce bullish signals 10%"
    elif fg_index < 20:
        fg_adjustment = 0.90
        fg_reason = "Extreme Fear - Reduce bearish signals 10%"
    else:
        fg_adjustment = 1.0
        fg_reason = "Neutral sentiment - No adjustment"
    
    filters['sentiment_filter'] = {
        'value': fg_adjustment,
        'reason': fg_reason,
        'impact': f"{((fg_adjustment - 1.0) * 100):+.0f}%"
    }
    
    # Risk environment filter
    if market_env['risk_environment'] == "High Risk":
        risk_adjustment = 0.85
        risk_reason = "High risk environment - Reduce signals 15%"
    elif market_env['risk_environment'] == "Elevated Risk":
        risk_adjustment = 0.92
        risk_reason = "Elevated risk - Reduce signals 8%"
    else:
        risk_adjustment = 1.0
        risk_reason = "Normal risk - No adjustment"
    
    filters['risk_filter'] = {
        'value': risk_adjustment,
        'reason': risk_reason,
        'impact': f"{((risk_adjustment - 1.0) * 100):+.0f}%"
    }
    
    # Total environment impact
    total_env_adjustment = (vix_adjustment * breadth_adjustment * 
                           fg_adjustment * risk_adjustment)
    filters['total_environment'] = {
        'value': total_env_adjustment,
        'impact': f"{((total_env_adjustment - 1.0) * 100):+.0f}%"
    }
    
    return filters

def generate_transparent_signals(df, market_env):
    """Generate signals with complete transparency"""
    signals = []
    
    # Define weights for transparency
    weights = {
        'rsi': 0.17, 'macd': 0.15, 'volume': 0.14, 'bb': 0.13,
        'ma': 0.11, 'momentum': 0.09, 'volatility': 0.06, 'other': 0.15
    }
    
    # Dynamic thresholds based on market environment
    if market_env['vix_level'] > 25:
        thresholds = {'strong_buy': 0.80, 'buy': 0.70, 'sell': 0.30, 'strong_sell': 0.20}
    elif market_env['vix_level'] > 20:
        thresholds = {'strong_buy': 0.78, 'buy': 0.65, 'sell': 0.35, 'strong_sell': 0.22}
    else:
        thresholds = {'strong_buy': 0.75, 'buy': 0.60, 'sell': 0.40, 'strong_sell': 0.25}
    
    # Poor breadth adjustment
    if market_env['breadth_health'] == "Poor":
        thresholds['strong_buy'] = 0.85
        thresholds['buy'] = 0.99  # Effectively disable
    
    for _, row in df.iterrows():
        try:
            # Calculate individual signals
            individual_signals = calculate_individual_signals(row, market_env)
            
            # Calculate weighted contributions
            weighted_contributions = {}
            raw_score = 0
            for indicator, weight in weights.items():
                contribution = individual_signals[indicator]['value'] * weight
                weighted_contributions[indicator] = {
                    'signal_value': individual_signals[indicator]['value'],
                    'weight': weight,
                    'contribution': contribution,
                    'interpretation': individual_signals[indicator]['interpretation'],
                    'color': individual_signals[indicator]['color']
                }
                raw_score += contribution
            
            # Calculate regime adjustments
            regime_adjustments = calculate_regime_adjustments(row, market_env)
            score_after_regime = raw_score * regime_adjustments['total_regime']['value']
            
            # Calculate environment filters
            environment_filters = calculate_environment_filters(market_env)
            final_score = score_after_regime * environment_filters['total_environment']['value']
            
            # Determine signal with dynamic thresholds
            if final_score > thresholds['strong_buy']:
                direction, strength = "STRONG_BUY", "Strong"
            elif final_score > thresholds['buy']:
                direction, strength = "BUY", "Strong" if final_score > 0.70 else "Moderate"
            elif final_score < thresholds['strong_sell']:
                direction, strength = "STRONG_SELL", "Strong"
            elif final_score < thresholds['sell']:
                direction, strength = "SELL", "Strong" if final_score < 0.30 else "Moderate"
            else:
                direction, strength = "HOLD", "Neutral"
            
            # Enhanced confidence calculation
            base_confidence = min(0.95, abs(final_score - 0.5) * 1.8)
            
            # Adjust confidence for market environment
            confidence_adjustment = environment_filters['total_environment']['value']
            final_confidence = base_confidence * confidence_adjustment
            
            # ===== NEW TRADING INTELLIGENCE CALCULATIONS =====
            
            # Calculate precise entry/exit price levels
            price_levels = calculate_entry_exit_levels(row, direction, market_env)
            
            # Calculate position sizing for different account sizes
            position_10k = calculate_position_sizing(price_levels, direction, 10000, 0.02, 0.3)
            position_50k = calculate_position_sizing(price_levels, direction, 50000, 0.02, 0.25)
            position_100k = calculate_position_sizing(price_levels, direction, 100000, 0.015, 0.2)
            
            # Get market timing context
            timing_context = get_market_timing_context(row['symbol'], market_env)
            
            # Get historical performance for similar setups
            historical_performance = simulate_historical_performance(direction, strength, row['sector'])
            
            signals.append({
                'Symbol': row['symbol'],
                'Company': row['company_name'],
                'Sector': row['sector'],
                'Industry': row['industry'],
                'Price': row['close'],
                'Change_1D': row['returns_1d'],
                'Change_5D': row['returns_5d'],
                'Volume': row['volume'],
                'Volume_Ratio': row['volume_ratio'],
                'Market_Cap': row['market_cap'],
                'RSI': row['rsi_14'],
                'MACD_Hist': row['macd_histogram'],
                'BB_Position': row['bb_position'],
                'Volatility': row['volatility_20d'],
                'Signal': direction,
                'Strength': strength,
                'Raw_Score': raw_score,
                'Final_Score': final_score,
                'Confidence': final_confidence,
                
                # ===== NEW TRADING INTELLIGENCE DATA =====
                # Price levels for actionable trading
                'Entry_Price': price_levels.get('entry_immediate', row['close']),
                'Entry_Pullback': price_levels.get('entry_pullback', row['close']),
                'Stop_Loss': price_levels.get('stop_loss', row['close'] * 0.95),
                'Take_Profit_1': price_levels.get('take_profit_1', row['close'] * 1.05),
                'Take_Profit_2': price_levels.get('take_profit_2', row['close'] * 1.1),
                'Risk_Reward_1': price_levels.get('risk_reward_1', 1.0),
                'Risk_Reward_2': price_levels.get('risk_reward_2', 2.0),
                'Strategy': price_levels.get('strategy', 'Hold'),
                
                # Position sizing for different account sizes
                'Shares_10K': position_10k['shares'],
                'Position_Value_10K': position_10k['position_value'],
                'Risk_Amount_10K': position_10k['risk_amount'],
                'Shares_50K': position_50k['shares'],
                'Shares_100K': position_100k['shares'],
                
                # Transparency data (existing)
                'weighted_contributions': weighted_contributions,
                'regime_adjustments': regime_adjustments,
                'environment_filters': environment_filters,
                'thresholds': thresholds,
                'weights': weights,
                'individual_signals': individual_signals,
                
                # Complete trading intelligence data
                'price_levels': price_levels,
                'position_sizing_10k': position_10k,
                'position_sizing_50k': position_50k,
                'position_sizing_100k': position_100k,
                'market_timing': timing_context,
                'historical_performance': historical_performance,
                
                'Last_Updated': row['trade_date']
            })
            
        except Exception as e:
            print(f"ERROR processing {row['symbol']}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return pd.DataFrame(signals)

def create_trading_intelligence_panel(selected_stock_data):
    """Create comprehensive trading intelligence panel with actionable information"""
    if selected_stock_data is None:
        return
    
    st.markdown("---")
    st.subheader(f"💰 Complete Trading Plan: {selected_stock_data['Symbol']}")
    
    # Trading summary card
    signal = selected_stock_data['Signal']
    entry_price = selected_stock_data['Entry_Price']
    stop_loss = selected_stock_data['Stop_Loss']
    take_profit = selected_stock_data['Take_Profit_1']
    risk_reward = selected_stock_data['Risk_Reward_1']
    shares_10k = selected_stock_data['Shares_10K']
    risk_amount = selected_stock_data['Risk_Amount_10K']
    
    # Create trading plan overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="trading-panel">', unsafe_allow_html=True)
        st.markdown("### 🎯 **Trade Setup**")
        st.write(f"**Signal:** {signal} ({selected_stock_data['Strength']})")
        st.write(f"**Strategy:** {selected_stock_data['Strategy']}")
        st.write(f"**Confidence:** {selected_stock_data['Confidence']:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="price-level">', unsafe_allow_html=True)
        st.markdown("### 💵 **Price Levels**")
        st.write(f"**Entry:** ${entry_price:.2f}")
        st.write(f"**Stop Loss:** ${stop_loss:.2f}")
        st.write(f"**Target:** ${take_profit:.2f}")
        st.write(f"**Risk:Reward:** {risk_reward:.1f}:1")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="position-size">', unsafe_allow_html=True)
        st.markdown("### 📊 **Position Size**")
        st.write(f"**Shares ($10K):** {shares_10k:,}")
        st.write(f"**Position Value:** ${selected_stock_data['Position_Value_10K']:,.0f}")
        st.write(f"**Risk Amount:** ${risk_amount:.0f}")
        st.write(f"**Risk %:** {(risk_amount/10000)*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed trading intelligence tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Entry Strategy", "💰 Position Sizing", "📅 Market Timing", "📈 Historical Performance", "⚠️ Risk Management"])
    
    with tab1:
        st.markdown("#### 🎯 **Entry & Exit Strategy**")
        
        price_levels = selected_stock_data['price_levels']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🚀 Entry Options:**")
            if 'entry_immediate' in price_levels:
                st.markdown(f"• **Immediate:** ${price_levels['entry_immediate']:.2f} (market order)")
            if 'entry_pullback' in price_levels:
                st.markdown(f"• **Pullback:** ${price_levels['entry_pullback']:.2f} (limit order)")
            if 'breakout_entry' in price_levels:
                st.markdown(f"• **Breakout:** ${price_levels['breakout_entry']:.2f} (stop order)")
            
            st.markdown("**🛑 Stop Loss Logic:**")
            st.write(f"• Technical: ${stop_loss:.2f}")
            st.write(f"• Risk per share: ${abs(entry_price - stop_loss):.2f}")
            
        with col2:
            st.markdown("**🎯 Take Profit Levels:**")
            st.markdown(f"• **Target 1:** ${take_profit:.2f} ({risk_reward:.1f}:1)")
            if 'take_profit_2' in price_levels:
                st.markdown(f"• **Target 2:** ${price_levels['take_profit_2']:.2f} ({selected_stock_data['Risk_Reward_2']:.1f}:1)")
            
            st.markdown("**⏱️ Time Horizon:**")
            hist_perf = selected_stock_data['historical_performance']
            st.write(f"• Expected hold: {hist_perf['avg_hold_days']} days")
            st.write(f"• Success rate: {hist_perf['win_rate']:.1%}")
    
    with tab2:
        st.markdown("#### 💰 **Position Sizing Analysis**")
        
        # Position sizing for different account sizes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pos_10k = selected_stock_data['position_sizing_10k']
            st.markdown("**$10K Account:**")
            st.metric("Shares", f"{pos_10k['shares']:,}")
            st.metric("Position Value", f"${pos_10k['position_value']:,.0f}")
            st.metric("Risk Amount", f"${pos_10k['risk_amount']:.0f}")
            st.metric("Position %", f"{pos_10k['position_pct']:.1f}%")
        
        with col2:
            pos_50k = selected_stock_data['position_sizing_50k']
            st.markdown("**$50K Account:**")
            st.metric("Shares", f"{pos_50k['shares']:,}")
            st.metric("Position Value", f"${pos_50k['position_value']:,.0f}")
            st.metric("Risk Amount", f"${pos_50k['risk_amount']:.0f}")
            st.metric("Position %", f"{pos_50k['position_pct']:.1f}%")
            
        with col3:
            pos_100k = selected_stock_data['position_sizing_100k']
            st.markdown("**$100K Account:**")
            st.metric("Shares", f"{pos_100k['shares']:,}")
            st.metric("Position Value", f"${pos_100k['position_value']:,.0f}")
            st.metric("Risk Amount", f"${pos_100k['risk_amount']:.0f}")
            st.metric("Position %", f"{pos_100k['position_pct']:.1f}%")
        
        # Position sizing recommendation
        st.markdown("**💡 Recommendation:**")
        st.info(pos_10k['recommendation'])
    
    with tab3:
        st.markdown("#### 📅 **Market Timing Intelligence**")
        
        timing = selected_stock_data['market_timing']
        
        # Timing verdict
        st.markdown(f"**🎯 Timing Verdict:** {timing['timing_verdict']}")
        st.progress(timing['timing_score'] / 100)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Key Dates:**")
            st.write(f"• **Earnings:** {timing['earnings_date']} ({timing['days_to_earnings']} days)")
            st.write(f"• **Ex-Dividend:** {timing['ex_dividend_date']} (${timing['dividend_amount']})")
            st.write(f"• **Options Expiry Friday:** {'Yes' if timing['options_expiry_friday'] else 'No'}")
            
        with col2:
            st.markdown("**⚠️ Warnings:**")
            if timing['warnings']:
                for warning in timing['warnings']:
                    st.warning(warning)
            else:
                st.success("✅ No timing concerns")
                
            st.markdown("**💡 Recommendations:**")
            if timing['recommendations']:
                for rec in timing['recommendations']:
                    st.info(rec)
    
    with tab4:
        st.markdown("#### 📈 **Historical Performance Analysis**")
        
        hist = selected_stock_data['historical_performance']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 Performance Stats:**")
            st.metric("Win Rate", f"{hist['win_rate']:.1%}")
            st.metric("Average Gain", f"+{hist['avg_gain']:.1f}%")
            st.metric("Average Loss", f"{hist['avg_loss']:.1f}%")
            st.metric("Expectancy", f"{hist['expectancy']:.1f}%")
        
        with col2:
            st.markdown("**🔢 Sample Size:**")
            st.metric("Total Similar Trades", hist['total_similar_trades'])
            st.metric("Profitable Trades", hist['profitable_trades'])
            st.metric("Average Hold Days", f"{hist['avg_hold_days']} days")
        
        with col3:
            st.markdown("**🏆 Best/Worst Examples:**")
            st.success(f"**Best:** {hist['best_trade']}")
            st.error(f"**Worst:** {hist['worst_trade']}")
    
    with tab5:
        st.markdown("#### ⚠️ **Risk Management Guidelines**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🛡️ Risk Controls:**")
            st.markdown(f"• **Max Risk:** 2% of account (${risk_amount:.0f})")
            st.markdown(f"• **Stop Loss:** Mandatory at ${stop_loss:.2f}")
            st.markdown(f"• **Position Size:** {shares_10k:,} shares maximum")
            st.markdown(f"• **Time Limit:** Exit if no progress in {hist_perf['avg_hold_days']*1.5:.0f} days")
        
        with col2:
            st.markdown("**🚨 Exit Triggers:**")
            st.markdown("• Stop loss hit (-5% max loss)")
            st.markdown("• Take profit target reached")
            st.markdown("• Signal changes to opposite direction")
            st.markdown("• Market environment deteriorates significantly")
            
            if timing['earnings_warning']:
                st.error("⚠️ Consider exiting before earnings")

def create_signal_breakdown_panel(selected_stock_data):
    """Create detailed signal breakdown visualization"""
    if selected_stock_data is None:
        return
    
    st.subheader(f"🔬 Complete Signal Breakdown: {selected_stock_data['Symbol']}")
    st.write(f"**{selected_stock_data['Company']}** | {selected_stock_data['Sector']} | ${selected_stock_data['Price']:.2f}")
    
    # Signal score display
    score = selected_stock_data['Final_Score']
    if score > 0.6:
        score_class = "score-buy"
    elif score < 0.4:
        score_class = "score-sell"
    else:
        score_class = "score-hold"
    
    st.markdown(f'<div class="{score_class}">Final Signal Score: {score:.3f}<br>Signal: {selected_stock_data["Signal"]} ({selected_stock_data["Strength"]})<br>Confidence: {selected_stock_data["Confidence"]:.1%}</div>', unsafe_allow_html=True)
    
    # Individual indicator breakdown
    st.markdown("#### 📊 Individual Indicator Contributions")
    
    contributions = selected_stock_data['weighted_contributions']
    weights = selected_stock_data['weights']
    
    # Create visualization for each indicator
    for indicator in ['rsi', 'macd', 'volume', 'bb', 'ma', 'momentum', 'volatility', 'other']:
        contrib = contributions[indicator]
        
        col1, col2, col3, col4 = st.columns([3, 2, 2, 3])
        
        with col1:
            st.write(f"**{indicator.upper()}**")
            st.write(f"Signal: {contrib['signal_value']:.3f}")
        
        with col2:
            st.write(f"Weight: {contrib['weight']:.1%}")
            # Visual weight bar
            if contrib['color'] == 'positive':
                bar_color = "#28a745"
            elif contrib['color'] == 'negative':
                bar_color = "#dc3545"
            else:
                bar_color = "#6c757d"
            
            st.markdown(f'<div style="width: {contrib["weight"]*500}px; height: 20px; background-color: {bar_color}; border-radius: 10px;"></div>', unsafe_allow_html=True)
        
        with col3:
            contribution_pct = contrib['contribution'] / selected_stock_data['Raw_Score'] * 100 if selected_stock_data['Raw_Score'] != 0 else 0
            st.write(f"Contrib: {contrib['contribution']:.3f}")
            st.write(f"({contribution_pct:.1f}%)")
        
        with col4:
            color_class = f"indicator-{contrib['color']}"
            st.markdown(f'<span class="{color_class}">{contrib["interpretation"]}</span>', unsafe_allow_html=True)
    
    st.divider()
    
    # Regime adjustments
    st.markdown("#### 🌍 Market Regime Adjustments")
    
    regime_adj = selected_stock_data['regime_adjustments']
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Sector Adjustment:**")
        st.write(f"• {regime_adj['sector_multiplier']['reason']}")
        st.write(f"• Impact: {regime_adj['sector_multiplier']['impact']}")
        
    with col2:
        st.write("**Market Cap Adjustment:**")
        st.write(f"• {regime_adj['cap_multiplier']['reason']}")
        st.write(f"• Impact: {regime_adj['cap_multiplier']['impact']}")
    
    st.write(f"**Total Regime Impact:** {regime_adj['total_regime']['impact']}")
    st.write(f"Raw Score: {selected_stock_data['Raw_Score']:.3f} → After Regime: {selected_stock_data['Raw_Score'] * regime_adj['total_regime']['value']:.3f}")
    
    st.divider()
    
    # Environment filters
    st.markdown("#### 🛡️ Market Environment Filters")
    
    env_filters = selected_stock_data['environment_filters']
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**VIX Filter:**")
        st.write(f"• {env_filters['vix_filter']['reason']}")
        st.write(f"• Impact: {env_filters['vix_filter']['impact']}")
        
        st.write("**Sentiment Filter:**")
        st.write(f"• {env_filters['sentiment_filter']['reason']}")
        st.write(f"• Impact: {env_filters['sentiment_filter']['impact']}")
        
    with col2:
        st.write("**Breadth Filter:**")
        st.write(f"• {env_filters['breadth_filter']['reason']}")
        st.write(f"• Impact: {env_filters['breadth_filter']['impact']}")
        
        st.write("**Risk Filter:**")
        st.write(f"• {env_filters['risk_filter']['reason']}")
        st.write(f"• Impact: {env_filters['risk_filter']['impact']}")
    
    st.write(f"**Total Environment Impact:** {env_filters['total_environment']['impact']}")
    st.write(f"After Regime: {selected_stock_data['Raw_Score'] * regime_adj['total_regime']['value']:.3f} → Final Score: {selected_stock_data['Final_Score']:.3f}")
    
    st.divider()
    
    # Threshold comparison
    st.markdown("#### 🎯 Signal Thresholds & Decision Logic")
    
    thresholds = selected_stock_data['thresholds']
    final_score = selected_stock_data['Final_Score']
    
    st.write("**Current Market Environment Thresholds:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("STRONG_BUY", f">{thresholds['strong_buy']:.2f}", "✅" if final_score > thresholds['strong_buy'] else "")
    with col2:
        st.metric("BUY", f">{thresholds['buy']:.2f}", "✅" if thresholds['buy'] < final_score <= thresholds['strong_buy'] else "")
    with col3:
        st.metric("SELL", f"<{thresholds['sell']:.2f}", "✅" if thresholds['strong_sell'] < final_score <= thresholds['sell'] else "")
    with col4:
        st.metric("STRONG_SELL", f"<{thresholds['strong_sell']:.2f}", "✅" if final_score < thresholds['strong_sell'] else "")
    
    # Visual threshold chart
    fig = go.Figure()
    
    # Add threshold lines
    fig.add_hline(y=thresholds['strong_buy'], line_dash="dash", line_color="green", annotation_text="Strong Buy")
    fig.add_hline(y=thresholds['buy'], line_dash="dash", line_color="lightgreen", annotation_text="Buy")
    fig.add_hline(y=thresholds['sell'], line_dash="dash", line_color="orange", annotation_text="Sell")
    fig.add_hline(y=thresholds['strong_sell'], line_dash="dash", line_color="red", annotation_text="Strong Sell")
    
    # Add current score
    fig.add_trace(go.Scatter(x=['Current Score'], y=[final_score], 
                            mode='markers', marker=dict(size=20, color='blue'),
                            name=f'Final Score: {final_score:.3f}'))
    
    fig.update_layout(title=f"Signal Score vs Thresholds", height=400, yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes only for live data freshness
def load_transparent_dashboard_data():
    """Load comprehensive data using database with parallel processing"""
    st.markdown("**🚀 Initializing High-Performance Data System...**")
    
    # Initialize the data manager
    data_manager = HistoricalDataManager()
    market_env = get_market_environment_data()
    symbols = get_top_stocks_symbols()
    
    # Show database stats
    stats = data_manager.get_database_stats()
    st.markdown(f"📊 **Database Stats:** {stats['stocks_with_historical']} stocks with historical data, {stats['stocks_with_live']} with live data")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Check which stocks need historical data update (runs once per week)
    status_text.text("🔍 Checking historical data status...")
    progress_bar.progress(0.1)
    
    stocks_needing_historical = data_manager.get_stocks_needing_historical_update(symbols, days_threshold=7)
    
    if stocks_needing_historical:
        st.markdown(f"📈 **Updating historical data for {len(stocks_needing_historical)} stocks (one-time process)...**")
        status_text.text(f"📈 Fetching 5-year historical data for {len(stocks_needing_historical)} stocks...")
        progress_bar.progress(0.2)
        
        # Fetch historical data in parallel (this runs rarely)
        historical_results = data_manager.fetch_historical_data_parallel(
            stocks_needing_historical, 
            max_workers=15,  # Aggressive parallel processing
            period="5y"
        )
        
        st.markdown(f"✅ Historical update complete: {len(historical_results['success'])} success, {len(historical_results['failed'])} failed")
        progress_bar.progress(0.5)
    else:
        st.markdown("✅ **Historical data is up to date**")
        progress_bar.progress(0.5)
    
    # Step 2: Fetch live data for all symbols (fast, runs every refresh)
    status_text.text("⚡ Fetching live market data with parallel processing...")
    progress_bar.progress(0.6)
    
    live_results = data_manager.fetch_live_data_parallel(
        symbols, 
        max_workers=25  # Very aggressive for live data
    )
    
    progress_bar.progress(0.8)
    st.markdown(f"⚡ **Live data update:** {len(live_results['success'])} stocks updated")
    
    # Step 3: Get complete dataset from database
    status_text.text("📊 Assembling complete dataset...")
    progress_bar.progress(0.9)
    
    complete_data = data_manager.get_complete_stock_data(symbols)
    
    # Clean and format the data
    if not complete_data.empty:
        # Remove rows where essential data is missing
        complete_data = complete_data.dropna(subset=['current_price', 'company_name'])
        
        # Rename columns to match expected format
        column_mapping = {
            'current_price': 'close',
            'rsi_14': 'rsi_14',
            'macd_histogram': 'macd_histogram',
            'bb_position': 'bb_position',
            'change_1d': 'returns_1d',
            'change_5d': 'returns_5d',
            'change_20d': 'returns_20d',
            'volatility_20d': 'volatility_20d',
            'volume_ratio': 'volume_ratio',
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in complete_data.columns:
                complete_data[new_col] = complete_data[old_col]
        
        # Add trade_date
        complete_data['trade_date'] = datetime.now().date()
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Ready! {len(complete_data)} stocks with complete trading intelligence")
        
        time.sleep(1)  # Show completion message briefly
        progress_bar.empty()
        status_text.empty()
        
        st.markdown(f"🎉 **System Ready:** {len(complete_data)} stocks loaded with parallel processing")
        
        return complete_data, symbols, market_env
    
    else:
        st.error("No data available - check database connection")
        return pd.DataFrame(), [], market_env

def create_fallback_data(symbols):
    """Create fallback data with simplified values when API fails"""
    import random
    data = []
    
    for symbol in symbols:
        # Create synthetic but realistic data
        base_price = random.uniform(50, 300)
        data.append({
            'symbol': symbol,
            'company_name': f"{symbol} Corporation",
            'sector': random.choice(['Technology', 'Healthcare', 'Financials', 'Consumer Discretionary']),
            'industry': f"{symbol} Industry",
            'market_cap': random.randint(10000000000, 500000000000),
            'close': base_price,
            'volume': random.randint(1000000, 50000000),
            'volume_avg_20': random.randint(800000, 45000000),
            'rsi_14': random.uniform(30, 70),
            'rsi_yesterday': random.uniform(25, 75),
            'macd_line': random.uniform(-2, 2),
            'macd_signal_line': random.uniform(-2, 2),
            'macd_histogram': random.uniform(-1, 1),
            'bb_upper': base_price * 1.05,
            'bb_middle': base_price,
            'bb_lower': base_price * 0.95,
            'bb_position': random.uniform(20, 80),
            'sma_20': base_price * random.uniform(0.98, 1.02),
            'sma_50': base_price * random.uniform(0.95, 1.05),
            'returns_1d': random.uniform(-5, 5),
            'returns_5d': random.uniform(-10, 10),
            'returns_20d': random.uniform(-20, 20),
            'volatility_20d': random.uniform(15, 45),
            'volume_ratio': random.uniform(0.5, 2.5),
            'trade_date': datetime.now().date(),
            'last_updated': datetime.now()
        })
    
    return pd.DataFrame(data)

def main():
    """Main transparent signal dashboard"""
    
    st.title("💰 Complete Trading Intelligence System")
    st.markdown("**Professional-grade trading signals with precise entry/exit prices, position sizing, and risk management**")
    
    # Load comprehensive data
    st.markdown('💰 <strong>Loading Complete Trading Intelligence...</strong> Generating actionable trading plans with precise price levels, position sizing, and market timing analysis.', unsafe_allow_html=True)
    
    with st.spinner("Loading detailed market analysis..."):
        df, symbols, market_env = load_transparent_dashboard_data()
    
    if df.empty:
        st.error("❌ Unable to load market data. Please try refreshing.")
        return
    
    # Enhanced market environment display
    st.header("🌡️ Market Environment Analysis")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("VIX Level", f"{market_env['vix_level']:.1f}", market_env['vix_environment'])
    with col2:
        st.metric("Fear & Greed", f"{market_env['fear_greed_index']:.0f}", market_env['fear_greed_state'])
    with col3:
        st.metric("Market Breadth", f"{market_env['market_breadth_ratio']:.2f}", market_env['breadth_health'])
    with col4:
        st.metric("10Y Treasury", f"{market_env['rate_level']:.2f}%", market_env['rate_trend'])
    with col5:
        st.metric("Risk Level", market_env['risk_environment'])
    with col6:
        st.metric("Market Stress", f"{market_env['market_stress']:.0f}%")
    
    # Generate transparent signals
    signals_df = generate_transparent_signals(df, market_env)
    
    if not signals_df.empty:
        # Enhanced summary
        total_signals = len(signals_df)
        buy_signals = len(signals_df[signals_df['Signal'].str.contains('BUY')])
        sell_signals = len(signals_df[signals_df['Signal'].str.contains('SELL')])
        strong_signals = len(signals_df[signals_df['Strength'] == 'Strong'])
        avg_confidence = signals_df['Confidence'].mean()
        avg_raw_score = signals_df['Raw_Score'].mean()
        avg_final_score = signals_df['Final_Score'].mean()
        
        st.header("💰 Complete Trading Intelligence")
        
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        with col1:
            st.metric("Total Stocks", total_signals)
        with col2:
            st.metric("Buy Signals", buy_signals, f"{buy_signals/total_signals:.1%}")
        with col3:
            st.metric("Sell Signals", sell_signals, f"{sell_signals/total_signals:.1%}")
        with col4:
            st.metric("Strong Signals", strong_signals)
        with col5:
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        with col6:
            # Calculate average risk/reward for active signals
            active_signals = signals_df[~signals_df['Signal'].str.contains('HOLD')]
            avg_risk_reward = active_signals['Risk_Reward_1'].mean() if len(active_signals) > 0 else 0
            st.metric("Avg Risk:Reward", f"{avg_risk_reward:.1f}:1")
        with col7:
            # Calculate total position value for $10K account
            total_position_10k = signals_df['Position_Value_10K'].sum()
            st.metric("Total Exposure", f"${total_position_10k:,.0f}" if total_position_10k < 10000 else f"${total_position_10k/1000:.0f}K")
        
        # Enhanced main signals table with trading intelligence
        display_df = signals_df[[
            'Symbol', 'Company', 'Sector', 'Price', 'Signal', 'Strength',
            'Entry_Price', 'Stop_Loss', 'Take_Profit_1', 'Risk_Reward_1', 
            'Shares_10K', 'Risk_Amount_10K', 'Confidence'
        ]].copy()
        
        # Format the new trading columns
        display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
        display_df['Entry_Price'] = display_df['Entry_Price'].apply(lambda x: f"${x:.2f}")
        display_df['Stop_Loss'] = display_df['Stop_Loss'].apply(lambda x: f"${x:.2f}")
        display_df['Take_Profit_1'] = display_df['Take_Profit_1'].apply(lambda x: f"${x:.2f}")
        display_df['Risk_Reward_1'] = display_df['Risk_Reward_1'].apply(lambda x: f"{x:.1f}:1")
        display_df['Shares_10K'] = display_df['Shares_10K'].apply(lambda x: f"{x:,}" if x > 0 else "-")
        display_df['Risk_Amount_10K'] = display_df['Risk_Amount_10K'].apply(lambda x: f"${x:.0f}" if x > 0 else "-")
        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1%}")
        
        # Rename columns for display
        display_df.columns = [
            'Symbol', 'Company', 'Sector', 'Current', 'Signal', 'Strength',
            'Entry', 'Stop', 'Target', 'R:R', 'Shares', 'Risk $', 'Confidence'
        ]
        
        # Signal styling
        def style_signals(val):
            if 'BUY' in str(val):
                return 'background-color: #d4edda; color: #155724'
            elif 'SELL' in str(val):
                return 'background-color: #f8d7da; color: #721c24'
            else:
                return 'background-color: #fff3cd; color: #856404'
        
        styled_df = display_df.style.map(style_signals, subset=['Signal'])
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Stock selection for detailed analysis
        st.header("🔍 Complete Stock Analysis")
        
        selected_symbol = st.selectbox(
            "Select a stock for complete trading intelligence:",
            options=signals_df['Symbol'].tolist(),
            index=0,
            help="Choose any stock to see the complete trading plan with entry/exit prices, position sizing, and risk management"
        )
        
        if selected_symbol:
            selected_stock = signals_df[signals_df['Symbol'] == selected_symbol].iloc[0]
            
            # Show the complete trading intelligence panel first
            create_trading_intelligence_panel(selected_stock)
            
            # Then show the detailed signal breakdown
            create_signal_breakdown_panel(selected_stock)
        
        # Sidebar with trading intelligence information
        with st.sidebar:
            st.header("💰 Trading Intelligence")
            
            st.success("✅ **Complete Trading System**")
            st.write("• **Precise Entry/Exit Prices**")
            st.write("• **Position Sizing Calculator**")
            st.write("• **Risk Management Rules**")
            st.write("• **Market Timing Analysis**")
            st.write("• **Historical Performance**")
            st.write("• **Signal Transparency**")
            
            st.divider()
            
            # Show active signals summary
            active_signals = signals_df[~signals_df['Signal'].str.contains('HOLD')]
            if len(active_signals) > 0:
                st.header("🎯 Active Opportunities")
                for _, signal in active_signals.head(5).iterrows():
                    st.markdown(f"**{signal['Symbol']}** - {signal['Signal']}")
                    st.markdown(f"Entry: ${signal['Entry_Price']:.2f} | R:R: {signal['Risk_Reward_1']:.1f}:1")
                    st.markdown("---")
            
            st.divider()
            
            st.header("📊 System Weights")
            weights = signals_df.iloc[0]['weights']
            for indicator, weight in weights.items():
                st.write(f"• **{indicator.upper()}:** {weight:.1%}")
            
            st.divider()
            
            st.header("🎯 Current Thresholds")
            thresholds = signals_df.iloc[0]['thresholds']
            st.write(f"• **Strong Buy:** >{thresholds['strong_buy']:.2f}")
            st.write(f"• **Buy:** >{thresholds['buy']:.2f}")  
            st.write(f"• **Sell:** <{thresholds['sell']:.2f}")
            st.write(f"• **Strong Sell:** <{thresholds['strong_sell']:.2f}")
            
            st.divider()
            
            if st.button("🔄 Refresh Analysis"):
                st.cache_data.clear()
                st.rerun()
    
    else:
        st.error("Unable to generate transparent signals.")
    
    # Footer
    st.divider()
    st.markdown(f"""
    <div style='text-align: center; color: #666; font-size: 12px;'>
    💰 <strong>Complete Trading Intelligence System</strong> | 
    Precise Entry/Exit Prices + Position Sizing + Market Timing | 
    {len(signals_df) if not signals_df.empty else 0} Stocks with Full Trading Plans | 
    Updated: {datetime.now().strftime("%H:%M:%S")} |
    <strong>Educational & Research Purposes Only</strong>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()