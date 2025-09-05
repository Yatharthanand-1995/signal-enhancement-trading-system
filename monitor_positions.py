#!/usr/bin/env python3
"""
Monitor Paper Trading Positions
Real-time monitoring and performance tracking of current positions
"""

import sys
import os
sys.path.append('src')

from src.dashboard.components.paper_trading import PaperTradingEngine
from src.dashboard.components.historical_signal_trader import HistoricalSignalTrader
import pandas as pd
from datetime import datetime
import time

def main():
    print("📊 Paper Trading Position Monitor")
    print("=" * 50)
    
    # Initialize paper trading engine
    paper_engine = PaperTradingEngine()
    historical_trader = HistoricalSignalTrader(paper_engine)
    
    print(f"🕐 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get current portfolio status
    print("💼 PORTFOLIO OVERVIEW")
    print("-" * 30)
    
    metrics = paper_engine.get_performance_metrics()
    positions_df = paper_engine.get_positions_summary()
    
    # Portfolio summary
    print(f"📈 Portfolio Value: ${metrics['current_value']:,.2f}")
    print(f"💰 Cash Available: ${metrics['cash']:,.2f}")
    print(f"📊 Total Return: {metrics['total_return']:+.2f}%")
    print(f"💸 Total P&L: ${metrics['total_pnl']:+,.2f}")
    print(f"🎯 Win Rate: {metrics['win_rate']:.1f}%")
    print()
    
    if not positions_df.empty:
        print("🔍 CURRENT POSITIONS")
        print("-" * 30)
        
        # Update all positions with current prices
        print("🔄 Updating positions with live prices...")
        paper_engine.update_positions()
        
        # Get updated positions
        updated_positions = paper_engine.get_positions_summary()
        
        total_unrealized_pnl = 0
        best_performer = {'symbol': '', 'return': -999}
        worst_performer = {'symbol': '', 'return': 999}
        
        for _, pos in updated_positions.iterrows():
            symbol = pos['Symbol']
            shares = pos['Shares']
            entry_price = pos['Entry Price']
            current_price = pos['Current Price']
            return_pct = pos['Return %']
            pnl = pos['P&L']
            value = pos['Value']
            days_held = pos['Days Held']
            signal = pos['Signal']
            confidence = pos['Confidence']
            
            # Clean numeric values if they're strings
            if isinstance(return_pct, str):
                return_num = float(return_pct.replace('%', '').replace('+', ''))
            else:
                return_num = return_pct
                
            if isinstance(pnl, str):
                pnl_num = float(pnl.replace('$', '').replace(',', '').replace('+', ''))
            else:
                pnl_num = pnl
            
            total_unrealized_pnl += pnl_num
            
            # Track best/worst performers
            if return_num > best_performer['return']:
                best_performer = {'symbol': symbol, 'return': return_num}
            if return_num < worst_performer['return']:
                worst_performer = {'symbol': symbol, 'return': return_num}
            
            # Position status
            status = "🟢" if return_num > 0 else "🔴" if return_num < 0 else "⚪"
            
            print(f"\n{status} {symbol} ({signal} @ {confidence:.1%} confidence)")
            print(f"   ├─ Position: {shares:.0f} shares")
            print(f"   ├─ Entry: ${entry_price} → Current: ${current_price}")
            print(f"   ├─ Return: {return_pct} | P&L: {pnl}")
            print(f"   ├─ Value: {value} | Days held: {days_held}")
            
            # Position alerts
            if return_num <= -5:
                print(f"   ⚠️  WARNING: Position down {abs(return_num):.1f}%")
            elif return_num >= 5:
                print(f"   🎉 WINNER: Position up {return_num:.1f}%")
        
        print(f"\n📊 POSITION SUMMARY")
        print(f"   Total Unrealized P&L: ${total_unrealized_pnl:+,.2f}")
        print(f"   Best Performer: {best_performer['symbol']} ({best_performer['return']:+.2f}%)")
        print(f"   Worst Performer: {worst_performer['symbol']} ({worst_performer['return']:+.2f}%)")
        
        # Risk analysis
        portfolio_value = metrics['current_value']
        cash_pct = (metrics['cash'] / portfolio_value) * 100
        stock_pct = 100 - cash_pct
        
        print(f"\n🎯 RISK ANALYSIS")
        print(f"   Cash Allocation: {cash_pct:.1f}%")
        print(f"   Stock Allocation: {stock_pct:.1f}%")
        print(f"   Active Positions: {len(updated_positions)}")
        print(f"   Max Drawdown: -{metrics['max_drawdown']:.2f}%")
        
    else:
        print("📭 No current positions")
    
    print()
    
    # Check for yesterday's signals to execute
    print("📅 YESTERDAY'S SIGNALS CHECK")
    print("-" * 30)
    
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    yesterdays_signals = historical_trader.get_yesterdays_signals()
    
    if yesterdays_signals:
        unprocessed_signals = []
        for symbol, signal_data in yesterdays_signals.items():
            signal_direction = signal_data.get('signal', 'NEUTRAL')
            confidence = signal_data.get('confidence', 0)
            
            # Check if this would be a new trade
            if signal_direction in ['BUY', 'STRONG_BUY'] and symbol not in paper_engine.positions:
                if confidence >= paper_engine.min_confidence:
                    unprocessed_signals.append((symbol, signal_direction, confidence))
        
        if unprocessed_signals:
            print(f"⚠️  Found {len(unprocessed_signals)} unprocessed signals:")
            for symbol, direction, conf in unprocessed_signals:
                print(f"   • {symbol}: {direction} (confidence: {conf:.1%})")
            print("   💡 Run execute_signals_auto.py to process these signals")
        else:
            print("✅ All applicable yesterday's signals have been processed")
    else:
        print("📭 No yesterday's signals found")
    
    print()
    
    # Store today's signals
    print("💾 TODAY'S SIGNALS STORAGE")
    print("-" * 30)
    
    # Check if today's signals are already stored
    today = datetime.now().strftime('%Y-%m-%d')
    history_summary = historical_trader.get_signal_history_summary()
    
    if today in historical_trader.get_available_dates():
        today_signals = historical_trader.signal_history.get(today, {})
        print(f"✅ Today's signals already stored: {len(today_signals)} signals")
    else:
        print("⚠️  Today's signals not yet stored")
        print("   💡 Visit the Market Signals dashboard to store today's signals")
    
    print(f"\n📈 Signal History: {history_summary['total_days']} days, {history_summary['total_signals']} total signals")
    
    print()
    print("🔗 Dashboard: http://localhost:8501 → Paper Trading")
    print("🎯 Next Actions:")
    print("   1. Monitor position performance throughout the day")
    print("   2. Store today's signals by visiting Market Signals dashboard")
    print("   3. Execute tomorrow's trades based on today's signals")

if __name__ == "__main__":
    # Import timedelta here since we use it
    from datetime import timedelta
    main()