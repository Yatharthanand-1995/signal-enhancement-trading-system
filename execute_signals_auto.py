#!/usr/bin/env python3
"""
Auto-Execute Yesterday's Signals for Paper Trading
Automatically executes the trades without user confirmation
"""

import sys
import os
sys.path.append('src')

from src.dashboard.components.paper_trading import PaperTradingEngine
from src.dashboard.components.historical_signal_trader import HistoricalSignalTrader
import pandas as pd
from datetime import datetime, timedelta

def main():
    print("🚀 Paper Trading - Auto-Executing Yesterday's Signals")
    print("=" * 60)
    
    # Initialize paper trading engine
    paper_engine = PaperTradingEngine()
    historical_trader = HistoricalSignalTrader(paper_engine)
    
    print(f"📊 Initial Portfolio Status:")
    print(f"   Cash: ${paper_engine.cash:,.2f}")
    print(f"   Positions: {len(paper_engine.positions)}")
    print(f"   Portfolio Value: ${paper_engine.get_current_portfolio_value():,.2f}")
    print()
    
    # The signals are already stored from the previous run
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"📅 Looking for signals from {yesterday}")
    yesterdays_signals = historical_trader.get_yesterdays_signals(yesterday)
    
    if yesterdays_signals:
        print(f"✅ Found {len(yesterdays_signals)} signals from {yesterday}")
        print()
        
        # Show signals preview
        print("📋 Yesterday's Signals:")
        for symbol, data in yesterdays_signals.items():
            signal = data.get('signal', 'N/A')
            confidence = data.get('confidence', 0)
            print(f"   • {symbol}: {signal} (confidence: {confidence:.1%})")
        print()
        
        print("💰 EXECUTING TRADES:")
        print("-" * 40)
        
        # Execute real trades
        result = historical_trader.execute_yesterdays_signals(target_date=yesterday, dry_run=False)
        
        if result['status'] == 'success':
            print(f"✅ Trade execution successful!")
            print(f"   Signals processed: {result['signals_processed']}")
            print(f"   Total trades executed: {result['trades_executed']}")
            print()
            
            if result['results']:
                print("📈 Executed Trades:")
                for i, trade_result in enumerate(result['results'], 1):
                    symbol = trade_result['symbol']
                    action = trade_result['action_type'] 
                    price = trade_result['current_price']
                    confidence = trade_result['confidence']
                    
                    print(f"   {i}. {action} {symbol}")
                    print(f"      ├─ Price: ${price:.2f}")
                    print(f"      ├─ Confidence: {confidence:.1%}")
                    print(f"      └─ Status: ✅ Executed")
            
            if result['errors']:
                print()
                print("⚠️ Errors encountered:")
                for error in result['errors']:
                    print(f"   • {error}")
            
            print()
            print("📊 Updated Portfolio Status:")
            print("-" * 40)
            metrics = paper_engine.get_performance_metrics()
            print(f"   Cash: ${metrics['cash']:,.2f}")
            print(f"   Active Positions: {metrics['positions_count']}")
            print(f"   Portfolio Value: ${metrics['current_value']:,.2f}")
            print(f"   Total Return: {metrics['total_return']:+.2f}%")
            print(f"   Total P&L: ${metrics['total_pnl']:+,.2f}")
            print(f"   Win Rate: {metrics['win_rate']:.1f}%")
            
            # Show current positions
            if paper_engine.positions:
                print()
                print("📋 Current Positions:")
                print("-" * 40)
                positions_df = paper_engine.get_positions_summary()
                total_position_value = 0
                
                for _, pos in positions_df.iterrows():
                    position_value = pos['Value']
                    if isinstance(position_value, str):
                        # Remove $ and , then convert to float
                        position_value = float(position_value.replace('$', '').replace(',', ''))
                    total_position_value += position_value
                    
                    return_pct = pos['Return %']
                    if isinstance(return_pct, str):
                        return_pct = return_pct.replace('%', '')
                    
                    print(f"   • {pos['Symbol']}:")
                    print(f"     ├─ Shares: {pos['Shares']:.0f}")
                    print(f"     ├─ Entry: ${pos['Entry Price']}")
                    print(f"     ├─ Current: ${pos['Current Price']}")
                    print(f"     ├─ Return: {return_pct}")
                    print(f"     ├─ P&L: {pos['P&L']}")
                    print(f"     └─ Value: {pos['Value']}")
                    print()
                
                print(f"📈 Total Position Value: ${total_position_value:,.2f}")
                cash_allocation = (metrics['cash'] / metrics['current_value']) * 100
                position_allocation = (total_position_value / metrics['current_value']) * 100
                print(f"💰 Cash Allocation: {cash_allocation:.1f}%")
                print(f"📊 Stock Allocation: {position_allocation:.1f}%")
            
            print()
            print("🎯 Paper Trading Summary:")
            print("-" * 40)
            print(f"   Started with: ${paper_engine.initial_capital:,.2f}")
            print(f"   Current value: ${metrics['current_value']:,.2f}")
            print(f"   Gain/Loss: ${metrics['current_value'] - paper_engine.initial_capital:+,.2f}")
            print(f"   Performance: {metrics['total_return']:+.2f}%")
            
            # Show recent trades
            trade_history = paper_engine.get_trade_history_df(limit=10)
            if not trade_history.empty:
                print()
                print("📜 Recent Trade History:")
                print("-" * 40)
                for _, trade in trade_history.tail(5).iterrows():
                    timestamp = trade['Timestamp']
                    symbol = trade['Symbol']
                    action = trade['Action']
                    shares = trade['Shares']
                    price = trade['Price']
                    
                    print(f"   {timestamp}: {action} {shares} shares of {symbol} @ {price}")
        
        else:
            print(f"❌ Trade execution failed: {result.get('error', 'Unknown error')}")
            if result.get('errors'):
                print("Detailed errors:")
                for error in result['errors']:
                    print(f"   • {error}")
    
    else:
        print(f"❌ No signals found for {yesterday}")
        print("ℹ️  Signals need to be stored first by visiting the Market Signals dashboard")
    
    print()
    print("🎉 Paper trading execution complete!")
    print("🔗 Visit http://localhost:8501 and go to 'Paper Trading' to see the full dashboard")

if __name__ == "__main__":
    main()