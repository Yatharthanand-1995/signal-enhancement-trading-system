#!/usr/bin/env python3
"""
Execute Yesterday's Signals for Paper Trading
Demonstrates executing signals based on the live signals we observed
"""

import sys
import os
sys.path.append('src')

from src.dashboard.components.paper_trading import PaperTradingEngine
from src.dashboard.components.historical_signal_trader import HistoricalSignalTrader
import pandas as pd
from datetime import datetime, timedelta

def main():
    print("🚀 Paper Trading - Executing Yesterday's Signals")
    print("=" * 50)
    
    # Initialize paper trading engine
    paper_engine = PaperTradingEngine()
    historical_trader = HistoricalSignalTrader(paper_engine)
    
    print(f"📊 Initial Portfolio Status:")
    print(f"   Cash: ${paper_engine.cash:,.2f}")
    print(f"   Positions: {len(paper_engine.positions)}")
    print(f"   Portfolio Value: ${paper_engine.get_current_portfolio_value():,.2f}")
    print()
    
    # Since we saw these signals in the logs, let's simulate executing them
    # as if they were yesterday's signals
    demo_signals = {
        # BUY Signals (from the logs)
        'CRM': {
            'signal': 'BUY',
            'confidence': 0.79,
            'strength': 0.65,
            'close_price': 280.0,  # Approximate current price
            'volume': 2500000,
            'market_cap': 280000000000,
            'volatility': 0.25,
            'rsi': 55.0,
            'macd': 0.5,
            'timestamp': (datetime.now() - timedelta(days=1)).isoformat()
        },
        'ELV': {
            'signal': 'BUY', 
            'confidence': 0.79,
            'strength': 0.65,
            'close_price': 500.0,
            'volume': 1200000,
            'market_cap': 120000000000,
            'volatility': 0.22,
            'rsi': 58.0,
            'macd': 0.8,
            'timestamp': (datetime.now() - timedelta(days=1)).isoformat()
        },
        'GOOG': {
            'signal': 'BUY',
            'confidence': 0.85,
            'strength': 0.80,
            'close_price': 160.0,
            'volume': 18000000,
            'market_cap': 2000000000000,
            'volatility': 0.28,
            'rsi': 60.0,
            'macd': 1.2,
            'timestamp': (datetime.now() - timedelta(days=1)).isoformat()
        },
        'LLY': {
            'signal': 'BUY',
            'confidence': 0.76,
            'strength': 1.00,
            'close_price': 900.0,
            'volume': 2800000,
            'market_cap': 850000000000,
            'volatility': 0.20,
            'rsi': 65.0,
            'macd': 2.1,
            'timestamp': (datetime.now() - timedelta(days=1)).isoformat()
        },
        'VLO': {
            'signal': 'BUY',
            'confidence': 0.81,
            'strength': 0.62,
            'close_price': 150.0,
            'volume': 3500000,
            'market_cap': 55000000000,
            'volatility': 0.30,
            'rsi': 62.0,
            'macd': 1.5,
            'timestamp': (datetime.now() - timedelta(days=1)).isoformat()
        },
        
        # STRONG_SELL Signals (from the logs)
        'BLK': {
            'signal': 'STRONG_SELL',
            'confidence': 0.66,
            'strength': 1.00,
            'close_price': 800.0,
            'volume': 650000,
            'market_cap': 120000000000,
            'volatility': 0.18,
            'rsi': 25.0,
            'macd': -1.5,
            'timestamp': (datetime.now() - timedelta(days=1)).isoformat()
        },
        'NVDA': {
            'signal': 'STRONG_SELL',
            'confidence': 0.62,
            'strength': 1.00,
            'close_price': 115.0,
            'volume': 45000000,
            'market_cap': 2800000000000,
            'volatility': 0.35,
            'rsi': 20.0,
            'macd': -2.8,
            'timestamp': (datetime.now() - timedelta(days=1)).isoformat()
        }
    }
    
    # Store these as "yesterday's" signals
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    historical_trader.signal_history[yesterday] = demo_signals
    historical_trader._save_signal_history()
    
    print(f"📅 Stored {len(demo_signals)} signals for {yesterday}")
    print()
    
    # First, do a dry run
    print("🧪 DRY RUN - Preview of trades that would be executed:")
    print("-" * 50)
    
    dry_result = historical_trader.execute_yesterdays_signals(target_date=yesterday, dry_run=True)
    
    if dry_result['status'] == 'success':
        print(f"✅ Dry run successful!")
        print(f"   Signals processed: {dry_result['signals_processed']}")
        print(f"   Trades that would be executed: {dry_result['trades_executed']}")
        print()
        
        if dry_result['results']:
            print("📋 Proposed Trades:")
            for i, result in enumerate(dry_result['results'], 1):
                symbol = result['symbol']
                action = result['action_type']
                confidence = result['confidence']
                price = result['current_price']
                reason = result['reason']
                
                print(f"   {i}. {action} {symbol}")
                print(f"      Price: ${price:.2f}")
                print(f"      Confidence: {confidence:.1%}")
                print(f"      Reason: {reason}")
                print()
        
        # Ask for confirmation to execute actual trades
        print("🎯 Ready to execute actual trades!")
        response = input("Execute real trades? (y/N): ").strip().lower()
        
        if response == 'y':
            print()
            print("💰 EXECUTING ACTUAL TRADES:")
            print("-" * 50)
            
            # Execute real trades
            real_result = historical_trader.execute_yesterdays_signals(target_date=yesterday, dry_run=False)
            
            if real_result['status'] == 'success':
                print(f"✅ Trade execution successful!")
                print(f"   Total trades executed: {real_result['trades_executed']}")
                print()
                
                if real_result['results']:
                    print("📈 Executed Trades:")
                    for i, result in enumerate(real_result['results'], 1):
                        symbol = result['symbol']
                        action = result['action_type'] 
                        price = result['current_price']
                        
                        print(f"   {i}. {action} {symbol} at ${price:.2f}")
                
                if real_result['errors']:
                    print()
                    print("⚠️ Errors encountered:")
                    for error in real_result['errors']:
                        print(f"   • {error}")
                
                print()
                print("📊 Updated Portfolio Status:")
                metrics = paper_engine.get_performance_metrics()
                print(f"   Cash: ${metrics['cash']:,.2f}")
                print(f"   Positions: {metrics['positions_count']}")
                print(f"   Portfolio Value: ${metrics['current_value']:,.2f}")
                print(f"   Total Return: {metrics['total_return']:+.2f}%")
                print(f"   Total P&L: ${metrics['total_pnl']:+,.2f}")
                
                # Show current positions
                if paper_engine.positions:
                    print()
                    print("📋 Current Positions:")
                    positions_df = paper_engine.get_positions_summary()
                    for _, pos in positions_df.iterrows():
                        print(f"   • {pos['Symbol']}: {pos['Shares']:.0f} shares @ ${pos['Entry Price']:.2f} (Return: {pos['Return %']:+.2f}%)")
                
            else:
                print(f"❌ Trade execution failed: {real_result.get('error', 'Unknown error')}")
        
        else:
            print("❌ Trade execution cancelled by user")
    
    else:
        print(f"❌ Dry run failed: {dry_result.get('error', 'Unknown error')}")
        if dry_result.get('errors'):
            print("Errors:")
            for error in dry_result['errors']:
                print(f"   • {error}")

if __name__ == "__main__":
    main()