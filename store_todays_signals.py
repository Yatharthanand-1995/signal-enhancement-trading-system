#!/usr/bin/env python3
"""
Store Today's Signals for Tomorrow's Paper Trading
Captures current signals for tomorrow's trading execution
"""

import sys
import os
sys.path.append('src')

from src.dashboard.components.paper_trading import PaperTradingEngine
from src.dashboard.components.historical_signal_trader import HistoricalSignalTrader
import pandas as pd
from datetime import datetime

# Import the main dashboard data loading function
from src.dashboard.main import load_transparent_dashboard_data

def main():
    print("💾 Store Today's Signals for Tomorrow's Paper Trading")
    print("=" * 60)
    
    # Initialize components
    paper_engine = PaperTradingEngine()
    historical_trader = HistoricalSignalTrader(paper_engine)
    
    print("📡 Loading current market signals...")
    
    try:
        # Load signals from main dashboard
        df, symbols, market_env = load_transparent_dashboard_data()
        
        if df.empty:
            print("❌ No signals data available")
            return
        
        print(f"✅ Loaded {len(df)} signals from {len(symbols)} symbols")
        print()
        
        # Show signal summary
        signal_counts = df['Signal'].value_counts()
        print("📊 Today's Signal Distribution:")
        for signal, count in signal_counts.items():
            print(f"   • {signal}: {count} stocks")
        
        print()
        
        # Show high confidence signals that might be traded tomorrow
        high_conf_signals = df[df['Confidence'] >= 0.6]  # Above min confidence threshold
        tradeable_signals = high_conf_signals[high_conf_signals['Signal'].isin(['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL'])]
        
        if not tradeable_signals.empty:
            print(f"🎯 High Confidence Tradeable Signals (≥60% confidence): {len(tradeable_signals)}")
            for _, row in tradeable_signals.iterrows():
                symbol = row['Symbol']
                signal = row['Signal']
                confidence = row['Confidence']
                print(f"   • {symbol}: {signal} (confidence: {confidence:.1%})")
            print()
        
        # Store today's signals
        today = datetime.now().strftime('%Y-%m-%d')
        print(f"💾 Storing signals for date: {today}")
        
        success = historical_trader.store_daily_signals(df, today)
        
        if success:
            print("✅ Successfully stored today's signals!")
            
            # Get updated history summary
            history_summary = historical_trader.get_signal_history_summary()
            
            print()
            print("📈 Updated Signal History:")
            print(f"   Total days: {history_summary['total_days']}")
            print(f"   Total signals: {history_summary['total_signals']}")
            print(f"   Date range: {history_summary['date_range']}")
            print(f"   Symbols tracked: {len(history_summary['symbols_tracked'])}")
            
            # Preview tomorrow's potential trades
            tomorrow_potential = []
            for _, row in tradeable_signals.iterrows():
                symbol = row['Symbol']
                signal = row['Signal']
                confidence = row['Confidence']
                
                # Check if this would be a new trade tomorrow
                if signal in ['BUY', 'STRONG_BUY'] and symbol not in paper_engine.positions:
                    tomorrow_potential.append((symbol, signal, confidence))
                elif signal in ['SELL', 'STRONG_SELL'] and symbol in paper_engine.positions:
                    tomorrow_potential.append((symbol, signal, confidence))
            
            if tomorrow_potential:
                print()
                print(f"🚀 Potential trades for tomorrow: {len(tomorrow_potential)}")
                for symbol, signal, conf in tomorrow_potential:
                    action = "BUY" if signal in ['BUY', 'STRONG_BUY'] else "SELL"
                    print(f"   • {action} {symbol} (confidence: {conf:.1%})")
            
        else:
            print("❌ Failed to store today's signals")
        
        print()
        print("🎯 Next Steps:")
        print("   1. Tomorrow morning: Run execute_signals_auto.py")
        print("   2. Monitor positions throughout the day")
        print("   3. Repeat signal storage process daily")
        print("   4. Visit dashboard for detailed analysis: http://localhost:8501")
        
    except Exception as e:
        print(f"❌ Error loading signals: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()