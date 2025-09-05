#!/usr/bin/env python3
"""
Daily Paper Trading Workflow Automation
Complete daily routine for paper trading - works during market hours and after close
"""

import sys
import os
sys.path.append('src')

from src.dashboard.components.paper_trading import PaperTradingEngine
from src.dashboard.components.historical_signal_trader import HistoricalSignalTrader
import pandas as pd
from datetime import datetime, timedelta
import time
import schedule
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paper_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DailyTradingWorkflow:
    """Automated daily paper trading workflow"""
    
    def __init__(self):
        self.paper_engine = PaperTradingEngine()
        self.historical_trader = HistoricalSignalTrader(self.paper_engine)
        logger.info("Daily Trading Workflow initialized")
    
    def morning_routine(self):
        """Execute morning routine - process yesterday's signals"""
        logger.info("🌅 Starting morning routine...")
        
        print("🌅 MORNING PAPER TRADING ROUTINE")
        print("=" * 40)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 1. Check portfolio status
        print("💼 Portfolio Status Check:")
        metrics = self.paper_engine.get_performance_metrics()
        print(f"   Portfolio Value: ${metrics['current_value']:,.2f}")
        print(f"   Total Return: {metrics['total_return']:+.2f}%")
        print(f"   Active Positions: {metrics['positions_count']}")
        print()
        
        # 2. Update existing positions
        if self.paper_engine.positions:
            print("🔄 Updating positions with current market data...")
            self.paper_engine.update_positions()
            print("✅ Position updates complete")
        else:
            print("📭 No current positions to update")
        print()
        
        # 3. Execute yesterday's signals
        print("📅 Processing Yesterday's Signals:")
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        yesterdays_signals = self.historical_trader.get_yesterdays_signals(yesterday)
        
        if yesterdays_signals:
            print(f"   Found {len(yesterdays_signals)} signals from {yesterday}")
            
            # Execute trades
            result = self.historical_trader.execute_yesterdays_signals(target_date=yesterday, dry_run=False)
            
            if result['status'] == 'success':
                trades_executed = result['trades_executed']
                print(f"✅ Successfully executed {trades_executed} trades")
                
                if result['results']:
                    for trade in result['results']:
                        action = trade.get('action_type', 'N/A')
                        symbol = trade.get('symbol', 'N/A')
                        price = trade.get('current_price', 0)
                        print(f"   • {action} {symbol} at ${price:.2f}")
                
                if result['errors']:
                    print("   ⚠️ Some errors occurred:")
                    for error in result['errors'][:3]:  # Show first 3 errors
                        print(f"   • {error}")
            else:
                print(f"❌ Signal execution failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"   📭 No signals found for {yesterday}")
        
        print()
        
        # 4. Updated portfolio summary
        updated_metrics = self.paper_engine.get_performance_metrics()
        print("📊 Updated Portfolio Summary:")
        print(f"   Portfolio Value: ${updated_metrics['current_value']:,.2f}")
        print(f"   Cash: ${updated_metrics['cash']:,.2f}")
        print(f"   Total Return: {updated_metrics['total_return']:+.2f}%")
        print(f"   Win Rate: {updated_metrics['win_rate']:.1f}%")
        
        logger.info(f"Morning routine completed. Portfolio value: ${updated_metrics['current_value']:,.2f}")
        print("\n🎯 Morning routine complete!")
    
    def evening_routine(self):
        """Execute evening routine - store today's signals"""
        logger.info("🌆 Starting evening routine...")
        
        print("🌆 EVENING PAPER TRADING ROUTINE")
        print("=" * 40)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 1. End-of-day portfolio review
        print("📊 End-of-Day Portfolio Review:")
        self.paper_engine.update_positions()  # Final price update
        
        metrics = self.paper_engine.get_performance_metrics()
        positions_df = self.paper_engine.get_positions_summary()
        
        print(f"   Portfolio Value: ${metrics['current_value']:,.2f}")
        print(f"   Daily P&L: ${metrics['total_pnl']:+,.2f}")
        print(f"   Total Return: {metrics['total_return']:+.2f}%")
        
        if not positions_df.empty:
            print(f"   Active Positions: {len(positions_df)}")
            
            # Show top performers of the day
            best_position = positions_df.loc[positions_df['Return %'].astype(str).str.replace('%','').astype(float).idxmax()]
            worst_position = positions_df.loc[positions_df['Return %'].astype(str).str.replace('%','').astype(float).idxmin()]
            
            print(f"   Best: {best_position['Symbol']} ({best_position['Return %']})")
            print(f"   Worst: {worst_position['Symbol']} ({worst_position['Return %']})")
        print()
        
        # 2. Store today's signals for tomorrow (when market is open)
        print("💾 Signal Storage for Tomorrow:")
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Check current market hours (simplified - US market 9:30 AM - 4:00 PM ET)
        current_hour = datetime.now().hour
        is_likely_market_hours = 9 <= current_hour <= 16
        
        if is_likely_market_hours:
            print("   📡 Market appears to be open - attempting signal capture...")
            try:
                # This would need to be integrated with the main dashboard data
                # For now, we'll simulate successful storage
                print("   ⚠️ Signal storage requires dashboard integration")
                print("   💡 Visit http://localhost:8501 (Market Signals) to capture live signals")
            except Exception as e:
                print(f"   ❌ Signal capture failed: {e}")
        else:
            print("   🕐 Market is closed - signals will be captured during next market session")
        
        # 3. Generate daily report
        print()
        print("📝 Daily Performance Report:")
        print(f"   Trades Today: {self.paper_engine.total_trades}")
        
        # Recent trades
        recent_trades = self.paper_engine.get_trade_history_df(limit=5)
        if not recent_trades.empty:
            print("   Recent Activity:")
            for _, trade in recent_trades.tail(3).iterrows():
                timestamp = trade['Timestamp']
                action = trade['Action']
                symbol = trade['Symbol']
                print(f"   • {timestamp}: {action} {symbol}")
        
        history_summary = self.historical_trader.get_signal_history_summary()
        print(f"   Signal History: {history_summary['total_days']} days stored")
        
        logger.info(f"Evening routine completed. Portfolio value: ${metrics['current_value']:,.2f}")
        print("\n🌙 Evening routine complete!")
    
    def midday_check(self):
        """Quick midday position check"""
        logger.info("🕐 Midday check...")
        
        print("🕐 MIDDAY PORTFOLIO CHECK")
        print("=" * 30)
        
        # Quick update and status
        self.paper_engine.update_positions()
        metrics = self.paper_engine.get_performance_metrics()
        
        print(f"📊 Portfolio: ${metrics['current_value']:,.2f} ({metrics['total_return']:+.2f}%)")
        print(f"💰 Cash: ${metrics['cash']:,.2f}")
        
        positions_df = self.paper_engine.get_positions_summary()
        if not positions_df.empty:
            # Check for significant moves
            big_movers = []
            for _, pos in positions_df.iterrows():
                return_pct = pos['Return %']
                if isinstance(return_pct, str):
                    return_num = float(return_pct.replace('%', '').replace('+', ''))
                    if abs(return_num) >= 2:  # 2% or more move
                        big_movers.append((pos['Symbol'], return_num))
            
            if big_movers:
                print("🚨 Significant position moves:")
                for symbol, return_pct in big_movers:
                    status = "🟢" if return_pct > 0 else "🔴"
                    print(f"   {status} {symbol}: {return_pct:+.2f}%")
            else:
                print("✅ All positions stable")
        
        print()
    
    def setup_schedule(self):
        """Set up automated schedule"""
        print("⏰ Setting up daily paper trading schedule...")
        
        # Morning routine: 9:45 AM (after market opens)
        schedule.every().day.at("09:45").do(self.morning_routine)
        
        # Midday check: 12:30 PM
        schedule.every().day.at("12:30").do(self.midday_check)
        
        # Evening routine: 5:00 PM (after market closes)
        schedule.every().day.at("17:00").do(self.evening_routine)
        
        print("✅ Schedule configured:")
        print("   • 09:45 - Morning routine (execute yesterday's signals)")
        print("   • 12:30 - Midday check (position monitoring)")
        print("   • 17:00 - Evening routine (end-of-day review)")
        print()
        print("📅 Running scheduled tasks... (Press Ctrl+C to stop)")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("\n⏹️ Scheduler stopped by user")
    
    def run_manual_workflow(self):
        """Run manual workflow for immediate execution"""
        print("🎮 MANUAL PAPER TRADING WORKFLOW")
        print("=" * 50)
        
        print("Choose an option:")
        print("1. Morning Routine (Execute yesterday's signals)")
        print("2. Evening Routine (End-of-day review)")  
        print("3. Midday Check (Quick status)")
        print("4. Full Day Simulation")
        print("5. Setup Automated Schedule")
        
        try:
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                self.morning_routine()
            elif choice == '2':
                self.evening_routine()
            elif choice == '3':
                self.midday_check()
            elif choice == '4':
                print("\n🔄 Running full day simulation...")
                self.morning_routine()
                print("\n" + "="*30)
                self.midday_check()
                print("\n" + "="*30)
                self.evening_routine()
            elif choice == '5':
                self.setup_schedule()
            else:
                print("❌ Invalid choice")
                
        except (KeyboardInterrupt, EOFError):
            print("\n⏹️ Workflow stopped by user")

def main():
    print("🚀 Daily Paper Trading Workflow System")
    print("=" * 50)
    
    workflow = DailyTradingWorkflow()
    
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == 'morning':
            workflow.morning_routine()
        elif command == 'evening':
            workflow.evening_routine()
        elif command == 'check':
            workflow.midday_check()
        elif command == 'schedule':
            workflow.setup_schedule()
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: morning, evening, check, schedule")
    else:
        # Interactive mode
        workflow.run_manual_workflow()

if __name__ == "__main__":
    main()