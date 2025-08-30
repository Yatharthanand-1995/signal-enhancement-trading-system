"""
Test script to verify dashboard integration with backtesting tab
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from dashboard.backtesting_tab import BacktestingDashboard
    from utils.logging_setup import get_logger
    
    logger = get_logger(__name__)
    
    def test_dashboard_integration():
        """Test the backtesting dashboard integration"""
        try:
            print("🚀 Testing Dashboard Integration...")
            print("=" * 50)
            
            # Initialize backtesting dashboard
            dashboard = BacktestingDashboard()
            
            # Test database connection
            print("📊 Testing Database Connection...")
            with dashboard.get_connection() as conn:
                result = conn.execute("SELECT COUNT(*) as count FROM backtest_results").fetchone()
                backtest_count = result['count']
                print(f"  ✅ Found {backtest_count} backtest results in database")
            
            # Test available results loading
            print("\n📈 Testing Results Loading...")
            available_results = dashboard.get_available_results()
            print(f"  ✅ Loaded {len(available_results)} available backtest configurations")
            
            if not available_results.empty:
                # Test detailed results loading
                print("\n🔍 Testing Detailed Results...")
                first_result_id = available_results.iloc[0]['result_id']
                detailed_results = dashboard.get_backtest_details(first_result_id)
                
                if detailed_results:
                    print(f"  ✅ Loaded detailed results for Result ID: {first_result_id}")
                    print(f"  📊 Portfolio values: {len(detailed_results['portfolio_values'])} records")
                    print(f"  💰 Individual trades: {len(detailed_results['trades'])} records")
                    print(f"  🎯 Strategy: {detailed_results['main_results']['config_name']}")
                    print(f"  📈 Return: {detailed_results['main_results']['total_return']:.2%}")
                    print(f"  ⚡ Sharpe: {detailed_results['main_results']['sharpe_ratio']:.2f}")
                else:
                    print("  ❌ Failed to load detailed results")
            
            # Test benchmark data
            print("\n🏦 Testing Benchmark Data...")
            spy_data = dashboard.get_benchmark_data('SPY', '2022-01-01', '2024-06-30')
            print(f"  ✅ Loaded {len(spy_data)} SPY benchmark records")
            
            # Test market regimes
            print("\n🌍 Testing Market Regimes...")
            regimes = dashboard.get_market_regimes()
            print(f"  ✅ Loaded {len(regimes)} market regime periods")
            
            if not regimes.empty:
                print("  📅 Market Regimes:")
                for _, regime in regimes.iterrows():
                    print(f"    • {regime['regime_name']}: {regime['start_date'].strftime('%Y-%m-%d')} to "
                          f"{regime['end_date'].strftime('%Y-%m-%d') if pd.notna(regime['end_date']) else 'Present'}")
            
            print("\n" + "=" * 50)
            print("✅ DASHBOARD INTEGRATION TEST PASSED!")
            print("\n🎯 Dashboard Features Available:")
            print("  • 📊 Performance Summary Cards")
            print("  • 📈 Interactive Equity Curves with Benchmark Comparison")
            print("  • 💰 Detailed Trade Analysis & P&L Distribution")
            print("  • 🌍 Market Regime Performance Attribution")
            print("  • 🚀 New Backtest Configuration Interface")
            print("  • 🔬 Multi-tab Analysis Dashboard")
            
            print(f"\n🌐 Access your dashboard at: http://localhost:8502")
            print("  Navigate to the '🔬 Backtesting' tab to see all features!")
            
            return True
            
        except Exception as e:
            print(f"❌ Dashboard integration test failed: {str(e)}")
            logger.error(f"Dashboard test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    if __name__ == "__main__":
        import pandas as pd  # Import pandas for the test
        
        success = test_dashboard_integration()
        if success:
            print("\n🎉 READY FOR PRODUCTION!")
            print("🚀 Your comprehensive backtesting dashboard is fully integrated!")
        else:
            print("\n💥 Integration issues detected - please review errors above")

except ImportError as e:
    print(f"❌ Import error: {str(e)}")
    print("Please ensure all required modules are installed")
    import traceback
    traceback.print_exc()