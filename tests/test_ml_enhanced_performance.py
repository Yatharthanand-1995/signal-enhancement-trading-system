#!/usr/bin/env python3
"""
ML-Enhanced Performance Test
Compare baseline vs ML-enhanced system performance
"""

import sys
import os
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.backtesting.enhanced_backtest_engine import run_backtest_analysis
    from src.utils.logging_setup import get_logger
    
    logger = get_logger(__name__)
    
    def test_ml_enhanced_performance():
        """Test ML-enhanced backtesting performance vs baseline"""
        
        print("🚀 ML-ENHANCED PERFORMANCE TEST")
        print("=" * 60)
        print("Testing ML integration performance improvements")
        print()
        
        # Configuration for ML-enhanced test
        config_data = {
            'config_name': 'ML-Enhanced Signal Test',
            'start_date': '2022-01-01',
            'end_date': '2024-06-30', 
            'initial_capital': 100000,
            'strategy_type': 'ml_enhanced',
            'position_sizing_method': 'equal_weight',
            'max_position_size': 0.08,
            'max_positions': 12,
            'transaction_costs': 0.0008,
            'slippage_rate': 0.0005,
            'commission_per_trade': 1.0,
            'signal_threshold': 0.55,
            'parameters': {
                'description': 'ML-enhanced multi-component signal strategy',
                'ml_integration': True,
                'ml_weight': 0.25
            }
        }
        
        # Use same symbols as baseline for fair comparison
        test_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
            'JPM', 'JNJ', 'PG', 'WMT', 'HD',
            'NVDA', 'CRM', 'ADBE', 'NFLX', 'DIS'
        ]
        
        print(f"🎯 Test Configuration:")
        print(f"  Strategy: {config_data['config_name']}")
        print(f"  Period: {config_data['start_date']} to {config_data['end_date']}")
        print(f"  Capital: ${config_data['initial_capital']:,}")
        print(f"  Symbols: {len(test_symbols)} stocks")
        print(f"  Max Position: {config_data['max_position_size']:.1%}")
        print(f"  ML Integration: {config_data['parameters']['ml_integration']}")
        print(f"  ML Weight: {config_data['parameters']['ml_weight']:.1%}")
        print()
        
        # Run ML-enhanced backtest
        print("🚀 Running ML-enhanced backtest...")
        start_time = datetime.now()
        
        try:
            results = run_backtest_analysis(
                config_data=config_data,
                symbols=test_symbols,
                benchmark='SPY'
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            print(f"✅ ML-enhanced backtest completed in {execution_time:.1f} seconds")
            print()
            
            # Display results
            print("📊 ML-ENHANCED RESULTS")
            print("=" * 60)
            
            print("🎯 PERFORMANCE METRICS:")
            print(f"  Total Return:           {results.total_return:>8.2%}")
            print(f"  Annualized Return:      {results.annualized_return:>8.2%}")
            print(f"  Volatility:             {results.volatility:>8.2%}")
            print(f"  Sharpe Ratio:           {results.sharpe_ratio:>8.2f}")
            print(f"  Calmar Ratio:           {results.calmar_ratio:>8.2f}")
            print()
            
            print("⚠️  RISK METRICS:")
            print(f"  Maximum Drawdown:       {results.max_drawdown:>8.2%}")
            print(f"  Average Drawdown:       {results.avg_drawdown:>8.2%}")
            print(f"  95% VaR:                {results.var_95:>8.2%}")
            print()
            
            print("📈 TRADING STATISTICS:")
            print(f"  Total Trades:           {results.total_trades:>8}")
            print(f"  Winning Trades:         {results.profitable_trades:>8}")
            print(f"  Win Rate:               {results.win_rate:>8.2%}")
            print(f"  Profit Factor:          {results.profit_factor:>8.2f}")
            print(f"  Average Holding Days:   {results.avg_holding_days:>8.1f}")
            print()
            
            print("📊 BENCHMARK COMPARISON:")
            print(f"  Benchmark Return:       {results.benchmark_return:>8.2%}")
            print(f"  Excess Return:          {results.excess_return:>8.2%}")
            print(f"  Information Ratio:      {results.information_ratio:>8.2f}")
            print(f"  Beta:                   {results.beta:>8.2f}")
            print(f"  Alpha:                  {results.alpha:>8.2%}")
            print()
            
            # Performance vs baseline
            baseline_return = 0.1056  # 10.56%
            baseline_sharpe = 0.58
            baseline_win_rate = 0.511  # 51.1%
            baseline_trades = 358
            
            print("🔥 IMPROVEMENT vs BASELINE:")
            print(f"  Return Improvement:     {results.total_return - baseline_return:>8.2%}")
            print(f"  Sharpe Improvement:     {results.sharpe_ratio - baseline_sharpe:>8.2f}")
            print(f"  Win Rate Change:        {results.win_rate - baseline_win_rate:>8.2%}")
            print(f"  Trade Count Change:     {results.total_trades - baseline_trades:>8}")
            
            # Calculate improvement percentages
            return_improvement = ((results.total_return / baseline_return) - 1) * 100
            sharpe_improvement = ((results.sharpe_ratio / baseline_sharpe) - 1) * 100
            
            print()
            print("📈 PERCENTAGE IMPROVEMENTS:")
            print(f"  Total Return:           {return_improvement:>8.1f}%")
            print(f"  Sharpe Ratio:           {sharpe_improvement:>8.1f}%")
            
            # Success criteria check
            print()
            print("🎯 SUCCESS CRITERIA CHECK:")
            
            target_return_improvement = 15  # 15% minimum improvement target
            target_sharpe_improvement = 20  # 20% minimum improvement target
            
            return_success = return_improvement >= target_return_improvement
            sharpe_success = sharpe_improvement >= target_sharpe_improvement
            
            print(f"  Return Target (≥15%):   {'✅ ACHIEVED' if return_success else '❌ MISSED'} ({return_improvement:.1f}%)")
            print(f"  Sharpe Target (≥20%):   {'✅ ACHIEVED' if sharpe_success else '❌ MISSED'} ({sharpe_improvement:.1f}%)")
            
            overall_success = return_success and sharpe_success
            
            print()
            if overall_success:
                print("🎉 ML INTEGRATION SUCCESS!")
                print("✅ Performance targets achieved")
                print("🚀 System ready for production deployment")
            elif return_success or sharpe_success:
                print("🎯 PARTIAL SUCCESS - Good progress")
                print("⚡ ML integration showing benefits")
                print("🔧 Further optimization recommended")
            else:
                print("📊 BASELINE ESTABLISHED")
                print("🔧 ML models need training for full benefits")
                print("🚀 Integration structure is working")
            
            return results
            
        except Exception as e:
            print(f"❌ ML-enhanced backtest failed: {str(e)}")
            logger.error(f"ML backtest error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def compare_with_baseline():
        """Display baseline comparison"""
        
        print(f"\n📋 BASELINE vs ML-ENHANCED COMPARISON")
        print("=" * 60)
        
        print("📊 BASELINE PERFORMANCE (Before ML Integration):")
        print("  Total Return:     10.56%")
        print("  Sharpe Ratio:      0.58")
        print("  Win Rate:         51.1%")
        print("  Total Trades:       358")
        print("  Signal Quality:    70% calculated, 30% hardcoded")
        print()
        
        print("🚀 ML-ENHANCED SYSTEM (After Integration):")
        print("  ML Contribution:   25% of every signal")
        print("  Signal Quality:   100% calculated (no hardcoded)")
        print("  Regime Awareness:  Dynamic ML weight adjustments")
        print("  Architecture:      Advanced ensemble scoring")
        print()
        
        print("🎯 EXPECTED IMPROVEMENTS:")
        print("  Total Return:     13-15% (+25-40% improvement)")
        print("  Sharpe Ratio:     0.75+ (+30%+ improvement)")
        print("  Signal Quality:   Professional-grade ML predictions")
        print("  Adaptability:     Market regime awareness")
    
    if __name__ == "__main__":
        # Show baseline comparison first
        compare_with_baseline()
        
        # Run ML-enhanced test
        results = test_ml_enhanced_performance()
        
        if results:
            print(f"\n💾 DATABASE STORAGE:")
            print(f"  Result ID: {results.result_id}")
            print(f"  Config ID: {results.config_id}")
            print(f"  Stored for dashboard analysis")
            
            print(f"\n🌐 View results in dashboard:")
            print(f"  URL: http://localhost:8504")
            print(f"  Navigate to: 🔬 Backtesting tab")
            
        print(f"\n🎯 ML-Enhanced Performance Test Complete!")

except ImportError as e:
    print(f"❌ Import error: {str(e)}")
    print("Please ensure all dependencies are installed")
except Exception as e:
    print(f"❌ Test error: {str(e)}")
    import traceback
    traceback.print_exc()