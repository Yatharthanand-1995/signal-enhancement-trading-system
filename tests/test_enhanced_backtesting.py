"""
Test script for enhanced backtesting system
Demonstrates comprehensive backtesting with benchmark comparison
"""

import sys
import os
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.backtesting.enhanced_backtest_engine import run_backtest_analysis
    from src.utils.logging_setup import get_logger
    
    logger = get_logger(__name__)
    
    def test_enhanced_backtesting():
        """Test the enhanced backtesting system"""
        try:
            print("🚀 Testing Enhanced Backtesting System...")
            print("=" * 60)
            
            # Define test configuration
            config_data = {
                'config_name': 'Signal-Based Strategy Test',
                'start_date': '2022-01-01',  # 2+ years of data
                'end_date': '2024-06-30',
                'initial_capital': 100000,  # $100k starting capital
                'strategy_type': 'signal_based',
                'position_sizing_method': 'equal_weight',
                'max_position_size': 0.08,  # 8% max per position
                'max_positions': 12,  # Up to 12 positions
                'transaction_costs': 0.0008,  # 0.08% transaction costs
                'slippage_rate': 0.0005,  # 0.05% slippage
                'commission_per_trade': 1.0,  # $1 per trade
                'signal_threshold': 0.55,  # 55% minimum signal strength
                'parameters': {
                    'description': 'Multi-component signal strategy with regime awareness',
                    'risk_level': 'moderate',
                    'target_volatility': 0.15
                }
            }
            
            # Test symbols - mix of different sectors and market caps
            test_symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Large cap tech
                'JPM', 'JNJ', 'PG', 'WMT', 'HD',  # Large cap diversified
                'NVDA', 'CRM', 'ADBE', 'NFLX', 'DIS'  # Growth stocks
            ]
            
            print(f"Configuration:")
            print(f"  Period: {config_data['start_date']} to {config_data['end_date']}")
            print(f"  Capital: ${config_data['initial_capital']:,}")
            print(f"  Symbols: {len(test_symbols)} stocks")
            print(f"  Max Position Size: {config_data['max_position_size']:.1%}")
            print(f"  Transaction Costs: {config_data['transaction_costs']:.3%}")
            print(f"  Benchmark: SPY")
            print()
            
            # Run comprehensive backtest
            print("Running comprehensive backtest analysis...")
            start_time = datetime.now()
            
            results = run_backtest_analysis(
                config_data=config_data,
                symbols=test_symbols,
                benchmark='SPY'
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            print(f"✅ Backtest completed in {execution_time:.1f} seconds")
            print()
            
            # Display results
            print("📊 BACKTEST RESULTS SUMMARY")
            print("=" * 60)
            
            print("🎯 PERFORMANCE METRICS:")
            print(f"  Total Return:           {results.total_return:>8.2%}")
            print(f"  Annualized Return:      {results.annualized_return:>8.2%}")
            print(f"  Volatility:             {results.volatility:>8.2%}")
            print(f"  Sharpe Ratio:           {results.sharpe_ratio:>8.2f}")
            print(f"  Calmar Ratio:           {results.calmar_ratio:>8.2f}")
            print(f"  Sortino Ratio:          {results.sortino_ratio:>8.2f}")
            print()
            
            print("⚠️  RISK METRICS:")
            print(f"  Maximum Drawdown:       {results.max_drawdown:>8.2%}")
            print(f"  Max DD Duration:        {results.max_drawdown_duration:>8} days")
            print(f"  Average Drawdown:       {results.avg_drawdown:>8.2%}")
            print(f"  95% VaR:                {results.var_95:>8.2%}")
            print(f"  95% CVaR:               {results.cvar_95:>8.2%}")
            print()
            
            print("📈 TRADING STATISTICS:")
            print(f"  Total Trades:           {results.total_trades:>8}")
            print(f"  Winning Trades:         {results.profitable_trades:>8}")
            print(f"  Losing Trades:          {results.losing_trades:>8}")
            print(f"  Win Rate:               {results.win_rate:>8.2%}")
            print(f"  Profit Factor:          {results.profit_factor:>8.2f}")
            print(f"  Expectancy:             ${results.expectancy:>7.2f}")
            print(f"  Avg Holding Days:       {results.avg_holding_days:>8.1f}")
            print(f"  Total Fees:             ${results.total_fees:>7.2f}")
            print()
            
            print("📊 BENCHMARK COMPARISON (vs SPY):")
            print(f"  Benchmark Return:       {results.benchmark_return:>8.2%}")
            print(f"  Excess Return:          {results.excess_return:>8.2%}")
            print(f"  Information Ratio:      {results.information_ratio:>8.2f}")
            print(f"  Tracking Error:         {results.tracking_error:>8.2%}")
            print(f"  Beta:                   {results.beta:>8.2f}")
            print(f"  Alpha:                  {results.alpha:>8.2%}")
            print(f"  Correlation:            {results.correlation_with_benchmark:>8.2f}")
            print(f"  Upside Capture:         {results.upside_capture:>8.2%}")
            print(f"  Downside Capture:       {results.downside_capture:>8.2%}")
            print()
            
            print("📅 MONTHLY ANALYSIS:")
            print(f"  Positive Months:        {results.up_months:>8}")
            print(f"  Negative Months:        {results.down_months:>8}")
            print(f"  Best Month:             {results.best_month:>8.2%}")
            print(f"  Worst Month:            {results.worst_month:>8.2%}")
            print()
            
            print("💰 PORTFOLIO VALUE:")
            if not results.daily_portfolio_values.empty:
                initial_value = results.daily_portfolio_values.iloc[0]['portfolio_value']
                print(f"  Initial Capital:        ${initial_value:>10,.2f}")
                print(f"  Final Value:            ${results.final_portfolio_value:>10,.2f}")
                print(f"  Total Profit/Loss:      ${results.final_portfolio_value - initial_value:>10,.2f}")
            else:
                print(f"  Initial Capital:        ${config_data['initial_capital']:>10,.2f}")
                print(f"  Final Value:            ${results.final_portfolio_value:>10,.2f}")
                print(f"  Total Profit/Loss:      ${results.final_portfolio_value - config_data['initial_capital']:>10,.2f}")
            print()
            
            # Performance vs benchmarks
            excess_vs_spy = results.excess_return
            if excess_vs_spy > 0:
                print(f"🎉 STRATEGY OUTPERFORMED SPY BY {excess_vs_spy:.2%}")
            else:
                print(f"📉 Strategy underperformed SPY by {abs(excess_vs_spy):.2%}")
            
            # Risk-adjusted performance assessment
            if results.sharpe_ratio > 1.0:
                risk_assessment = "EXCELLENT"
            elif results.sharpe_ratio > 0.7:
                risk_assessment = "GOOD"
            elif results.sharpe_ratio > 0.5:
                risk_assessment = "ACCEPTABLE"
            else:
                risk_assessment = "POOR"
            
            print(f"📊 Risk-Adjusted Performance: {risk_assessment} (Sharpe: {results.sharpe_ratio:.2f})")
            print()
            
            print("💾 DATABASE STORAGE:")
            print(f"  Config ID: {results.config_id}")
            print(f"  Result ID: {results.result_id}")
            print(f"  Stored {len(results.trades)} individual trades")
            print(f"  Stored {len(results.daily_portfolio_values)} daily portfolio values")
            print()
            
            print("=" * 60)
            print("✅ Enhanced backtesting system test completed successfully!")
            print("🎯 Ready for dashboard visualization development")
            
            return True
            
        except Exception as e:
            print(f"❌ Enhanced backtesting test failed: {str(e)}")
            logger.error(f"Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    if __name__ == "__main__":
        success = test_enhanced_backtesting()
        if success:
            print("\n🎉 Ready for Phase 3: Dashboard Development!")
        else:
            print("\n💥 Please fix backtesting issues before proceeding")
            sys.exit(1)

except ImportError as e:
    print(f"❌ Import error: {str(e)}")
    print("Please ensure all required modules are installed and paths are correct")
    sys.exit(1)