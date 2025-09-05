#!/usr/bin/env python3
"""
Comprehensive Backtesting Runner
Execute complete signal methodology validation across market regimes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
import json
from typing import Dict, Any

from src.backtesting.comprehensive_signal_backtester import (
    ComprehensiveSignalBacktester, BacktestConfig
)
from src.backtesting.top_100_universe import Top100Universe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtesting.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_historical_backtest(start_date: datetime, end_date: datetime,
                          initial_capital: float = 1_000_000,
                          save_results: bool = True) -> Dict[str, Any]:
    """Run historical backtest with specified parameters"""
    
    logger.info(f"Starting comprehensive backtest: {start_date} to {end_date}")
    logger.info(f"Initial capital: ${initial_capital:,}")
    
    # Configure backtest
    config = BacktestConfig(
        initial_capital=initial_capital,
        max_position_size=0.02,      # 2% max per position
        transaction_cost=0.001,      # 0.1% transaction cost
        min_confidence=0.60,         # 60% minimum confidence
        benchmark_symbols=['SPY', 'QQQ', 'IWM', 'VTI']
    )
    
    # Create backtester
    backtester = ComprehensiveSignalBacktester(config)
    
    try:
        # Run backtest
        results = backtester.run_backtest(start_date, end_date)
        
        # Add configuration to results
        results['config'] = {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'initial_capital': initial_capital,
            'max_position_size': config.max_position_size,
            'transaction_cost': config.transaction_cost,
            'min_confidence': config.min_confidence,
            'benchmark_symbols': config.benchmark_symbols
        }
        
        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f'backtest_results_{timestamp}.json'
            
            # Convert results for JSON serialization
            serializable_results = prepare_results_for_json(results)
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {results_file}")
            
            # Also save CSV of daily values for Excel analysis
            if 'daily_values' in results and not results['daily_values'].empty:
                csv_file = f'backtest_daily_values_{timestamp}.csv'
                results['daily_values'].to_csv(csv_file)
                logger.info(f"Daily values saved to {csv_file}")
            
            # Save trades CSV
            if results.get('completed_trades'):
                trades_data = []
                for trade in results['completed_trades']:
                    trades_data.append({
                        'Symbol': trade.symbol,
                        'Entry Date': trade.entry_date.strftime('%Y-%m-%d'),
                        'Exit Date': trade.exit_date.strftime('%Y-%m-%d') if trade.exit_date else '',
                        'Entry Price': trade.entry_price,
                        'Exit Price': trade.exit_price or 0,
                        'Quantity': trade.quantity,
                        'Days Held': trade.days_held or 0,
                        'P&L': trade.pnl or 0,
                        'Return %': trade.return_pct or 0,
                        'Signal Direction': trade.entry_signal.get('direction', ''),
                        'Signal Confidence': trade.entry_signal.get('confidence', 0),
                        'Signal Strength': trade.entry_signal.get('strength', 0),
                        'Exit Reason': trade.exit_reason
                    })
                
                trades_df = pd.DataFrame(trades_data)
                trades_file = f'backtest_trades_{timestamp}.csv'
                trades_df.to_csv(trades_file, index=False)
                logger.info(f"Trades saved to {trades_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise

def prepare_results_for_json(results: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare results for JSON serialization"""
    
    serializable = {}
    
    for key, value in results.items():
        if key == 'daily_values':
            if hasattr(value, 'to_dict'):
                serializable[key] = value.to_dict('records')
            else:
                serializable[key] = value
        elif key == 'trades' or key == 'completed_trades':
            # Convert trades to dictionaries
            serializable[key] = []
            for trade in value:
                trade_dict = {
                    'symbol': trade.symbol,
                    'entry_date': trade.entry_date.isoformat(),
                    'exit_date': trade.exit_date.isoformat() if trade.exit_date else None,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'quantity': trade.quantity,
                    'direction': trade.direction,
                    'entry_signal': trade.entry_signal,
                    'exit_reason': trade.exit_reason,
                    'pnl': trade.pnl,
                    'return_pct': trade.return_pct,
                    'days_held': trade.days_held
                }
                serializable[key].append(trade_dict)
        elif key == 'performance_metrics':
            # Convert performance metrics
            serializable[key] = {}
            for metric_name, metric_obj in value.items():
                if hasattr(metric_obj, '__dict__'):
                    serializable[key][metric_name] = metric_obj.__dict__
                else:
                    serializable[key][metric_name] = metric_obj
        elif key == 'benchmark_returns':
            # Convert pandas Series to lists
            serializable[key] = {}
            for benchmark, returns in value.items():
                if hasattr(returns, 'to_dict'):
                    serializable[key][benchmark] = returns.to_dict()
                else:
                    serializable[key][benchmark] = returns
        else:
            serializable[key] = value
    
    return serializable

def generate_summary_report(results: Dict[str, Any]) -> str:
    """Generate executive summary report"""
    
    config = results.get('config', {})
    performance_metrics = results.get('performance_metrics', {})
    trade_analysis = results.get('trade_analysis', {})
    regime_analysis = results.get('regime_analysis', {})
    
    # Strategy metrics
    strategy_metrics = performance_metrics.get('strategy')
    
    report = f"""
# 📊 COMPREHENSIVE SIGNAL BACKTESTING RESULTS

## Executive Summary

**Backtesting Period**: {config.get('start_date', 'N/A')} to {config.get('end_date', 'N/A')}
**Initial Capital**: ${config.get('initial_capital', 0):,}
**Final Portfolio Value**: ${results.get('final_portfolio_value', 0):,.2f}
**Total Return**: {results.get('total_return', 0):+.2f}%

## Key Performance Indicators

### Returns & Risk
- **Total Return**: {results.get('total_return', 0):+.2f}%
- **Annualized Return**: {strategy_metrics.annualized_return * 100 if strategy_metrics else 0:+.2f}%
- **Sharpe Ratio**: {strategy_metrics.sharpe_ratio if strategy_metrics else 0:.2f}
- **Sortino Ratio**: {strategy_metrics.sortino_ratio if strategy_metrics else 0:.2f}
- **Maximum Drawdown**: -{strategy_metrics.max_drawdown * 100 if strategy_metrics else 0:.2f}%
- **Volatility**: {strategy_metrics.volatility * 100 if strategy_metrics else 0:.1f}%

### Trading Statistics
- **Total Trades**: {trade_analysis.get('total_trades', 0)}
- **Completed Trades**: {len(results.get('completed_trades', []))}
- **Win Rate**: {trade_analysis.get('win_rate', 0) * 100:.1f}%
- **Profit Factor**: {trade_analysis.get('profit_factor', 0):.2f}
- **Average Trade Return**: {trade_analysis.get('avg_trade_return', 0):+.2f}%
- **Average Holding Period**: {trade_analysis.get('avg_days_held', 0):.1f} days

## Benchmark Comparison

"""
    
    # Add benchmark comparisons
    for benchmark_name, benchmark_metrics in performance_metrics.items():
        if benchmark_name != 'strategy':
            excess_return = benchmark_metrics.excess_return * 100
            report += f"- **vs {benchmark_name}**: {excess_return:+.2f}% excess return\n"
    
    report += f"""

## Market Regime Performance

"""
    
    # Add regime analysis
    for regime_name, regime_data in regime_analysis.items():
        report += f"### {regime_name}\n"
        report += f"- Period: {regime_data.get('period', 'N/A')}\n"
        report += f"- Return: {regime_data.get('total_return', 0) * 100:+.2f}%\n"
        report += f"- Sharpe: {regime_data.get('sharpe_ratio', 0):.2f}\n"
        report += f"- Trades: {regime_data.get('trades', 0)}\n"
        report += f"- Win Rate: {regime_data.get('win_rate', 0) * 100:.1f}%\n\n"
    
    # Success criteria assessment
    if strategy_metrics:
        success_score = 0
        total_criteria = 5
        
        criteria_results = []
        
        if strategy_metrics.sharpe_ratio > 1.0:
            success_score += 1
            criteria_results.append("✅ Sharpe Ratio > 1.0")
        else:
            criteria_results.append("❌ Sharpe Ratio > 1.0")
        
        if strategy_metrics.max_drawdown < 0.25:
            success_score += 1
            criteria_results.append("✅ Max Drawdown < 25%")
        else:
            criteria_results.append("❌ Max Drawdown < 25%")
        
        if trade_analysis.get('win_rate', 0) > 0.50:
            success_score += 1
            criteria_results.append("✅ Win Rate > 50%")
        else:
            criteria_results.append("❌ Win Rate > 50%")
        
        if trade_analysis.get('profit_factor', 0) > 1.2:
            success_score += 1
            criteria_results.append("✅ Profit Factor > 1.2")
        else:
            criteria_results.append("❌ Profit Factor > 1.2")
        
        if trade_analysis.get('total_trades', 0) > 50:
            success_score += 1
            criteria_results.append("✅ Sufficient Trade Sample (>50)")
        else:
            criteria_results.append("❌ Sufficient Trade Sample (>50)")
        
        success_rate = success_score / total_criteria
        
        report += f"""
## System Validation Assessment

**Success Criteria Passed**: {success_score}/{total_criteria} ({success_rate:.1%})

"""
        
        for criterion in criteria_results:
            report += f"{criterion}\n"
        
        if success_rate >= 0.8:
            recommendation = "🟢 **RECOMMENDED FOR DEPLOYMENT**"
            details = "System shows strong performance across market conditions with acceptable risk levels."
        elif success_rate >= 0.6:
            recommendation = "🟡 **CONDITIONAL DEPLOYMENT**"
            details = "System shows promise but consider reduced position sizes or additional filters."
        else:
            recommendation = "🔴 **NOT RECOMMENDED FOR DEPLOYMENT**"
            details = "System requires significant improvements before live trading."
        
        report += f"""

## Final Recommendation

{recommendation}

{details}

"""
    
    # Best and worst trades
    completed_trades = results.get('completed_trades', [])
    if completed_trades:
        sorted_trades = sorted(completed_trades, key=lambda t: t.pnl or 0, reverse=True)
        
        report += f"""
## Trade Highlights

### Best Trade
- **Symbol**: {sorted_trades[0].symbol}
- **Entry Date**: {sorted_trades[0].entry_date.strftime('%Y-%m-%d')}
- **Return**: {sorted_trades[0].return_pct or 0:+.2f}%
- **P&L**: ${sorted_trades[0].pnl or 0:+,.0f}
- **Signal**: {sorted_trades[0].entry_signal.get('direction', 'N/A')}

### Worst Trade
- **Symbol**: {sorted_trades[-1].symbol}
- **Entry Date**: {sorted_trades[-1].entry_date.strftime('%Y-%m-%d')}
- **Return**: {sorted_trades[-1].return_pct or 0:+.2f}%
- **P&L**: ${sorted_trades[-1].pnl or 0:+,.0f}
- **Signal**: {sorted_trades[-1].entry_signal.get('direction', 'N/A')}

"""
    
    report += f"""
## Data Files Generated

- Detailed results: `backtest_results_[timestamp].json`
- Daily portfolio values: `backtest_daily_values_[timestamp].csv`
- Individual trades: `backtest_trades_[timestamp].csv`
- This report: `backtest_summary_report_[timestamp].md`

---

*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return report.strip()

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='Run comprehensive signal backtesting')
    parser.add_argument('--start-date', type=str, default='2019-01-01', 
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-09-30',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=1_000_000,
                       help='Initial capital')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to files')
    
    args = parser.parse_args()
    
    try:
        # Parse dates
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        
        # Run backtest
        logger.info("Starting comprehensive backtesting...")
        results = run_historical_backtest(
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.capital,
            save_results=not args.no_save
        )
        
        # Generate summary report
        report = generate_summary_report(results)
        
        # Print summary to console
        print("\\n" + "="*80)
        print(report)
        print("="*80)
        
        # Save report
        if not args.no_save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = f'backtest_summary_report_{timestamp}.md'
            with open(report_file, 'w') as f:
                f.write(report)
            logger.info(f"Summary report saved to {report_file}")
        
        logger.info("Backtesting completed successfully!")
        
        return results
        
    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        return None

if __name__ == "__main__":
    main()