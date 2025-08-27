#!/usr/bin/env python3
"""
Comprehensive Backtesting Suite
Validates enhanced signal system performance against baseline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import our components
from src.backtesting.backtest_engine import BacktestEngine, WalkForwardOptimizer
from src.backtesting.enhanced_signal_strategy import EnhancedSignalStrategy, BaselineStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backtest_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveBacktester:
    """Comprehensive backtesting framework"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.engine = BacktestEngine(initial_capital=initial_capital)
        self.results = {}
        
        # Define test universes
        self.test_universes = {
            'large_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM'],
            'diverse_sectors': ['AAPL', 'JPM', 'JNJ', 'PFE', 'XOM', 'WMT', 'DIS', 'BA'],
            'small_sample': ['AAPL', 'MSFT', 'GOOGL']
        }
        
        # Define test periods
        self.test_periods = {
            '2023_full': ('2023-01-01', '2023-12-31'),
            '2022_2023': ('2022-01-01', '2023-12-31'), 
            'recent_6m': ('2023-07-01', '2023-12-31')
        }
    
    def run_comprehensive_tests(self) -> Dict:
        """Run all backtesting scenarios"""
        
        logger.info("Starting comprehensive backtesting suite...")
        
        all_results = {}
        
        # Test 1: Enhanced vs Baseline Comparison
        logger.info("=== Test 1: Enhanced vs Baseline Comparison ===")
        all_results['comparison_tests'] = self._run_comparison_tests()
        
        # Test 2: Parameter Sensitivity Analysis
        logger.info("=== Test 2: Parameter Sensitivity Analysis ===")
        all_results['sensitivity_analysis'] = self._run_sensitivity_analysis()
        
        # Test 3: Walk-Forward Optimization
        logger.info("=== Test 3: Walk-Forward Optimization ===")
        all_results['walk_forward'] = self._run_walk_forward_tests()
        
        # Test 4: Market Regime Performance
        logger.info("=== Test 4: Market Regime Analysis ===")
        all_results['regime_analysis'] = self._run_regime_analysis()
        
        # Test 5: Risk Metrics Analysis
        logger.info("=== Test 5: Risk Metrics Analysis ===")
        all_results['risk_analysis'] = self._analyze_risk_metrics(all_results)
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report(all_results)
        
        # Save results
        self._save_results(all_results, report)
        
        logger.info("Comprehensive backtesting completed!")
        
        return {
            'results': all_results,
            'report': report
        }
    
    def _run_comparison_tests(self) -> Dict:
        """Compare enhanced strategy vs baseline across different scenarios"""
        
        comparison_results = {}
        
        for universe_name, symbols in self.test_universes.items():
            logger.info(f"Testing universe: {universe_name}")
            comparison_results[universe_name] = {}
            
            for period_name, (start_date, end_date) in self.test_periods.items():
                logger.info(f"  Testing period: {period_name}")
                
                # Test Enhanced Strategy
                enhanced_strategy = EnhancedSignalStrategy()
                enhanced_results = self.engine.run_backtest(
                    strategy=enhanced_strategy,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Test Baseline Strategy
                baseline_strategy = BaselineStrategy()
                baseline_results = self.engine.run_backtest(
                    strategy=baseline_strategy,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Calculate improvement metrics
                improvement = self._calculate_improvement_metrics(
                    enhanced_results, baseline_results
                )
                
                comparison_results[universe_name][period_name] = {
                    'enhanced': self._extract_key_metrics(enhanced_results),
                    'baseline': self._extract_key_metrics(baseline_results),
                    'improvement': improvement
                }
                
                logger.info(f"    Enhanced Return: {enhanced_results.total_return:.2%}")
                logger.info(f"    Baseline Return: {baseline_results.total_return:.2%}")
                logger.info(f"    Improvement: {improvement['return_improvement']:.2%}")
        
        return comparison_results
    
    def _run_sensitivity_analysis(self) -> Dict:
        """Test parameter sensitivity for the enhanced strategy"""
        
        sensitivity_results = {}
        
        # Define parameter ranges to test
        param_ranges = {
            'buy_threshold': [0.60, 0.65, 0.70, 0.75],
            'volume_weight': [0.15, 0.25, 0.35, 0.45],
            'regime_weight': [0.20, 0.30, 0.40, 0.50],
            'base_position_size': [0.03, 0.04, 0.05, 0.06]
        }
        
        base_params = {
            'buy_threshold': 0.65,
            'volume_weight': 0.25,
            'regime_weight': 0.30,
            'base_position_size': 0.04
        }
        
        # Use small sample for faster testing
        symbols = self.test_universes['small_sample']
        start_date, end_date = self.test_periods['2023_full']
        
        for param_name, param_values in param_ranges.items():
            logger.info(f"Testing sensitivity for {param_name}")
            sensitivity_results[param_name] = {}
            
            for param_value in param_values:
                # Create strategy with modified parameter
                test_params = base_params.copy()
                test_params[param_name] = param_value
                
                strategy = EnhancedSignalStrategy(**test_params)
                results = self.engine.run_backtest(
                    strategy=strategy,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date
                )
                
                sensitivity_results[param_name][param_value] = {
                    'total_return': results.total_return,
                    'sharpe_ratio': results.sharpe_ratio,
                    'max_drawdown': results.max_drawdown,
                    'win_rate': results.win_rate,
                    'total_trades': results.total_trades
                }
                
                logger.info(f"  {param_name}={param_value}: Return={results.total_return:.2%}, Sharpe={results.sharpe_ratio:.2f}")
        
        return sensitivity_results
    
    def _run_walk_forward_tests(self) -> Dict:
        """Run walk-forward optimization tests"""
        
        wf_results = {}
        
        try:
            optimizer = WalkForwardOptimizer(self.engine)
            
            # Define parameter grid for optimization
            parameter_grid = {
                'buy_threshold': [0.60, 0.65, 0.70],
                'volume_weight': [0.20, 0.25, 0.30],
                'regime_weight': [0.25, 0.30, 0.35]
            }
            
            # Test on small sample with shorter period
            symbols = self.test_universes['small_sample']
            
            wf_results = optimizer.optimize(
                strategy_class=EnhancedSignalStrategy,
                symbols=symbols,
                start_date='2022-01-01',
                end_date='2023-12-31',
                parameter_grid=parameter_grid
            )
            
            logger.info(f"Walk-forward efficiency: {wf_results['walk_forward_efficiency']:.2f}")
            logger.info(f"Average OOS return: {wf_results['aggregate_performance']['avg_return']:.2%}")
            
        except Exception as e:
            logger.error(f"Error in walk-forward testing: {str(e)}")
            wf_results = {'error': str(e)}
        
        return wf_results
    
    def _run_regime_analysis(self) -> Dict:
        """Analyze performance across different market regimes"""
        
        regime_results = {}
        
        # This would typically require regime classification data
        # For now, we'll use volatility-based regime proxy
        symbols = self.test_universes['diverse_sectors']
        start_date, end_date = self.test_periods['2022_2023']
        
        try:
            # Run enhanced strategy backtest
            enhanced_strategy = EnhancedSignalStrategy()
            results = self.engine.run_backtest(
                strategy=enhanced_strategy,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date
            )
            
            # Analyze regime performance from results
            regime_performance = results.regime_performance
            
            if regime_performance:
                regime_results = {
                    'regime_breakdown': regime_performance,
                    'total_performance': self._extract_key_metrics(results)
                }
                
                for regime, stats in regime_performance.items():
                    logger.info(f"Regime {regime}: Trades={stats.get('trade_count', 0)}, Win Rate={stats.get('win_rate', 0):.2%}")
            else:
                logger.warning("No regime performance data available")
                regime_results = {'regime_breakdown': {}, 'total_performance': {}}
                
        except Exception as e:
            logger.error(f"Error in regime analysis: {str(e)}")
            regime_results = {'error': str(e)}
        
        return regime_results
    
    def _analyze_risk_metrics(self, all_results: Dict) -> Dict:
        """Analyze risk metrics across all tests"""
        
        risk_analysis = {
            'max_drawdown_analysis': {},
            'sharpe_ratio_distribution': {},
            'win_rate_analysis': {},
            'volatility_analysis': {}
        }
        
        try:
            # Extract risk metrics from comparison tests
            if 'comparison_tests' in all_results:
                enhanced_metrics = []
                baseline_metrics = []
                
                for universe_name, universe_results in all_results['comparison_tests'].items():
                    for period_name, period_results in universe_results.items():
                        if 'enhanced' in period_results and 'baseline' in period_results:
                            enhanced_metrics.append(period_results['enhanced'])
                            baseline_metrics.append(period_results['baseline'])
                
                if enhanced_metrics and baseline_metrics:
                    # Max drawdown analysis
                    enhanced_dd = [m['max_drawdown'] for m in enhanced_metrics]
                    baseline_dd = [m['max_drawdown'] for m in baseline_metrics]
                    
                    risk_analysis['max_drawdown_analysis'] = {
                        'enhanced_avg': np.mean(enhanced_dd),
                        'baseline_avg': np.mean(baseline_dd),
                        'enhanced_max': max(enhanced_dd),
                        'baseline_max': max(baseline_dd),
                        'improvement': np.mean(baseline_dd) - np.mean(enhanced_dd)
                    }
                    
                    # Sharpe ratio distribution
                    enhanced_sharpe = [m['sharpe_ratio'] for m in enhanced_metrics]
                    baseline_sharpe = [m['sharpe_ratio'] for m in baseline_metrics]
                    
                    risk_analysis['sharpe_ratio_distribution'] = {
                        'enhanced_avg': np.mean(enhanced_sharpe),
                        'baseline_avg': np.mean(baseline_sharpe),
                        'enhanced_std': np.std(enhanced_sharpe),
                        'baseline_std': np.std(baseline_sharpe),
                        'improvement': np.mean(enhanced_sharpe) - np.mean(baseline_sharpe)
                    }
                    
                    # Win rate analysis
                    enhanced_wr = [m['win_rate'] for m in enhanced_metrics]
                    baseline_wr = [m['win_rate'] for m in baseline_metrics]
                    
                    risk_analysis['win_rate_analysis'] = {
                        'enhanced_avg': np.mean(enhanced_wr),
                        'baseline_avg': np.mean(baseline_wr),
                        'improvement': np.mean(enhanced_wr) - np.mean(baseline_wr)
                    }
                    
                    logger.info(f"Risk Analysis - Enhanced DD: {np.mean(enhanced_dd):.2%}, Baseline DD: {np.mean(baseline_dd):.2%}")
                    logger.info(f"Risk Analysis - Enhanced Sharpe: {np.mean(enhanced_sharpe):.2f}, Baseline Sharpe: {np.mean(baseline_sharpe):.2f}")
                    
        except Exception as e:
            logger.error(f"Error in risk analysis: {str(e)}")
            risk_analysis['error'] = str(e)
        
        return risk_analysis
    
    def _calculate_improvement_metrics(self, enhanced_results, baseline_results) -> Dict:
        """Calculate improvement metrics between strategies"""
        
        return {
            'return_improvement': enhanced_results.total_return - baseline_results.total_return,
            'sharpe_improvement': enhanced_results.sharpe_ratio - baseline_results.sharpe_ratio,
            'drawdown_improvement': baseline_results.max_drawdown - enhanced_results.max_drawdown,
            'win_rate_improvement': enhanced_results.win_rate - baseline_results.win_rate,
            'trade_efficiency': (enhanced_results.total_return / max(enhanced_results.total_trades, 1)) - 
                               (baseline_results.total_return / max(baseline_results.total_trades, 1)),
            'relative_return': (enhanced_results.total_return / max(baseline_results.total_return, 0.001)) - 1 if baseline_results.total_return != 0 else 0
        }
    
    def _extract_key_metrics(self, results) -> Dict:
        """Extract key performance metrics"""
        
        return {
            'total_return': results.total_return,
            'annualized_return': results.annualized_return,
            'volatility': results.volatility,
            'sharpe_ratio': results.sharpe_ratio,
            'calmar_ratio': results.calmar_ratio,
            'max_drawdown': results.max_drawdown,
            'win_rate': results.win_rate,
            'profit_factor': results.profit_factor,
            'total_trades': results.total_trades,
            'avg_holding_period': results.avg_holding_period
        }
    
    def _generate_comprehensive_report(self, all_results: Dict) -> Dict:
        """Generate comprehensive performance report"""
        
        report = {
            'executive_summary': {},
            'detailed_analysis': {},
            'recommendations': []
        }
        
        try:
            # Executive Summary
            if 'comparison_tests' in all_results:
                comparison_data = all_results['comparison_tests']
                
                # Calculate aggregate improvements
                all_improvements = []
                all_enhanced_returns = []
                all_enhanced_sharpes = []
                
                for universe_results in comparison_data.values():
                    for period_results in universe_results.values():
                        if 'improvement' in period_results:
                            all_improvements.append(period_results['improvement'])
                        if 'enhanced' in period_results:
                            all_enhanced_returns.append(period_results['enhanced']['total_return'])
                            all_enhanced_sharpes.append(period_results['enhanced']['sharpe_ratio'])
                
                if all_improvements:
                    avg_return_imp = np.mean([imp['return_improvement'] for imp in all_improvements])
                    avg_sharpe_imp = np.mean([imp['sharpe_improvement'] for imp in all_improvements])
                    avg_dd_imp = np.mean([imp['drawdown_improvement'] for imp in all_improvements])
                    
                    report['executive_summary'] = {
                        'avg_return_improvement': avg_return_imp,
                        'avg_sharpe_improvement': avg_sharpe_imp,
                        'avg_drawdown_improvement': avg_dd_imp,
                        'avg_enhanced_return': np.mean(all_enhanced_returns) if all_enhanced_returns else 0,
                        'avg_enhanced_sharpe': np.mean(all_enhanced_sharpes) if all_enhanced_sharpes else 0,
                        'positive_improvement_rate': len([imp for imp in all_improvements if imp['return_improvement'] > 0]) / len(all_improvements) if all_improvements else 0
                    }
                    
                    # Generate recommendations based on results
                    if avg_return_imp > 0.02:  # 2% improvement
                        report['recommendations'].append("Enhanced strategy shows significant return improvement - recommend deployment")
                    if avg_sharpe_imp > 0.2:
                        report['recommendations'].append("Risk-adjusted performance improvement is substantial")
                    if avg_dd_imp > 0.01:  # 1% better drawdown
                        report['recommendations'].append("Improved risk management demonstrated")
            
            # Detailed Analysis
            report['detailed_analysis'] = {
                'best_performing_scenario': self._find_best_scenario(all_results),
                'worst_performing_scenario': self._find_worst_scenario(all_results),
                'parameter_sensitivity_summary': self._summarize_sensitivity(all_results),
                'regime_performance_summary': self._summarize_regime_performance(all_results)
            }
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            report['error'] = str(e)
        
        return report
    
    def _find_best_scenario(self, all_results: Dict) -> Dict:
        """Find the best performing scenario"""
        
        best_scenario = {'return': -999, 'scenario': 'None'}
        
        try:
            if 'comparison_tests' in all_results:
                for universe_name, universe_results in all_results['comparison_tests'].items():
                    for period_name, period_results in universe_results.items():
                        if 'enhanced' in period_results:
                            enhanced_return = period_results['enhanced']['total_return']
                            if enhanced_return > best_scenario['return']:
                                best_scenario = {
                                    'return': enhanced_return,
                                    'scenario': f"{universe_name}_{period_name}",
                                    'sharpe': period_results['enhanced']['sharpe_ratio'],
                                    'drawdown': period_results['enhanced']['max_drawdown']
                                }
        except:
            pass
        
        return best_scenario
    
    def _find_worst_scenario(self, all_results: Dict) -> Dict:
        """Find the worst performing scenario"""
        
        worst_scenario = {'return': 999, 'scenario': 'None'}
        
        try:
            if 'comparison_tests' in all_results:
                for universe_name, universe_results in all_results['comparison_tests'].items():
                    for period_name, period_results in universe_results.items():
                        if 'enhanced' in period_results:
                            enhanced_return = period_results['enhanced']['total_return']
                            if enhanced_return < worst_scenario['return']:
                                worst_scenario = {
                                    'return': enhanced_return,
                                    'scenario': f"{universe_name}_{period_name}",
                                    'sharpe': period_results['enhanced']['sharpe_ratio'],
                                    'drawdown': period_results['enhanced']['max_drawdown']
                                }
        except:
            pass
        
        return worst_scenario
    
    def _summarize_sensitivity(self, all_results: Dict) -> Dict:
        """Summarize parameter sensitivity results"""
        
        sensitivity_summary = {}
        
        try:
            if 'sensitivity_analysis' in all_results:
                for param_name, param_results in all_results['sensitivity_analysis'].items():
                    returns = [result['total_return'] for result in param_results.values()]
                    if returns:
                        sensitivity_summary[param_name] = {
                            'return_range': max(returns) - min(returns),
                            'best_value': max(param_results.items(), key=lambda x: x[1]['total_return'])[0],
                            'sensitivity_score': np.std(returns) / np.mean(np.abs(returns)) if returns else 0
                        }
        except:
            pass
        
        return sensitivity_summary
    
    def _summarize_regime_performance(self, all_results: Dict) -> Dict:
        """Summarize regime performance results"""
        
        regime_summary = {}
        
        try:
            if 'regime_analysis' in all_results and 'regime_breakdown' in all_results['regime_analysis']:
                regime_breakdown = all_results['regime_analysis']['regime_breakdown']
                
                for regime, stats in regime_breakdown.items():
                    regime_summary[regime] = {
                        'trade_count': stats.get('trade_count', 0),
                        'win_rate': stats.get('win_rate', 0),
                        'avg_pnl': stats.get('avg_pnl', 0)
                    }
        except:
            pass
        
        return regime_summary
    
    def _save_results(self, all_results: Dict, report: Dict):
        """Save backtest results and report"""
        
        # Create results directory
        results_dir = Path('backtest_results')
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        with open(results_dir / f'detailed_results_{timestamp}.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Save summary report
        with open(results_dir / f'summary_report_{timestamp}.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_dir}")


def main():
    """Main execution function"""
    
    try:
        # Ensure logs directory exists
        Path('logs').mkdir(exist_ok=True)
        
        # Create and run comprehensive backtester
        backtester = ComprehensiveBacktester(initial_capital=100000)
        
        # Run all tests
        final_results = backtester.run_comprehensive_tests()
        
        # Print summary
        print("\n" + "="*60)
        print("COMPREHENSIVE BACKTEST RESULTS SUMMARY")
        print("="*60)
        
        if 'report' in final_results and 'executive_summary' in final_results['report']:
            summary = final_results['report']['executive_summary']
            
            print(f"Average Return Improvement: {summary.get('avg_return_improvement', 0):.2%}")
            print(f"Average Sharpe Improvement: {summary.get('avg_sharpe_improvement', 0):.2f}")
            print(f"Average Drawdown Improvement: {summary.get('avg_drawdown_improvement', 0):.2%}")
            print(f"Enhanced Strategy Average Return: {summary.get('avg_enhanced_return', 0):.2%}")
            print(f"Enhanced Strategy Average Sharpe: {summary.get('avg_enhanced_sharpe', 0):.2f}")
            print(f"Positive Improvement Rate: {summary.get('positive_improvement_rate', 0):.1%}")
            
            print("\nRecommendations:")
            for rec in final_results['report'].get('recommendations', []):
                print(f"- {rec}")
        else:
            print("Summary not available - check detailed logs for results")
        
        print("\nDetailed results saved in backtest_results/ directory")
        print("="*60)
        
        return final_results
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()