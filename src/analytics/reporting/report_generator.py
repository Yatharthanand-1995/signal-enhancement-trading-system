"""
Automated Reporting and Visualization System
Generates comprehensive reports with performance metrics, attribution analysis, and visualizations
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.analytics.performance.attribution import PerformanceAttributionAnalyzer
from src.analytics.signals.effectiveness import SignalEffectivenessTracker
from src.analytics.portfolio.optimization import PortfolioOptimizer

logger = logging.getLogger(__name__)

@dataclass
class ReportMetrics:
    """Core metrics for reporting"""
    portfolio_return: float
    benchmark_return: float
    active_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    tracking_error: float
    information_ratio: float
    win_rate: float
    signal_count: int
    
@dataclass
class ReportConfig:
    """Configuration for report generation"""
    output_dir: str = "reports"
    include_charts: bool = True
    chart_format: str = "png"
    report_format: str = "html"
    time_period: int = 30  # days
    benchmark: str = "SPY"
    
class ReportGenerator:
    """
    Automated reporting and visualization system
    """
    
    def __init__(self, 
                 config: Optional[ReportConfig] = None,
                 db_path: str = "data/analytics.db"):
        self.config = config or ReportConfig()
        self.db_path = db_path
        
        # Initialize analytics components
        self.attribution_analyzer = PerformanceAttributionAnalyzer(db_path)
        self.signal_tracker = SignalEffectivenessTracker(db_path)
        self.portfolio_optimizer = PortfolioOptimizer()
        
        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "charts").mkdir(exist_ok=True)
        
        # Setup plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_comprehensive_report(self, 
                                    start_date: Optional[datetime] = None,
                                    end_date: Optional[datetime] = None) -> str:
        """Generate comprehensive trading report"""
        
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=self.config.time_period)
        
        logger.info(f"Generating report for period: {start_date} to {end_date}")
        
        try:
            # Collect all report data
            metrics = self._calculate_report_metrics(start_date, end_date)
            attribution_data = self._get_attribution_analysis(start_date, end_date)
            signal_data = self._get_signal_analysis(start_date, end_date)
            portfolio_data = self._get_portfolio_analysis()
            
            # Generate visualizations
            if self.config.include_charts:
                self._generate_performance_charts(metrics, start_date, end_date)
                self._generate_attribution_charts(attribution_data)
                self._generate_signal_charts(signal_data)
            
            # Generate report
            if self.config.report_format == "html":
                report_path = self._generate_html_report(
                    metrics, attribution_data, signal_data, portfolio_data,
                    start_date, end_date
                )
            else:
                report_path = self._generate_json_report(
                    metrics, attribution_data, signal_data, portfolio_data,
                    start_date, end_date
                )
            
            logger.info(f"Report generated: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    def _calculate_report_metrics(self, 
                                start_date: datetime,
                                end_date: datetime) -> ReportMetrics:
        """Calculate core metrics for the reporting period"""
        
        # Generate sample data for demonstration
        days = (end_date - start_date).days
        returns = np.random.normal(0.001, 0.02, days)  # Daily returns
        benchmark_returns = np.random.normal(0.0008, 0.015, days)
        
        portfolio_return = np.prod(1 + returns) - 1
        benchmark_return = np.prod(1 + benchmark_returns) - 1
        active_return = portfolio_return - benchmark_return
        
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        # Max drawdown calculation
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Tracking error and information ratio
        active_returns = returns - benchmark_returns
        tracking_error = np.std(active_returns) * np.sqrt(252)
        information_ratio = np.mean(active_returns) / np.std(active_returns) * np.sqrt(252)
        
        win_rate = np.sum(returns > 0) / len(returns)
        signal_count = len(returns)  # Simplified
        
        return ReportMetrics(
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
            active_return=active_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            win_rate=win_rate,
            signal_count=signal_count
        )
    
    def _get_attribution_analysis(self, 
                                start_date: datetime,
                                end_date: datetime) -> Dict[str, Any]:
        """Get performance attribution analysis"""
        
        # Sample attribution data
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        
        # Factor attribution
        factor_attribution = {
            'market_beta': 0.0025,
            'value_factor': 0.0015,
            'momentum_factor': -0.0008,
            'quality_factor': 0.0012,
            'size_factor': 0.0003
        }
        
        # Sector attribution
        sector_attribution = {
            'Technology': 0.0018,
            'Healthcare': 0.0012,
            'Financials': -0.0005,
            'Consumer Discretionary': 0.0008,
            'Industrials': 0.0003
        }
        
        # Signal attribution
        signal_attribution = {
            'momentum_signals': 0.0015,
            'mean_reversion_signals': 0.0008,
            'sentiment_signals': 0.0005,
            'fundamental_signals': 0.0012
        }
        
        return {
            'factor_attribution': factor_attribution,
            'sector_attribution': sector_attribution,
            'signal_attribution': signal_attribution,
            'total_attribution': sum(factor_attribution.values())
        }
    
    def _get_signal_analysis(self, 
                           start_date: datetime,
                           end_date: datetime) -> Dict[str, Any]:
        """Get signal effectiveness analysis"""
        
        # Signal performance data
        signal_performance = {
            'momentum_1d': {'accuracy': 0.62, 'avg_return': 0.0015, 'count': 45},
            'momentum_5d': {'accuracy': 0.58, 'avg_return': 0.0012, 'count': 32},
            'mean_reversion': {'accuracy': 0.65, 'avg_return': 0.0018, 'count': 28},
            'sentiment': {'accuracy': 0.55, 'avg_return': 0.0008, 'count': 52},
            'volume_anomaly': {'accuracy': 0.68, 'avg_return': 0.0022, 'count': 23}
        }
        
        # Correlation analysis
        signal_correlation = np.array([
            [1.00, 0.25, -0.15, 0.08, 0.12],
            [0.25, 1.00, -0.08, 0.05, 0.18],
            [-0.15, -0.08, 1.00, 0.02, -0.05],
            [0.08, 0.05, 0.02, 1.00, 0.15],
            [0.12, 0.18, -0.05, 0.15, 1.00]
        ])
        
        return {
            'signal_performance': signal_performance,
            'signal_correlation': signal_correlation,
            'top_signals': ['volume_anomaly', 'mean_reversion', 'momentum_1d'],
            'total_signals': sum(s['count'] for s in signal_performance.values())
        }
    
    def _get_portfolio_analysis(self) -> Dict[str, Any]:
        """Get portfolio optimization analysis"""
        
        # Sample portfolio weights
        portfolio_weights = {
            'AAPL': 0.15,
            'MSFT': 0.12,
            'GOOGL': 0.10,
            'TSLA': 0.08,
            'AMZN': 0.10,
            'META': 0.07,
            'NVDA': 0.09,
            'Others': 0.29
        }
        
        # Risk metrics
        risk_metrics = {
            'portfolio_volatility': 0.18,
            'expected_return': 0.12,
            'sharpe_ratio': 0.67,
            'var_95': -0.025,
            'cvar_95': -0.038
        }
        
        return {
            'portfolio_weights': portfolio_weights,
            'risk_metrics': risk_metrics,
            'rebalance_frequency': 'Weekly',
            'last_rebalance': datetime.now() - timedelta(days=3)
        }
    
    def _generate_performance_charts(self, 
                                   metrics: ReportMetrics,
                                   start_date: datetime,
                                   end_date: datetime):
        """Generate performance visualization charts"""
        
        # Performance comparison chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cumulative returns
        days = (end_date - start_date).days
        returns = np.random.normal(0.001, 0.02, days)
        benchmark_returns = np.random.normal(0.0008, 0.015, days)
        
        cum_returns = np.cumprod(1 + returns)
        cum_benchmark = np.cumprod(1 + benchmark_returns)
        dates = pd.date_range(start_date, end_date, freq='D')[:len(cum_returns)]
        
        axes[0, 0].plot(dates, cum_returns, label='Portfolio', linewidth=2)
        axes[0, 0].plot(dates, cum_benchmark, label='Benchmark', linewidth=2)
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Drawdown chart
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        axes[0, 1].fill_between(dates, drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Risk-return scatter
        risk_data = np.array([[0.12, 0.18], [0.08, 0.15]])  # [return, volatility]
        axes[1, 0].scatter(risk_data[:, 1], risk_data[:, 0], s=100)
        axes[1, 0].set_xlabel('Volatility')
        axes[1, 0].set_ylabel('Return')
        axes[1, 0].set_title('Risk-Return Profile')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Key metrics bar chart
        metric_names = ['Sharpe Ratio', 'Information Ratio', 'Win Rate']
        metric_values = [metrics.sharpe_ratio, metrics.information_ratio, metrics.win_rate]
        axes[1, 1].bar(metric_names, metric_values)
        axes[1, 1].set_title('Key Performance Metrics')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'charts' / f'performance_{datetime.now().strftime("%Y%m%d")}.png')
        plt.close()
    
    def _generate_attribution_charts(self, attribution_data: Dict[str, Any]):
        """Generate attribution analysis charts"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Factor attribution
        factors = list(attribution_data['factor_attribution'].keys())
        values = list(attribution_data['factor_attribution'].values())
        axes[0].barh(factors, values)
        axes[0].set_title('Factor Attribution')
        axes[0].set_xlabel('Contribution to Return')
        axes[0].grid(True, alpha=0.3)
        
        # Sector attribution pie chart
        sectors = list(attribution_data['sector_attribution'].keys())
        sector_values = list(attribution_data['sector_attribution'].values())
        axes[1].pie([abs(v) for v in sector_values], labels=sectors, autopct='%1.1f%%')
        axes[1].set_title('Sector Attribution')
        
        # Signal attribution
        signals = list(attribution_data['signal_attribution'].keys())
        signal_values = list(attribution_data['signal_attribution'].values())
        axes[2].bar(signals, signal_values)
        axes[2].set_title('Signal Attribution')
        axes[2].set_ylabel('Contribution to Return')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'charts' / f'attribution_{datetime.now().strftime("%Y%m%d")}.png')
        plt.close()
    
    def _generate_signal_charts(self, signal_data: Dict[str, Any]):
        """Generate signal analysis charts"""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Signal performance scatter
        signals = list(signal_data['signal_performance'].keys())
        accuracies = [signal_data['signal_performance'][s]['accuracy'] for s in signals]
        returns = [signal_data['signal_performance'][s]['avg_return'] for s in signals]
        counts = [signal_data['signal_performance'][s]['count'] for s in signals]
        
        scatter = axes[0].scatter(accuracies, returns, s=[c*3 for c in counts], alpha=0.6)
        axes[0].set_xlabel('Accuracy')
        axes[0].set_ylabel('Average Return')
        axes[0].set_title('Signal Performance (size = count)')
        for i, signal in enumerate(signals):
            axes[0].annotate(signal, (accuracies[i], returns[i]))
        axes[0].grid(True, alpha=0.3)
        
        # Signal correlation heatmap
        correlation_matrix = signal_data['signal_correlation']
        im = axes[1].imshow(correlation_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
        axes[1].set_title('Signal Correlation Matrix')
        axes[1].set_xticks(range(len(signals)))
        axes[1].set_yticks(range(len(signals)))
        axes[1].set_xticklabels(signals, rotation=45)
        axes[1].set_yticklabels(signals)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1])
        cbar.set_label('Correlation')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'charts' / f'signals_{datetime.now().strftime("%Y%m%d")}.png')
        plt.close()
    
    def _generate_html_report(self, 
                            metrics: ReportMetrics,
                            attribution_data: Dict[str, Any],
                            signal_data: Dict[str, Any],
                            portfolio_data: Dict[str, Any],
                            start_date: datetime,
                            end_date: datetime) -> str:
        """Generate HTML report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"trading_report_{timestamp}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Signal Trading System - Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 20px; border-left: 4px solid #3498db; }}
                .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .metric {{ background: #ecf0f1; padding: 15px; border-radius: 5px; min-width: 200px; }}
                .chart {{ text-align: center; margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #34495e; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Signal Trading System Performance Report</h1>
                <p>Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}</p>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Key Performance Metrics</h2>
                <div class="metrics">
                    <div class="metric">
                        <h3>Portfolio Return</h3>
                        <p>{metrics.portfolio_return:.2%}</p>
                    </div>
                    <div class="metric">
                        <h3>Benchmark Return</h3>
                        <p>{metrics.benchmark_return:.2%}</p>
                    </div>
                    <div class="metric">
                        <h3>Active Return</h3>
                        <p>{metrics.active_return:.2%}</p>
                    </div>
                    <div class="metric">
                        <h3>Sharpe Ratio</h3>
                        <p>{metrics.sharpe_ratio:.2f}</p>
                    </div>
                    <div class="metric">
                        <h3>Max Drawdown</h3>
                        <p>{metrics.max_drawdown:.2%}</p>
                    </div>
                    <div class="metric">
                        <h3>Win Rate</h3>
                        <p>{metrics.win_rate:.1%}</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 Performance Attribution</h2>
                <h3>Factor Attribution</h3>
                <table>
                    <tr><th>Factor</th><th>Contribution</th></tr>
        """
        
        for factor, contribution in attribution_data['factor_attribution'].items():
            html_content += f"<tr><td>{factor}</td><td>{contribution:.4f}</td></tr>"
        
        html_content += f"""
                </table>
                
                <h3>Signal Attribution</h3>
                <table>
                    <tr><th>Signal Type</th><th>Contribution</th></tr>
        """
        
        for signal, contribution in attribution_data['signal_attribution'].items():
            html_content += f"<tr><td>{signal}</td><td>{contribution:.4f}</td></tr>"
        
        html_content += f"""
                </table>
            </div>
            
            <div class="section">
                <h2>🔍 Signal Analysis</h2>
                <table>
                    <tr><th>Signal</th><th>Accuracy</th><th>Avg Return</th><th>Count</th></tr>
        """
        
        for signal, data in signal_data['signal_performance'].items():
            html_content += f"""
                <tr>
                    <td>{signal}</td>
                    <td>{data['accuracy']:.1%}</td>
                    <td>{data['avg_return']:.4f}</td>
                    <td>{data['count']}</td>
                </tr>
            """
        
        html_content += f"""
                </table>
                <p><strong>Total Signals Generated:</strong> {signal_data['total_signals']}</p>
                <p><strong>Top Performing Signals:</strong> {', '.join(signal_data['top_signals'])}</p>
            </div>
            
            <div class="section">
                <h2>💼 Portfolio Analysis</h2>
                <h3>Current Allocation</h3>
                <table>
                    <tr><th>Asset</th><th>Weight</th></tr>
        """
        
        for asset, weight in portfolio_data['portfolio_weights'].items():
            html_content += f"<tr><td>{asset}</td><td>{weight:.1%}</td></tr>"
        
        html_content += f"""
                </table>
                
                <h3>Risk Metrics</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Portfolio Volatility</td><td>{portfolio_data['risk_metrics']['portfolio_volatility']:.1%}</td></tr>
                    <tr><td>Expected Return</td><td>{portfolio_data['risk_metrics']['expected_return']:.1%}</td></tr>
                    <tr><td>Sharpe Ratio</td><td>{portfolio_data['risk_metrics']['sharpe_ratio']:.2f}</td></tr>
                    <tr><td>VaR (95%)</td><td>{portfolio_data['risk_metrics']['var_95']:.1%}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>📈 Charts</h2>
                <div class="chart">
                    <img src="charts/performance_{datetime.now().strftime("%Y%m%d")}.png" alt="Performance Charts" style="max-width: 100%;">
                </div>
                <div class="chart">
                    <img src="charts/attribution_{datetime.now().strftime("%Y%m%d")}.png" alt="Attribution Charts" style="max-width: 100%;">
                </div>
                <div class="chart">
                    <img src="charts/signals_{datetime.now().strftime("%Y%m%d")}.png" alt="Signal Charts" style="max-width: 100%;">
                </div>
            </div>
            
            <div class="section">
                <h2>📋 Summary</h2>
                <p>The trading system generated <strong>{metrics.signal_count}</strong> signals during the reporting period, 
                achieving a <strong>{metrics.win_rate:.1%}</strong> win rate and outperforming the benchmark by 
                <strong>{metrics.active_return:.2%}</strong>.</p>
                
                <p>Top performing signal categories were: <strong>{', '.join(signal_data['top_signals'])}</strong>.</p>
                
                <p>Risk-adjusted performance (Sharpe ratio: <strong>{metrics.sharpe_ratio:.2f}</strong>) indicates 
                efficient use of risk capital with maximum drawdown of <strong>{metrics.max_drawdown:.2%}</strong>.</p>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def _generate_json_report(self,
                            metrics: ReportMetrics,
                            attribution_data: Dict[str, Any],
                            signal_data: Dict[str, Any],
                            portfolio_data: Dict[str, Any],
                            start_date: datetime,
                            end_date: datetime) -> str:
        """Generate JSON report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"trading_report_{timestamp}.json"
        
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'period_start': start_date.isoformat(),
                'period_end': end_date.isoformat(),
                'report_type': 'comprehensive_trading_report'
            },
            'performance_metrics': asdict(metrics),
            'attribution_analysis': attribution_data,
            'signal_analysis': {
                **signal_data,
                'signal_correlation': signal_data['signal_correlation'].tolist()  # Convert numpy array
            },
            'portfolio_analysis': portfolio_data
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return str(report_path)
    
    def generate_daily_summary(self) -> str:
        """Generate quick daily summary report"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        metrics = self._calculate_report_metrics(start_date, end_date)
        
        summary = f"""
🚀 Daily Trading Summary - {end_date.strftime('%Y-%m-%d')}
================================
📈 Portfolio Return: {metrics.portfolio_return:.2%}
📊 Benchmark Return: {metrics.benchmark_return:.2%}
⚡ Active Return: {metrics.active_return:.2%}
🎯 Win Rate: {metrics.win_rate:.1%}
📊 Signals: {metrics.signal_count}
🏆 Sharpe Ratio: {metrics.sharpe_ratio:.2f}
        """
        
        return summary
    
    def schedule_reports(self, frequency: str = "daily"):
        """Schedule automated report generation"""
        logger.info(f"Report scheduling for {frequency} frequency would be implemented here")
        # In production, this would integrate with a task scheduler like Celery or APScheduler
        pass

# Convenience functions for easy usage
def generate_report(days: int = 30, output_dir: str = "reports") -> str:
    """Generate a comprehensive report for the last N days"""
    config = ReportConfig(output_dir=output_dir, time_period=days)
    generator = ReportGenerator(config)
    return generator.generate_comprehensive_report()

def generate_daily_summary() -> str:
    """Generate quick daily summary"""
    generator = ReportGenerator()
    return generator.generate_daily_summary()

if __name__ == "__main__":
    # Demo report generation
    print("🚀 Generating sample trading report...")
    
    config = ReportConfig(
        output_dir="reports",
        include_charts=True,
        time_period=30
    )
    
    generator = ReportGenerator(config)
    report_path = generator.generate_comprehensive_report()
    print(f"✅ Report generated: {report_path}")
    
    # Daily summary
    print("\n" + generator.generate_daily_summary())