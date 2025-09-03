#!/usr/bin/env python3
"""
Performance Analysis Engine for Backtesting Results
Analyzes portfolio performance across different market regimes and generates comprehensive reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for backtesting results
    """
    
    def __init__(self):
        self.portfolio_history = None
        self.trade_history = None
        self.benchmark_data = None
        self.signal_data = None
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_data(self, portfolio_history: pd.DataFrame, trade_history: pd.DataFrame,
                 signal_data: pd.DataFrame, benchmark_data: pd.DataFrame = None):
        """
        Load all necessary data for analysis
        """
        self.portfolio_history = portfolio_history.copy()
        self.trade_history = trade_history.copy()
        self.signal_data = signal_data.copy()
        self.benchmark_data = benchmark_data.copy() if benchmark_data is not None else None
        
        logger.info(f"Loaded data: {len(self.portfolio_history)} days, {len(self.trade_history)} trades")
    
    def calculate_returns_metrics(self, returns: pd.Series) -> Dict:
        """
        Calculate comprehensive returns metrics for a return series
        """
        if len(returns) == 0 or returns.isna().all():
            return {}
        
        # Clean returns
        returns = returns.dropna()
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_vol if downside_vol > 0 else 0
        
        # Drawdown calculation
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # VaR calculation (95% confidence)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Win rate (% of positive periods)
        win_rate = (returns > 0).mean()
        
        # Average win/loss
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
    
    def analyze_portfolio_performance(self) -> Dict:
        """
        Analyze overall portfolio performance
        """
        if self.portfolio_history is None:
            raise ValueError("Portfolio history not loaded")
        
        # Calculate daily returns
        portfolio_df = self.portfolio_history.copy()
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        
        # Overall performance metrics
        portfolio_metrics = self.calculate_returns_metrics(portfolio_df['daily_return'])
        
        # Additional portfolio-specific metrics
        initial_value = portfolio_df['portfolio_value'].iloc[0]
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        
        portfolio_metrics.update({
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return_dollar': final_value - initial_value,
            'average_cash_pct': portfolio_df['cash_pct'].mean(),
            'average_positions': portfolio_df['num_positions'].mean(),
            'max_positions': portfolio_df['num_positions'].max(),
            'min_positions': portfolio_df['num_positions'].min()
        })
        
        return portfolio_metrics
    
    def analyze_trading_performance(self) -> Dict:
        """
        Analyze trading-specific performance metrics
        """
        if self.trade_history is None or len(self.trade_history) == 0:
            return {}
        
        trades_df = self.trade_history.copy()
        
        # Filter to completed trades (sells)
        completed_trades = trades_df[trades_df['action'] == 'SELL'].copy()
        
        if len(completed_trades) == 0:
            return {}
        
        # Basic trading metrics
        total_trades = len(completed_trades)
        profitable_trades = len(completed_trades[completed_trades['realized_pnl'] > 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # P&L analysis
        total_pnl = completed_trades['realized_pnl'].sum()
        avg_pnl = completed_trades['realized_pnl'].mean()
        avg_pnl_pct = completed_trades['realized_pnl_pct'].mean()
        
        # Separate wins and losses
        wins = completed_trades[completed_trades['realized_pnl'] > 0]
        losses = completed_trades[completed_trades['realized_pnl'] <= 0]
        
        avg_win = wins['realized_pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss = losses['realized_pnl_pct'].mean() if len(losses) > 0 else 0
        
        # Risk metrics
        profit_factor = abs(wins['realized_pnl'].sum() / losses['realized_pnl'].sum()) if len(losses) > 0 and losses['realized_pnl'].sum() != 0 else float('inf')
        
        # Holding period analysis
        avg_holding_days = completed_trades['days_held'].mean()
        
        # Best and worst trades
        best_trade_pct = completed_trades['realized_pnl_pct'].max()
        worst_trade_pct = completed_trades['realized_pnl_pct'].min()
        
        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'losing_trades': total_trades - profitable_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_pnl_pct': avg_pnl_pct,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_days': avg_holding_days,
            'best_trade_pct': best_trade_pct,
            'worst_trade_pct': worst_trade_pct,
            'expectancy': win_rate * avg_win + (1 - win_rate) * avg_loss
        }
    
    def analyze_performance_by_regime(self) -> Dict:
        """
        Analyze performance broken down by market regime
        """
        if self.trade_history is None or self.portfolio_history is None:
            return {}
        
        # Analyze trading performance by regime
        trades_df = self.trade_history.copy()
        completed_trades = trades_df[trades_df['action'] == 'SELL'].copy()
        
        regime_analysis = {}
        
        # Get unique regimes
        if 'market_regime' in completed_trades.columns:
            regimes = completed_trades['market_regime'].unique()
            
            for regime in regimes:
                regime_trades = completed_trades[completed_trades['market_regime'] == regime]
                
                if len(regime_trades) > 0:
                    win_rate = (regime_trades['realized_pnl'] > 0).mean()
                    
                    regime_analysis[regime] = {
                        'total_trades': len(regime_trades),
                        'win_rate': win_rate,
                        'total_pnl': regime_trades['realized_pnl'].sum(),
                        'avg_pnl_pct': regime_trades['realized_pnl_pct'].mean(),
                        'avg_holding_days': regime_trades['days_held'].mean(),
                        'best_trade': regime_trades['realized_pnl_pct'].max(),
                        'worst_trade': regime_trades['realized_pnl_pct'].min(),
                        'profitable_trades': len(regime_trades[regime_trades['realized_pnl'] > 0])
                    }
        
        # Analyze portfolio returns by regime if we have the data
        if self.signal_data is not None and 'market_regime' in self.signal_data.columns:
            portfolio_df = self.portfolio_history.copy()
            portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
            
            # Add market regime to portfolio data (simplified)
            portfolio_df['market_regime'] = 'UNKNOWN'  # Would need proper mapping
            
        return regime_analysis
    
    def analyze_signal_accuracy(self) -> Dict:
        """
        Analyze the accuracy of our signal generation system
        """
        if self.signal_data is None or self.trade_history is None:
            return {}
        
        signals_df = self.signal_data.copy()
        trades_df = self.trade_history.copy()
        
        # Analyze signal distribution
        signal_counts = signals_df['signal'].value_counts()
        
        # Match trades with signals for accuracy analysis
        completed_trades = trades_df[trades_df['action'] == 'SELL'].copy()
        
        signal_accuracy = {}
        
        for signal_type in ['STRONG_BUY', 'BUY', 'SELL', 'STRONG_SELL']:
            # Find trades that entered with this signal
            signal_trades = completed_trades[completed_trades.get('signal') == signal_type]
            
            if len(signal_trades) > 0:
                win_rate = (signal_trades['realized_pnl'] > 0).mean()
                avg_return = signal_trades['realized_pnl_pct'].mean()
                
                signal_accuracy[signal_type] = {
                    'total_signals_generated': signal_counts.get(signal_type, 0),
                    'total_trades_executed': len(signal_trades),
                    'execution_rate': len(signal_trades) / signal_counts.get(signal_type, 1),
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'total_pnl': signal_trades['realized_pnl'].sum()
                }
        
        return signal_accuracy
    
    def compare_to_benchmark(self, benchmark_symbol: str = 'SPY') -> Dict:
        """
        Compare portfolio performance to benchmark
        """
        if self.benchmark_data is None or self.portfolio_history is None:
            logger.warning("Benchmark data not available for comparison")
            return {}
        
        # Calculate portfolio returns
        portfolio_df = self.portfolio_history.copy()
        portfolio_df['portfolio_return'] = portfolio_df['portfolio_value'].pct_change()
        
        # Calculate benchmark returns
        benchmark_df = self.benchmark_data.copy()
        benchmark_df['benchmark_return'] = benchmark_df['Close'].pct_change()
        
        # Align dates
        common_dates = portfolio_df.index.intersection(benchmark_df.index)
        portfolio_returns = portfolio_df.loc[common_dates, 'portfolio_return']
        benchmark_returns = benchmark_df.loc[common_dates, 'benchmark_return']
        
        # Calculate metrics for both
        portfolio_metrics = self.calculate_returns_metrics(portfolio_returns)
        benchmark_metrics = self.calculate_returns_metrics(benchmark_returns)
        
        # Calculate alpha and beta
        excess_returns = portfolio_returns - benchmark_returns
        alpha_annualized = excess_returns.mean() * 252
        
        # Beta calculation
        covariance = np.cov(portfolio_returns.dropna(), benchmark_returns.dropna())[0, 1]
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 1
        
        # Information ratio
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = alpha_annualized / tracking_error if tracking_error > 0 else 0
        
        comparison = {
            'portfolio': portfolio_metrics,
            'benchmark': benchmark_metrics,
            'alpha_annualized': alpha_annualized,
            'beta': beta,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'excess_return': portfolio_metrics['annualized_return'] - benchmark_metrics['annualized_return'],
            'excess_sharpe': portfolio_metrics['sharpe_ratio'] - benchmark_metrics['sharpe_ratio']
        }
        
        return comparison
    
    def create_performance_visualizations(self, save_path: str = None):
        """
        Create comprehensive performance visualization charts
        """
        if self.portfolio_history is None:
            return
        
        portfolio_df = self.portfolio_history.copy()
        portfolio_df['cumulative_return'] = (portfolio_df['portfolio_value'] / portfolio_df['portfolio_value'].iloc[0] - 1) * 100
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Portfolio Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Cumulative returns
        axes[0, 0].plot(portfolio_df.index, portfolio_df['cumulative_return'], linewidth=2, color='blue')
        axes[0, 0].set_title('Cumulative Returns (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 2. Portfolio value over time
        axes[0, 1].plot(portfolio_df.index, portfolio_df['portfolio_value'], linewidth=2, color='green')
        axes[0, 1].set_title('Portfolio Value Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].ticklabel_format(style='plain', axis='y')
        
        # 3. Number of positions over time
        axes[1, 0].plot(portfolio_df.index, portfolio_df['num_positions'], linewidth=2, color='orange')
        axes[1, 0].set_title('Number of Positions')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, portfolio_df['num_positions'].max() * 1.1)
        
        # 4. Cash percentage over time
        axes[1, 1].plot(portfolio_df.index, portfolio_df['cash_pct'], linewidth=2, color='purple')
        axes[1, 1].set_title('Cash Allocation (%)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 100)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/portfolio_performance.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_regime_analysis_charts(self, save_path: str = None):
        """
        Create charts analyzing performance by market regime
        """
        regime_analysis = self.analyze_performance_by_regime()
        
        if not regime_analysis:
            logger.warning("No regime analysis data available")
            return
        
        # Prepare data for plotting
        regimes = list(regime_analysis.keys())
        win_rates = [regime_analysis[r]['win_rate'] * 100 for r in regimes]
        avg_returns = [regime_analysis[r]['avg_pnl_pct'] for r in regimes]
        total_trades = [regime_analysis[r]['total_trades'] for r in regimes]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Performance by Market Regime', fontsize=16, fontweight='bold')
        
        # Win rate by regime
        bars1 = axes[0].bar(regimes, win_rates, color='skyblue', alpha=0.7)
        axes[0].set_title('Win Rate by Market Regime (%)')
        axes[0].set_ylabel('Win Rate (%)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].axhline(y=50, color='red', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom')
        
        # Average return by regime
        bars2 = axes[1].bar(regimes, avg_returns, color='lightgreen', alpha=0.7)
        axes[1].set_title('Average Return by Market Regime (%)')
        axes[1].set_ylabel('Average Return (%)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., 
                        height + (0.1 if height >= 0 else -0.3),
                        f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # Number of trades by regime
        bars3 = axes[2].bar(regimes, total_trades, color='coral', alpha=0.7)
        axes[2].set_title('Number of Trades by Market Regime')
        axes[2].set_ylabel('Number of Trades')
        axes[2].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/regime_analysis.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_comprehensive_report(self) -> str:
        """
        Generate a comprehensive performance analysis report
        """
        portfolio_metrics = self.analyze_portfolio_performance()
        trading_metrics = self.analyze_trading_performance()
        regime_analysis = self.analyze_performance_by_regime()
        signal_accuracy = self.analyze_signal_accuracy()
        
        report = f"""
# COMPREHENSIVE BACKTESTING PERFORMANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## EXECUTIVE SUMMARY
Initial Portfolio Value: ${portfolio_metrics.get('initial_value', 0):,.0f}
Final Portfolio Value: ${portfolio_metrics.get('final_value', 0):,.0f}
Total Return: {portfolio_metrics.get('total_return', 0)*100:.2f}%
Annualized Return: {portfolio_metrics.get('annualized_return', 0)*100:.2f}%
Sharpe Ratio: {portfolio_metrics.get('sharpe_ratio', 0):.2f}
Maximum Drawdown: {portfolio_metrics.get('max_drawdown', 0)*100:.2f}%

## DETAILED PERFORMANCE METRICS

### Risk-Adjusted Performance
- Sortino Ratio: {portfolio_metrics.get('sortino_ratio', 0):.2f}
- Calmar Ratio: {portfolio_metrics.get('calmar_ratio', 0):.2f}
- Volatility (Annualized): {portfolio_metrics.get('volatility', 0)*100:.2f}%
- Skewness: {portfolio_metrics.get('skewness', 0):.2f}
- Kurtosis: {portfolio_metrics.get('kurtosis', 0):.2f}

### Risk Metrics
- Value at Risk (95%): {portfolio_metrics.get('var_95', 0)*100:.2f}%
- Conditional VaR (95%): {portfolio_metrics.get('cvar_95', 0)*100:.2f}%
- Win Rate (Daily): {portfolio_metrics.get('win_rate', 0)*100:.1f}%

### Portfolio Management
- Average Cash Allocation: {portfolio_metrics.get('average_cash_pct', 0):.1f}%
- Average Number of Positions: {portfolio_metrics.get('average_positions', 0):.1f}
- Maximum Positions Held: {portfolio_metrics.get('max_positions', 0)}
- Minimum Positions Held: {portfolio_metrics.get('min_positions', 0)}

## TRADING PERFORMANCE ANALYSIS

### Trading Statistics
- Total Completed Trades: {trading_metrics.get('total_trades', 0)}
- Profitable Trades: {trading_metrics.get('profitable_trades', 0)}
- Losing Trades: {trading_metrics.get('losing_trades', 0)}
- Win Rate: {trading_metrics.get('win_rate', 0)*100:.1f}%

### Profit/Loss Analysis  
- Total P&L: ${trading_metrics.get('total_pnl', 0):,.0f}
- Average Trade Return: {trading_metrics.get('avg_pnl_pct', 0):.2f}%
- Average Winning Trade: {trading_metrics.get('avg_win', 0):.2f}%
- Average Losing Trade: {trading_metrics.get('avg_loss', 0):.2f}%
- Profit Factor: {trading_metrics.get('profit_factor', 0):.2f}
- Trade Expectancy: {trading_metrics.get('expectancy', 0):.2f}%

### Trading Behavior
- Average Holding Period: {trading_metrics.get('avg_holding_days', 0):.0f} days
- Best Trade: {trading_metrics.get('best_trade_pct', 0):.2f}%
- Worst Trade: {trading_metrics.get('worst_trade_pct', 0):.2f}%

## PERFORMANCE BY MARKET REGIME
"""
        
        for regime, metrics in regime_analysis.items():
            report += f"""
### {regime.replace('_', ' ').title()}
- Total Trades: {metrics['total_trades']}
- Win Rate: {metrics['win_rate']*100:.1f}%
- Average Return: {metrics['avg_pnl_pct']:.2f}%
- Total P&L: ${metrics['total_pnl']:,.0f}
- Average Hold Days: {metrics['avg_holding_days']:.0f}
- Best Trade: {metrics['best_trade']:.2f}%
- Worst Trade: {metrics['worst_trade']:.2f}%
"""
        
        report += "\n## SIGNAL ACCURACY ANALYSIS\n"
        
        for signal, metrics in signal_accuracy.items():
            report += f"""
### {signal} Signals
- Total Signals Generated: {metrics['total_signals_generated']}
- Trades Executed: {metrics['total_trades_executed']}
- Execution Rate: {metrics['execution_rate']*100:.1f}%
- Win Rate: {metrics['win_rate']*100:.1f}%
- Average Return: {metrics['avg_return']:.2f}%
- Total P&L: ${metrics['total_pnl']:,.0f}
"""
        
        report += f"""
## CONCLUSIONS AND INSIGHTS

### Key Strengths
{"- Strong risk-adjusted returns with Sharpe ratio above 1.5" if portfolio_metrics.get('sharpe_ratio', 0) > 1.5 else "- Risk-adjusted returns need improvement"}
{"- Excellent drawdown control with max drawdown < 20%" if abs(portfolio_metrics.get('max_drawdown', 0)) < 0.20 else "- Drawdown control could be improved"}  
{"- High win rate indicating good signal quality" if trading_metrics.get('win_rate', 0) > 0.65 else "- Win rate could be improved"}

### Areas for Improvement
{"- Consider increasing position sizing for better returns" if portfolio_metrics.get('annualized_return', 0) < 0.15 else "- Strong returns achieved"}
{"- Review signal accuracy, particularly for sell signals" if trading_metrics.get('win_rate', 0) < 0.6 else "- Signal accuracy is satisfactory"}
{"- Portfolio diversification looks good" if portfolio_metrics.get('average_positions', 0) > 15 else "- Consider increasing diversification"}

### Market Regime Performance
{"- Strategy adapts well across different market conditions" if len(regime_analysis) >= 4 else "- Need more market regime data for full analysis"}
{"- Strong performance during volatile periods" if any(r.get('win_rate', 0) > 0.6 and 'CRASH' in regime for regime, r in regime_analysis.items()) else "- Performance during stress periods needs attention"}

---

*This report provides a comprehensive analysis of the backtesting results. 
For detailed technical analysis and recommendations, please review the individual metric sections.*
"""
        
        return report

def main():
    """
    Test the performance analyzer
    """
    analyzer = PerformanceAnalyzer()
    print("Performance Analyzer initialized successfully!")
    print("Ready to analyze backtesting results.")

if __name__ == "__main__":
    main()