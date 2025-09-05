"""
Performance Attribution Analysis
Advanced attribution analysis for trading strategy performance
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import sqlite3
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class AttributionMetrics:
    """Performance attribution metrics"""
    total_return: float
    active_return: float
    tracking_error: float
    information_ratio: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    
@dataclass
class AttributionBreakdown:
    """Attribution breakdown by source"""
    factor_name: str
    contribution: float
    contribution_pct: float
    t_stat: float
    p_value: float
    confidence_interval: Tuple[float, float]

class PerformanceAttributionAnalyzer:
    """
    Advanced performance attribution analyzer
    Breaks down portfolio returns by various factors and sources
    """
    
    def __init__(self, db_path: str = "analytics.db"):
        self.db_path = db_path
        self.factor_loadings = {}
        self.benchmark_data = {}
        self._init_database()
        
        logger.info("Performance attribution analyzer initialized")
    
    def _init_database(self):
        """Initialize analytics database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS performance_attribution (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        portfolio_id TEXT NOT NULL,
                        date DATE NOT NULL,
                        total_return REAL NOT NULL,
                        benchmark_return REAL,
                        active_return REAL,
                        attribution_factors TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS factor_returns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE NOT NULL,
                        factor_name TEXT NOT NULL,
                        factor_return REAL NOT NULL,
                        factor_loading REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS attribution_reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        portfolio_id TEXT NOT NULL,
                        report_date DATE NOT NULL,
                        period_start DATE NOT NULL,
                        period_end DATE NOT NULL,
                        metrics TEXT NOT NULL,
                        breakdown TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
        except Exception as e:
            logger.error(f"Error initializing attribution database: {e}")
            raise
    
    def calculate_factor_attribution(self, portfolio_returns: pd.Series, 
                                   factor_returns: pd.DataFrame,
                                   portfolio_weights: pd.DataFrame = None) -> Dict[str, AttributionBreakdown]:
        """
        Calculate factor-based attribution using Fama-French style analysis
        
        Args:
            portfolio_returns: Portfolio return time series
            factor_returns: DataFrame with factor returns (columns = factors)
            portfolio_weights: Optional position weights for more precise attribution
        """
        try:
            # Align data
            common_dates = portfolio_returns.index.intersection(factor_returns.index)
            if len(common_dates) < 30:
                logger.warning(f"Insufficient common dates for attribution: {len(common_dates)}")
                return {}
            
            portfolio_aligned = portfolio_returns.loc[common_dates]
            factors_aligned = factor_returns.loc[common_dates]
            
            attribution_breakdown = {}
            
            # Multi-factor regression for each factor
            for factor_name in factors_aligned.columns:
                factor_data = factors_aligned[factor_name]
                
                # Remove NaN values
                valid_idx = ~(portfolio_aligned.isna() | factor_data.isna())
                if valid_idx.sum() < 20:
                    continue
                
                port_ret = portfolio_aligned[valid_idx]
                factor_ret = factor_data[valid_idx]
                
                # Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(factor_ret, port_ret)
                
                # Calculate contribution
                factor_contribution = slope * factor_ret.mean() * len(common_dates)
                contribution_pct = factor_contribution / portfolio_aligned.sum() * 100
                
                # T-statistic
                t_stat = slope / std_err if std_err > 0 else 0
                
                # Confidence interval (95%)
                margin_error = 1.96 * std_err
                confidence_interval = (slope - margin_error, slope + margin_error)
                
                attribution_breakdown[factor_name] = AttributionBreakdown(
                    factor_name=factor_name,
                    contribution=float(factor_contribution),
                    contribution_pct=float(contribution_pct),
                    t_stat=float(t_stat),
                    p_value=float(p_value),
                    confidence_interval=confidence_interval
                )
            
            return attribution_breakdown
            
        except Exception as e:
            logger.error(f"Error calculating factor attribution: {e}")
            return {}
    
    def calculate_sector_attribution(self, portfolio_returns: pd.Series,
                                   sector_weights: pd.DataFrame,
                                   benchmark_weights: pd.DataFrame,
                                   sector_returns: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate sector-based attribution (allocation + selection effects)
        """
        try:
            # Ensure all data has common dates
            common_dates = (portfolio_returns.index
                          .intersection(sector_weights.index)
                          .intersection(benchmark_weights.index)
                          .intersection(sector_returns.index))
            
            if len(common_dates) < 5:
                logger.warning("Insufficient data for sector attribution")
                return {}
            
            # Align all data
            port_weights = sector_weights.loc[common_dates]
            bench_weights = benchmark_weights.loc[common_dates]
            sect_returns = sector_returns.loc[common_dates]
            
            attribution_results = {}
            
            for sector in port_weights.columns:
                if sector not in bench_weights.columns or sector not in sect_returns.columns:
                    continue
                
                # Calculate allocation effect
                weight_diff = port_weights[sector] - bench_weights[sector]
                benchmark_return = (bench_weights * sect_returns).sum(axis=1)
                sector_return = sect_returns[sector]
                
                allocation_effect = (weight_diff * (sector_return - benchmark_return)).mean()
                
                # Calculate selection effect
                avg_port_weight = port_weights[sector].mean()
                excess_return = sector_return.mean()
                
                selection_effect = avg_port_weight * excess_return
                
                # Total effect
                total_effect = allocation_effect + selection_effect
                
                attribution_results[sector] = {
                    'allocation_effect': float(allocation_effect),
                    'selection_effect': float(selection_effect),
                    'total_effect': float(total_effect),
                    'avg_weight': float(avg_port_weight),
                    'avg_return': float(sector_return.mean())
                }
            
            return attribution_results
            
        except Exception as e:
            logger.error(f"Error calculating sector attribution: {e}")
            return {}
    
    def calculate_signal_attribution(self, portfolio_returns: pd.Series,
                                   signal_data: pd.DataFrame,
                                   position_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate attribution by signal types
        """
        try:
            # Ensure data alignment
            common_dates = (portfolio_returns.index
                          .intersection(signal_data.index)
                          .intersection(position_data.index))
            
            if len(common_dates) < 10:
                logger.warning("Insufficient data for signal attribution")
                return {}
            
            # Align data
            returns = portfolio_returns.loc[common_dates]
            signals = signal_data.loc[common_dates]
            positions = position_data.loc[common_dates]
            
            attribution_by_signal = {}
            
            # Get unique signal types
            signal_types = signals['signal_direction'].unique()
            
            for signal_type in signal_types:
                if pd.isna(signal_type):
                    continue
                
                # Filter positions by signal type
                signal_mask = signals['signal_direction'] == signal_type
                signal_positions = positions[signal_mask]
                
                if len(signal_positions) == 0:
                    continue
                
                # Calculate signal-specific returns
                signal_returns = []
                for date in common_dates:
                    if date in signal_positions.index:
                        daily_signal_positions = signal_positions.loc[date]
                        if hasattr(daily_signal_positions, 'sum'):
                            daily_contribution = daily_signal_positions.sum()
                        else:
                            daily_contribution = daily_signal_positions
                        signal_returns.append(daily_contribution)
                
                if len(signal_returns) > 0:
                    avg_contribution = np.mean(signal_returns)
                    total_contribution = np.sum(signal_returns)
                    
                    attribution_by_signal[signal_type] = {
                        'total_contribution': float(total_contribution),
                        'average_contribution': float(avg_contribution),
                        'num_signals': len(signal_returns),
                        'contribution_pct': float(total_contribution / returns.sum() * 100)
                    }
            
            return attribution_by_signal
            
        except Exception as e:
            logger.error(f"Error calculating signal attribution: {e}")
            return {}
    
    def calculate_performance_metrics(self, portfolio_returns: pd.Series,
                                    benchmark_returns: pd.Series = None,
                                    risk_free_rate: float = 0.02) -> AttributionMetrics:
        """
        Calculate comprehensive performance metrics
        """
        try:
            if len(portfolio_returns) == 0:
                raise ValueError("Empty portfolio returns")
            
            # Basic returns
            total_return = (1 + portfolio_returns).prod() - 1
            annualized_return = (1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1
            
            # Volatility
            volatility = portfolio_returns.std() * np.sqrt(252)
            
            # Sharpe ratio
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Drawdown analysis
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Win rate
            win_rate = (portfolio_returns > 0).mean()
            
            # Profit factor
            gross_profits = portfolio_returns[portfolio_returns > 0].sum()
            gross_losses = abs(portfolio_returns[portfolio_returns < 0].sum())
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
            
            # Sortino ratio
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Active return and tracking error (if benchmark provided)
            active_return = 0
            tracking_error = 0
            information_ratio = 0
            
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                # Align data
                common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
                if len(common_dates) > 0:
                    port_aligned = portfolio_returns.loc[common_dates]
                    bench_aligned = benchmark_returns.loc[common_dates]
                    
                    active_returns = port_aligned - bench_aligned
                    active_return = active_returns.mean() * 252
                    tracking_error = active_returns.std() * np.sqrt(252)
                    information_ratio = active_return / tracking_error if tracking_error > 0 else 0
            
            return AttributionMetrics(
                total_return=float(total_return),
                active_return=float(active_return),
                tracking_error=float(tracking_error),
                information_ratio=float(information_ratio),
                sharpe_ratio=float(sharpe_ratio),
                max_drawdown=float(max_drawdown),
                win_rate=float(win_rate),
                profit_factor=float(profit_factor),
                calmar_ratio=float(calmar_ratio),
                sortino_ratio=float(sortino_ratio)
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return AttributionMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def generate_attribution_report(self, portfolio_id: str, 
                                  start_date: datetime, end_date: datetime,
                                  portfolio_returns: pd.Series,
                                  benchmark_returns: pd.Series = None,
                                  factor_returns: pd.DataFrame = None,
                                  signal_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Generate comprehensive attribution report
        """
        try:
            report = {
                'portfolio_id': portfolio_id,
                'period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat(),
                    'days': (end_date - start_date).days
                },
                'generated_at': datetime.now().isoformat()
            }
            
            # Filter returns for period
            period_mask = (portfolio_returns.index >= start_date) & (portfolio_returns.index <= end_date)
            period_returns = portfolio_returns[period_mask]
            
            if len(period_returns) == 0:
                report['error'] = 'No data for specified period'
                return report
            
            # Performance metrics
            if benchmark_returns is not None:
                benchmark_period = benchmark_returns[period_mask]
                metrics = self.calculate_performance_metrics(period_returns, benchmark_period)
            else:
                metrics = self.calculate_performance_metrics(period_returns)
            
            report['performance_metrics'] = {
                'total_return': metrics.total_return,
                'active_return': metrics.active_return,
                'sharpe_ratio': metrics.sharpe_ratio,
                'information_ratio': metrics.information_ratio,
                'max_drawdown': metrics.max_drawdown,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'calmar_ratio': metrics.calmar_ratio,
                'sortino_ratio': metrics.sortino_ratio
            }
            
            # Factor attribution
            if factor_returns is not None:
                factor_period = factor_returns[period_mask]
                factor_attribution = self.calculate_factor_attribution(period_returns, factor_period)
                
                report['factor_attribution'] = {}
                for factor_name, breakdown in factor_attribution.items():
                    report['factor_attribution'][factor_name] = {
                        'contribution': breakdown.contribution,
                        'contribution_pct': breakdown.contribution_pct,
                        't_stat': breakdown.t_stat,
                        'p_value': breakdown.p_value,
                        'significant': breakdown.p_value < 0.05
                    }
            
            # Signal attribution
            if signal_data is not None:
                signal_period = signal_data[period_mask]
                # Mock position data for example
                position_data = pd.DataFrame(index=period_returns.index, 
                                           data={'position': np.random.uniform(-0.1, 0.1, len(period_returns))})
                
                signal_attribution = self.calculate_signal_attribution(period_returns, signal_period, position_data)
                report['signal_attribution'] = signal_attribution
            
            # Risk decomposition
            report['risk_analysis'] = {
                'volatility': float(period_returns.std() * np.sqrt(252)),
                'var_95': float(np.percentile(period_returns, 5)),
                'cvar_95': float(period_returns[period_returns <= np.percentile(period_returns, 5)].mean()),
                'skewness': float(stats.skew(period_returns)),
                'kurtosis': float(stats.kurtosis(period_returns))
            }
            
            # Store report
            self._store_attribution_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating attribution report: {e}")
            return {'error': str(e)}
    
    def _store_attribution_report(self, report: Dict[str, Any]):
        """Store attribution report in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO attribution_reports 
                    (portfolio_id, report_date, period_start, period_end, metrics, breakdown)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    report['portfolio_id'],
                    datetime.now().date(),
                    report['period']['start'],
                    report['period']['end'],
                    str(report.get('performance_metrics', {})),
                    str(report.get('factor_attribution', {}))
                ))
                
        except Exception as e:
            logger.error(f"Error storing attribution report: {e}")
    
    def get_historical_attribution(self, portfolio_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get historical attribution reports"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                reports = pd.read_sql_query('''
                    SELECT * FROM attribution_reports 
                    WHERE portfolio_id = ? AND created_at >= ?
                    ORDER BY created_at DESC
                ''', conn, params=(portfolio_id, cutoff_date))
            
            return reports.to_dict('records') if not reports.empty else []
            
        except Exception as e:
            logger.error(f"Error getting historical attribution: {e}")
            return []