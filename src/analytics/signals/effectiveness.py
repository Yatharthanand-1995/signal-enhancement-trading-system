"""
Signal Effectiveness Analysis
Advanced tracking and analysis of signal performance and predictive power
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import sqlite3
import json
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class SignalMetrics:
    """Signal performance metrics"""
    signal_type: str
    total_signals: int
    hit_rate: float
    precision: float
    recall: float
    f1_score: float
    average_return: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_holding_period: float
    roi: float
    
@dataclass
class SignalDecayAnalysis:
    """Signal decay analysis results"""
    signal_type: str
    decay_half_life: float
    initial_effectiveness: float
    decay_rate: float
    effectiveness_timeline: Dict[str, float]
    statistical_significance: bool

class SignalEffectivenessTracker:
    """
    Advanced signal effectiveness tracking and analysis system
    Monitors signal performance, decay patterns, and predictive power
    """
    
    def __init__(self, db_path: str = "signal_analytics.db"):
        self.db_path = db_path
        self.signal_history = {}
        self.effectiveness_cache = {}
        
        # Initialize database
        self._init_database()
        
        logger.info("Signal effectiveness tracker initialized")
    
    def _init_database(self):
        """Initialize signal analytics database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS signal_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_id TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        generated_at DATETIME NOT NULL,
                        direction TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        target_price REAL,
                        entry_price REAL,
                        exit_price REAL,
                        exit_date DATETIME,
                        holding_period_hours REAL,
                        realized_return REAL,
                        max_favorable_move REAL,
                        max_adverse_move REAL,
                        outcome TEXT,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS signal_effectiveness (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_type TEXT NOT NULL,
                        measurement_date DATE NOT NULL,
                        total_signals INTEGER NOT NULL,
                        successful_signals INTEGER NOT NULL,
                        hit_rate REAL NOT NULL,
                        avg_return REAL NOT NULL,
                        sharpe_ratio REAL,
                        max_drawdown REAL,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS signal_decay (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_type TEXT NOT NULL,
                        analysis_date DATE NOT NULL,
                        decay_half_life REAL,
                        initial_effectiveness REAL,
                        decay_rate REAL,
                        effectiveness_timeline TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indices
                conn.execute('CREATE INDEX IF NOT EXISTS idx_signal_perf_type_date ON signal_performance(signal_type, generated_at)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_signal_symbol_date ON signal_performance(symbol, generated_at)')
                
        except Exception as e:
            logger.error(f"Error initializing signal analytics database: {e}")
            raise
    
    def record_signal(self, signal_id: str, signal_type: str, symbol: str, 
                     direction: str, confidence: float, target_price: float = None,
                     entry_price: float = None, metadata: Dict = None):
        """Record a new signal for tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO signal_performance 
                    (signal_id, signal_type, symbol, generated_at, direction, 
                     confidence, target_price, entry_price, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_id, signal_type, symbol, datetime.now(), direction,
                    confidence, target_price, entry_price, json.dumps(metadata or {})
                ))
            
            logger.debug(f"Recorded signal: {signal_id} ({signal_type}) for {symbol}")
            
        except Exception as e:
            logger.error(f"Error recording signal {signal_id}: {e}")
    
    def update_signal_outcome(self, signal_id: str, exit_price: float, 
                            exit_date: datetime = None, outcome: str = None):
        """Update signal with final outcome"""
        try:
            exit_date = exit_date or datetime.now()
            
            # Get original signal data
            with sqlite3.connect(self.db_path) as conn:
                signal_data = pd.read_sql_query('''
                    SELECT * FROM signal_performance 
                    WHERE signal_id = ?
                ''', conn, params=(signal_id,))
                
                if signal_data.empty:
                    logger.warning(f"Signal not found: {signal_id}")
                    return
                
                signal = signal_data.iloc[0]
                
                # Calculate metrics
                entry_price = signal['entry_price'] or signal['target_price']
                if entry_price:
                    if signal['direction'].upper() in ['BUY', 'LONG']:
                        realized_return = (exit_price - entry_price) / entry_price
                    else:  # SELL or SHORT
                        realized_return = (entry_price - exit_price) / entry_price
                else:
                    realized_return = 0.0
                
                # Calculate holding period
                generated_at = pd.to_datetime(signal['generated_at'])
                holding_period_hours = (exit_date - generated_at).total_seconds() / 3600
                
                # Determine outcome if not provided
                if outcome is None:
                    outcome = 'success' if realized_return > 0 else 'failure'
                
                # Update database
                conn.execute('''
                    UPDATE signal_performance 
                    SET exit_price = ?, exit_date = ?, holding_period_hours = ?,
                        realized_return = ?, outcome = ?
                    WHERE signal_id = ?
                ''', (exit_price, exit_date, holding_period_hours, realized_return, outcome, signal_id))
            
            logger.debug(f"Updated signal outcome: {signal_id} -> {outcome} ({realized_return:.3f})")
            
        except Exception as e:
            logger.error(f"Error updating signal outcome {signal_id}: {e}")
    
    def calculate_signal_metrics(self, signal_type: str, 
                               days_lookback: int = 30) -> Optional[SignalMetrics]:
        """Calculate comprehensive metrics for a signal type"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_lookback)
            
            with sqlite3.connect(self.db_path) as conn:
                signals_df = pd.read_sql_query('''
                    SELECT * FROM signal_performance 
                    WHERE signal_type = ? AND generated_at >= ? 
                    AND exit_price IS NOT NULL
                    ORDER BY generated_at DESC
                ''', conn, params=(signal_type, cutoff_date))
            
            if signals_df.empty:
                logger.warning(f"No completed signals found for {signal_type}")
                return None
            
            # Basic metrics
            total_signals = len(signals_df)
            successful_signals = (signals_df['realized_return'] > 0).sum()
            hit_rate = successful_signals / total_signals
            
            # Return metrics
            returns = signals_df['realized_return'].fillna(0)
            average_return = returns.mean()
            total_return = returns.sum()
            
            # Risk-adjusted metrics
            return_std = returns.std()
            sharpe_ratio = average_return / return_std if return_std > 0 else 0
            
            # Drawdown analysis
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Win rate
            win_rate = (returns > 0).mean()
            
            # Holding period
            avg_holding_period = signals_df['holding_period_hours'].mean()
            
            # ROI calculation
            roi = total_return * 100  # As percentage
            
            # Classification metrics (for directional signals)
            if 'direction' in signals_df.columns:
                y_true = (returns > 0).astype(int)
                y_pred = (signals_df['direction'].isin(['BUY', 'LONG'])).astype(int)
                
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            else:
                precision = hit_rate
                recall = hit_rate  
                f1 = hit_rate
            
            return SignalMetrics(
                signal_type=signal_type,
                total_signals=int(total_signals),
                hit_rate=float(hit_rate),
                precision=float(precision),
                recall=float(recall),
                f1_score=float(f1),
                average_return=float(average_return),
                total_return=float(total_return),
                sharpe_ratio=float(sharpe_ratio),
                max_drawdown=float(max_drawdown),
                win_rate=float(win_rate),
                avg_holding_period=float(avg_holding_period),
                roi=float(roi)
            )
            
        except Exception as e:
            logger.error(f"Error calculating signal metrics for {signal_type}: {e}")
            return None
    
    def analyze_signal_decay(self, signal_type: str, max_days_back: int = 90) -> Optional[SignalDecayAnalysis]:
        """Analyze signal effectiveness decay over time"""
        try:
            cutoff_date = datetime.now() - timedelta(days=max_days_back)
            
            with sqlite3.connect(self.db_path) as conn:
                signals_df = pd.read_sql_query('''
                    SELECT * FROM signal_performance 
                    WHERE signal_type = ? AND generated_at >= ?
                    AND exit_price IS NOT NULL
                    ORDER BY generated_at ASC
                ''', conn, params=(signal_type, cutoff_date))
            
            if len(signals_df) < 20:
                logger.warning(f"Insufficient data for decay analysis: {len(signals_df)} signals")
                return None
            
            # Convert dates
            signals_df['generated_at'] = pd.to_datetime(signals_df['generated_at'])
            
            # Group by weeks to analyze decay
            signals_df['week'] = signals_df['generated_at'].dt.to_period('W')
            weekly_effectiveness = signals_df.groupby('week').agg({
                'realized_return': ['mean', 'count', 'std'],
                'outcome': lambda x: (x == 'success').mean()
            }).reset_index()
            
            # Flatten column names
            weekly_effectiveness.columns = ['week', 'avg_return', 'signal_count', 'return_std', 'success_rate']
            
            # Only include weeks with sufficient signals
            weekly_effectiveness = weekly_effectiveness[weekly_effectiveness['signal_count'] >= 3]
            
            if len(weekly_effectiveness) < 4:
                logger.warning(f"Insufficient weekly data for decay analysis: {len(weekly_effectiveness)}")
                return None
            
            # Time series analysis
            weeks_from_start = np.arange(len(weekly_effectiveness))
            effectiveness_values = weekly_effectiveness['success_rate'].values
            
            # Fit exponential decay model: effectiveness = initial * exp(-decay_rate * time)
            try:
                # Log transformation for linear regression
                log_effectiveness = np.log(np.maximum(effectiveness_values, 0.01))  # Avoid log(0)
                
                # Linear regression on log values
                slope, intercept, r_value, p_value, std_err = stats.linregress(weeks_from_start, log_effectiveness)
                
                # Convert back to exponential parameters
                initial_effectiveness = np.exp(intercept)
                decay_rate = -slope  # Negative slope means decay
                
                # Calculate half-life
                if decay_rate > 0:
                    decay_half_life = np.log(2) / decay_rate
                else:
                    decay_half_life = float('inf')  # No decay
                
                # Statistical significance
                statistical_significance = p_value < 0.05
                
                # Create effectiveness timeline
                effectiveness_timeline = {}
                for i, row in weekly_effectiveness.iterrows():
                    week_str = str(row['week'])
                    effectiveness_timeline[week_str] = float(row['success_rate'])
                
                return SignalDecayAnalysis(
                    signal_type=signal_type,
                    decay_half_life=float(decay_half_life),
                    initial_effectiveness=float(initial_effectiveness),
                    decay_rate=float(decay_rate),
                    effectiveness_timeline=effectiveness_timeline,
                    statistical_significance=statistical_significance
                )
                
            except (ValueError, np.linalg.LinAlgError) as e:
                logger.warning(f"Could not fit decay model for {signal_type}: {e}")
                return None
            
        except Exception as e:
            logger.error(f"Error analyzing signal decay for {signal_type}: {e}")
            return None
    
    def get_signal_ranking(self, days_lookback: int = 30) -> List[Dict[str, Any]]:
        """Get ranking of all signal types by performance"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_lookback)
            
            with sqlite3.connect(self.db_path) as conn:
                # Get all signal types with sufficient data
                signal_types = pd.read_sql_query('''
                    SELECT signal_type, COUNT(*) as signal_count
                    FROM signal_performance 
                    WHERE generated_at >= ? AND exit_price IS NOT NULL
                    GROUP BY signal_type
                    HAVING signal_count >= 5
                    ORDER BY signal_count DESC
                ''', conn, params=(cutoff_date,))
            
            if signal_types.empty:
                return []
            
            rankings = []
            for _, row in signal_types.iterrows():
                signal_type = row['signal_type']
                metrics = self.calculate_signal_metrics(signal_type, days_lookback)
                
                if metrics:
                    # Calculate composite score
                    composite_score = (
                        metrics.hit_rate * 0.3 +
                        min(metrics.sharpe_ratio, 3) / 3 * 0.3 +  # Cap Sharpe at 3
                        metrics.roi / 100 * 0.2 +  # Convert ROI to 0-1 scale
                        (1 + metrics.max_drawdown) * 0.2  # Drawdown penalty
                    )
                    
                    rankings.append({
                        'signal_type': signal_type,
                        'composite_score': float(composite_score),
                        'hit_rate': metrics.hit_rate,
                        'total_signals': metrics.total_signals,
                        'avg_return': metrics.average_return,
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'max_drawdown': metrics.max_drawdown,
                        'roi': metrics.roi
                    })
            
            # Sort by composite score
            rankings.sort(key=lambda x: x['composite_score'], reverse=True)
            
            return rankings
            
        except Exception as e:
            logger.error(f"Error getting signal ranking: {e}")
            return []
    
    def analyze_signal_correlation(self, days_lookback: int = 60) -> Dict[str, Dict[str, float]]:
        """Analyze correlation between different signal types"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_lookback)
            
            with sqlite3.connect(self.db_path) as conn:
                signals_df = pd.read_sql_query('''
                    SELECT signal_type, symbol, generated_at, realized_return
                    FROM signal_performance 
                    WHERE generated_at >= ? AND exit_price IS NOT NULL
                    ORDER BY generated_at
                ''', conn, params=(cutoff_date,))
            
            if len(signals_df) < 50:
                logger.warning("Insufficient data for correlation analysis")
                return {}
            
            # Create pivot table: dates x signal_types with returns
            signals_df['date'] = pd.to_datetime(signals_df['generated_at']).dt.date
            
            # Aggregate by signal type and date
            daily_returns = signals_df.groupby(['date', 'signal_type'])['realized_return'].mean().unstack(fill_value=0)
            
            if daily_returns.shape[1] < 2:
                logger.warning("Need at least 2 signal types for correlation analysis")
                return {}
            
            # Calculate correlation matrix
            correlation_matrix = daily_returns.corr()
            
            # Convert to nested dictionary
            correlation_dict = {}
            for signal1 in correlation_matrix.index:
                correlation_dict[signal1] = {}
                for signal2 in correlation_matrix.columns:
                    if signal1 != signal2:
                        correlation_dict[signal1][signal2] = float(correlation_matrix.loc[signal1, signal2])
            
            return correlation_dict
            
        except Exception as e:
            logger.error(f"Error analyzing signal correlation: {e}")
            return {}
    
    def get_effectiveness_trend(self, signal_type: str, days_lookback: int = 90) -> Dict[str, List]:
        """Get effectiveness trend over time for visualization"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_lookback)
            
            with sqlite3.connect(self.db_path) as conn:
                signals_df = pd.read_sql_query('''
                    SELECT generated_at, realized_return, outcome
                    FROM signal_performance 
                    WHERE signal_type = ? AND generated_at >= ?
                    AND exit_price IS NOT NULL
                    ORDER BY generated_at
                ''', conn, params=(signal_type, cutoff_date))
            
            if signals_df.empty:
                return {'dates': [], 'hit_rates': [], 'avg_returns': [], 'signal_counts': []}
            
            # Convert dates
            signals_df['generated_at'] = pd.to_datetime(signals_df['generated_at'])
            signals_df['date'] = signals_df['generated_at'].dt.date
            
            # Calculate rolling metrics
            daily_stats = signals_df.groupby('date').agg({
                'realized_return': ['mean', 'count'],
                'outcome': lambda x: (x == 'success').mean()
            }).reset_index()
            
            # Flatten column names
            daily_stats.columns = ['date', 'avg_return', 'signal_count', 'hit_rate']
            
            # Sort by date
            daily_stats = daily_stats.sort_values('date')
            
            return {
                'dates': [str(d) for d in daily_stats['date'].tolist()],
                'hit_rates': daily_stats['hit_rate'].tolist(),
                'avg_returns': daily_stats['avg_return'].tolist(),
                'signal_counts': daily_stats['signal_count'].tolist()
            }
            
        except Exception as e:
            logger.error(f"Error getting effectiveness trend for {signal_type}: {e}")
            return {'dates': [], 'hit_rates': [], 'avg_returns': [], 'signal_counts': []}
    
    def store_effectiveness_metrics(self, signal_type: str, metrics: SignalMetrics):
        """Store effectiveness metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO signal_effectiveness 
                    (signal_type, measurement_date, total_signals, successful_signals,
                     hit_rate, avg_return, sharpe_ratio, max_drawdown, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_type, datetime.now().date(), metrics.total_signals,
                    int(metrics.hit_rate * metrics.total_signals), metrics.hit_rate,
                    metrics.average_return, metrics.sharpe_ratio, metrics.max_drawdown,
                    json.dumps({
                        'precision': metrics.precision,
                        'recall': metrics.recall,
                        'f1_score': metrics.f1_score,
                        'roi': metrics.roi
                    })
                ))
                
            logger.debug(f"Stored effectiveness metrics for {signal_type}")
            
        except Exception as e:
            logger.error(f"Error storing effectiveness metrics: {e}")
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all signal performance"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Overall statistics
                overall_stats = pd.read_sql_query('''
                    SELECT 
                        COUNT(*) as total_signals,
                        COUNT(CASE WHEN exit_price IS NOT NULL THEN 1 END) as completed_signals,
                        AVG(CASE WHEN exit_price IS NOT NULL THEN realized_return END) as avg_return,
                        COUNT(CASE WHEN outcome = 'success' THEN 1 END) as successful_signals
                    FROM signal_performance
                    WHERE generated_at >= ?
                ''', conn, params=(datetime.now() - timedelta(days=30),))
                
                # By signal type
                by_signal_type = pd.read_sql_query('''
                    SELECT 
                        signal_type,
                        COUNT(*) as total_signals,
                        COUNT(CASE WHEN exit_price IS NOT NULL THEN 1 END) as completed_signals,
                        AVG(CASE WHEN exit_price IS NOT NULL THEN realized_return END) as avg_return,
                        COUNT(CASE WHEN outcome = 'success' THEN 1 END) as successful_signals
                    FROM signal_performance
                    WHERE generated_at >= ?
                    GROUP BY signal_type
                    ORDER BY successful_signals DESC
                ''', conn, params=(datetime.now() - timedelta(days=30),))
            
            stats = overall_stats.iloc[0] if not overall_stats.empty else {}
            
            return {
                'overall': {
                    'total_signals': int(stats.get('total_signals', 0)),
                    'completed_signals': int(stats.get('completed_signals', 0)),
                    'completion_rate': float(stats.get('completed_signals', 0) / max(stats.get('total_signals', 1), 1)),
                    'avg_return': float(stats.get('avg_return', 0) or 0),
                    'success_rate': float(stats.get('successful_signals', 0) / max(stats.get('completed_signals', 1), 1))
                },
                'by_signal_type': by_signal_type.to_dict('records') if not by_signal_type.empty else [],
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting signal summary: {e}")
            return {'error': str(e)}