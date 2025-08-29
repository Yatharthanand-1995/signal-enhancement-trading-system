"""
Enhanced Backtesting Engine with SQLite Integration and Benchmark Comparison
Integrates with existing signal generation system and stores comprehensive results
"""

import numpy as np
import pandas as pd
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import time
import json
from concurrent.futures import ThreadPoolExecutor

from src.utils.backtesting_schema_sqlite import sqlite_backtesting_schema
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

@dataclass
class EnhancedTrade:
    """Enhanced trade record with comprehensive details"""
    trade_id: Optional[int]
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    trade_direction: str  # 'LONG' or 'SHORT'
    entry_signal_strength: float
    exit_signal_strength: Optional[float]
    signal_components: Dict[str, float]  # Individual indicator contributions
    entry_regime: str
    exit_regime: Optional[str]
    market_conditions: Dict[str, Any]
    sector: str
    market_cap: str
    entry_reason: str
    exit_reason: Optional[str]
    gross_pnl: Optional[float]
    net_pnl: Optional[float]
    pnl_percent: Optional[float]
    commission: float
    slippage: float
    total_costs: float
    holding_days: Optional[int]
    portfolio_weight: float
    position_value: float
    is_open: bool = True

@dataclass
class EnhancedBacktestResults:
    """Comprehensive backtest results with benchmark comparison"""
    config_id: int
    result_id: Optional[int]
    
    # Basic Performance
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    calmar_ratio: float
    sortino_ratio: float
    
    # Risk Metrics
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float
    var_95: float
    cvar_95: float
    
    # Trade Metrics
    total_trades: int
    profitable_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    avg_win_loss_ratio: float
    profit_factor: float
    expectancy: float
    avg_holding_days: float
    turnover_rate: float
    
    # Benchmark Comparison
    benchmark_return: float
    benchmark_volatility: float
    excess_return: float
    information_ratio: float
    tracking_error: float
    beta: float
    alpha: float
    correlation_with_benchmark: float
    upside_capture: float
    downside_capture: float
    
    # Additional Metrics
    up_months: int
    down_months: int
    best_month: float
    worst_month: float
    final_portfolio_value: float
    total_fees: float
    
    # Data structures
    trades: List[EnhancedTrade] = field(default_factory=list)
    daily_portfolio_values: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime_performance: Dict[str, Dict] = field(default_factory=dict)
    signal_performance: Dict[str, Dict] = field(default_factory=dict)

class EnhancedBacktestEngine:
    """Enhanced backtesting engine with SQLite integration and benchmark comparison"""
    
    def __init__(self, db_path: str = "data/historical_stocks.db"):
        self.db_path = db_path
        self.benchmark_symbols = ['SPY', 'QQQ', 'IWM', 'RSP', 'VTI']
        
    def get_connection(self):
        """Get SQLite database connection"""
        return sqlite_backtesting_schema.get_connection()
        
    def create_backtest_config(self, config_data: Dict[str, Any]) -> int:
        """Create a new backtest configuration and return config_id"""
        
        required_fields = {
            'config_name': config_data.get('config_name', 'Default Strategy'),
            'start_date': config_data['start_date'],
            'end_date': config_data['end_date'],
            'initial_capital': config_data.get('initial_capital', 100000),
            'strategy_type': config_data.get('strategy_type', 'signal_based'),
            'position_sizing_method': config_data.get('position_sizing_method', 'equal'),
            'max_position_size': config_data.get('max_position_size', 0.10),
            'max_positions': config_data.get('max_positions', 10),
            'transaction_costs': config_data.get('transaction_costs', 0.001),
            'slippage_rate': config_data.get('slippage_rate', 0.0005),
            'commission_per_trade': config_data.get('commission_per_trade', 1.0),
            'signal_threshold': config_data.get('signal_threshold', 0.5),
            'parameters': json.dumps(config_data.get('parameters', {}))
        }
        
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO backtest_configs 
                (config_name, start_date, end_date, initial_capital, strategy_type, 
                 position_sizing_method, max_position_size, max_positions, 
                 transaction_costs, slippage_rate, commission_per_trade, 
                 signal_threshold, parameters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, tuple(required_fields.values()))
            
            conn.commit()
            config_id = cursor.lastrowid
            
        logger.info(f"Created backtest config {config_id}: {config_data['config_name']}")
        return config_id
    
    def load_historical_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical stock data from SQLite database and calculate technical indicators"""
        
        logger.info(f"Loading historical data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        with self.get_connection() as conn:
            # Create placeholders for symbols
            symbols_placeholder = ','.join(['?' for _ in symbols])
            
            # Query basic OHLCV data
            query = f"""
            SELECT 
                symbol,
                date,
                open,
                high, 
                low,
                close,
                volume
            FROM historical_data 
            WHERE symbol IN ({symbols_placeholder})
              AND date BETWEEN ? AND ?
            ORDER BY date, symbol
            """
            
            params = symbols + [start_date, end_date]
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                
                # Calculate technical indicators for each symbol
                logger.info("Calculating technical indicators...")
                df = self._calculate_technical_indicators(df)
                
                logger.info(f"Loaded {len(df)} records with technical indicators")
            else:
                logger.warning("No historical data found for the specified criteria")
                
            return df
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the historical data"""
        
        result_dfs = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy().sort_values('date').reset_index(drop=True)
            
            if len(symbol_df) < 50:  # Need at least 50 days for indicators
                continue
            
            # RSI calculation
            delta = symbol_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            symbol_df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # MACD calculation
            ema_12 = symbol_df['close'].ewm(span=12).mean()
            ema_26 = symbol_df['close'].ewm(span=26).mean()
            symbol_df['macd'] = ema_12 - ema_26
            symbol_df['macd_signal'] = symbol_df['macd'].ewm(span=9).mean()
            symbol_df['macd_histogram'] = symbol_df['macd'] - symbol_df['macd_signal']
            
            # Bollinger Bands
            symbol_df['sma_20'] = symbol_df['close'].rolling(window=20).mean()
            rolling_std = symbol_df['close'].rolling(window=20).std()
            symbol_df['bb_upper'] = symbol_df['sma_20'] + (rolling_std * 2)
            symbol_df['bb_middle'] = symbol_df['sma_20']
            symbol_df['bb_lower'] = symbol_df['sma_20'] - (rolling_std * 2)
            
            # Additional moving averages
            symbol_df['sma_50'] = symbol_df['close'].rolling(window=50).mean()
            symbol_df['ema_12'] = ema_12
            symbol_df['ema_26'] = ema_26
            
            # ATR calculation
            high_low = symbol_df['high'] - symbol_df['low']
            high_close = np.abs(symbol_df['high'] - symbol_df['close'].shift())
            low_close = np.abs(symbol_df['low'] - symbol_df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            symbol_df['atr_14'] = true_range.rolling(window=14).mean()
            
            # Volume indicators
            symbol_df['volume_sma_20'] = symbol_df['volume'].rolling(window=20).mean()
            
            # Stochastic Oscillator
            low_min = symbol_df['low'].rolling(window=14).min()
            high_max = symbol_df['high'].rolling(window=14).max()
            symbol_df['stoch_k'] = 100 * ((symbol_df['close'] - low_min) / (high_max - low_min))
            symbol_df['stoch_d'] = symbol_df['stoch_k'].rolling(window=3).mean()
            
            # Williams %R
            symbol_df['williams_r'] = -100 * ((high_max - symbol_df['close']) / (high_max - low_min))
            
            result_dfs.append(symbol_df)
        
        if result_dfs:
            return pd.concat(result_dfs, ignore_index=True).sort_values(['date', 'symbol']).reset_index(drop=True)
        else:
            return df
    
    def load_benchmark_data(self, benchmark_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load benchmark performance data"""
        
        with self.get_connection() as conn:
            query = """
            SELECT date, adj_close_price, daily_return, cumulative_return
            FROM benchmark_performance 
            WHERE symbol = ? AND date BETWEEN ? AND ?
            ORDER BY date
            """
            
            df = pd.read_sql_query(query, conn, params=[benchmark_symbol, start_date, end_date])
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                
            return df
    
    def reconstruct_historical_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reconstruct historical signals using point-in-time data"""
        
        logger.info("Reconstructing historical signals...")
        
        signals_data = []
        
        # Get unique dates and symbols
        dates = sorted(data['date'].unique())
        symbols = data['symbol'].unique()
        
        for date in dates:
            # Get data up to current date for signal generation
            available_data = data[data['date'] <= date].copy()
            current_day_data = data[data['date'] == date].copy()
            
            if available_data.empty or current_day_data.empty:
                continue
                
            # Generate signals for each symbol on this date
            for symbol in symbols:
                symbol_data = current_day_data[current_day_data['symbol'] == symbol]
                if symbol_data.empty:
                    continue
                    
                row = symbol_data.iloc[0]
                
                # Skip if missing essential data
                if pd.isna(row['rsi_14']) or pd.isna(row['macd_histogram']):
                    continue
                
                # Generate signal using multiple indicators
                signal_strength, signal_components = self._calculate_signal_strength(row)
                
                if signal_strength > 0.3:  # Lowered threshold for more signals
                    signals_data.append({
                        'date': date,
                        'symbol': symbol,
                        'signal_strength': signal_strength,
                        'signal_components': signal_components,
                        'price': row['close'],
                        'volume': row['volume'],
                        'atr_14': row.get('atr_14', 0),
                        'regime': self._classify_market_regime(date, row)
                    })
        
        signals_df = pd.DataFrame(signals_data)
        logger.info(f"Generated {len(signals_df)} historical signals")
        
        return signals_df
    
    def _calculate_signal_strength(self, row: pd.Series) -> Tuple[float, Dict[str, float]]:
        """🚀 ENHANCED: Calculate signal strength using integrated ML system"""
        
        try:
            # Convert row to DataFrame for enhanced signal processing
            symbol_data = pd.DataFrame([row])
            
            # Try to use enhanced signal integration
            from src.strategy.enhanced_signal_integration import get_enhanced_signal
            
            enhanced_signal = get_enhanced_signal('BACKTEST', symbol_data)
            
            if enhanced_signal and enhanced_signal.confidence > 0.5:
                signal_strength = enhanced_signal.strength
                
                # Extract component contributions from enhanced signal
                components = {
                    'Technical': enhanced_signal.technical_contribution,
                    'Volume': enhanced_signal.volume_contribution, 
                    'Regime': enhanced_signal.regime_contribution,
                    'Momentum': enhanced_signal.momentum_contribution,
                    'ML_Ensemble': enhanced_signal.ml_contribution  # 🚀 NEW: ML contribution
                }
                
                logger.debug(f"Enhanced signal used: strength={signal_strength:.3f}, ML_contrib={enhanced_signal.ml_contribution:.3f}")
                return min(signal_strength, 1.0), components
        
        except Exception as e:
            logger.warning(f"Enhanced signal generation failed, using improved fallback: {e}")
        
        # 🚀 IMPROVED FALLBACK: Better basic calculation (removed hardcoded values)
        signal_strength = 0.0
        components = {}
        
        # RSI component (20% weight)
        rsi_score = 0.0
        if not pd.isna(row.get('rsi_14', np.nan)):
            rsi = row['rsi_14']
            if rsi < 30:
                rsi_score = (30 - rsi) / 30 * 0.8
            elif rsi < 40:
                rsi_score = (40 - rsi) / 10 * 0.4
        components['RSI'] = rsi_score * 0.20
        
        # MACD component (20% weight) 
        macd_score = 0.0
        if not pd.isna(row.get('macd_histogram', np.nan)):
            if row['macd_histogram'] > 0:
                macd_score = min(row['macd_histogram'] * 10, 1.0)
        components['MACD'] = macd_score * 0.20
        
        # Volume component (20% weight)
        volume_score = 0.0
        if not pd.isna(row.get('volume_sma_20', np.nan)) and row['volume_sma_20'] > 0:
            volume_ratio = row['volume'] / row['volume_sma_20']
            if volume_ratio > 1.2:
                volume_score = min((volume_ratio - 1.0) / 2.0, 1.0)
        components['Volume'] = volume_score * 0.20
        
        # Bollinger Bands component (20% weight)
        bb_score = 0.0
        if not pd.isna(row.get('bb_lower', np.nan)) and not pd.isna(row.get('bb_middle', np.nan)):
            if row['close'] < row['bb_lower']:
                bb_score = (row['bb_lower'] - row['close']) / (row['bb_middle'] - row['bb_lower'])
                bb_score = min(bb_score, 1.0)
        components['Bollinger'] = bb_score * 0.20
        
        # Momentum component (20% weight) - FIXED from hardcoded
        momentum_score = 0.0
        try:
            # Calculate actual momentum from price data
            if 'open' in row and not pd.isna(row['open']):
                recent_return = (row['close'] / row['open']) - 1
                momentum_score = 0.5 + np.tanh(recent_return * 20) * 0.5  # Normalize to 0-1
            else:
                momentum_score = 0.5  # Neutral if no data
        except:
            momentum_score = 0.5
        components['Momentum'] = momentum_score * 0.20
        
        # Sum all components
        signal_strength = sum(components.values())
        
        logger.debug(f"Fallback signal used: strength={signal_strength:.3f}")
        return min(signal_strength, 1.0), components
    
    def _classify_market_regime(self, date: datetime, row: pd.Series) -> str:
        """Classify market regime for a given date"""
        
        with self.get_connection() as conn:
            # Find the market regime for this date
            regime_query = """
            SELECT regime_name 
            FROM market_regimes 
            WHERE start_date <= ? AND (end_date >= ? OR end_date IS NULL)
            ORDER BY start_date DESC
            LIMIT 1
            """
            
            result = conn.execute(regime_query, [date.strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d')]).fetchone()
            
            if result:
                return result['regime_name']
            else:
                return 'Unknown'
    
    def run_comprehensive_backtest(self, config_data: Dict[str, Any], 
                                 symbols: List[str], 
                                 benchmark_symbol: str = 'SPY') -> EnhancedBacktestResults:
        """Run a comprehensive backtest with benchmark comparison"""
        
        start_time = time.time()
        logger.info(f"Starting comprehensive backtest: {config_data['start_date']} to {config_data['end_date']}")
        
        # Create backtest configuration
        config_id = self.create_backtest_config(config_data)
        
        # Load historical data
        historical_data = self.load_historical_data(symbols, config_data['start_date'], config_data['end_date'])
        if historical_data.empty:
            logger.error("No historical data available")
            return self._create_empty_results(config_id)
        
        # Load benchmark data
        benchmark_data = self.load_benchmark_data(benchmark_symbol, config_data['start_date'], config_data['end_date'])
        
        # Reconstruct historical signals
        signals_data = self.reconstruct_historical_signals(historical_data)
        if signals_data.empty:
            logger.error("No signals generated")
            return self._create_empty_results(config_id)
        
        # Run portfolio simulation
        portfolio_results = self._simulate_portfolio(
            signals_data, historical_data, config_data, symbols
        )
        
        # Calculate comprehensive performance metrics
        results = self._calculate_comprehensive_metrics(
            portfolio_results, benchmark_data, config_data, config_id
        )
        
        # Store results in database
        result_id = self._store_backtest_results(results)
        results.result_id = result_id
        
        execution_time = time.time() - start_time
        logger.info(f"Backtest completed in {execution_time:.2f} seconds")
        logger.info(f"Results: {results.total_return:.2%} return, {results.sharpe_ratio:.2f} Sharpe, vs {benchmark_symbol}: {results.excess_return:.2%}")
        
        return results
    
    def _simulate_portfolio(self, signals_data: pd.DataFrame, 
                          historical_data: pd.DataFrame,
                          config_data: Dict[str, Any], 
                          symbols: List[str]) -> Dict[str, Any]:
        """Simulate portfolio performance based on signals"""
        
        logger.info("Running portfolio simulation...")
        
        initial_capital = config_data.get('initial_capital', 100000)
        max_positions = config_data.get('max_positions', 10)
        max_position_size = config_data.get('max_position_size', 0.10)
        transaction_costs = config_data.get('transaction_costs', 0.001)
        slippage_rate = config_data.get('slippage_rate', 0.0005)
        
        # Initialize portfolio tracking
        cash = initial_capital
        positions = {}  # symbol -> position_info
        closed_trades = []
        daily_portfolio_values = []
        
        # Get trading dates
        trading_dates = sorted(signals_data['date'].unique())
        
        for current_date in trading_dates:
            # Get signals for current date
            current_signals = signals_data[signals_data['date'] == current_date].copy()
            
            # Get current prices
            current_prices = {}
            for symbol in symbols:
                symbol_data = historical_data[
                    (historical_data['symbol'] == symbol) & 
                    (historical_data['date'] == current_date)
                ]
                if not symbol_data.empty:
                    current_prices[symbol] = symbol_data.iloc[0]['close']
            
            # Process new signals (entries)
            for _, signal_row in current_signals.iterrows():
                symbol = signal_row['symbol']
                
                # Check if we can open a new position
                if (len(positions) < max_positions and 
                    symbol not in positions and 
                    symbol in current_prices):
                    
                    # Calculate position size
                    current_portfolio_value = cash + sum(
                        pos['quantity'] * current_prices.get(pos['symbol'], pos['entry_price'])
                        for pos in positions.values()
                    )
                    
                    position_value = current_portfolio_value * max_position_size
                    shares = int(position_value / current_prices[symbol])
                    
                    if shares > 0:
                        # Calculate costs
                        trade_value = shares * current_prices[symbol]
                        commission = max(trade_value * transaction_costs, 1.0)
                        slippage_cost = trade_value * slippage_rate
                        total_cost = trade_value + commission + slippage_cost
                        
                        if cash >= total_cost:
                            # Open position
                            cash -= total_cost
                            positions[symbol] = {
                                'symbol': symbol,
                                'quantity': shares,
                                'entry_price': current_prices[symbol],
                                'entry_date': current_date,
                                'signal_strength': signal_row['signal_strength'],
                                'signal_components': signal_row['signal_components'],
                                'regime': signal_row['regime'],
                                'commission_paid': commission,
                                'slippage_paid': slippage_cost
                            }
            
            # Check exit conditions for existing positions
            positions_to_close = []
            for symbol, position in positions.items():
                if symbol in current_prices:
                    current_price = current_prices[symbol]
                    entry_price = position['entry_price']
                    holding_days = (current_date - position['entry_date']).days
                    
                    # Simple exit rules
                    exit_reason = None
                    
                    # Profit target (8%)
                    if current_price >= entry_price * 1.08:
                        exit_reason = 'profit_target'
                    # Stop loss (4%)
                    elif current_price <= entry_price * 0.96:
                        exit_reason = 'stop_loss'
                    # Maximum holding period (15 days)
                    elif holding_days >= 15:
                        exit_reason = 'time_limit'
                    
                    if exit_reason:
                        positions_to_close.append((symbol, current_price, exit_reason))
            
            # Close positions
            for symbol, exit_price, exit_reason in positions_to_close:
                position = positions[symbol]
                shares = position['quantity']
                
                # Calculate proceeds
                gross_proceeds = shares * exit_price
                commission = max(gross_proceeds * transaction_costs, 1.0)
                slippage_cost = gross_proceeds * slippage_rate
                net_proceeds = gross_proceeds - commission - slippage_cost
                
                cash += net_proceeds
                
                # Calculate P&L
                entry_cost = shares * position['entry_price']
                total_commission = position['commission_paid'] + commission
                total_slippage = position['slippage_paid'] + slippage_cost
                
                gross_pnl = gross_proceeds - entry_cost
                net_pnl = net_proceeds - entry_cost - position['commission_paid'] - position['slippage_paid']
                pnl_percent = net_pnl / entry_cost
                holding_days = (current_date - position['entry_date']).days
                
                # Create trade record
                trade = EnhancedTrade(
                    trade_id=None,
                    symbol=symbol,
                    entry_date=position['entry_date'],
                    exit_date=current_date,
                    entry_price=position['entry_price'],
                    exit_price=exit_price,
                    quantity=shares,
                    trade_direction='LONG',
                    entry_signal_strength=position['signal_strength'],
                    exit_signal_strength=None,
                    signal_components=position['signal_components'],
                    entry_regime=position['regime'],
                    exit_regime=None,
                    market_conditions={},
                    sector='Unknown',
                    market_cap='Unknown',
                    entry_reason='signal',
                    exit_reason=exit_reason,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    pnl_percent=pnl_percent,
                    commission=total_commission,
                    slippage=total_slippage,
                    total_costs=total_commission + total_slippage,
                    holding_days=holding_days,
                    portfolio_weight=entry_cost / initial_capital,
                    position_value=entry_cost,
                    is_open=False
                )
                
                closed_trades.append(trade)
                del positions[symbol]
            
            # Calculate portfolio value for this date
            positions_value = sum(
                pos['quantity'] * current_prices.get(pos['symbol'], pos['entry_price'])
                for pos in positions.values()
            )
            total_portfolio_value = cash + positions_value
            
            daily_portfolio_values.append({
                'date': current_date,
                'portfolio_value': total_portfolio_value,
                'cash_balance': cash,
                'invested_amount': positions_value,
                'active_positions': len(positions)
            })
        
        return {
            'initial_capital': initial_capital,
            'final_capital': daily_portfolio_values[-1]['portfolio_value'] if daily_portfolio_values else initial_capital,
            'closed_trades': closed_trades,
            'daily_portfolio_values': pd.DataFrame(daily_portfolio_values)
        }
    
    def _calculate_comprehensive_metrics(self, portfolio_results: Dict[str, Any],
                                       benchmark_data: pd.DataFrame,
                                       config_data: Dict[str, Any],
                                       config_id: int) -> EnhancedBacktestResults:
        """Calculate comprehensive performance metrics"""
        
        logger.info("Calculating performance metrics...")
        
        initial_capital = portfolio_results['initial_capital']
        final_capital = portfolio_results['final_capital']
        trades = portfolio_results['closed_trades']
        daily_values = portfolio_results['daily_portfolio_values']
        
        # Basic performance
        total_return = (final_capital - initial_capital) / initial_capital
        
        # Calculate daily returns
        daily_values['daily_return'] = daily_values['portfolio_value'].pct_change()
        returns = daily_values['daily_return'].dropna()
        
        # Risk-adjusted metrics
        if len(returns) > 0:
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (returns.mean() * 252) / (volatility) if volatility > 0 else 0
            
            # Annualized return
            days = len(returns)
            annualized_return = ((final_capital / initial_capital) ** (252 / days) - 1) if days > 0 else 0
            
            # Downside deviation for Sortino ratio
            negative_returns = returns[returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
            sortino_ratio = (annualized_return) / downside_deviation if downside_deviation > 0 else 0
        else:
            volatility = annualized_return = sharpe_ratio = sortino_ratio = 0
        
        # Drawdown analysis
        daily_values['running_max'] = daily_values['portfolio_value'].expanding().max()
        daily_values['drawdown'] = (daily_values['portfolio_value'] / daily_values['running_max'] - 1)
        
        max_drawdown = abs(daily_values['drawdown'].min())
        max_dd_duration = self._calculate_max_drawdown_duration(daily_values)
        avg_drawdown = abs(daily_values['drawdown'].mean())
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # VaR and CVaR (simplified)
        var_95 = returns.quantile(0.05) if len(returns) > 0 else 0
        cvar_95 = returns[returns <= var_95].mean() if len(returns) > 0 else 0
        
        # Trade statistics
        total_trades = len(trades)
        if total_trades > 0:
            profitable_trades = len([t for t in trades if t.net_pnl > 0])
            losing_trades = total_trades - profitable_trades
            win_rate = profitable_trades / total_trades
            
            wins = [t.net_pnl for t in trades if t.net_pnl > 0]
            losses = [t.net_pnl for t in trades if t.net_pnl < 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
            expectancy = np.mean([t.net_pnl for t in trades])
            avg_holding_days = np.mean([t.holding_days for t in trades])
            
            total_fees = sum([t.total_costs for t in trades])
        else:
            profitable_trades = losing_trades = 0
            win_rate = avg_win = avg_loss = avg_win_loss_ratio = 0
            profit_factor = expectancy = avg_holding_days = total_fees = 0
        
        # Benchmark comparison
        benchmark_metrics = self._calculate_benchmark_comparison(daily_values, benchmark_data)
        
        # Monthly analysis
        monthly_metrics = self._calculate_monthly_metrics(daily_values)
        
        return EnhancedBacktestResults(
            config_id=config_id,
            result_id=None,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            avg_drawdown=avg_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            total_trades=total_trades,
            profitable_trades=profitable_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_win_loss_ratio=avg_win_loss_ratio,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_holding_days=avg_holding_days,
            turnover_rate=0.0,  # TODO: Calculate
            benchmark_return=benchmark_metrics.get('benchmark_return', 0),
            benchmark_volatility=benchmark_metrics.get('benchmark_volatility', 0),
            excess_return=benchmark_metrics.get('excess_return', 0),
            information_ratio=benchmark_metrics.get('information_ratio', 0),
            tracking_error=benchmark_metrics.get('tracking_error', 0),
            beta=benchmark_metrics.get('beta', 0),
            alpha=benchmark_metrics.get('alpha', 0),
            correlation_with_benchmark=benchmark_metrics.get('correlation', 0),
            upside_capture=benchmark_metrics.get('upside_capture', 0),
            downside_capture=benchmark_metrics.get('downside_capture', 0),
            up_months=monthly_metrics.get('up_months', 0),
            down_months=monthly_metrics.get('down_months', 0),
            best_month=monthly_metrics.get('best_month', 0),
            worst_month=monthly_metrics.get('worst_month', 0),
            final_portfolio_value=final_capital,
            total_fees=total_fees,
            trades=trades,
            daily_portfolio_values=daily_values
        )
    
    def _calculate_benchmark_comparison(self, daily_values: pd.DataFrame, 
                                      benchmark_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate benchmark comparison metrics"""
        
        if benchmark_data.empty or daily_values.empty:
            return {}
        
        # Align data by date
        portfolio_returns = daily_values.set_index('date')['daily_return'].dropna()
        benchmark_returns = benchmark_data.set_index('date')['daily_return'].dropna()
        
        # Find common dates
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) == 0:
            return {}
        
        portfolio_aligned = portfolio_returns.loc[common_dates]
        benchmark_aligned = benchmark_returns.loc[common_dates]
        
        # Calculate metrics
        benchmark_return = (1 + benchmark_aligned).prod() - 1
        benchmark_volatility = benchmark_aligned.std() * np.sqrt(252)
        
        portfolio_total_return = (1 + portfolio_aligned).prod() - 1
        excess_return = portfolio_total_return - benchmark_return
        
        # Tracking error and information ratio
        active_returns = portfolio_aligned - benchmark_aligned
        tracking_error = active_returns.std() * np.sqrt(252)
        information_ratio = (excess_return) / tracking_error if tracking_error > 0 else 0
        
        # Beta and alpha
        if len(portfolio_aligned) > 1:
            covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1]
            benchmark_variance = np.var(benchmark_aligned)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            portfolio_mean = portfolio_aligned.mean() * 252
            benchmark_mean = benchmark_aligned.mean() * 252
            alpha = portfolio_mean - beta * benchmark_mean
            
            correlation = np.corrcoef(portfolio_aligned, benchmark_aligned)[0, 1]
        else:
            beta = alpha = correlation = 0
        
        # Capture ratios (simplified)
        upside_benchmark = benchmark_aligned[benchmark_aligned > 0]
        downside_benchmark = benchmark_aligned[benchmark_aligned < 0]
        
        if len(upside_benchmark) > 0:
            upside_portfolio = portfolio_aligned[benchmark_aligned > 0]
            upside_capture = upside_portfolio.mean() / upside_benchmark.mean() if upside_benchmark.mean() != 0 else 0
        else:
            upside_capture = 0
        
        if len(downside_benchmark) > 0:
            downside_portfolio = portfolio_aligned[benchmark_aligned < 0]
            downside_capture = downside_portfolio.mean() / downside_benchmark.mean() if downside_benchmark.mean() != 0 else 0
        else:
            downside_capture = 0
        
        return {
            'benchmark_return': benchmark_return,
            'benchmark_volatility': benchmark_volatility,
            'excess_return': excess_return,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha,
            'correlation': correlation,
            'upside_capture': upside_capture,
            'downside_capture': downside_capture
        }
    
    def _calculate_monthly_metrics(self, daily_values: pd.DataFrame) -> Dict[str, Any]:
        """Calculate monthly performance metrics"""
        
        if daily_values.empty:
            return {}
        
        # Resample to monthly
        monthly_values = daily_values.set_index('date')['portfolio_value'].resample('M').last()
        monthly_returns = monthly_values.pct_change().dropna()
        
        if len(monthly_returns) == 0:
            return {}
        
        up_months = len(monthly_returns[monthly_returns > 0])
        down_months = len(monthly_returns[monthly_returns <= 0])
        best_month = monthly_returns.max()
        worst_month = monthly_returns.min()
        
        return {
            'up_months': up_months,
            'down_months': down_months,
            'best_month': best_month,
            'worst_month': worst_month
        }
    
    def _calculate_max_drawdown_duration(self, daily_values: pd.DataFrame) -> int:
        """Calculate maximum drawdown duration in days"""
        
        if daily_values.empty:
            return 0
        
        drawdowns = daily_values['drawdown']
        max_duration = 0
        current_duration = 0
        
        for dd in drawdowns:
            if dd < 0:  # In drawdown
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
                
        return max_duration
    
    def _store_backtest_results(self, results: EnhancedBacktestResults) -> int:
        """Store comprehensive backtest results in database"""
        
        logger.info("Storing backtest results...")
        
        with self.get_connection() as conn:
            # Store main results
            results_data = (
                results.config_id, results.total_return, results.annualized_return,
                results.volatility, results.sharpe_ratio, results.calmar_ratio,
                results.sortino_ratio, results.max_drawdown, results.max_drawdown_duration,
                results.avg_drawdown, results.win_rate, results.avg_win, results.avg_loss,
                results.avg_win_loss_ratio, results.profit_factor, results.expectancy,
                results.total_trades, results.profitable_trades, results.losing_trades,
                results.avg_holding_days, results.turnover_rate, results.information_ratio,
                results.tracking_error, results.beta, results.alpha, results.var_95,
                results.cvar_95, results.benchmark_return, results.benchmark_volatility,
                results.excess_return, results.correlation_with_benchmark,
                results.upside_capture, results.downside_capture, results.up_months,
                results.down_months, results.best_month, results.worst_month,
                results.final_portfolio_value, results.total_fees
            )
            
            cursor = conn.execute("""
                INSERT INTO backtest_results 
                (config_id, total_return, annualized_return, volatility, sharpe_ratio,
                 calmar_ratio, sortino_ratio, max_drawdown, max_drawdown_duration,
                 avg_drawdown, win_rate, avg_win, avg_loss, avg_win_loss_ratio,
                 profit_factor, expectancy, total_trades, profitable_trades,
                 losing_trades, avg_holding_days, turnover_rate, information_ratio,
                 tracking_error, beta, alpha, var_95, cvar_95, benchmark_return,
                 benchmark_volatility, excess_return, correlation_with_benchmark,
                 upside_capture, downside_capture, up_months, down_months,
                 best_month, worst_month, final_portfolio_value, total_fees)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, results_data)
            
            result_id = cursor.lastrowid
            
            # Store individual trades
            for trade in results.trades:
                trade_data = (
                    result_id, trade.symbol, trade.entry_date.strftime('%Y-%m-%d'),
                    trade.exit_date.strftime('%Y-%m-%d') if trade.exit_date else None,
                    trade.entry_price, trade.exit_price, trade.quantity, trade.trade_direction,
                    trade.entry_signal_strength, trade.exit_signal_strength,
                    json.dumps(trade.signal_components), trade.entry_regime, trade.exit_regime,
                    json.dumps(trade.market_conditions), trade.sector, trade.market_cap,
                    trade.entry_reason, trade.exit_reason, trade.gross_pnl, trade.net_pnl,
                    trade.pnl_percent, trade.commission, trade.slippage, trade.total_costs,
                    trade.holding_days, trade.portfolio_weight, trade.position_value,
                    1 if trade.is_open else 0
                )
                
                conn.execute("""
                    INSERT INTO backtest_trades 
                    (result_id, symbol, entry_date, exit_date, entry_price, exit_price,
                     quantity, trade_direction, entry_signal_strength, exit_signal_strength,
                     signal_components, entry_regime, exit_regime, market_conditions,
                     sector, market_cap, entry_reason, exit_reason, gross_pnl, net_pnl,
                     pnl_percent, commission, slippage, total_costs, holding_days,
                     portfolio_weight, position_value, is_open)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, trade_data)
            
            # Store daily portfolio values
            for _, row in results.daily_portfolio_values.iterrows():
                daily_return = row.get('daily_return', 0) if not pd.isna(row.get('daily_return')) else 0
                cumulative_return = (row['portfolio_value'] / results.daily_portfolio_values.iloc[0]['portfolio_value'] - 1)
                
                portfolio_data = (
                    result_id, row['date'].strftime('%Y-%m-%d'), row['portfolio_value'],
                    row['cash_balance'], row['invested_amount'], daily_return,
                    cumulative_return, row.get('drawdown', 0), row['active_positions']
                )
                
                conn.execute("""
                    INSERT INTO backtest_portfolio_values 
                    (result_id, date, portfolio_value, cash_balance, invested_amount,
                     daily_return, cumulative_return, drawdown, active_positions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, portfolio_data)
            
            conn.commit()
            
        logger.info(f"Stored backtest results with ID: {result_id}")
        return result_id
    
    def _create_empty_results(self, config_id: int) -> EnhancedBacktestResults:
        """Create empty results structure"""
        return EnhancedBacktestResults(
            config_id=config_id,
            result_id=None,
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            avg_drawdown=0.0,
            var_95=0.0,
            cvar_95=0.0,
            total_trades=0,
            profitable_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_win_loss_ratio=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            avg_holding_days=0.0,
            turnover_rate=0.0,
            benchmark_return=0.0,
            benchmark_volatility=0.0,
            excess_return=0.0,
            information_ratio=0.0,
            tracking_error=0.0,
            beta=0.0,
            alpha=0.0,
            correlation_with_benchmark=0.0,
            upside_capture=0.0,
            downside_capture=0.0,
            up_months=0,
            down_months=0,
            best_month=0.0,
            worst_month=0.0,
            final_portfolio_value=0.0,
            total_fees=0.0
        )

# Global enhanced backtest engine instance
enhanced_backtest_engine = EnhancedBacktestEngine()

# Convenience functions
def run_backtest_analysis(config_data: Dict[str, Any], symbols: List[str], benchmark: str = 'SPY'):
    """Run comprehensive backtest analysis"""
    return enhanced_backtest_engine.run_comprehensive_backtest(config_data, symbols, benchmark)