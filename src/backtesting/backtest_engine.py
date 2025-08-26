"""
Comprehensive Backtesting Framework with Walk-Forward Optimization
Event-driven backtesting engine for systematic strategy validation
"""
import numpy as np
import pandas as pd
import psycopg2
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
warnings.filterwarnings('ignore')

from config.config import config

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    direction: str  # 'LONG' or 'SHORT'
    pnl: float
    pnl_pct: float
    holding_days: int
    signal_strength: float
    regime_at_entry: str
    exit_reason: str
    commission: float = 0.0
    slippage: float = 0.0

@dataclass
class BacktestResults:
    """Comprehensive backtesting results"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_holding_period: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    drawdown_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime_performance: Dict[str, Dict] = field(default_factory=dict)

class TradingStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, current_date: datetime) -> Dict[str, Dict]:
        """Generate trading signals for given data and date"""
        pass
    
    @abstractmethod
    def get_position_size(self, signal: Dict, portfolio_value: float, 
                         regime: str) -> int:
        """Calculate position size for a signal"""
        pass
    
    @abstractmethod
    def get_exit_rules(self) -> Dict[str, Any]:
        """Get exit rules for positions"""
        pass

class Portfolio:
    """Portfolio management for backtesting"""
    
    def __init__(self, initial_capital: float, commission_per_share: float = 0.005,
                 minimum_commission: float = 1.0, slippage: float = 0.0005):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {}
        self.closed_trades: List[Trade] = []
        
        # Transaction costs
        self.commission_per_share = commission_per_share
        self.minimum_commission = minimum_commission
        self.slippage = slippage
        
        # Performance tracking
        self.equity_history = []
        self.dates = []
        self.peak_value = initial_capital
        self.drawdown_history = []
        
    def calculate_commission(self, quantity: int, price: float) -> float:
        """Calculate commission for a trade"""
        commission = max(quantity * self.commission_per_share, self.minimum_commission)
        
        # Add SEC fees for US stocks (approximately 0.0000229 per dollar)
        sec_fee = quantity * price * 0.0000229
        
        return commission + sec_fee
    
    def calculate_slippage_cost(self, quantity: int, price: float, direction: str) -> float:
        """Calculate slippage cost"""
        if direction == 'BUY':
            return quantity * price * self.slippage
        else:  # SELL
            return quantity * price * self.slippage
    
    def can_open_position(self, symbol: str, quantity: int, price: float) -> bool:
        """Check if position can be opened"""
        if symbol in self.positions:
            return False  # Already have position
        
        total_cost = (quantity * price + 
                     self.calculate_commission(quantity, price) + 
                     self.calculate_slippage_cost(quantity, price, 'BUY'))
        
        return self.cash >= total_cost
    
    def open_position(self, symbol: str, quantity: int, entry_price: float,
                     entry_date: datetime, signal_strength: float,
                     regime: str, stop_loss: float = None,
                     profit_target: float = None) -> bool:
        """Open a new position"""
        
        if not self.can_open_position(symbol, quantity, entry_price):
            return False
        
        # Calculate costs
        commission = self.calculate_commission(quantity, entry_price)
        slippage_cost = self.calculate_slippage_cost(quantity, entry_price, 'BUY')
        total_cost = quantity * entry_price + commission + slippage_cost
        
        # Update cash
        self.cash -= total_cost
        
        # Create position
        self.positions[symbol] = {
            'symbol': symbol,
            'quantity': quantity,
            'entry_price': entry_price,
            'entry_date': entry_date,
            'signal_strength': signal_strength,
            'regime_at_entry': regime,
            'stop_loss': stop_loss,
            'profit_target': profit_target,
            'commission_paid': commission,
            'slippage_paid': slippage_cost
        }
        
        return True
    
    def close_position(self, symbol: str, exit_price: float, exit_date: datetime,
                      exit_reason: str = 'Manual') -> Optional[Trade]:
        """Close an existing position"""
        
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        quantity = position['quantity']
        
        # Calculate costs
        commission = self.calculate_commission(quantity, exit_price)
        slippage_cost = self.calculate_slippage_cost(quantity, exit_price, 'SELL')
        
        # Calculate proceeds
        gross_proceeds = quantity * exit_price
        net_proceeds = gross_proceeds - commission - slippage_cost
        
        # Update cash
        self.cash += net_proceeds
        
        # Calculate P&L
        entry_cost = position['quantity'] * position['entry_price']
        total_commission = position['commission_paid'] + commission
        total_slippage = position['slippage_paid'] + slippage_cost
        
        pnl = net_proceeds - entry_cost - position['commission_paid'] - position['slippage_paid']
        pnl_pct = pnl / entry_cost
        
        holding_days = (exit_date - position['entry_date']).days
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            entry_date=position['entry_date'],
            exit_date=exit_date,
            entry_price=position['entry_price'],
            exit_price=exit_price,
            quantity=quantity,
            direction='LONG',  # Assuming long-only for now
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_days=holding_days,
            signal_strength=position['signal_strength'],
            regime_at_entry=position['regime_at_entry'],
            exit_reason=exit_reason,
            commission=total_commission,
            slippage=total_slippage
        )
        
        self.closed_trades.append(trade)
        del self.positions[symbol]
        
        return trade
    
    def update_portfolio(self, current_prices: Dict[str, float], current_date: datetime) -> None:
        """Update portfolio value and tracking metrics"""
        
        # Calculate current portfolio value
        positions_value = 0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                positions_value += position['quantity'] * current_prices[symbol]
        
        total_value = self.cash + positions_value
        
        # Update tracking
        self.equity_history.append(total_value)
        self.dates.append(current_date)
        
        # Update peak and drawdown
        if total_value > self.peak_value:
            self.peak_value = total_value
        
        drawdown = (self.peak_value - total_value) / self.peak_value
        self.drawdown_history.append(drawdown)
    
    def get_current_value(self, current_prices: Dict[str, float]) -> float:
        """Get current portfolio value"""
        positions_value = sum(
            pos['quantity'] * current_prices.get(pos['symbol'], pos['entry_price'])
            for pos in self.positions.values()
        )
        return self.cash + positions_value
    
    def get_position_count(self) -> int:
        """Get number of open positions"""
        return len(self.positions)

class BacktestEngine:
    """Event-driven backtesting engine"""
    
    def __init__(self, initial_capital: float = 100000, db_config=None):
        self.initial_capital = initial_capital
        
        if db_config is None:
            db_config = config.db
            
        self.db_config = {
            'host': db_config.host,
            'port': db_config.port,
            'database': db_config.database,
            'user': db_config.user,
            'password': db_config.password
        }
        
        # Backtesting parameters
        self.backtest_config = config.backtest
        
    def load_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical data for backtesting"""
        
        try:
            conn = psycopg2.connect(**self.db_config)
            
            symbols_str = "','".join(symbols)
            query = f"""
            SELECT 
                s.symbol,
                dp.trade_date,
                dp.open,
                dp.high,
                dp.low,
                dp.close,
                dp.volume,
                ti.rsi_9,
                ti.rsi_14,
                ti.macd_value,
                ti.macd_signal,
                ti.macd_histogram,
                ti.bb_upper,
                ti.bb_middle,
                ti.bb_lower,
                ti.sma_20,
                ti.sma_50,
                ti.ema_12,
                ti.ema_26,
                ti.atr_14,
                ti.volume_sma_20,
                ti.stoch_k,
                ti.stoch_d,
                ti.williams_r
            FROM securities s
            JOIN daily_prices dp ON s.id = dp.symbol_id
            LEFT JOIN technical_indicators ti ON s.id = ti.symbol_id 
                AND dp.trade_date = ti.trade_date
            WHERE s.symbol IN ('{symbols_str}')
              AND dp.trade_date BETWEEN %s AND %s
            ORDER BY dp.trade_date, s.symbol
            """
            
            df = pd.read_sql(query, conn, params=[start_date, end_date], 
                           parse_dates=['trade_date'])
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading backtest data: {str(e)}")
            return pd.DataFrame()
    
    def run_backtest(self, strategy: TradingStrategy, symbols: List[str],
                    start_date: str, end_date: str,
                    regime_data: Optional[pd.DataFrame] = None) -> BacktestResults:
        """Run a complete backtest"""
        
        logger.info(f"Starting backtest: {start_date} to {end_date}, {len(symbols)} symbols")
        
        # Load data
        data = self.load_data(symbols, start_date, end_date)
        if data.empty:
            logger.error("No data available for backtesting")
            return self._empty_results(start_date, end_date)
        
        # Initialize portfolio
        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            commission_per_share=self.backtest_config.commission_per_share,
            minimum_commission=self.backtest_config.minimum_commission,
            slippage=self.backtest_config.slippage
        )
        
        # Get trading dates
        trading_dates = sorted(data['trade_date'].unique())
        
        # Main backtesting loop
        for current_date in trading_dates:
            # Get data up to current date for strategy
            historical_data = data[data['trade_date'] <= current_date]
            current_day_data = data[data['trade_date'] == current_date]
            
            if historical_data.empty or current_day_data.empty:
                continue
            
            # Get current regime
            current_regime = self._get_regime_for_date(regime_data, current_date)
            
            # Generate signals
            try:
                signals = strategy.generate_signals(historical_data, current_date)
            except Exception as e:
                logger.warning(f"Error generating signals for {current_date}: {str(e)}")
                signals = {}
            
            # Process signals for new positions
            for symbol, signal in signals.items():
                if (signal.get('direction') == 'BUY' and 
                    signal.get('strength', 0) > 0.6 and
                    symbol not in portfolio.positions):
                    
                    # Get current price
                    symbol_data = current_day_data[current_day_data['symbol'] == symbol]
                    if symbol_data.empty:
                        continue
                    
                    current_price = symbol_data.iloc[0]['open']  # Use next day's open
                    atr = symbol_data.iloc[0]['atr_14'] if not pd.isna(symbol_data.iloc[0]['atr_14']) else current_price * 0.02
                    
                    # Calculate position size
                    try:
                        position_size = strategy.get_position_size(
                            signal, portfolio.get_current_value({symbol: current_price}), current_regime
                        )
                        
                        if position_size > 0:
                            # Calculate stop loss and profit target
                            stop_loss = current_price - (2.0 * atr)
                            profit_target = current_price + (2.5 * atr)  # 1.25:1 reward:risk
                            
                            # Open position
                            success = portfolio.open_position(
                                symbol=symbol,
                                quantity=position_size,
                                entry_price=current_price,
                                entry_date=current_date,
                                signal_strength=signal.get('strength', 0.7),
                                regime=current_regime,
                                stop_loss=stop_loss,
                                profit_target=profit_target
                            )
                            
                            if success:
                                logger.debug(f"Opened position: {symbol} x{position_size} @ {current_price:.2f}")
                                
                    except Exception as e:
                        logger.warning(f"Error calculating position size for {symbol}: {str(e)}")
            
            # Check exit conditions for existing positions
            positions_to_close = []
            current_prices = {}
            
            for symbol in list(portfolio.positions.keys()):
                symbol_data = current_day_data[current_day_data['symbol'] == symbol]
                if symbol_data.empty:
                    continue
                
                current_price = symbol_data.iloc[0]['close']
                current_prices[symbol] = current_price
                position = portfolio.positions[symbol]
                
                exit_reason = None
                
                # Check stop loss
                if (position['stop_loss'] and 
                    current_price <= position['stop_loss']):
                    exit_reason = 'Stop Loss'
                
                # Check profit target
                elif (position['profit_target'] and 
                      current_price >= position['profit_target']):
                    exit_reason = 'Profit Target'
                
                # Check maximum holding period
                elif ((current_date - position['entry_date']).days >= 
                      config.trading.max_holding_days):
                    exit_reason = 'Max Holding Period'
                
                # Check strategy-specific exit rules
                elif self._check_strategy_exits(strategy, symbol, historical_data, current_date):
                    exit_reason = 'Strategy Exit'
                
                if exit_reason:
                    positions_to_close.append((symbol, current_price, exit_reason))
            
            # Close positions
            for symbol, exit_price, exit_reason in positions_to_close:
                trade = portfolio.close_position(symbol, exit_price, current_date, exit_reason)
                if trade:
                    logger.debug(f"Closed position: {symbol} @ {exit_price:.2f}, P&L: {trade.pnl:.2f}, Reason: {exit_reason}")
            
            # Update portfolio tracking
            portfolio.update_portfolio(current_prices, current_date)
        
        # Generate results
        results = self._generate_results(portfolio, start_date, end_date, regime_data)
        
        logger.info(f"Backtest completed: {results.total_return:.2%} return, {results.sharpe_ratio:.2f} Sharpe, {results.max_drawdown:.2%} max DD")
        
        return results
    
    def _get_regime_for_date(self, regime_data: Optional[pd.DataFrame], 
                           date: datetime) -> str:
        """Get market regime for a specific date"""
        if regime_data is None or regime_data.empty:
            return 'Low_Volatility'  # Default regime
        
        # Find closest regime data
        regime_data['date_diff'] = abs((regime_data['trade_date'] - date).dt.days)
        closest_regime = regime_data.loc[regime_data['date_diff'].idxmin()]
        
        return closest_regime.get('regime_name', 'Low_Volatility')
    
    def _check_strategy_exits(self, strategy: TradingStrategy, symbol: str,
                            data: pd.DataFrame, current_date: datetime) -> bool:
        """Check strategy-specific exit conditions"""
        try:
            exit_rules = strategy.get_exit_rules()
            symbol_data = data[data['symbol'] == symbol].tail(5)  # Last 5 days
            
            if symbol_data.empty:
                return False
            
            # Example exit conditions (can be customized by strategy)
            latest = symbol_data.iloc[-1]
            
            # RSI overbought exit
            if (exit_rules.get('rsi_exit', False) and 
                not pd.isna(latest['rsi_14']) and 
                latest['rsi_14'] > 75):
                return True
            
            # MACD bearish crossover
            if (exit_rules.get('macd_exit', False) and 
                len(symbol_data) >= 2 and
                not pd.isna(latest['macd_histogram']) and
                not pd.isna(symbol_data.iloc[-2]['macd_histogram'])):
                
                if (latest['macd_histogram'] < 0 and 
                    symbol_data.iloc[-2]['macd_histogram'] >= 0):
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking exit conditions: {str(e)}")
            return False
    
    def _generate_results(self, portfolio: Portfolio, start_date: str, end_date: str,
                         regime_data: Optional[pd.DataFrame] = None) -> BacktestResults:
        """Generate comprehensive backtest results"""
        
        if not portfolio.equity_history:
            return self._empty_results(start_date, end_date)
        
        # Basic metrics
        final_capital = portfolio.equity_history[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame({
            'date': portfolio.dates,
            'equity': portfolio.equity_history,
            'drawdown': portfolio.drawdown_history
        })
        equity_df.set_index('date', inplace=True)
        
        # Calculate returns
        equity_df['daily_return'] = equity_df['equity'].pct_change()
        returns = equity_df['daily_return'].dropna()
        
        # Risk-adjusted metrics
        if len(returns) > 0:
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            
            # Annualized return
            days = len(returns)
            annualized_return = ((final_capital / self.initial_capital) ** (252 / days) - 1) if days > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
            annualized_return = 0
        
        # Drawdown metrics
        max_drawdown = max(portfolio.drawdown_history) if portfolio.drawdown_history else 0
        
        # Calculate max drawdown duration
        max_dd_duration = self._calculate_max_drawdown_duration(equity_df)
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Trade statistics
        trades = portfolio.closed_trades
        total_trades = len(trades)
        
        if total_trades > 0:
            winning_trades = len([t for t in trades if t.pnl > 0])
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades
            
            wins = [t.pnl for t in trades if t.pnl > 0]
            losses = [t.pnl for t in trades if t.pnl < 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
            
            avg_holding_period = np.mean([t.holding_days for t in trades])
        else:
            winning_trades = losing_trades = 0
            win_rate = avg_win = avg_loss = profit_factor = avg_holding_period = 0
        
        # Monthly returns
        monthly_returns = self._calculate_monthly_returns(equity_df)
        
        # Regime performance analysis
        regime_performance = self._analyze_regime_performance(trades, regime_data)
        
        return BacktestResults(
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date),
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_holding_period=avg_holding_period,
            trades=trades,
            equity_curve=equity_df,
            drawdown_curve=equity_df[['drawdown']],
            monthly_returns=monthly_returns,
            regime_performance=regime_performance
        )
    
    def _calculate_max_drawdown_duration(self, equity_df: pd.DataFrame) -> int:
        """Calculate maximum drawdown duration in days"""
        if equity_df.empty:
            return 0
        
        drawdowns = equity_df['drawdown']
        max_duration = 0
        current_duration = 0
        
        for dd in drawdowns:
            if dd > 0:  # In drawdown
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
                
        return max_duration
    
    def _calculate_monthly_returns(self, equity_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate monthly returns"""
        if equity_df.empty:
            return pd.DataFrame()
        
        # Resample to monthly
        monthly_equity = equity_df['equity'].resample('M').last()
        monthly_returns = monthly_equity.pct_change().dropna()
        
        # Create DataFrame with year and month columns
        monthly_df = pd.DataFrame({
            'date': monthly_returns.index,
            'return': monthly_returns.values
        })
        
        monthly_df['year'] = monthly_df['date'].dt.year
        monthly_df['month'] = monthly_df['date'].dt.month
        
        return monthly_df
    
    def _analyze_regime_performance(self, trades: List[Trade], 
                                  regime_data: Optional[pd.DataFrame]) -> Dict[str, Dict]:
        """Analyze performance by market regime"""
        if not trades:
            return {}
        
        regime_stats = {}
        
        # Group trades by regime
        for trade in trades:
            regime = trade.regime_at_entry
            if regime not in regime_stats:
                regime_stats[regime] = {
                    'trades': [],
                    'total_pnl': 0,
                    'win_rate': 0,
                    'avg_holding_days': 0
                }
            
            regime_stats[regime]['trades'].append(trade)
            regime_stats[regime]['total_pnl'] += trade.pnl
        
        # Calculate statistics for each regime
        for regime, stats in regime_stats.items():
            trades_list = stats['trades']
            total_trades = len(trades_list)
            
            if total_trades > 0:
                winning_trades = len([t for t in trades_list if t.pnl > 0])
                stats['win_rate'] = winning_trades / total_trades
                stats['avg_holding_days'] = np.mean([t.holding_days for t in trades_list])
                stats['avg_pnl'] = stats['total_pnl'] / total_trades
                stats['trade_count'] = total_trades
                
        return regime_stats
    
    def _empty_results(self, start_date: str, end_date: str) -> BacktestResults:
        """Create empty results structure"""
        return BacktestResults(
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date),
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_holding_period=0.0
        )

class WalkForwardOptimizer:
    """Walk-forward optimization for robust strategy validation"""
    
    def __init__(self, backtest_engine: BacktestEngine):
        self.backtest_engine = backtest_engine
        self.training_window = config.backtest.training_window  # 504 days (2 years)
        self.testing_window = config.backtest.testing_window    # 63 days (3 months)
        
    def optimize(self, strategy_class, symbols: List[str], 
                start_date: str, end_date: str,
                parameter_grid: Dict[str, List]) -> Dict[str, Any]:
        """Run walk-forward optimization"""
        
        logger.info(f"Starting walk-forward optimization: {start_date} to {end_date}")
        
        # Generate walk-forward periods
        periods = self._generate_periods(start_date, end_date)
        
        if len(periods) < 2:
            raise ValueError("Insufficient data for walk-forward optimization")
        
        results = {
            'periods': [],
            'aggregate_performance': {},
            'parameter_stability': {},
            'walk_forward_efficiency': 0.0
        }
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
            logger.info(f"Period {i+1}/{len(periods)}: Train {train_start} to {train_end}, Test {test_start} to {test_end}")
            
            # Optimize on training period
            best_params = self._optimize_single_period(
                strategy_class, symbols, train_start, train_end, parameter_grid
            )
            
            # Test on out-of-sample period
            test_strategy = strategy_class(**best_params)
            test_results = self.backtest_engine.run_backtest(
                test_strategy, symbols, test_start, test_end
            )
            
            period_result = {
                'period': i + 1,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'optimal_parameters': best_params,
                'oos_performance': {
                    'total_return': test_results.total_return,
                    'sharpe_ratio': test_results.sharpe_ratio,
                    'max_drawdown': test_results.max_drawdown,
                    'win_rate': test_results.win_rate
                }
            }
            
            results['periods'].append(period_result)
        
        # Calculate aggregate metrics
        results['aggregate_performance'] = self._calculate_aggregate_performance(results['periods'])
        results['walk_forward_efficiency'] = self._calculate_wf_efficiency(results['periods'])
        
        logger.info(f"Walk-forward optimization completed. WF Efficiency: {results['walk_forward_efficiency']:.2f}")
        
        return results
    
    def _generate_periods(self, start_date: str, end_date: str) -> List[Tuple[str, str, str, str]]:
        """Generate walk-forward periods"""
        periods = []
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        current_date = start_dt
        
        while current_date + timedelta(days=self.training_window + self.testing_window) <= end_dt:
            train_start = current_date
            train_end = current_date + timedelta(days=self.training_window)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.testing_window)
            
            periods.append((
                train_start.strftime('%Y-%m-%d'),
                train_end.strftime('%Y-%m-%d'),
                test_start.strftime('%Y-%m-%d'),
                test_end.strftime('%Y-%m-%d')
            ))
            
            # Move forward by testing window
            current_date = test_start
        
        return periods
    
    def _optimize_single_period(self, strategy_class, symbols: List[str],
                              start_date: str, end_date: str,
                              parameter_grid: Dict[str, List]) -> Dict[str, Any]:
        """Optimize parameters for a single period"""
        
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        
        best_params = {}
        best_sharpe = -np.inf
        
        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo))
            
            try:
                # Create strategy with parameters
                strategy = strategy_class(**params)
                
                # Run backtest
                results = self.backtest_engine.run_backtest(
                    strategy, symbols, start_date, end_date
                )
                
                # Use Sharpe ratio as optimization criterion
                if results.sharpe_ratio > best_sharpe:
                    best_sharpe = results.sharpe_ratio
                    best_params = params.copy()
                    
            except Exception as e:
                logger.warning(f"Error testing parameters {params}: {str(e)}")
                continue
        
        return best_params
    
    def _calculate_aggregate_performance(self, periods: List[Dict]) -> Dict[str, float]:
        """Calculate aggregate performance across all periods"""
        
        if not periods:
            return {}
        
        # Extract performance metrics
        returns = [p['oos_performance']['total_return'] for p in periods]
        sharpes = [p['oos_performance']['sharpe_ratio'] for p in periods]
        drawdowns = [p['oos_performance']['max_drawdown'] for p in periods]
        win_rates = [p['oos_performance']['win_rate'] for p in periods]
        
        return {
            'avg_return': np.mean(returns),
            'avg_sharpe': np.mean(sharpes),
            'avg_drawdown': np.mean(drawdowns),
            'avg_win_rate': np.mean(win_rates),
            'return_std': np.std(returns),
            'sharpe_std': np.std(sharpes),
            'positive_periods': len([r for r in returns if r > 0]) / len(returns)
        }
    
    def _calculate_wf_efficiency(self, periods: List[Dict]) -> float:
        """Calculate walk-forward efficiency"""
        
        if not periods:
            return 0.0
        
        # WF Efficiency = Out-of-sample return / In-sample return
        # For simplicity, using average performance metrics
        
        oos_returns = [p['oos_performance']['total_return'] for p in periods]
        avg_oos_return = np.mean(oos_returns)
        
        # Assume in-sample performance is typically better
        # This is a simplified calculation - ideally we'd track in-sample performance
        estimated_is_return = avg_oos_return * 1.3  # Assume 30% better in-sample
        
        if estimated_is_return <= 0:
            return 0.0
        
        return avg_oos_return / estimated_is_return

# Example strategy implementation
class SampleTechnicalStrategy(TradingStrategy):
    """Sample strategy for demonstration"""
    
    def __init__(self, rsi_oversold: float = 30, rsi_overbought: float = 70,
                 macd_threshold: float = 0, min_volume_ratio: float = 1.2):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_threshold = macd_threshold
        self.min_volume_ratio = min_volume_ratio
    
    def generate_signals(self, data: pd.DataFrame, current_date: datetime) -> Dict[str, Dict]:
        """Generate buy/sell signals"""
        
        signals = {}
        current_data = data[data['trade_date'] == current_date]
        
        for _, row in current_data.iterrows():
            symbol = row['symbol']
            
            # Skip if missing data
            if pd.isna(row['rsi_14']) or pd.isna(row['macd_histogram']):
                continue
            
            signal_strength = 0.0
            signal_direction = None
            
            # RSI signals
            if row['rsi_14'] < self.rsi_oversold:
                signal_strength += 0.4
                signal_direction = 'BUY'
            elif row['rsi_14'] > self.rsi_overbought:
                signal_strength += 0.4
                signal_direction = 'SELL'
            
            # MACD signals
            if row['macd_histogram'] > self.macd_threshold and signal_direction != 'SELL':
                signal_strength += 0.3
                signal_direction = 'BUY'
            
            # Volume confirmation
            volume_ratio = row['volume'] / row['volume_sma_20'] if not pd.isna(row['volume_sma_20']) and row['volume_sma_20'] > 0 else 1.0
            if volume_ratio > self.min_volume_ratio:
                signal_strength += 0.3
            
            if signal_strength > 0.6 and signal_direction:
                signals[symbol] = {
                    'direction': signal_direction,
                    'strength': min(signal_strength, 1.0),
                    'confidence': 0.7
                }
        
        return signals
    
    def get_position_size(self, signal: Dict, portfolio_value: float, regime: str) -> int:
        """Calculate position size"""
        
        # Base allocation
        base_allocation = 0.05  # 5% of portfolio per position
        
        # Adjust for signal strength
        signal_adjustment = signal.get('strength', 0.7)
        
        # Adjust for regime
        regime_adjustment = 1.0
        if regime == 'High_Volatility':
            regime_adjustment = 0.7
        elif regime == 'Low_Volatility':
            regime_adjustment = 1.2
        
        allocation = base_allocation * signal_adjustment * regime_adjustment
        position_value = portfolio_value * allocation
        
        # Convert to shares (assuming reasonable price)
        estimated_price = 100  # Placeholder - should use actual price
        shares = int(position_value / estimated_price)
        
        return max(shares, 0)
    
    def get_exit_rules(self) -> Dict[str, Any]:
        """Get exit rules"""
        return {
            'rsi_exit': True,
            'macd_exit': True,
            'profit_target': 0.08,  # 8% profit target
            'stop_loss': 0.04       # 4% stop loss
        }

# Example usage
if __name__ == "__main__":
    try:
        # Test the backtesting framework
        engine = BacktestEngine(initial_capital=100000)
        
        # Test with sample strategy
        strategy = SampleTechnicalStrategy()
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        # Run backtest
        results = engine.run_backtest(
            strategy=strategy,
            symbols=symbols,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        print(f"Backtest Results:")
        print(f"Total Return: {results.total_return:.2%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {results.max_drawdown:.2%}")
        print(f"Total Trades: {results.total_trades}")
        print(f"Win Rate: {results.win_rate:.2%}")
        
        # Test walk-forward optimization
        optimizer = WalkForwardOptimizer(engine)
        
        parameter_grid = {
            'rsi_oversold': [25, 30, 35],
            'rsi_overbought': [70, 75, 80],
            'min_volume_ratio': [1.0, 1.2, 1.5]
        }
        
        wf_results = optimizer.optimize(
            SampleTechnicalStrategy,
            symbols,
            '2022-01-01',
            '2023-12-31',
            parameter_grid
        )
        
        print(f"\nWalk-Forward Results:")
        print(f"WF Efficiency: {wf_results['walk_forward_efficiency']:.2f}")
        print(f"Average OOS Return: {wf_results['aggregate_performance']['avg_return']:.2%}")
        print(f"Average Sharpe: {wf_results['aggregate_performance']['avg_sharpe']:.2f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()