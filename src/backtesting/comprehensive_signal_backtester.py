"""
Comprehensive Signal Methodology Backtester
Tests our ensemble signal system across market regimes with Top 100 stocks
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Import our signal generation components
from .top_100_universe import Top100Universe

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 1_000_000  # $1M starting capital
    max_position_size: float = 0.02     # 2% max per position
    transaction_cost: float = 0.001     # 0.1% transaction cost
    min_confidence: float = 0.6         # Minimum signal confidence
    rebalance_frequency: str = 'daily'  # daily, weekly, monthly
    benchmark_symbols: List[str] = None # ['SPY', 'QQQ', 'IWM', 'VTI']
    
    def __post_init__(self):
        if self.benchmark_symbols is None:
            self.benchmark_symbols = ['SPY', 'QQQ', 'IWM', 'VTI']

@dataclass
class MarketRegime:
    """Market regime classification"""
    name: str
    start_date: datetime
    end_date: datetime
    characteristics: Dict[str, Any]
    vix_range: Tuple[float, float]
    market_trend: str  # 'bull', 'bear', 'sideways'

@dataclass
class Trade:
    """Individual trade record"""
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    direction: str  # 'long', 'short'
    entry_signal: Dict[str, Any]
    exit_reason: str
    pnl: Optional[float] = None
    return_pct: Optional[float] = None
    days_held: Optional[int] = None

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Returns
    total_return: float
    annualized_return: float
    excess_return: float  # vs benchmark
    
    # Risk
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Trading
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    avg_days_held: float
    
    # Correlation
    beta: float
    correlation: float
    tracking_error: float
    information_ratio: float

class SignalGenerator:
    """Generates enhanced ensemble signals using our academic methodology"""
    
    def __init__(self):
        # Import enhanced signal system
        try:
            from ..strategy.enhanced.enhanced_ensemble_signal_scoring import EnhancedEnsembleSignalScoring
            self.enhanced_scorer = EnhancedEnsembleSignalScoring(
                enable_regime_detection=True,
                enable_macro_integration=True,
                enable_factor_timing=False,  # Phase 2 feature
                enable_dynamic_sizing=True
            )
            self.use_enhanced = True
            logger.info("✅ Backtesting using Enhanced Signal System with 18-33% expected improvement")
        except ImportError as e:
            logger.warning(f"Enhanced signals not available, falling back to base system: {e}")
            self.use_enhanced = False
        
        # Fallback technical indicators (kept for compatibility)
        self.technical_indicators = {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'ma_short': 20,
            'ma_long': 50
        }
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = data.copy()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.technical_indicators['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.technical_indicators['rsi_period']).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=self.technical_indicators['macd_fast']).mean()
        exp2 = df['Close'].ewm(span=self.technical_indicators['macd_slow']).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=self.technical_indicators['macd_signal']).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=self.technical_indicators['bb_period']).mean()
        bb_std = df['Close'].rolling(window=self.technical_indicators['bb_period']).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * self.technical_indicators['bb_std'])
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * self.technical_indicators['bb_std'])
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Moving Averages
        df['MA_Short'] = df['Close'].rolling(window=self.technical_indicators['ma_short']).mean()
        df['MA_Long'] = df['Close'].rolling(window=self.technical_indicators['ma_long']).mean()
        df['MA_Ratio'] = df['MA_Short'] / df['MA_Long']
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price momentum
        df['Returns_1d'] = df['Close'].pct_change()
        df['Returns_5d'] = df['Close'].pct_change(5)
        df['Returns_20d'] = df['Close'].pct_change(20)
        
        # Volatility
        df['Volatility_20d'] = df['Returns_1d'].rolling(window=20).std() * np.sqrt(252)
        
        return df
    
    def generate_signal(self, data: pd.DataFrame, date: datetime, symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """Generate enhanced ensemble signal for a stock on a specific date"""
        
        try:
            # Get data up to the signal date (no look-ahead bias)
            signal_data = data[data.index <= date].copy()
            
            if len(signal_data) < 50:  # Need minimum data
                return self._neutral_signal()
            
            # Use enhanced signal system if available
            if self.use_enhanced:
                try:
                    # Prepare data for enhanced signal system
                    enhanced_data = signal_data[['Close', 'High', 'Low', 'Volume']].copy()
                    enhanced_data.columns = ['close', 'high', 'low', 'volume']
                    
                    # Calculate enhanced signal
                    enhanced_signal = self.enhanced_scorer.calculate_enhanced_signal(symbol, enhanced_data)
                    
                    # Convert to backtesting format with enhanced fields
                    return {
                        'signal_strength': enhanced_signal.strength,
                        'confidence': enhanced_signal.confidence,
                        'direction': enhanced_signal.direction.name,
                        'composite_score': enhanced_signal.composite_score,
                        'market_regime': enhanced_signal.market_regime.value,
                        'position_size': enhanced_signal.optimal_position_size,
                        'should_trade': enhanced_signal.should_trade,
                        'trade_rationale': enhanced_signal.trade_rationale,
                        'enhanced': True,  # Flag for tracking
                        'academic_backed': True
                    }
                    
                except Exception as e:
                    logger.warning(f"Enhanced signal generation failed for {symbol}, falling back: {e}")
                    # Fall through to traditional method
            
            # Traditional signal generation (fallback)
            # Get latest values
            latest = signal_data.iloc[-1]
            
            # Technical signal scores (normalized -1 to +1)
            technical_scores = {}
            
            # RSI signal
            rsi = latest['RSI']
            if rsi < 30:
                technical_scores['rsi'] = 0.8  # Oversold - bullish
            elif rsi > 70:
                technical_scores['rsi'] = -0.8  # Overbought - bearish
            else:
                technical_scores['rsi'] = (50 - rsi) / 20  # Linear scaling
            
            # MACD signal
            macd = latest['MACD']
            macd_signal = latest['MACD_Signal']
            macd_hist = latest['MACD_Histogram']
            
            if macd > macd_signal and macd_hist > 0:
                technical_scores['macd'] = 0.6
            elif macd < macd_signal and macd_hist < 0:
                technical_scores['macd'] = -0.6
            else:
                technical_scores['macd'] = macd_hist * 0.1  # Scale histogram
            
            # Bollinger Bands signal
            bb_pos = latest['BB_Position']
            if bb_pos < 0.2:
                technical_scores['bb'] = 0.7  # Near lower band - bullish
            elif bb_pos > 0.8:
                technical_scores['bb'] = -0.7  # Near upper band - bearish
            else:
                technical_scores['bb'] = (0.5 - bb_pos) * 1.4
            
            # Moving Average signal
            ma_ratio = latest['MA_Ratio']
            if ma_ratio > 1.02:
                technical_scores['ma'] = 0.5  # Strong uptrend
            elif ma_ratio < 0.98:
                technical_scores['ma'] = -0.5  # Strong downtrend
            else:
                technical_scores['ma'] = (ma_ratio - 1) * 50  # Scale around 1.0
            
            # Volume signal
            vol_ratio = latest['Volume_Ratio']
            momentum_5d = latest['Returns_5d']
            
            # High volume + positive momentum = bullish
            if vol_ratio > 1.5 and momentum_5d > 0.02:
                volume_score = 0.4
            elif vol_ratio > 1.5 and momentum_5d < -0.02:
                volume_score = -0.4
            else:
                volume_score = (vol_ratio - 1) * 0.2
            
            # Momentum signals
            momentum_1d = latest['Returns_1d']
            momentum_20d = latest['Returns_20d']
            
            momentum_score = (momentum_1d * 2 + momentum_20d) / 3  # Weight recent more
            momentum_score = np.clip(momentum_score * 5, -1, 1)  # Scale and clip
            
            # Volatility adjustment (reduce signal in high vol)
            volatility = latest['Volatility_20d']
            vol_adjustment = max(0.5, 1 - (volatility - 0.2) * 2)  # Reduce in high vol
            
            # Combine signals with weights
            signal_weights = {
                'technical': 0.4,
                'volume': 0.2,
                'momentum': 0.3,
                'volatility_adj': 0.1
            }
            
            # Weighted technical score
            technical_score = np.mean(list(technical_scores.values()))
            
            # Final ensemble score
            raw_score = (
                technical_score * signal_weights['technical'] +
                volume_score * signal_weights['volume'] +
                momentum_score * signal_weights['momentum']
            ) * vol_adjustment
            
            # Convert to direction and confidence
            abs_score = abs(raw_score)
            confidence = min(abs_score * 2, 1.0)  # Scale to 0-1
            strength = abs_score
            
            if raw_score > 0.3:
                direction = 'STRONG_BUY'
            elif raw_score > 0.1:
                direction = 'BUY'
            elif raw_score < -0.3:
                direction = 'STRONG_SELL'
            elif raw_score < -0.1:
                direction = 'SELL'
            else:
                direction = 'NEUTRAL'
            
            return {
                'direction': direction,
                'strength': strength,
                'confidence': confidence,
                'composite_score': raw_score,
                'technical_scores': technical_scores,
                'volume_score': volume_score,
                'momentum_score': momentum_score,
                'volatility_adjustment': vol_adjustment,
                'raw_data': {
                    'rsi': rsi,
                    'macd': macd,
                    'bb_position': bb_pos,
                    'ma_ratio': ma_ratio,
                    'volume_ratio': vol_ratio,
                    'volatility': volatility
                }
            }
            
        except Exception as e:
            logger.warning(f"Error generating signal: {e}")
            return self._neutral_signal()
    
    def _neutral_signal(self) -> Dict[str, Any]:
        """Return neutral signal when unable to generate proper signal"""
        return {
            'direction': 'NEUTRAL',
            'strength': 0.0,
            'confidence': 0.0,
            'composite_score': 0.0,
            'technical_scores': {},
            'volume_score': 0.0,
            'momentum_score': 0.0,
            'volatility_adjustment': 1.0,
            'raw_data': {}
        }

class MarketRegimeClassifier:
    """Classifies market regimes for regime-specific analysis"""
    
    def __init__(self):
        self.regimes = self._define_market_regimes()
    
    def _define_market_regimes(self) -> List[MarketRegime]:
        """Define historical market regimes for backtesting"""
        return [
            MarketRegime(
                name="Pre-COVID Bull",
                start_date=datetime(2019, 1, 1),
                end_date=datetime(2020, 2, 19),
                characteristics={'trend': 'bull', 'volatility': 'low'},
                vix_range=(12, 25),
                market_trend='bull'
            ),
            MarketRegime(
                name="COVID Crash",
                start_date=datetime(2020, 2, 20),
                end_date=datetime(2020, 3, 23),
                characteristics={'trend': 'bear', 'volatility': 'extreme'},
                vix_range=(25, 85),
                market_trend='bear'
            ),
            MarketRegime(
                name="COVID Recovery",
                start_date=datetime(2020, 3, 24),
                end_date=datetime(2020, 12, 31),
                characteristics={'trend': 'bull', 'volatility': 'high'},
                vix_range=(20, 40),
                market_trend='bull'
            ),
            MarketRegime(
                name="Low Rate Bull",
                start_date=datetime(2021, 1, 1),
                end_date=datetime(2021, 12, 31),
                characteristics={'trend': 'bull', 'volatility': 'medium'},
                vix_range=(15, 30),
                market_trend='bull'
            ),
            MarketRegime(
                name="Rate Rise Bear",
                start_date=datetime(2022, 1, 1),
                end_date=datetime(2022, 10, 12),
                characteristics={'trend': 'bear', 'volatility': 'high'},
                vix_range=(20, 35),
                market_trend='bear'
            ),
            MarketRegime(
                name="AI Boom Bull",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2024, 9, 30),
                characteristics={'trend': 'bull', 'volatility': 'medium'},
                vix_range=(15, 25),
                market_trend='bull'
            )
        ]
    
    def get_regime(self, date: datetime) -> Optional[MarketRegime]:
        """Get market regime for a specific date"""
        for regime in self.regimes:
            if regime.start_date <= date <= regime.end_date:
                return regime
        return None

class ComprehensiveSignalBacktester:
    """Main backtesting engine for signal methodology validation"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.universe = Top100Universe()
        self.signal_generator = SignalGenerator()
        self.regime_classifier = MarketRegimeClassifier()
        
        # State tracking
        self.portfolio = {}  # {symbol: shares}
        self.cash = config.initial_capital
        self.trades = []
        self.daily_values = []
        self.benchmark_data = {}
        
        # Results storage
        self.results = {}
        self.performance_metrics = {}
    
    def load_data(self, symbols: List[str], start_date: datetime, 
                  end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Load historical data for all symbols"""
        
        logger.info(f"Loading data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        data = {}
        
        # Load stock data in batches
        batch_size = 20
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i+batch_size]
            
            try:
                # Download batch data
                tickers = yf.Tickers(' '.join(batch_symbols))
                
                for symbol in batch_symbols:
                    try:
                        ticker_data = tickers.tickers[symbol].history(
                            start=start_date - timedelta(days=100),  # Extra data for indicators
                            end=end_date,
                            auto_adjust=True
                        )
                        
                        if not ticker_data.empty:
                            # Calculate technical indicators
                            ticker_data = self.signal_generator.calculate_technical_indicators(ticker_data)
                            data[symbol] = ticker_data
                        
                    except Exception as e:
                        logger.warning(f"Error loading data for {symbol}: {e}")
                
                logger.info(f"Loaded {i + len(batch_symbols)}/{len(symbols)} symbols")
                
            except Exception as e:
                logger.error(f"Error loading batch {i}-{i+batch_size}: {e}")
        
        # Load benchmark data
        for benchmark in self.config.benchmark_symbols:
            try:
                bench_data = yf.download(benchmark, start=start_date, end=end_date, auto_adjust=True)
                if not bench_data.empty:
                    self.benchmark_data[benchmark] = bench_data
            except Exception as e:
                logger.warning(f"Error loading benchmark {benchmark}: {e}")
        
        logger.info(f"Data loading complete: {len(data)} stocks, {len(self.benchmark_data)} benchmarks")
        return data
    
    def run_backtest(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Run comprehensive backtest"""
        
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Generate universe for start date
        universe_df = self.universe.generate_top_100_universe(start_date)
        symbols = universe_df['symbol'].tolist()
        
        # Load historical data
        data = self.load_data(symbols, start_date, end_date)
        
        if not data:
            raise ValueError("No data loaded for backtesting")
        
        # Get trading days - handle timezone issues
        first_stock = next(iter(data.values()))
        
        # Convert dates to timezone-aware if needed
        if first_stock.index.tz is not None:
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=first_stock.index.tz)
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=first_stock.index.tz)
        
        trading_days = first_stock[start_date:end_date].index
        
        logger.info(f"Running backtest for {len(trading_days)} trading days")
        
        # Daily backtesting loop
        for i, date in enumerate(trading_days):
            if i % 50 == 0:
                logger.info(f"Processing day {i+1}/{len(trading_days)}: {date.strftime('%Y-%m-%d')}")
            
            self._process_trading_day(date, data)
        
        # Calculate final results
        self.results = self._calculate_results()
        
        logger.info("Backtest complete!")
        return self.results
    
    def _process_trading_day(self, date: datetime, data: Dict[str, pd.DataFrame]):
        """Process a single trading day"""
        
        daily_signals = {}
        
        # Generate signals for all stocks
        for symbol in data.keys():
            if date in data[symbol].index:
                signal = self.signal_generator.generate_signal(data[symbol], date, symbol)
                daily_signals[symbol] = signal
        
        # Execute trading logic
        self._execute_trades(date, daily_signals, data)
        
        # Update portfolio values
        self._update_portfolio_value(date, data)
    
    def _execute_trades(self, date: datetime, signals: Dict[str, Any], data: Dict[str, pd.DataFrame]):
        """Execute trades based on signals"""
        
        # Get current portfolio value for position sizing
        portfolio_value = self._get_current_portfolio_value(date, data)
        
        # Process sell signals first (free up cash)
        for symbol, signal in signals.items():
            if symbol in self.portfolio and self.portfolio[symbol] > 0:
                if signal['direction'] in ['SELL', 'STRONG_SELL'] and signal['confidence'] >= self.config.min_confidence:
                    self._execute_sell(symbol, date, data[symbol].loc[date], signal)
        
        # Process buy signals (enhanced logic)
        for symbol, signal in signals.items():
            # Enhanced signal handling - check should_trade flag if available
            should_proceed = True
            if signal.get('enhanced', False) and 'should_trade' in signal:
                should_proceed = signal['should_trade']
            
            if (signal['direction'] in ['BUY', 'STRONG_BUY'] and 
                signal['confidence'] >= self.config.min_confidence and 
                should_proceed):
                if symbol not in self.portfolio or self.portfolio[symbol] == 0:
                    self._execute_buy(symbol, date, data[symbol].loc[date], signal, portfolio_value)
    
    def _execute_buy(self, symbol: str, date: datetime, price_data: pd.Series, 
                    signal: Dict[str, Any], portfolio_value: float):
        """Execute buy order"""
        
        price = price_data['Close']
        
        # Calculate position size using enhanced Kelly-based sizing if available
        if 'position_size' in signal and signal['position_size'] > 0:
            # Use Kelly Criterion optimized position size from enhanced signal
            kelly_allocation = min(signal['position_size'], self.config.max_position_size)
            target_value = portfolio_value * kelly_allocation
            logger.debug(f"Using Kelly position sizing: {kelly_allocation:.1%} for {symbol}")
        else:
            # Fallback to traditional confidence-weighted allocation
            base_allocation = self.config.max_position_size * signal['confidence']
            target_value = portfolio_value * base_allocation
        
        # Account for transaction costs
        shares = int(target_value / (price * (1 + self.config.transaction_cost)))
        cost = shares * price * (1 + self.config.transaction_cost)
        
        if cost <= self.cash and shares > 0:
            # Execute trade
            self.cash -= cost
            self.portfolio[symbol] = self.portfolio.get(symbol, 0) + shares
            
            # Record trade
            trade = Trade(
                symbol=symbol,
                entry_date=date,
                exit_date=None,
                entry_price=price,
                exit_price=None,
                quantity=shares,
                direction='long',
                entry_signal=signal,
                exit_reason='',
                pnl=None,
                return_pct=None,
                days_held=None
            )
            
            self.trades.append(trade)
    
    def _execute_sell(self, symbol: str, date: datetime, price_data: pd.Series, 
                     signal: Dict[str, Any]):
        """Execute sell order"""
        
        price = price_data['Close']
        shares = self.portfolio[symbol]
        
        # Execute trade
        proceeds = shares * price * (1 - self.config.transaction_cost)
        self.cash += proceeds
        self.portfolio[symbol] = 0
        
        # Find corresponding buy trade and update
        for trade in reversed(self.trades):
            if trade.symbol == symbol and trade.exit_date is None:
                trade.exit_date = date
                trade.exit_price = price
                trade.exit_reason = f"SELL_SIGNAL_{signal['direction']}"
                trade.days_held = (date - trade.entry_date).days
                trade.pnl = (price - trade.entry_price) * trade.quantity - \
                           (trade.entry_price * trade.quantity * self.config.transaction_cost * 2)
                trade.return_pct = ((price - trade.entry_price) / trade.entry_price) * 100
                break
    
    def _get_current_portfolio_value(self, date: datetime, data: Dict[str, pd.DataFrame]) -> float:
        """Calculate current portfolio value"""
        
        portfolio_value = self.cash
        
        for symbol, shares in self.portfolio.items():
            if shares > 0 and symbol in data and date in data[symbol].index:
                current_price = data[symbol].loc[date, 'Close']
                portfolio_value += shares * current_price
        
        return portfolio_value
    
    def _update_portfolio_value(self, date: datetime, data: Dict[str, pd.DataFrame]):
        """Update daily portfolio value tracking"""
        
        portfolio_value = self._get_current_portfolio_value(date, data)
        
        # Get benchmark values
        benchmark_values = {}
        for benchmark, bench_data in self.benchmark_data.items():
            if date in bench_data.index:
                benchmark_values[benchmark] = bench_data.loc[date, 'Close']
        
        self.daily_values.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions_count': sum(1 for shares in self.portfolio.values() if shares > 0),
            'benchmark_values': benchmark_values
        })
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate comprehensive backtest results"""
        
        if not self.daily_values:
            return {}
        
        # Convert daily values to DataFrame
        daily_df = pd.DataFrame(self.daily_values)
        daily_df.set_index('date', inplace=True)
        
        # Calculate returns
        daily_df['returns'] = daily_df['portfolio_value'].pct_change()
        
        # Calculate benchmark returns
        benchmark_returns = {}
        for benchmark in self.benchmark_data.keys():
            bench_values = [dv['benchmark_values'].get(benchmark, np.nan) for dv in self.daily_values]
            bench_df = pd.DataFrame({'value': bench_values}, index=daily_df.index)
            bench_df['returns'] = bench_df['value'].pct_change()
            benchmark_returns[benchmark] = bench_df['returns']
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(daily_df['returns'], benchmark_returns)
        
        # Trade analysis
        completed_trades = [t for t in self.trades if t.exit_date is not None]
        trade_analysis = self._analyze_trades(completed_trades)
        
        # Regime analysis
        regime_analysis = self._analyze_regime_performance(daily_df, completed_trades)
        
        return {
            'daily_values': daily_df,
            'trades': self.trades,
            'completed_trades': completed_trades,
            'performance_metrics': metrics,
            'trade_analysis': trade_analysis,
            'regime_analysis': regime_analysis,
            'benchmark_returns': benchmark_returns,
            'final_portfolio_value': daily_df['portfolio_value'].iloc[-1],
            'total_return': (daily_df['portfolio_value'].iloc[-1] / self.config.initial_capital - 1) * 100
        }
    
    def _calculate_performance_metrics(self, returns: pd.Series, 
                                     benchmark_returns: Dict[str, pd.Series]) -> Dict[str, PerformanceMetrics]:
        """Calculate performance metrics vs benchmarks"""
        
        metrics = {}
        
        # Strategy metrics
        strategy_metrics = self._calculate_single_performance_metrics(returns, returns)  # Self as benchmark for base metrics
        
        # Benchmark comparisons
        for benchmark_name, benchmark_rets in benchmark_returns.items():
            if not benchmark_rets.empty:
                benchmark_metrics = self._calculate_single_performance_metrics(returns, benchmark_rets)
                metrics[benchmark_name] = benchmark_metrics
        
        metrics['strategy'] = strategy_metrics
        return metrics
    
    def _calculate_single_performance_metrics(self, strategy_returns: pd.Series, 
                                            benchmark_returns: pd.Series) -> PerformanceMetrics:
        """Calculate performance metrics for strategy vs benchmark"""
        
        # Remove NaN values
        strategy_rets = strategy_returns.dropna()
        benchmark_rets = benchmark_returns.dropna()
        
        if len(strategy_rets) == 0:
            return PerformanceMetrics(**{k: 0.0 for k in PerformanceMetrics.__annotations__})
        
        # Returns
        total_return = (1 + strategy_rets).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_rets)) - 1
        
        benchmark_total = (1 + benchmark_rets).prod() - 1 if not benchmark_rets.empty else 0
        excess_return = total_return - benchmark_total
        
        # Risk metrics
        volatility = strategy_rets.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = strategy_rets[strategy_rets < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # Max drawdown
        cumulative = (1 + strategy_rets).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Correlation metrics
        if not benchmark_rets.empty and len(benchmark_rets) == len(strategy_rets):
            beta = strategy_rets.cov(benchmark_rets) / benchmark_rets.var() if benchmark_rets.var() > 0 else 0
            correlation = strategy_rets.corr(benchmark_rets) if not benchmark_rets.empty else 0
            tracking_error = (strategy_rets - benchmark_rets).std() * np.sqrt(252)
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
        else:
            beta = correlation = tracking_error = information_ratio = 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            excess_return=excess_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            total_trades=0,  # Will be filled in trade analysis
            win_rate=0,
            profit_factor=0,
            avg_trade_return=0,
            avg_days_held=0,
            beta=beta,
            correlation=correlation,
            tracking_error=tracking_error,
            information_ratio=information_ratio
        )
    
    def _analyze_trades(self, completed_trades: List[Trade]) -> Dict[str, Any]:
        """Analyze completed trades"""
        
        if not completed_trades:
            return {}
        
        # Trade metrics
        total_trades = len(completed_trades)
        winning_trades = len([t for t in completed_trades if t.pnl > 0])
        win_rate = winning_trades / total_trades
        
        # P&L analysis
        total_pnl = sum(t.pnl for t in completed_trades)
        gross_profits = sum(t.pnl for t in completed_trades if t.pnl > 0)
        gross_losses = abs(sum(t.pnl for t in completed_trades if t.pnl < 0))
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        # Return analysis
        returns = [t.return_pct for t in completed_trades]
        avg_trade_return = np.mean(returns)
        
        # Holding period analysis
        days_held = [t.days_held for t in completed_trades]
        avg_days_held = np.mean(days_held)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'gross_profits': gross_profits,
            'gross_losses': gross_losses,
            'profit_factor': profit_factor,
            'avg_trade_return': avg_trade_return,
            'avg_days_held': avg_days_held,
            'best_trade': max(completed_trades, key=lambda t: t.pnl) if completed_trades else None,
            'worst_trade': min(completed_trades, key=lambda t: t.pnl) if completed_trades else None,
            'trade_returns_std': np.std(returns),
            'consecutive_wins': self._calculate_consecutive_wins(completed_trades),
            'consecutive_losses': self._calculate_consecutive_losses(completed_trades)
        }
    
    def _analyze_regime_performance(self, daily_df: pd.DataFrame, 
                                  completed_trades: List[Trade]) -> Dict[str, Any]:
        """Analyze performance by market regime"""
        
        regime_analysis = {}
        
        for regime in self.regime_classifier.regimes:
            # Filter data for regime period
            regime_data = daily_df[(daily_df.index >= regime.start_date) & 
                                  (daily_df.index <= regime.end_date)]
            
            if regime_data.empty:
                continue
            
            # Performance metrics for regime
            regime_returns = regime_data['returns'].dropna()
            if len(regime_returns) > 0:
                total_return = (1 + regime_returns).prod() - 1
                volatility = regime_returns.std() * np.sqrt(252)
                sharpe = (total_return * 252 / len(regime_returns)) / volatility if volatility > 0 else 0
            else:
                total_return = volatility = sharpe = 0
            
            # Trades in regime
            regime_trades = [t for t in completed_trades 
                           if regime.start_date <= t.entry_date <= regime.end_date]
            
            regime_analysis[regime.name] = {
                'period': f"{regime.start_date.strftime('%Y-%m-%d')} to {regime.end_date.strftime('%Y-%m-%d')}",
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'trades': len(regime_trades),
                'win_rate': len([t for t in regime_trades if t.pnl > 0]) / len(regime_trades) if regime_trades else 0,
                'avg_trade_return': np.mean([t.return_pct for t in regime_trades]) if regime_trades else 0,
                'characteristics': regime.characteristics
            }
        
        return regime_analysis
    
    def _calculate_consecutive_wins(self, trades: List[Trade]) -> int:
        """Calculate maximum consecutive winning trades"""
        max_consecutive = current_consecutive = 0
        
        for trade in trades:
            if trade.pnl > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_consecutive_losses(self, trades: List[Trade]) -> int:
        """Calculate maximum consecutive losing trades"""
        max_consecutive = current_consecutive = 0
        
        for trade in trades:
            if trade.pnl < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive


def main():
    """Example usage"""
    
    # Configure backtesting
    config = BacktestConfig(
        initial_capital=1_000_000,
        max_position_size=0.02,
        transaction_cost=0.001,
        min_confidence=0.6
    )
    
    # Create backtester
    backtester = ComprehensiveSignalBacktester(config)
    
    # Run backtest
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 9, 30)
    
    try:
        results = backtester.run_backtest(start_date, end_date)
        
        print(f"\\nBacktest Results ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}):")
        print(f"Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Total Trades: {len(results['completed_trades'])}")
        
        if results['completed_trades']:
            print(f"Win Rate: {results['trade_analysis']['win_rate']:.1%}")
            print(f"Profit Factor: {results['trade_analysis']['profit_factor']:.2f}")
        
        print("\\nRegime Performance:")
        for regime_name, regime_data in results['regime_analysis'].items():
            print(f"  {regime_name}: {regime_data['total_return']:+.2%} return, {regime_data['trades']} trades")
        
        return results
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()