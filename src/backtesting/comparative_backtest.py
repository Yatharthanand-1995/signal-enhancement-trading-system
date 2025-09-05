"""
Comparative Backtesting Framework
Tests Original vs Adaptive Signal Systems on specific market periods

Focus periods:
1. COVID Crash (Feb-Apr 2020) - where we underperformed
2. 2022 Rate Rise Bear (Jan-Oct 2022) - persistent underperformance  
3. Bull Market 2021 (Jan-Nov 2021) - check for profit reduction

This will provide concrete evidence of improvement or degradation
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
import json

# Import both signal systems for comparison
from ..strategy.ensemble_signal_scoring import EnsembleSignalScorer
from ..strategy.adaptive_signal_system import AdaptiveSignalSystem

logger = logging.getLogger(__name__)

@dataclass
class BacktestPeriod:
    """Define backtesting periods for comparison"""
    name: str
    start_date: datetime
    end_date: datetime
    expected_spy_return: float  # For comparison
    market_type: str  # 'bull', 'bear', 'crisis', 'sideways'
    description: str

@dataclass 
class BacktestResults:
    """Results from backtesting a signal system"""
    period_name: str
    system_name: str
    
    # Performance metrics
    total_return: float
    annualized_return: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    
    # Trading metrics
    total_trades: int
    win_rate: float
    avg_trade_return: float
    
    # Comparison metrics
    vs_spy_return: float
    vs_spy_sharpe: float
    
    # Signal-specific metrics
    avg_confidence: float
    signal_distribution: Dict[str, int]  # BUY, SELL, NEUTRAL counts

class ComparativeBacktester:
    """
    Backtesting framework to compare Original vs Adaptive signal systems
    """
    
    def __init__(self):
        # Initialize both signal systems
        self.original_system = EnsembleSignalScorer()
        self.adaptive_system = AdaptiveSignalSystem()
        
        # Test periods where we expect different performance
        self.test_periods = [
            BacktestPeriod(
                name="COVID_CRASH",
                start_date=datetime(2020, 2, 20),
                end_date=datetime(2020, 4, 30),
                expected_spy_return=-0.13,  # SPY lost ~13% during crash + recovery
                market_type="crisis",
                description="COVID-19 market crash and initial recovery"
            ),
            BacktestPeriod(
                name="RATE_RISE_BEAR",
                start_date=datetime(2022, 1, 3),
                end_date=datetime(2022, 10, 31),
                expected_spy_return=-0.20,  # SPY lost ~20% during rate rise
                market_type="bear",
                description="Fed rate rise bear market"
            ),
            BacktestPeriod(
                name="BULL_MARKET_2021", 
                start_date=datetime(2021, 1, 4),
                end_date=datetime(2021, 11, 30),
                expected_spy_return=0.22,   # SPY gained ~22% in 2021
                market_type="bull",
                description="Strong bull market - check for profit reduction"
            ),
            BacktestPeriod(
                name="RECOVERY_2023",
                start_date=datetime(2023, 1, 3),
                end_date=datetime(2023, 12, 29),
                expected_spy_return=0.24,   # SPY gained ~24% in 2023
                market_type="bull",
                description="AI boom recovery year"
            )
        ]
        
        # Test stocks (focus on liquid, representative stocks)
        self.test_symbols = [
            'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN',  # Tech leaders
            'JPM', 'BAC', 'WFC',                        # Financials (rate sensitive)
            'XOM', 'CVX',                               # Energy (volatile)
            'JNJ', 'PFE',                               # Healthcare (defensive)
            'WMT', 'PG'                                 # Consumer staples (defensive)
        ]
        
        logger.info(f"Comparative backtester initialized with {len(self.test_periods)} periods and {len(self.test_symbols)} symbols")
    
    def get_vix_data(self, start_date: datetime, end_date: datetime) -> pd.Series:
        """
        Get VIX data for the period (for adaptive system)
        """
        try:
            vix_ticker = yf.Ticker('^VIX')
            vix_data = vix_ticker.history(start=start_date - timedelta(days=5), end=end_date + timedelta(days=1))
            return vix_data['Close'] if not vix_data.empty else None
        except Exception as e:
            logger.warning(f"Could not fetch VIX data: {e}")
            return None
    
    def simulate_trading(self, signals: List[Dict], prices: pd.DataFrame, initial_capital: float = 100000) -> Dict[str, Any]:
        """
        Simulate trading based on signals and return performance metrics
        """
        portfolio_value = initial_capital
        cash = initial_capital
        positions = {}  # symbol -> shares
        trades = []
        portfolio_history = []
        
        # Process each signal
        for signal in signals:
            symbol = signal['symbol']
            date = signal['timestamp']
            direction = signal['direction']
            confidence = signal['confidence']
            
            if symbol not in prices.columns or date not in prices.index:
                continue
                
            current_price = prices.loc[date, symbol]
            
            # Simple trading logic
            position_size = min(confidence * 0.02, 0.02)  # Max 2% per position
            
            if direction == 'BUY' and confidence > 0.6:
                # Buy signal
                investment = portfolio_value * position_size
                shares = investment / current_price
                
                if cash >= investment:
                    positions[symbol] = positions.get(symbol, 0) + shares
                    cash -= investment
                    trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'BUY',
                        'shares': shares,
                        'price': current_price,
                        'confidence': confidence
                    })
            
            elif direction == 'SELL' and symbol in positions and confidence > 0.6:
                # Sell signal
                if positions[symbol] > 0:
                    shares_to_sell = positions[symbol] * 0.5  # Sell half
                    cash += shares_to_sell * current_price
                    positions[symbol] -= shares_to_sell
                    trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'shares': shares_to_sell,
                        'price': current_price,
                        'confidence': confidence
                    })
            
            # Calculate current portfolio value
            position_value = sum(positions.get(symbol, 0) * prices.loc[date, symbol] 
                               for symbol in positions if symbol in prices.columns and date in prices.index)
            total_value = cash + position_value
            portfolio_history.append({
                'date': date,
                'portfolio_value': total_value,
                'cash': cash,
                'positions_value': position_value
            })
        
        if not portfolio_history:
            return self._empty_results()
        
        # Calculate performance metrics
        portfolio_df = pd.DataFrame(portfolio_history).set_index('date')
        returns = portfolio_df['portfolio_value'].pct_change().dropna()
        
        total_return = (portfolio_df['portfolio_value'].iloc[-1] - initial_capital) / initial_capital
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0
        
        # Trading metrics
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        win_trades = 0
        total_trade_return = 0
        
        # Simple win rate calculation (buy followed by higher price)
        for i, trade in enumerate(buy_trades):
            if i < len(trades) - 1:
                entry_price = trade['price']
                # Look for exit or end price
                symbol = trade['symbol']
                entry_date = trade['date']
                
                # Find price 30 days later or end of period
                future_date = entry_date + timedelta(days=30)
                future_prices = prices.loc[prices.index > entry_date, symbol].dropna()
                
                if not future_prices.empty:
                    exit_price = future_prices.iloc[min(30, len(future_prices)-1)]
                    trade_return = (exit_price - entry_price) / entry_price
                    total_trade_return += trade_return
                    if trade_return > 0:
                        win_trades += 1
        
        win_rate = win_trades / len(buy_trades) if buy_trades else 0
        avg_trade_return = total_trade_return / len(buy_trades) if buy_trades else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'portfolio_history': portfolio_df,
            'trades': trades
        }
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results when no trading occurs"""
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'avg_trade_return': 0.0,
            'portfolio_history': pd.DataFrame(),
            'trades': []
        }
    
    def backtest_period(self, period: BacktestPeriod, system_name: str) -> BacktestResults:
        """
        Backtest a single period with specified system
        """
        logger.info(f"Backtesting {system_name} on {period.name} ({period.start_date.date()} to {period.end_date.date()})")
        
        # Get market data
        try:
            tickers_str = ' '.join(self.test_symbols)
            data = yf.download(tickers_str, start=period.start_date - timedelta(days=30), 
                             end=period.end_date + timedelta(days=1), progress=False)
            
            if data.empty:
                logger.error(f"No data available for period {period.name}")
                return self._create_empty_backtest_result(period.name, system_name)
            
            # Get VIX data for adaptive system
            vix_data = None
            if system_name == 'Adaptive':
                vix_data = self.get_vix_data(period.start_date, period.end_date)
            
            # Handle multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close']
                volumes = data['Volume'] if 'Volume' in data.columns.levels[0] else None
            else:
                prices = data
                volumes = None
            
            # Generate signals for each day and symbol
            signals = []
            trading_dates = prices.index[prices.index >= period.start_date]
            
            for date in trading_dates:
                for symbol in self.test_symbols:
                    if symbol in prices.columns:
                        # Get historical data up to this date
                        symbol_data = pd.DataFrame({
                            'Close': prices[symbol].loc[:date],
                            'High': prices[symbol].loc[:date],  # Simplified
                            'Low': prices[symbol].loc[:date],   # Simplified
                            'Volume': volumes[symbol].loc[:date] if volumes is not None and symbol in volumes.columns else prices[symbol].loc[:date]
                        }).dropna()
                        
                        if len(symbol_data) >= 30:  # Need sufficient history
                            # Get VIX value for this date
                            current_vix = None
                            if vix_data is not None and date in vix_data.index:
                                current_vix = vix_data[date]
                            
                            # Generate signal
                            if system_name == 'Original':
                                # Use original system (simplified)
                                signal = self._generate_original_signal(symbol_data, date, symbol)
                            else:  # Adaptive
                                signal = self.adaptive_system.generate_adaptive_signal(
                                    symbol_data, date, symbol, current_vix
                                )
                            
                            signals.append(signal)
            
            # Simulate trading
            trading_results = self.simulate_trading(signals, prices)
            
            # Get SPY benchmark for comparison
            spy_data = yf.download('SPY', start=period.start_date, end=period.end_date, progress=False)
            spy_return = 0
            spy_sharpe = 0
            
            if not spy_data.empty:
                spy_returns = spy_data['Close'].pct_change().dropna()
                spy_return = (spy_data['Close'].iloc[-1] - spy_data['Close'].iloc[0]) / spy_data['Close'].iloc[0]
                spy_vol = spy_returns.std() * np.sqrt(252)
                spy_sharpe = (spy_return * 252 / len(spy_returns)) / spy_vol if spy_vol > 0 else 0
            
            # Calculate signal distribution
            signal_distribution = {}
            for signal in signals:
                direction = signal.get('direction', 'NEUTRAL')
                signal_distribution[direction] = signal_distribution.get(direction, 0) + 1
            
            avg_confidence = np.mean([s.get('confidence', 0) for s in signals]) if signals else 0
            
            # Create results
            return BacktestResults(
                period_name=period.name,
                system_name=system_name,
                total_return=trading_results['total_return'],
                annualized_return=trading_results['annualized_return'],
                max_drawdown=trading_results['max_drawdown'],
                volatility=trading_results['volatility'],
                sharpe_ratio=trading_results['sharpe_ratio'],
                total_trades=trading_results['total_trades'],
                win_rate=trading_results['win_rate'],
                avg_trade_return=trading_results['avg_trade_return'],
                vs_spy_return=trading_results['total_return'] - spy_return,
                vs_spy_sharpe=trading_results['sharpe_ratio'] - spy_sharpe,
                avg_confidence=avg_confidence,
                signal_distribution=signal_distribution
            )
            
        except Exception as e:
            logger.error(f"Error backtesting {system_name} on {period.name}: {e}")
            return self._create_empty_backtest_result(period.name, system_name)
    
    def _generate_original_signal(self, market_data: pd.DataFrame, date: datetime, symbol: str) -> Dict[str, Any]:
        """
        Generate signal using original system (simplified version)
        """
        if len(market_data) < 14:
            return {'symbol': symbol, 'timestamp': date, 'direction': 'NEUTRAL', 'confidence': 0.0}
        
        prices = market_data['Close'].values
        
        # Simple RSI calculation
        price_changes = np.diff(prices[-15:])
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses) + 1e-8
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        
        # Simple MACD
        if len(prices) >= 26:
            ema12 = pd.Series(prices).ewm(span=12).mean().iloc[-1]
            ema26 = pd.Series(prices).ewm(span=26).mean().iloc[-1]
            macd = ema12 - ema26
        else:
            macd = 0
        
        # Simple signal logic
        if rsi < 30 and macd > 0:
            return {'symbol': symbol, 'timestamp': date, 'direction': 'BUY', 'confidence': 0.7}
        elif rsi > 70 and macd < 0:
            return {'symbol': symbol, 'timestamp': date, 'direction': 'SELL', 'confidence': 0.7}
        else:
            return {'symbol': symbol, 'timestamp': date, 'direction': 'NEUTRAL', 'confidence': 0.5}
    
    def _create_empty_backtest_result(self, period_name: str, system_name: str) -> BacktestResults:
        """Create empty backtest result for error cases"""
        return BacktestResults(
            period_name=period_name,
            system_name=system_name,
            total_return=0.0,
            annualized_return=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            total_trades=0,
            win_rate=0.0,
            avg_trade_return=0.0,
            vs_spy_return=0.0,
            vs_spy_sharpe=0.0,
            avg_confidence=0.0,
            signal_distribution={}
        )
    
    def run_comparative_analysis(self) -> Dict[str, Any]:
        """
        Run complete comparative analysis across all periods
        """
        logger.info("🚀 Starting Comparative Backtesting Analysis")
        
        all_results = []
        
        # Test each period with both systems
        for period in self.test_periods:
            # Test original system
            original_results = self.backtest_period(period, 'Original')
            all_results.append(original_results)
            
            # Test adaptive system  
            adaptive_results = self.backtest_period(period, 'Adaptive')
            all_results.append(adaptive_results)
        
        # Analyze results
        analysis = self._analyze_comparative_results(all_results)
        
        logger.info("✅ Comparative analysis complete")
        return {
            'individual_results': all_results,
            'comparative_analysis': analysis,
            'test_periods': [{'name': p.name, 'description': p.description, 'market_type': p.market_type} for p in self.test_periods]
        }
    
    def _analyze_comparative_results(self, results: List[BacktestResults]) -> Dict[str, Any]:
        """
        Analyze and compare results between systems
        """
        original_results = [r for r in results if r.system_name == 'Original']
        adaptive_results = [r for r in results if r.system_name == 'Adaptive']
        
        # Calculate improvements
        improvements = {}
        
        for orig, adapt in zip(original_results, adaptive_results):
            period_name = orig.period_name
            improvements[period_name] = {
                'return_improvement': adapt.total_return - orig.total_return,
                'sharpe_improvement': adapt.sharpe_ratio - orig.sharpe_ratio,
                'drawdown_improvement': orig.max_drawdown - adapt.max_drawdown,  # Positive = better
                'vs_spy_improvement': adapt.vs_spy_return - orig.vs_spy_return,
                'confidence_improvement': adapt.avg_confidence - orig.avg_confidence,
                'trades_change': adapt.total_trades - orig.total_trades
            }
        
        # Overall statistics
        avg_return_improvement = np.mean([imp['return_improvement'] for imp in improvements.values()])
        avg_sharpe_improvement = np.mean([imp['sharpe_improvement'] for imp in improvements.values()])
        periods_improved = sum(1 for imp in improvements.values() if imp['return_improvement'] > 0)
        
        return {
            'period_improvements': improvements,
            'summary': {
                'avg_return_improvement': avg_return_improvement,
                'avg_sharpe_improvement': avg_sharpe_improvement,
                'periods_improved': periods_improved,
                'total_periods_tested': len(improvements),
                'improvement_rate': periods_improved / len(improvements)
            }
        }
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'comparative_backtest_results_{timestamp}.json'
        
        # Convert results to serializable format
        serializable_results = self._make_serializable(results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")
        return filename
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, BacktestResults):
            return obj.__dict__
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        else:
            return obj

def main():
    """Run comparative backtesting"""
    print("🧪 Starting Comparative Backtesting: Original vs Adaptive Signals")
    
    # Initialize backtester
    backtester = ComparativeBacktester()
    
    # Run comparative analysis
    results = backtester.run_comparative_analysis()
    
    # Save results
    results_file = backtester.save_results(results)
    
    # Print summary
    summary = results['comparative_analysis']['summary']
    print(f"\n📊 COMPARATIVE RESULTS SUMMARY")
    print(f"Periods tested: {summary['total_periods_tested']}")
    print(f"Periods improved: {summary['periods_improved']}")
    print(f"Improvement rate: {summary['improvement_rate']:.1%}")
    print(f"Avg return improvement: {summary['avg_return_improvement']:+.2%}")
    print(f"Avg Sharpe improvement: {summary['avg_sharpe_improvement']:+.2f}")
    
    # Period-by-period breakdown
    print(f"\n📈 PERIOD-BY-PERIOD BREAKDOWN:")
    for period_name, improvements in results['comparative_analysis']['period_improvements'].items():
        print(f"\n{period_name}:")
        print(f"  Return improvement: {improvements['return_improvement']:+.2%}")
        print(f"  Sharpe improvement: {improvements['sharpe_improvement']:+.2f}")
        print(f"  vs SPY improvement: {improvements['vs_spy_improvement']:+.2%}")
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    return results

if __name__ == "__main__":
    main()