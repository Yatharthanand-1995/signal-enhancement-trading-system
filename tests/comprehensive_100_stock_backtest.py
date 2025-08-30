#!/usr/bin/env python3
"""
Comprehensive Backtesting Framework for 100+ Stocks
Validates ML signal accuracy across the full stock universe with detailed performance metrics
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')

@dataclass
class BacktestResult:
    """Individual stock backtest result"""
    symbol: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    best_trade: float
    worst_trade: float
    avg_holding_period: float
    ml_accuracy: float  # How often ML signals were correct
    signal_strength_correlation: float  # Correlation between signal strength and returns
    
@dataclass
class ComprehensiveBacktestSummary:
    """Comprehensive backtest summary across all stocks"""
    total_stocks_tested: int
    successful_backtests: int
    avg_total_return: float
    avg_annualized_return: float
    avg_sharpe_ratio: float
    avg_win_rate: float
    avg_ml_accuracy: float
    best_performers: List[str]
    worst_performers: List[str]
    sector_performance: Dict[str, Dict]
    market_cap_performance: Dict[str, Dict]
    overall_portfolio_return: float
    portfolio_sharpe: float
    benchmark_comparison: Dict[str, float]

class Comprehensive100StockBacktester:
    """Comprehensive backtesting system for 100+ stocks"""
    
    def __init__(self, initial_capital=10000, commission=0.005):
        self.initial_capital = initial_capital
        self.commission = commission  # 0.5% commission per trade
        
        # Stock categorization for analysis
        self.stock_categories = {
            'Mega Cap Tech': ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA'],
            'Large Cap Tech': ['ORCL', 'ADBE', 'CRM', 'AMD', 'INTC', 'CSCO', 'QCOM', 'AMAT'],
            'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'C', 'AXP', 'BLK', 'SCHW', 'USB'],
            'Healthcare': ['UNH', 'JNJ', 'PFE', 'ABBV', 'LLY', 'TMO', 'ABT', 'AMGN', 'SYK', 'BSX'],
            'Consumer': ['WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'TJX', 'WM', 'INTU'],
            'Industrial': ['CAT', 'BA', 'RTX', 'HON', 'DE', 'GE', 'MMC', 'ITW'],
            'Energy': ['XOM', 'CVX', 'SLB', 'EOG', 'FCX'],
            'Utilities': ['NEE', 'DUK', 'SO'],
            'Real Estate': ['AMT', 'PLD', 'EQIX']
        }
        
        # Results storage
        self.results_dir = Path('backtest_results')
        self.results_dir.mkdir(exist_ok=True)
        
    def load_full_dataset(self):
        """Load the complete 106-stock dataset"""
        
        print("📊 LOADING COMPREHENSIVE DATASET")
        print("-" * 40)
        
        # Load all three datasets for comprehensive backtesting
        data_files = {
            'train': 'data/full_market/train_data.csv',
            'validation': 'data/full_market/validation_data.csv', 
            'test': 'data/full_market/test_data.csv'
        }
        
        all_data = []
        
        for dataset_name, file_path in data_files.items():
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                data['date'] = pd.to_datetime(data['date'])
                data['dataset'] = dataset_name
                all_data.append(data)
                print(f"✅ {dataset_name.title()}: {len(data):,} records")
            else:
                print(f"❌ {dataset_name.title()}: File not found")
        
        if not all_data:
            raise FileNotFoundError("No dataset files found")
        
        # Combine all datasets
        self.full_data = pd.concat(all_data, ignore_index=True)
        self.full_data = self.full_data.sort_values(['symbol', 'date'])
        
        self.available_symbols = sorted(self.full_data['symbol'].unique())
        
        print(f"\n✅ COMBINED DATASET")
        print(f"   Total records: {len(self.full_data):,}")
        print(f"   Symbols: {len(self.available_symbols)}")
        print(f"   Date range: {self.full_data['date'].min().date()} to {self.full_data['date'].max().date()}")
        print(f"   Period: {(self.full_data['date'].max() - self.full_data['date'].min()).days} days")
        
        return True
    
    def initialize_ml_system(self):
        """Initialize the ML signal generation system"""
        
        print(f"\n🤖 INITIALIZING ML SIGNAL SYSTEM")
        print("-" * 35)
        
        try:
            from strategy.enhanced_signal_integration import get_enhanced_signal, initialize_enhanced_signal_integration
            
            self.integrator = initialize_enhanced_signal_integration()
            self.get_enhanced_signal = get_enhanced_signal
            
            print(f"✅ ML System: {self.integrator.name}")
            print(f"✅ Correlations: {len(self.integrator.signal_correlations)} proven features")
            
            return True
        except Exception as e:
            print(f"❌ ML initialization failed: {e}")
            return False
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators for backtesting"""
        
        # Price-based indicators
        data['sma_10'] = data['close'].rolling(10).mean()
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        
        # MACD
        exp1 = data['close'].ewm(span=12).mean()
        exp2 = data['close'].ewm(span=26).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        rolling_mean = data['close'].rolling(20).mean()
        rolling_std = data['close'].rolling(20).std()
        data['bb_upper'] = rolling_mean + (rolling_std * 2)
        data['bb_lower'] = rolling_mean - (rolling_std * 2)
        
        # Volume indicators
        data['volume_sma'] = data['volume'].rolling(20).mean()
        
        return data
    
    def backtest_single_stock(self, symbol: str) -> Optional[BacktestResult]:
        """Comprehensive backtest for a single stock"""
        
        try:
            # Get symbol data
            symbol_data = self.full_data[self.full_data['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date').reset_index(drop=True)
            
            if len(symbol_data) < 200:
                return None  # Insufficient data
            
            # Add technical indicators
            symbol_data = self.calculate_technical_indicators(symbol_data)
            
            # Initialize variables
            capital = self.initial_capital
            position = 0
            position_entry_price = 0
            position_entry_date = None
            trades = []
            portfolio_values = []
            signals_generated = []
            
            # Backtesting loop
            for i in range(100, len(symbol_data)):  # Start after 100 days for indicators
                current_data = symbol_data.iloc[:i+1]
                current_row = current_data.iloc[-1]
                current_price = current_row['close']
                current_date = current_row['date']
                
                # Generate ML signal
                if len(current_data) >= 100:
                    try:
                        test_data = current_data.tail(100)
                        ml_signal = self.get_enhanced_signal(
                            symbol=symbol,
                            data=test_data,
                            current_price=current_price,
                            current_regime='normal'
                        )
                        
                        if ml_signal:
                            signal_strength = ml_signal.signal_strength
                            confidence = ml_signal.confidence
                            position_size = ml_signal.recommended_position_size
                        else:
                            signal_strength = 0
                            confidence = 0.5
                            position_size = 0.02
                            
                    except Exception:
                        signal_strength = 0
                        confidence = 0.5
                        position_size = 0.02
                else:
                    signal_strength = 0
                    confidence = 0.5
                    position_size = 0.02
                
                signals_generated.append({
                    'date': current_date,
                    'signal_strength': signal_strength,
                    'confidence': confidence,
                    'price': current_price
                })
                
                # Trading logic
                if position == 0:  # No position
                    # Entry conditions
                    if abs(signal_strength) > 0.1 and confidence > 0.7:  # Strong signal
                        # Calculate position size based on capital and ML recommendation
                        max_position_value = capital * position_size
                        shares_to_buy = int(max_position_value / current_price)
                        
                        if shares_to_buy > 0:
                            # Enter position
                            position = shares_to_buy if signal_strength > 0 else -shares_to_buy
                            position_entry_price = current_price
                            position_entry_date = current_date
                            
                            # Account for commission
                            commission_cost = abs(position * current_price) * self.commission
                            capital -= commission_cost
                
                elif position != 0:  # Have position
                    # Exit conditions
                    should_exit = False
                    exit_reason = ""
                    
                    # Signal-based exit
                    if position > 0 and signal_strength < -0.05:  # Long position, sell signal
                        should_exit = True
                        exit_reason = "Signal reversal"
                    elif position < 0 and signal_strength > 0.05:  # Short position, buy signal
                        should_exit = True
                        exit_reason = "Signal reversal"
                    
                    # Stop loss (10% loss)
                    if position > 0 and current_price < position_entry_price * 0.9:
                        should_exit = True
                        exit_reason = "Stop loss"
                    elif position < 0 and current_price > position_entry_price * 1.1:
                        should_exit = True
                        exit_reason = "Stop loss"
                    
                    # Take profit (20% gain)
                    if position > 0 and current_price > position_entry_price * 1.2:
                        should_exit = True
                        exit_reason = "Take profit"
                    elif position < 0 and current_price < position_entry_price * 0.8:
                        should_exit = True
                        exit_reason = "Take profit"
                    
                    # Time-based exit (hold for max 30 days)
                    if (current_date - position_entry_date).days > 30:
                        should_exit = True
                        exit_reason = "Time limit"
                    
                    if should_exit:
                        # Exit position
                        exit_value = position * current_price
                        commission_cost = abs(exit_value) * self.commission
                        
                        if position > 0:  # Long position
                            capital += exit_value - commission_cost
                        else:  # Short position
                            capital += -exit_value - commission_cost
                        
                        # Record trade
                        entry_value = position * position_entry_price
                        if position > 0:
                            trade_return = (exit_value - abs(entry_value)) / abs(entry_value)
                        else:
                            trade_return = (abs(entry_value) - exit_value) / abs(entry_value)
                        
                        # Account for commission in trade return
                        total_commission = 2 * abs(entry_value) * self.commission  # Entry + Exit
                        trade_return -= total_commission / abs(entry_value)
                        
                        trades.append({
                            'entry_date': position_entry_date,
                            'exit_date': current_date,
                            'entry_price': position_entry_price,
                            'exit_price': current_price,
                            'position': position,
                            'return': trade_return,
                            'holding_days': (current_date - position_entry_date).days,
                            'exit_reason': exit_reason
                        })
                        
                        position = 0
                        position_entry_price = 0
                        position_entry_date = None
                
                # Calculate current portfolio value
                current_portfolio_value = capital
                if position != 0:
                    current_portfolio_value += position * current_price
                
                portfolio_values.append({
                    'date': current_date,
                    'portfolio_value': current_portfolio_value,
                    'capital': capital,
                    'position_value': position * current_price if position != 0 else 0
                })
            
            # Calculate performance metrics
            if not trades or not portfolio_values:
                return None
            
            # Basic performance metrics
            final_value = portfolio_values[-1]['portfolio_value']
            total_return = (final_value - self.initial_capital) / self.initial_capital
            
            # Annualized return
            total_days = (symbol_data['date'].iloc[-1] - symbol_data['date'].iloc[100]).days
            years = total_days / 365.25
            annualized_return = (final_value / self.initial_capital) ** (1/years) - 1 if years > 0 else 0
            
            # Trade statistics
            trade_returns = [t['return'] for t in trades]
            win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
            avg_trade_return = np.mean(trade_returns)
            best_trade = max(trade_returns)
            worst_trade = min(trade_returns)
            avg_holding_period = np.mean([t['holding_days'] for t in trades])
            
            # Risk metrics
            portfolio_df = pd.DataFrame(portfolio_values)
            portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
            
            volatility = portfolio_df['daily_return'].std() * np.sqrt(252) if len(portfolio_df) > 1 else 0
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Max drawdown
            portfolio_df['cummax'] = portfolio_df['portfolio_value'].cummax()
            portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['cummax']) / portfolio_df['cummax']
            max_drawdown = abs(portfolio_df['drawdown'].min())
            
            # Sortino ratio (downside deviation)
            negative_returns = portfolio_df['daily_return'][portfolio_df['daily_return'] < 0]
            downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
            sortino_ratio = annualized_return / downside_std if downside_std > 0 else 0
            
            # Calmar ratio
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            
            # ML accuracy analysis
            ml_accuracy = self.calculate_ml_accuracy(trades, signals_generated)
            signal_strength_correlation = self.calculate_signal_correlation(trades, signals_generated)
            
            return BacktestResult(
                symbol=symbol,
                total_return=total_return,
                annualized_return=annualized_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_trades=len(trades),
                avg_trade_return=avg_trade_return,
                volatility=volatility,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                best_trade=best_trade,
                worst_trade=worst_trade,
                avg_holding_period=avg_holding_period,
                ml_accuracy=ml_accuracy,
                signal_strength_correlation=signal_strength_correlation
            )
            
        except Exception as e:
            print(f"❌ Backtest failed for {symbol}: {e}")
            return None
    
    def calculate_ml_accuracy(self, trades, signals):
        """Calculate how accurate ML signals were"""
        if not trades or not signals:
            return 0.0
        
        # Match trades with signals to see if signal direction matched trade outcome
        correct_predictions = 0
        total_predictions = 0
        
        for trade in trades:
            # Find signal closest to trade entry
            entry_date = trade['entry_date']
            closest_signal = min(signals, key=lambda s: abs((s['date'] - entry_date).total_seconds()))
            
            if abs((closest_signal['date'] - entry_date).total_seconds()) < 86400:  # Within 1 day
                signal_direction = 1 if closest_signal['signal_strength'] > 0 else -1
                trade_success = 1 if trade['return'] > 0 else -1
                
                if signal_direction == trade_success:
                    correct_predictions += 1
                total_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def calculate_signal_correlation(self, trades, signals):
        """Calculate correlation between signal strength and trade returns"""
        if not trades or not signals:
            return 0.0
        
        signal_strengths = []
        trade_returns = []
        
        for trade in trades:
            entry_date = trade['entry_date']
            closest_signal = min(signals, key=lambda s: abs((s['date'] - entry_date).total_seconds()))
            
            if abs((closest_signal['date'] - entry_date).total_seconds()) < 86400:
                signal_strengths.append(abs(closest_signal['signal_strength']))
                trade_returns.append(trade['return'])
        
        if len(signal_strengths) > 2:
            return np.corrcoef(signal_strengths, trade_returns)[0, 1]
        return 0.0
    
    def run_comprehensive_backtest(self, max_workers=8):
        """Run comprehensive backtest across all stocks"""
        
        print(f"🚀 COMPREHENSIVE 100+ STOCK BACKTESTING")
        print("=" * 55)
        print(f"Testing {len(self.available_symbols)} stocks with ML signals")
        print(f"Initial capital: ${self.initial_capital:,} per stock")
        print(f"Commission: {self.commission:.1%} per trade")
        print()
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all backtest tasks
            future_to_symbol = {
                executor.submit(self.backtest_single_stock, symbol): symbol
                for symbol in self.available_symbols
            }
            
            # Progress tracking
            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                result = future.result()
                
                completed += 1
                
                if result:
                    results.append(result)
                    print(f"✅ {symbol:5}: {result.total_return:+6.1%} return, "
                          f"{result.win_rate:5.1%} win rate, "
                          f"{result.total_trades:3d} trades, "
                          f"ML accuracy: {result.ml_accuracy:5.1%}")
                else:
                    print(f"❌ {symbol:5}: Backtest failed")
                
                # Progress update
                if completed % 20 == 0 or completed == len(self.available_symbols):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (len(self.available_symbols) - completed) / rate if rate > 0 else 0
                    
                    print(f"\n📊 Progress: {completed}/{len(self.available_symbols)} "
                          f"({completed/len(self.available_symbols)*100:.1f}%) "
                          f"- ETA: {eta:.0f}s")
                    print()
        
        total_time = time.time() - start_time
        
        print(f"\n⏱️ BACKTESTING COMPLETED")
        print(f"   Total time: {total_time:.1f} seconds")
        print(f"   Average time per stock: {total_time/len(self.available_symbols):.2f}s")
        print(f"   Successful backtests: {len(results)}/{len(self.available_symbols)}")
        
        return results
    
    def analyze_results(self, results: List[BacktestResult]) -> ComprehensiveBacktestSummary:
        """Comprehensive analysis of backtest results"""
        
        print(f"\n📊 COMPREHENSIVE RESULTS ANALYSIS")
        print("=" * 45)
        
        if not results:
            print("❌ No results to analyze")
            return None
        
        # Basic statistics
        total_returns = [r.total_return for r in results]
        annualized_returns = [r.annualized_return for r in results]
        sharpe_ratios = [r.sharpe_ratio for r in results if not np.isnan(r.sharpe_ratio)]
        win_rates = [r.win_rate for r in results]
        ml_accuracies = [r.ml_accuracy for r in results]
        
        print(f"📈 OVERALL PERFORMANCE")
        print("-" * 25)
        print(f"Stocks tested: {len(results)}")
        print(f"Average total return: {np.mean(total_returns):.1%}")
        print(f"Median total return: {np.median(total_returns):.1%}")
        print(f"Average annualized return: {np.mean(annualized_returns):.1%}")
        print(f"Average Sharpe ratio: {np.mean(sharpe_ratios):.2f}")
        print(f"Average win rate: {np.mean(win_rates):.1%}")
        print(f"Average ML accuracy: {np.mean(ml_accuracies):.1%}")
        
        # Performance distribution
        positive_returns = sum(1 for r in total_returns if r > 0)
        print(f"\n📊 RETURN DISTRIBUTION")
        print("-" * 22)
        print(f"Positive returns: {positive_returns}/{len(results)} ({positive_returns/len(results):.1%})")
        print(f"Returns > 10%: {sum(1 for r in total_returns if r > 0.1)} stocks")
        print(f"Returns > 20%: {sum(1 for r in total_returns if r > 0.2)} stocks")
        print(f"Returns < -10%: {sum(1 for r in total_returns if r < -0.1)} stocks")
        
        # Best and worst performers
        sorted_results = sorted(results, key=lambda r: r.total_return, reverse=True)
        best_performers = [r.symbol for r in sorted_results[:10]]
        worst_performers = [r.symbol for r in sorted_results[-10:]]
        
        print(f"\n🏆 TOP 10 PERFORMERS")
        print("-" * 20)
        for i, result in enumerate(sorted_results[:10]):
            print(f"{i+1:2d}. {result.symbol}: {result.total_return:+6.1%} "
                  f"(Sharpe: {result.sharpe_ratio:.2f}, ML Acc: {result.ml_accuracy:.1%})")
        
        print(f"\n📉 BOTTOM 10 PERFORMERS")
        print("-" * 25)
        for i, result in enumerate(sorted_results[-10:]):
            print(f"{i+1:2d}. {result.symbol}: {result.total_return:+6.1%} "
                  f"(Sharpe: {result.sharpe_ratio:.2f}, ML Acc: {result.ml_accuracy:.1%})")
        
        # Category analysis
        sector_performance = self.analyze_by_category(results)
        
        # ML effectiveness analysis
        print(f"\n🤖 ML EFFECTIVENESS ANALYSIS")
        print("-" * 28)
        high_ml_accuracy = [r for r in results if r.ml_accuracy > 0.6]
        print(f"Stocks with >60% ML accuracy: {len(high_ml_accuracy)}/{len(results)} ({len(high_ml_accuracy)/len(results):.1%})")
        
        if high_ml_accuracy:
            avg_return_high_ml = np.mean([r.total_return for r in high_ml_accuracy])
            print(f"Average return for high ML accuracy stocks: {avg_return_high_ml:.1%}")
        
        # Signal strength correlation
        correlations = [r.signal_strength_correlation for r in results if not np.isnan(r.signal_strength_correlation)]
        if correlations:
            avg_correlation = np.mean(correlations)
            print(f"Average signal strength correlation: {avg_correlation:.3f}")
        
        # Portfolio simulation
        portfolio_return = self.calculate_portfolio_performance(results)
        
        return ComprehensiveBacktestSummary(
            total_stocks_tested=len(results),
            successful_backtests=len(results),
            avg_total_return=np.mean(total_returns),
            avg_annualized_return=np.mean(annualized_returns),
            avg_sharpe_ratio=np.mean(sharpe_ratios),
            avg_win_rate=np.mean(win_rates),
            avg_ml_accuracy=np.mean(ml_accuracies),
            best_performers=best_performers,
            worst_performers=worst_performers,
            sector_performance=sector_performance,
            market_cap_performance={},  # To be implemented
            overall_portfolio_return=portfolio_return,
            portfolio_sharpe=0.0,  # To be calculated
            benchmark_comparison={}  # To be implemented
        )
    
    def analyze_by_category(self, results: List[BacktestResult]) -> Dict[str, Dict]:
        """Analyze results by stock category"""
        
        print(f"\n📈 SECTOR PERFORMANCE ANALYSIS")
        print("-" * 32)
        
        # Create symbol to category mapping
        symbol_to_category = {}
        for category, symbols in self.stock_categories.items():
            for symbol in symbols:
                symbol_to_category[symbol] = category
        
        # Group results by category
        category_results = {}
        for result in results:
            category = symbol_to_category.get(result.symbol, 'Other')
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(result)
        
        # Analyze each category
        sector_performance = {}
        for category, cat_results in category_results.items():
            if cat_results:
                avg_return = np.mean([r.total_return for r in cat_results])
                avg_sharpe = np.mean([r.sharpe_ratio for r in cat_results if not np.isnan(r.sharpe_ratio)])
                avg_win_rate = np.mean([r.win_rate for r in cat_results])
                avg_ml_accuracy = np.mean([r.ml_accuracy for r in cat_results])
                
                sector_performance[category] = {
                    'count': len(cat_results),
                    'avg_return': avg_return,
                    'avg_sharpe': avg_sharpe,
                    'avg_win_rate': avg_win_rate,
                    'avg_ml_accuracy': avg_ml_accuracy
                }
                
                print(f"{category:15}: {len(cat_results):2d} stocks, "
                      f"{avg_return:+6.1%} return, "
                      f"{avg_win_rate:5.1%} win rate, "
                      f"{avg_ml_accuracy:5.1%} ML accuracy")
        
        return sector_performance
    
    def calculate_portfolio_performance(self, results: List[BacktestResult]) -> float:
        """Calculate equal-weighted portfolio performance"""
        
        print(f"\n💼 PORTFOLIO ANALYSIS")
        print("-" * 20)
        
        # Equal weight portfolio
        total_returns = [r.total_return for r in results]
        portfolio_return = np.mean(total_returns)
        
        print(f"Equal-weighted portfolio return: {portfolio_return:.1%}")
        print(f"Best single stock: {max(total_returns):.1%}")
        print(f"Worst single stock: {min(total_returns):.1%}")
        print(f"Portfolio Sharpe advantage: Diversification reduces risk")
        
        return portfolio_return
    
    def save_results(self, results: List[BacktestResult], summary: ComprehensiveBacktestSummary):
        """Save detailed results to files"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save individual results
        results_data = [asdict(r) for r in results]
        results_file = self.results_dir / f'comprehensive_backtest_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save summary
        summary_file = self.results_dir / f'backtest_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(asdict(summary), f, indent=2, default=str)
        
        # Save as CSV for analysis
        df = pd.DataFrame(results_data)
        csv_file = self.results_dir / f'backtest_results_{timestamp}.csv'
        df.to_csv(csv_file, index=False)
        
        print(f"\n💾 RESULTS SAVED")
        print("-" * 16)
        print(f"Results: {results_file}")
        print(f"Summary: {summary_file}")
        print(f"CSV: {csv_file}")
        
        return results_file, summary_file, csv_file

def main():
    """Run comprehensive backtesting"""
    
    print("🧪 COMPREHENSIVE 100+ STOCK BACKTESTING FRAMEWORK")
    print("=" * 70)
    print("Validating ML signal accuracy across the full stock universe")
    print()
    
    backtester = Comprehensive100StockBacktester(initial_capital=10000, commission=0.005)
    
    try:
        # Initialize
        if not backtester.load_full_dataset():
            return False
        
        if not backtester.initialize_ml_system():
            return False
        
        # Run comprehensive backtest
        results = backtester.run_comprehensive_backtest(max_workers=6)
        
        if not results:
            print("❌ No successful backtests")
            return False
        
        # Analyze results
        summary = backtester.analyze_results(results)
        
        # Save results
        backtester.save_results(results, summary)
        
        # Final assessment
        print(f"\n🎯 FINAL ASSESSMENT")
        print("=" * 20)
        
        success_rate = len(results) / len(backtester.available_symbols)
        avg_return = summary.avg_total_return
        avg_ml_accuracy = summary.avg_ml_accuracy
        
        print(f"✅ Success Rate: {success_rate:.1%}")
        print(f"📊 Average Return: {avg_return:.1%}")
        print(f"🤖 Average ML Accuracy: {avg_ml_accuracy:.1%}")
        print(f"💼 Portfolio Return: {summary.overall_portfolio_return:.1%}")
        
        # Decision making insight
        if avg_ml_accuracy > 0.55 and avg_return > 0.05:
            print(f"\n🎉 STRONG SIGNAL ACCURACY VALIDATED")
            print("ML signals show predictive power across the stock universe")
            print("Recommendation: Proceed with live trading deployment")
        elif avg_ml_accuracy > 0.50:
            print(f"\n✅ MODERATE SIGNAL ACCURACY")
            print("ML signals show some predictive power")
            print("Recommendation: Consider refinement before full deployment")
        else:
            print(f"\n⚠️ SIGNAL ACCURACY NEEDS IMPROVEMENT")
            print("ML signals require optimization")
            print("Recommendation: Refine model before deployment")
        
        return True
        
    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()