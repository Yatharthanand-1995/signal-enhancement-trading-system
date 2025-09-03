#!/usr/bin/env python3
"""
Portfolio Simulation Engine for Backtesting
Executes trades based on signals and tracks performance across market regimes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class Position:
    """Represents a single position in the portfolio"""
    
    def __init__(self, symbol: str, shares: float, entry_price: float, entry_date: pd.Timestamp, 
                 signal_strength: str, confidence: float, stop_loss: float = None):
        self.symbol = symbol
        self.shares = shares
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.signal_strength = signal_strength
        self.confidence = confidence
        self.stop_loss = stop_loss or entry_price * 0.90  # 10% stop loss default
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        self.days_held = 0
        
    def update_price(self, current_price: float, current_date: pd.Timestamp):
        """Update position with current market price"""
        self.current_price = current_price
        self.unrealized_pnl = (current_price - self.entry_price) * self.shares
        self.days_held = (current_date - self.entry_date).days
        
    def get_position_value(self) -> float:
        """Get current position value"""
        return self.shares * self.current_price
        
    def should_stop_out(self) -> bool:
        """Check if position should be stopped out"""
        return self.current_price <= self.stop_loss

class PortfolioSimulator:
    """
    Portfolio simulation engine that executes trades based on signals
    and tracks performance across different market regimes
    """
    
    def __init__(self, initial_capital: float = 1_000_000, max_positions: int = 20,
                 transaction_cost: float = 0.001, max_position_pct: float = 0.05):
        
        self.initial_capital = initial_capital
        self.max_positions = max_positions  
        self.transaction_cost = transaction_cost
        self.max_position_pct = max_position_pct
        
        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.portfolio_value = initial_capital
        self.total_return = 0.0
        
        # Performance tracking
        self.daily_values = []
        self.trade_history = []
        self.performance_by_regime = {}
        
        logger.info(f"Portfolio initialized with ${initial_capital:,.0f}")
    
    def calculate_position_size(self, price: float, confidence: float, volatility: float = 0.2) -> int:
        """
        Calculate optimal position size based on price, confidence, and volatility
        Uses Kelly Criterion-inspired approach with risk management overlays
        """
        # Base position size as % of portfolio
        base_position_pct = self.max_position_pct * confidence
        
        # Volatility adjustment (reduce size for volatile stocks)
        vol_adjustment = min(1.0, 0.2 / max(volatility, 0.1))
        
        # Final position size
        position_value = self.portfolio_value * base_position_pct * vol_adjustment
        
        # Convert to shares
        shares = int(position_value / price)
        
        # Ensure we don't exceed available cash
        max_affordable_shares = int((self.cash * 0.95) / price)  # 5% cash buffer
        
        return min(shares, max_affordable_shares)
    
    def execute_buy_order(self, symbol: str, price: float, signal_data: Dict, 
                         current_date: pd.Timestamp) -> bool:
        """
        Execute a buy order for the given symbol
        """
        # Check if we already have a position
        if symbol in self.positions:
            logger.debug(f"Already have position in {symbol}, skipping buy")
            return False
        
        # Check if we've reached max positions
        if len(self.positions) >= self.max_positions:
            logger.debug(f"Max positions ({self.max_positions}) reached, skipping buy of {symbol}")
            return False
        
        # Calculate position size
        confidence = signal_data.get('confidence', 0.5)
        volatility = signal_data.get('volatility', 0.2)
        shares = self.calculate_position_size(price, confidence, volatility)
        
        if shares <= 0:
            logger.debug(f"Position size too small for {symbol}, skipping")
            return False
        
        # Calculate total cost including transaction costs
        gross_cost = shares * price
        transaction_cost = gross_cost * self.transaction_cost
        total_cost = gross_cost + transaction_cost
        
        if total_cost > self.cash:
            logger.debug(f"Insufficient cash for {symbol}: need ${total_cost:,.0f}, have ${self.cash:,.0f}")
            return False
        
        # Execute the trade
        self.cash -= total_cost
        
        # Create position
        stop_loss = price * 0.90  # 10% stop loss
        position = Position(
            symbol=symbol,
            shares=shares,
            entry_price=price,
            entry_date=current_date,
            signal_strength=signal_data.get('strength', 'Moderate'),
            confidence=confidence,
            stop_loss=stop_loss
        )
        
        self.positions[symbol] = position
        
        # Record trade
        trade_record = {
            'date': current_date,
            'symbol': symbol,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'total_cost': total_cost,
            'transaction_cost': transaction_cost,
            'signal': signal_data.get('signal', 'BUY'),
            'confidence': confidence,
            'market_regime': signal_data.get('market_regime', 'UNKNOWN'),
            'portfolio_value_before': self.portfolio_value
        }
        self.trade_history.append(trade_record)
        
        logger.info(f"BUY: {shares} shares of {symbol} at ${price:.2f} (Total: ${total_cost:,.0f})")
        return True
    
    def execute_sell_order(self, symbol: str, price: float, reason: str, 
                          current_date: pd.Timestamp) -> bool:
        """
        Execute a sell order for the given symbol
        """
        if symbol not in self.positions:
            logger.debug(f"No position in {symbol} to sell")
            return False
        
        position = self.positions[symbol]
        
        # Calculate proceeds
        gross_proceeds = position.shares * price
        transaction_cost = gross_proceeds * self.transaction_cost
        net_proceeds = gross_proceeds - transaction_cost
        
        # Add proceeds to cash
        self.cash += net_proceeds
        
        # Calculate P&L
        total_cost = position.shares * position.entry_price
        realized_pnl = net_proceeds - total_cost
        realized_pnl_pct = realized_pnl / total_cost * 100
        
        # Record trade
        trade_record = {
            'date': current_date,
            'symbol': symbol,
            'action': 'SELL',
            'shares': position.shares,
            'price': price,
            'net_proceeds': net_proceeds,
            'transaction_cost': transaction_cost,
            'entry_price': position.entry_price,
            'entry_date': position.entry_date,
            'days_held': (current_date - position.entry_date).days,
            'realized_pnl': realized_pnl,
            'realized_pnl_pct': realized_pnl_pct,
            'reason': reason,
            'signal_strength': position.signal_strength,
            'confidence': position.confidence,
            'portfolio_value_before': self.portfolio_value
        }
        self.trade_history.append(trade_record)
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"SELL: {position.shares} shares of {symbol} at ${price:.2f} "
                   f"(P&L: ${realized_pnl:,.0f} / {realized_pnl_pct:.1f}%) - {reason}")
        return True
    
    def update_portfolio(self, current_prices: Dict[str, float], current_date: pd.Timestamp):
        """
        Update portfolio with current market prices
        """
        # Update positions with current prices
        positions_value = 0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.update_price(current_prices[symbol], current_date)
                positions_value += position.get_position_value()
        
        # Calculate total portfolio value
        self.portfolio_value = self.cash + positions_value
        self.total_return = (self.portfolio_value / self.initial_capital - 1) * 100
        
        # Record daily value
        daily_record = {
            'date': current_date,
            'cash': self.cash,
            'positions_value': positions_value,
            'portfolio_value': self.portfolio_value,
            'total_return': self.total_return,
            'num_positions': len(self.positions),
            'cash_pct': self.cash / self.portfolio_value * 100
        }
        self.daily_values.append(daily_record)
    
    def check_stop_losses(self, current_prices: Dict[str, float], current_date: pd.Timestamp):
        """
        Check and execute stop loss orders
        """
        symbols_to_sell = []
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                if position.should_stop_out():
                    symbols_to_sell.append((symbol, current_price))
        
        # Execute stop loss sales
        for symbol, price in symbols_to_sell:
            self.execute_sell_order(symbol, price, "STOP_LOSS", current_date)
    
    def rebalance_portfolio(self, new_signals: pd.DataFrame, current_prices: Dict[str, float],
                           current_date: pd.Timestamp):
        """
        Rebalance portfolio based on new signals
        """
        # Get buy and sell signals
        buy_signals = new_signals[new_signals['signal'].isin(['BUY', 'STRONG_BUY'])]
        sell_signals = new_signals[new_signals['signal'].isin(['SELL', 'STRONG_SELL'])]
        
        # Execute sell orders first (to free up capital)
        current_positions = set(self.positions.keys())
        
        # Sell positions with sell signals
        for _, signal in sell_signals.iterrows():
            symbol = signal['symbol']
            if symbol in current_positions and symbol in current_prices:
                self.execute_sell_order(symbol, current_prices[symbol], "SELL_SIGNAL", current_date)
        
        # Check for positions without signals (HOLD or no signal) - consider selling weak positions
        signals_dict = {row['symbol']: row for _, row in new_signals.iterrows()}
        for symbol in list(current_positions):
            if symbol not in signals_dict:  # No signal for this position
                # Consider selling if held for more than 60 days with poor performance
                position = self.positions.get(symbol)
                if position and position.days_held > 60 and position.unrealized_pnl < 0:
                    if symbol in current_prices:
                        self.execute_sell_order(symbol, current_prices[symbol], "NO_SIGNAL_CLEANUP", current_date)
        
        # Execute buy orders (sorted by confidence)
        buy_signals = buy_signals.sort_values('confidence', ascending=False)
        
        for _, signal in buy_signals.iterrows():
            symbol = signal['symbol']
            if symbol in current_prices and symbol not in self.positions:
                signal_data = {
                    'signal': signal['signal'],
                    'strength': signal['strength'],
                    'confidence': signal['confidence'],
                    'market_regime': signal['market_regime'],
                    'volatility': signal.get('volatility', 0.2)
                }
                self.execute_buy_order(symbol, current_prices[symbol], signal_data, current_date)
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        if not self.daily_values:
            return {}
        
        daily_df = pd.DataFrame(self.daily_values)
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df = daily_df.set_index('date')
        
        # Calculate daily returns
        daily_df['daily_return'] = daily_df['portfolio_value'].pct_change()
        returns = daily_df['daily_return'].dropna()
        
        # Basic metrics
        total_return = (self.portfolio_value / self.initial_capital - 1) * 100
        annualized_return = ((self.portfolio_value / self.initial_capital) ** (252 / len(returns)) - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = ((cumulative - rolling_max) / rolling_max) * 100
        max_drawdown = drawdown.min()
        
        # Trading metrics
        trades_df = pd.DataFrame([t for t in self.trade_history if t['action'] == 'SELL'])
        
        if len(trades_df) > 0:
            win_rate = (trades_df['realized_pnl'] > 0).mean() * 100
            avg_win = trades_df[trades_df['realized_pnl'] > 0]['realized_pnl_pct'].mean()
            avg_loss = trades_df[trades_df['realized_pnl'] <= 0]['realized_pnl_pct'].mean()
            avg_holding_days = trades_df['days_held'].mean()
            total_trades = len(trades_df)
        else:
            win_rate = avg_win = avg_loss = avg_holding_days = total_trades = 0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'current_cash': self.cash,
            'current_positions_value': self.portfolio_value - self.cash,
            'num_positions': len(self.positions),
            'win_rate': win_rate,
            'avg_win': avg_win or 0,
            'avg_loss': avg_loss or 0,
            'avg_holding_days': avg_holding_days,
            'total_trades': total_trades,
            'final_portfolio_value': self.portfolio_value
        }
        
        return metrics
    
    def get_performance_by_regime(self, signals_data: pd.DataFrame = None) -> Dict:
        """
        Calculate performance metrics broken down by market regime
        """
        if not self.trade_history:
            return {}
        
        trades_df = pd.DataFrame(self.trade_history)
        sell_trades = trades_df[trades_df['action'] == 'SELL'].copy()
        
        if len(sell_trades) == 0:
            return {}
        
        # Group by market regime (from entry date)
        regime_performance = {}
        
        for regime in sell_trades['market_regime'].unique():
            regime_trades = sell_trades[sell_trades['market_regime'] == regime]
            
            if len(regime_trades) > 0:
                regime_performance[regime] = {
                    'total_trades': len(regime_trades),
                    'win_rate': (regime_trades['realized_pnl'] > 0).mean() * 100,
                    'avg_return': regime_trades['realized_pnl_pct'].mean(),
                    'total_pnl': regime_trades['realized_pnl'].sum(),
                    'avg_holding_days': regime_trades['days_held'].mean(),
                    'best_trade': regime_trades['realized_pnl_pct'].max(),
                    'worst_trade': regime_trades['realized_pnl_pct'].min()
                }
        
        return regime_performance
    
    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report of portfolio performance
        """
        metrics = self.get_performance_metrics()
        regime_performance = self.get_performance_by_regime()
        
        report = f"""
=== PORTFOLIO PERFORMANCE SUMMARY ===
Initial Capital: ${self.initial_capital:,.0f}
Final Portfolio Value: ${self.portfolio_value:,.0f}
Total Return: {metrics.get('total_return', 0):.2f}%
Annualized Return: {metrics.get('annualized_return', 0):.2f}%

=== RISK METRICS ===
Volatility: {metrics.get('volatility', 0):.2f}%
Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
Maximum Drawdown: {metrics.get('max_drawdown', 0):.2f}%

=== TRADING METRICS ===
Total Trades: {metrics.get('total_trades', 0)}
Win Rate: {metrics.get('win_rate', 0):.1f}%
Average Win: {metrics.get('avg_win', 0):.2f}%
Average Loss: {metrics.get('avg_loss', 0):.2f}%
Average Holding Period: {metrics.get('avg_holding_days', 0):.0f} days

=== CURRENT PORTFOLIO ===
Cash: ${metrics.get('current_cash', 0):,.0f} ({metrics.get('current_cash', 0)/self.portfolio_value*100:.1f}%)
Positions Value: ${metrics.get('current_positions_value', 0):,.0f}
Number of Positions: {metrics.get('num_positions', 0)}

=== PERFORMANCE BY MARKET REGIME ===
"""
        
        for regime, perf in regime_performance.items():
            report += f"""
{regime}:
  Trades: {perf['total_trades']}
  Win Rate: {perf['win_rate']:.1f}%
  Avg Return: {perf['avg_return']:.2f}%
  Total P&L: ${perf['total_pnl']:,.0f}
  Avg Hold Days: {perf['avg_holding_days']:.0f}
"""
        
        return report

def main():
    """
    Test the portfolio simulator with sample data
    """
    # Create portfolio simulator
    portfolio = PortfolioSimulator(initial_capital=1_000_000, max_positions=20)
    
    # Sample test (would be replaced with real backtesting)
    print("Portfolio Simulator initialized successfully!")
    print(f"Initial capital: ${portfolio.initial_capital:,.0f}")
    print(f"Max positions: {portfolio.max_positions}")
    print(f"Transaction cost: {portfolio.transaction_cost:.3f}")

if __name__ == "__main__":
    main()