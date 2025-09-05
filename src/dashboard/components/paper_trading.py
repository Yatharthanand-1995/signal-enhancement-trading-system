"""
Paper Trading Engine for Dashboard
Real-time paper trading simulation with signal integration and performance tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

@dataclass
class PaperPosition:
    """Paper trading position"""
    symbol: str
    shares: float
    entry_price: float
    entry_date: datetime
    entry_signal: str
    signal_strength: float
    confidence: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    stop_loss: float = 0.0
    days_held: int = 0
    # Enhanced target price tracking
    target_price: float = 0.0
    trailing_stop_pct: float = 0.0
    trailing_stop_price: float = 0.0
    entry_reason: str = ""
    exit_target_reason: str = ""
    profit_target_pct: float = 0.20  # 20% profit target default
    max_loss_pct: float = 0.10       # 10% max loss default
    price_alerts_enabled: bool = True
    intraday_high: float = 0.0
    intraday_low: float = 0.0
    last_price_update: Optional[datetime] = None
    expected_holding_days: int = 0
    signal_components: Dict = None
    
    def update_price(self, current_price: float):
        """Update position with current price"""
        self.current_price = current_price
        self.unrealized_pnl = (current_price - self.entry_price) * self.shares
        self.days_held = (datetime.now() - self.entry_date).days
        self.last_price_update = datetime.now()
        
        # Update intraday high/low tracking
        if self.intraday_high == 0 or current_price > self.intraday_high:
            self.intraday_high = current_price
        if self.intraday_low == 0 or current_price < self.intraday_low:
            self.intraday_low = current_price
        
        # Update trailing stop if enabled
        if self.trailing_stop_pct > 0:
            self.update_trailing_stop()
        
    def get_position_value(self) -> float:
        """Get current position value"""
        return self.shares * self.current_price
        
    def get_return_pct(self) -> float:
        """Get position return percentage"""
        if self.entry_price == 0:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100
    
    def calculate_target_price(self) -> float:
        """Calculate target price based on signal strength and confidence"""
        if self.target_price > 0:
            return self.target_price
        
        # Base target on signal strength and confidence
        base_target_pct = self.profit_target_pct
        
        # Adjust based on signal strength (0.0 to 1.0)
        strength_multiplier = max(0.5, min(2.0, self.signal_strength))
        
        # Adjust based on confidence (0.6 to 1.0 typical range)
        confidence_multiplier = max(0.8, min(1.5, self.confidence))
        
        # Calculate target price
        adjusted_target_pct = base_target_pct * strength_multiplier * confidence_multiplier
        target_price = self.entry_price * (1 + adjusted_target_pct)
        
        self.target_price = target_price
        return target_price
    
    def update_trailing_stop(self):
        """Update trailing stop price"""
        if self.trailing_stop_pct <= 0:
            return
        
        new_trailing_stop = self.current_price * (1 - self.trailing_stop_pct)
        
        # Only move trailing stop up (for long positions)
        if new_trailing_stop > self.trailing_stop_price:
            self.trailing_stop_price = new_trailing_stop
    
    def check_exit_conditions(self) -> Tuple[bool, str]:
        """Check if position should be exited
        
        Returns:
            Tuple[bool, str]: (should_exit, reason)
        """
        if self.current_price <= 0:
            return False, ""
        
        # Check stop loss
        if self.current_price <= self.stop_loss:
            return True, "STOP_LOSS"
        
        # Check trailing stop
        if self.trailing_stop_price > 0 and self.current_price <= self.trailing_stop_price:
            return True, "TRAILING_STOP"
        
        # Check target price
        if self.target_price > 0 and self.current_price >= self.target_price:
            return True, "TARGET_REACHED"
        
        # Check maximum loss
        loss_pct = (self.entry_price - self.current_price) / self.entry_price
        if loss_pct >= self.max_loss_pct:
            return True, "MAX_LOSS"
        
        return False, ""
    
    def get_target_progress(self) -> float:
        """Get progress toward target as percentage (0.0 to 1.0+)"""
        if self.target_price <= self.entry_price:
            return 0.0
        
        progress = (self.current_price - self.entry_price) / (self.target_price - self.entry_price)
        return max(0.0, progress)
    
    def get_price_metrics(self) -> Dict:
        """Get comprehensive price metrics for this position"""
        return {
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'trailing_stop': self.trailing_stop_price,
            'intraday_high': self.intraday_high,
            'intraday_low': self.intraday_low,
            'unrealized_pnl': self.unrealized_pnl,
            'return_pct': self.get_return_pct(),
            'target_progress': self.get_target_progress(),
            'days_held': self.days_held,
            'max_favorable_excursion': max(0, self.intraday_high - self.entry_price) * self.shares,
            'max_adverse_excursion': min(0, self.intraday_low - self.entry_price) * self.shares
        }

@dataclass
class PaperTrade:
    """Paper trade record"""
    symbol: str
    action: str  # 'BUY' or 'SELL'
    shares: float
    price: float
    timestamp: datetime
    signal_info: Dict
    pnl: float = 0.0
    # Enhanced trade logging
    entry_target_price: float = 0.0
    exit_target_price: float = 0.0
    actual_vs_target_slippage: float = 0.0
    signal_components: Dict = None
    market_conditions: Dict = None
    trade_duration_planned: int = 0
    trade_duration_actual: int = 0
    exit_trigger: str = ""  # "SIGNAL", "TARGET", "STOP_LOSS", "TRAILING_STOP", "MANUAL"
    commissions_paid: float = 0.0
    unrealized_pnl_at_entry: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    confidence_at_entry: float = 0.0
    signal_strength_at_entry: float = 0.0
    entry_reason: str = ""
    market_regime_at_entry: str = ""
    
class PaperTradingEngine:
    """
    Paper trading engine for real-time signal validation
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, PaperPosition] = {}
        self.trade_history: List[PaperTrade] = []
        self.daily_values: List[Dict] = []
        
        # Trading parameters
        self.max_positions = 10
        self.max_position_pct = 0.10  # 10% max per position
        self.transaction_cost = 0.001  # 0.1% transaction cost
        self.min_confidence = 0.6  # Minimum signal confidence for trades
        self.stop_loss_pct = 0.10  # 10% stop loss
        self.default_profit_target_pct = 0.20  # 20% profit target
        self.trailing_stop_enabled = True
        self.trailing_stop_pct = 0.05  # 5% trailing stop
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_value = initial_capital
        
        # Load saved state
        self.load_state()
        
    def get_data_file_path(self) -> str:
        """Get path for saving paper trading data"""
        dashboard_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(dashboard_dir, '..', 'paper_trading_data.json')
        
    def save_state(self):
        """Save paper trading state to file"""
        try:
            state = {
                'initial_capital': self.initial_capital,
                'cash': self.cash,
                'positions': {
                    symbol: {
                        **asdict(position),
                        'entry_date': position.entry_date.isoformat(),
                    } for symbol, position in self.positions.items()
                },
                'trade_history': [
                    {
                        **asdict(trade),
                        'timestamp': trade.timestamp.isoformat(),
                    } for trade in self.trade_history
                ],
                'daily_values': self.daily_values,
                'performance_metrics': {
                    'total_trades': self.total_trades,
                    'winning_trades': self.winning_trades,
                    'total_pnl': self.total_pnl,
                    'max_drawdown': self.max_drawdown,
                    'peak_value': self.peak_value
                }
            }
            
            with open(self.get_data_file_path(), 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving paper trading state: {e}")
    
    def load_state(self):
        """Load paper trading state from file"""
        try:
            if os.path.exists(self.get_data_file_path()):
                with open(self.get_data_file_path(), 'r') as f:
                    state = json.load(f)
                
                self.cash = state.get('cash', self.initial_capital)
                
                # Load positions
                for symbol, pos_data in state.get('positions', {}).items():
                    pos_data['entry_date'] = datetime.fromisoformat(pos_data['entry_date'])
                    self.positions[symbol] = PaperPosition(**pos_data)
                
                # Load trade history
                for trade_data in state.get('trade_history', []):
                    trade_data['timestamp'] = datetime.fromisoformat(trade_data['timestamp'])
                    self.trade_history.append(PaperTrade(**trade_data))
                
                self.daily_values = state.get('daily_values', [])
                
                # Load performance metrics
                metrics = state.get('performance_metrics', {})
                self.total_trades = metrics.get('total_trades', 0)
                self.winning_trades = metrics.get('winning_trades', 0)
                self.total_pnl = metrics.get('total_pnl', 0.0)
                self.max_drawdown = metrics.get('max_drawdown', 0.0)
                self.peak_value = metrics.get('peak_value', self.initial_capital)
                
        except Exception as e:
            logger.error(f"Error loading paper trading state: {e}")
    
    def reset_portfolio(self):
        """Reset paper trading portfolio"""
        self.cash = self.initial_capital
        self.positions = {}
        self.trade_history = []
        self.daily_values = []
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_value = self.initial_capital
        self.save_state()
    
    def get_current_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        positions_value = sum(pos.get_position_value() for pos in self.positions.values())
        return self.cash + positions_value
    
    def calculate_position_size(self, price: float, confidence: float) -> int:
        """Calculate optimal position size"""
        portfolio_value = self.get_current_portfolio_value()
        
        # Base position size scaled by confidence
        base_allocation = portfolio_value * self.max_position_pct * confidence
        
        # Account for transaction costs
        available_cash = self.cash * 0.95  # Keep 5% cash buffer
        max_allocation = min(base_allocation, available_cash)
        
        # Calculate shares
        shares = int(max_allocation / price)
        
        return max(0, shares)
    
    def can_buy(self, symbol: str, shares: int, price: float) -> bool:
        """Check if buy order can be executed"""
        if symbol in self.positions:
            return False  # Already have position
        
        if len(self.positions) >= self.max_positions:
            return False  # Max positions reached
        
        cost = shares * price * (1 + self.transaction_cost)
        return cost <= self.cash
    
    def execute_buy_order(self, symbol: str, signal_data: Dict) -> bool:
        """Execute paper buy order"""
        try:
            # Get current price
            ticker = yf.Ticker(symbol)
            current_price = ticker.history(period='1d')['Close'].iloc[-1]
            
            confidence = signal_data.get('confidence', 0.0)
            
            # Check minimum confidence
            if confidence < self.min_confidence:
                return False
            
            # Calculate position size
            shares = self.calculate_position_size(current_price, confidence)
            
            if shares == 0 or not self.can_buy(symbol, shares, current_price):
                return False
            
            # Execute trade
            cost = shares * current_price * (1 + self.transaction_cost)
            self.cash -= cost
            
            # Create position with enhanced tracking
            stop_loss = current_price * (1 - self.stop_loss_pct)
            
            position = PaperPosition(
                symbol=symbol,
                shares=shares,
                entry_price=current_price,
                entry_date=datetime.now(),
                entry_signal=signal_data.get('direction', 'UNKNOWN'),
                signal_strength=signal_data.get('strength', 0.0),
                confidence=confidence,
                current_price=current_price,
                stop_loss=stop_loss,
                profit_target_pct=self.default_profit_target_pct,
                trailing_stop_pct=self.trailing_stop_pct if self.trailing_stop_enabled else 0.0,
                entry_reason=f"Signal: {signal_data.get('direction', 'UNKNOWN')} (Confidence: {confidence:.1%})",
                signal_components=signal_data.get('components', {}),
                intraday_high=current_price,
                intraday_low=current_price,
                last_price_update=datetime.now()
            )
            
            # Calculate target price
            position.calculate_target_price()
            
            self.positions[symbol] = position
            
            # Record enhanced trade
            trade = PaperTrade(
                symbol=symbol,
                action='BUY',
                shares=shares,
                price=current_price,
                timestamp=datetime.now(),
                signal_info=signal_data,
                entry_target_price=position.target_price,
                signal_components=signal_data.get('components', {}),
                confidence_at_entry=confidence,
                signal_strength_at_entry=signal_data.get('strength', 0.0),
                entry_reason=position.entry_reason,
                commissions_paid=cost - (shares * current_price),
                trade_duration_planned=position.expected_holding_days
            )
            
            self.trade_history.append(trade)
            self.total_trades += 1
            
            self.save_state()
            logger.info(f"Paper trade executed: BUY {shares} shares of {symbol} at ${current_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing buy order for {symbol}: {e}")
            return False
    
    def execute_sell_order(self, symbol: str, reason: str = "SIGNAL") -> bool:
        """Execute paper sell order"""
        try:
            if symbol not in self.positions:
                return False
            
            position = self.positions[symbol]
            
            # Get current price
            ticker = yf.Ticker(symbol)
            current_price = ticker.history(period='1d')['Close'].iloc[-1]
            
            # Calculate proceeds
            proceeds = position.shares * current_price * (1 - self.transaction_cost)
            pnl = proceeds - (position.shares * position.entry_price * (1 + self.transaction_cost))
            
            self.cash += proceeds
            self.total_pnl += pnl
            
            if pnl > 0:
                self.winning_trades += 1
            
            # Record enhanced sell trade
            position_metrics = position.get_price_metrics()
            trade = PaperTrade(
                symbol=symbol,
                action='SELL',
                shares=position.shares,
                price=current_price,
                timestamp=datetime.now(),
                signal_info={'reason': reason},
                pnl=pnl,
                exit_trigger=reason,
                exit_target_price=position.target_price,
                trade_duration_actual=position.days_held,
                max_favorable_excursion=position_metrics['max_favorable_excursion'],
                max_adverse_excursion=position_metrics['max_adverse_excursion'],
                commissions_paid=position.shares * current_price * self.transaction_cost
            )
            
            self.trade_history.append(trade)
            self.total_trades += 1
            
            # Remove position
            del self.positions[symbol]
            
            self.save_state()
            logger.info(f"Paper trade executed: SELL {position.shares} shares of {symbol} at ${current_price:.2f}, P&L: ${pnl:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing sell order for {symbol}: {e}")
            return False
    
    def update_positions(self, check_exits: bool = True):
        """Update all positions with current prices and optionally check exit conditions"""
        if not self.positions:
            return
        
        symbols = list(self.positions.keys())
        try:
            # Batch fetch current prices
            tickers = yf.Tickers(' '.join(symbols))
            
            positions_to_exit = []
            
            for symbol in symbols:
                try:
                    current_price = tickers.tickers[symbol].history(period='1d')['Close'].iloc[-1]
                    self.positions[symbol].update_price(current_price)
                    
                    # Check all exit conditions if enabled
                    if check_exits:
                        position = self.positions[symbol]
                        should_exit, reason = position.check_exit_conditions()
                        
                        if should_exit:
                            positions_to_exit.append((symbol, reason))
                        
                except Exception as e:
                    logger.error(f"Error updating price for {symbol}: {e}")
            
            # Execute exits after price updates to avoid dictionary changes during iteration
            for symbol, reason in positions_to_exit:
                try:
                    success = self.execute_sell_order(symbol, reason)
                    if success:
                        logger.info(f"Automated exit executed: {symbol} ({reason})")
                except Exception as e:
                    logger.error(f"Error executing automated exit for {symbol}: {e}")
        
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def process_signals(self, signals_data: Dict):
        """Process new signals and execute trades"""
        if not signals_data:
            return
        
        for symbol, signal_info in signals_data.items():
            try:
                direction = signal_info.get('direction', 'NEUTRAL')
                confidence = signal_info.get('confidence', 0.0)
                
                # Buy signals
                if direction in ['BUY', 'STRONG_BUY'] and symbol not in self.positions:
                    self.execute_buy_order(symbol, signal_info)
                
                # Sell signals
                elif direction in ['SELL', 'STRONG_SELL'] and symbol in self.positions:
                    self.execute_sell_order(symbol, "SELL_SIGNAL")
                    
            except Exception as e:
                logger.error(f"Error processing signal for {symbol}: {e}")
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        current_value = self.get_current_portfolio_value()
        total_return = ((current_value - self.initial_capital) / self.initial_capital) * 100
        
        # Update peak and drawdown
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        current_drawdown = ((self.peak_value - current_value) / self.peak_value) * 100
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
        
        return {
            'current_value': current_value,
            'total_return': total_return,
            'total_pnl': self.total_pnl,
            'win_rate': win_rate,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'max_drawdown': self.max_drawdown,
            'cash': self.cash,
            'positions_count': len(self.positions)
        }
    
    def get_positions_summary(self) -> pd.DataFrame:
        """Get current positions as DataFrame"""
        if not self.positions:
            return pd.DataFrame()
        
        data = []
        for symbol, position in self.positions.items():
            data.append({
                'Symbol': symbol,
                'Shares': position.shares,
                'Entry Price': position.entry_price,
                'Current Price': position.current_price,
                'Entry Date': position.entry_date.strftime('%Y-%m-%d'),
                'Days Held': position.days_held,
                'Return %': position.get_return_pct(),
                'P&L': position.unrealized_pnl,
                'Value': position.get_position_value(),
                'Signal': position.entry_signal,
                'Confidence': position.confidence
            })
        
        return pd.DataFrame(data)
    
    def get_trade_history_df(self, limit: int = 50) -> pd.DataFrame:
        """Get recent trade history as DataFrame"""
        if not self.trade_history:
            return pd.DataFrame()
        
        # Get recent trades
        recent_trades = self.trade_history[-limit:] if len(self.trade_history) > limit else self.trade_history
        
        data = []
        for trade in recent_trades:
            data.append({
                'Timestamp': trade.timestamp.strftime('%Y-%m-%d %H:%M'),
                'Symbol': trade.symbol,
                'Action': trade.action,
                'Shares': trade.shares,
                'Price': trade.price,
                'P&L': trade.pnl if trade.action == 'SELL' else None,
                'Signal Info': str(trade.signal_info.get('direction', 'N/A'))
            })
        
        return pd.DataFrame(data)