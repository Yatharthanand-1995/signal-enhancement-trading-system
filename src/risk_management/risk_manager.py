"""
Adaptive Risk Management System
Dynamic position sizing with regime-dependent parameters and Kelly Criterion optimization
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from config.config import config

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: int
    entry_price: float
    entry_date: datetime
    stop_loss: float
    profit_target: float
    signal_strength: float
    regime_at_entry: str
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    days_held: int = 0
    
    def update_price(self, current_price: float) -> None:
        """Update current price and calculate unrealized P&L"""
        self.current_price = current_price
        self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        self.days_held = (datetime.now() - self.entry_date).days

@dataclass
class RiskMetrics:
    """Risk metrics for the portfolio"""
    total_value: float
    total_exposure: float
    max_drawdown: float
    current_drawdown: float
    var_95: float
    expected_shortfall: float
    sharpe_ratio: float
    volatility: float
    correlation_risk: float
    concentration_risk: float

class AdaptiveRiskManager:
    """Advanced risk management system with regime-dependent parameters"""
    
    def __init__(self, initial_capital: float, risk_config=None):
        if risk_config is None:
            risk_config = config.trading
            
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.config = risk_config
        
        # Current positions
        self.positions: Dict[str, Position] = {}
        self.cash = initial_capital
        
        # Risk tracking
        self.equity_curve = [initial_capital]
        self.drawdown_history = [0.0]
        self.daily_returns = []
        
        # Performance tracking
        self.peak_value = initial_capital
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Regime-dependent parameters
        self.regime_adjustments = {
            'Low_Volatility': {
                'position_size_multiplier': 1.2,
                'stop_loss_multiplier': 1.0,
                'max_positions': 12,
                'max_single_position': 0.25
            },
            'High_Volatility': {
                'position_size_multiplier': 0.7,
                'stop_loss_multiplier': 0.8,
                'max_positions': 8,
                'max_single_position': 0.15
            },
            'Bull_Market': {
                'position_size_multiplier': 1.1,
                'stop_loss_multiplier': 1.1,
                'max_positions': 10,
                'max_single_position': 0.22
            },
            'Bear_Market': {
                'position_size_multiplier': 0.8,
                'stop_loss_multiplier': 0.9,
                'max_positions': 6,
                'max_single_position': 0.18
            },
            'Sideways_Market': {
                'position_size_multiplier': 0.9,
                'stop_loss_multiplier': 0.95,
                'max_positions': 8,
                'max_single_position': 0.20
            }
        }
        
        # Correlation matrix for risk management
        self.correlation_matrix = {}
        self.sector_exposure = {}
        
    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float,
                                signal_strength: float = 0.7) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        if avg_loss >= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        # Standard Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = abs(avg_win / avg_loss)
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Adjust Kelly based on signal strength
        signal_adjustment = 0.5 + 0.5 * signal_strength  # Scale between 0.5 and 1.0
        adjusted_kelly = kelly_fraction * signal_adjustment
        
        # Apply fractional Kelly (typically 25-50% of full Kelly)
        conservative_kelly = adjusted_kelly * 0.25
        
        # Ensure reasonable bounds
        return max(0.01, min(conservative_kelly, 0.15))  # Between 1% and 15%
    
    def calculate_position_size(self, symbol: str, signal_strength: float,
                              confidence: float, current_price: float,
                              atr: float, regime: str = 'Low_Volatility') -> Dict[str, Any]:
        """Calculate optimal position size based on multiple factors"""
        
        # Get regime adjustments
        regime_params = self.regime_adjustments.get(regime, 
                                                   self.regime_adjustments['Low_Volatility'])
        
        # Base position size calculation
        available_capital = self.current_capital
        base_allocation = self.config.max_position_size  # Default max per position
        
        # Adjust for regime
        regime_multiplier = regime_params['position_size_multiplier']
        max_single_position = regime_params['max_single_position']
        
        # Calculate base position size
        base_position_value = available_capital * min(base_allocation, max_single_position)
        
        # Adjust for signal strength and confidence
        signal_factor = 0.5 + 0.5 * signal_strength  # 0.5 to 1.0
        confidence_factor = 0.6 + 0.4 * confidence    # 0.6 to 1.0
        
        adjusted_position_value = base_position_value * signal_factor * confidence_factor * regime_multiplier
        
        # Volatility adjustment using ATR
        if atr > 0 and current_price > 0:
            volatility_ratio = atr / current_price
            if volatility_ratio > 0.03:  # High volatility (>3% daily ATR)
                volatility_adjustment = 0.8
            elif volatility_ratio < 0.015:  # Low volatility (<1.5% daily ATR)
                volatility_adjustment = 1.1
            else:
                volatility_adjustment = 1.0
                
            adjusted_position_value *= volatility_adjustment
        
        # Portfolio heat adjustment (reduce size if portfolio is stressed)
        current_drawdown = self.get_current_drawdown()
        if current_drawdown > 0.10:  # 10% drawdown
            heat_adjustment = 0.7
        elif current_drawdown > 0.05:  # 5% drawdown
            heat_adjustment = 0.85
        else:
            heat_adjustment = 1.0
            
        final_position_value = adjusted_position_value * heat_adjustment
        
        # Convert to shares
        shares = int(final_position_value / current_price) if current_price > 0 else 0
        
        # Ensure minimum viable position
        min_position_value = 1000  # Minimum $1000 position
        if final_position_value < min_position_value:
            shares = 0
        
        # Calculate stop loss and profit target
        stop_loss_atr = self.config.stop_loss_atr * regime_params['stop_loss_multiplier']
        stop_loss_price = current_price - (stop_loss_atr * atr)
        profit_target_price = current_price + (2.0 * stop_loss_atr * atr)  # 2:1 reward:risk
        
        return {
            'shares': shares,
            'position_value': shares * current_price,
            'stop_loss': max(stop_loss_price, current_price * 0.95),  # Max 5% stop
            'profit_target': profit_target_price,
            'risk_per_share': current_price - stop_loss_price,
            'reward_risk_ratio': (profit_target_price - current_price) / max(current_price - stop_loss_price, 0.01),
            'regime_used': regime,
            'adjustments': {
                'regime_multiplier': regime_multiplier,
                'signal_factor': signal_factor,
                'confidence_factor': confidence_factor,
                'volatility_adjustment': volatility_adjustment,
                'heat_adjustment': heat_adjustment
            }
        }
    
    def can_open_position(self, symbol: str, position_value: float, regime: str = 'Low_Volatility') -> Dict[str, Any]:
        """Check if new position can be opened based on risk limits"""
        checks = {
            'can_open': True,
            'reasons': [],
            'limits': {}
        }
        
        regime_params = self.regime_adjustments.get(regime, self.regime_adjustments['Low_Volatility'])
        
        # Check if symbol already has a position
        if symbol in self.positions:
            checks['can_open'] = False
            checks['reasons'].append("Position already exists for this symbol")
        
        # Check portfolio heat (current drawdown)
        current_drawdown = self.get_current_drawdown()
        if current_drawdown > self.config.max_drawdown:
            checks['can_open'] = False
            checks['reasons'].append(f"Max drawdown exceeded: {current_drawdown:.2%} > {self.config.max_drawdown:.2%}")
        
        # Check maximum number of positions
        max_positions = regime_params['max_positions']
        if len(self.positions) >= max_positions:
            checks['can_open'] = False
            checks['reasons'].append(f"Max positions reached: {len(self.positions)} >= {max_positions}")
        
        # Check single position size limit
        max_single = regime_params['max_single_position']
        position_pct = position_value / self.current_capital
        if position_pct > max_single:
            checks['can_open'] = False
            checks['reasons'].append(f"Position too large: {position_pct:.2%} > {max_single:.2%}")
        
        # Check available cash
        if position_value > self.cash:
            checks['can_open'] = False
            checks['reasons'].append(f"Insufficient cash: ${position_value:,.0f} > ${self.cash:,.0f}")
        
        # Check concentration risk (sector exposure)
        symbol_sector = self._get_symbol_sector(symbol)
        if symbol_sector:
            sector_exposure = self.sector_exposure.get(symbol_sector, 0)
            max_sector_exposure = 0.4  # Max 40% in any sector
            if (sector_exposure + position_pct) > max_sector_exposure:
                checks['can_open'] = False
                checks['reasons'].append(f"Sector concentration risk: {symbol_sector} exposure would exceed {max_sector_exposure:.0%}")
        
        checks['limits'] = {
            'max_drawdown': self.config.max_drawdown,
            'current_drawdown': current_drawdown,
            'max_positions': max_positions,
            'current_positions': len(self.positions),
            'max_single_position': max_single,
            'current_cash': self.cash,
            'position_value': position_value
        }
        
        return checks
    
    def open_position(self, symbol: str, shares: int, entry_price: float,
                     stop_loss: float, profit_target: float, signal_strength: float,
                     regime: str = 'Low_Volatility') -> Dict[str, Any]:
        """Open a new position"""
        
        position_value = shares * entry_price
        
        # Check if position can be opened
        can_open_check = self.can_open_position(symbol, position_value, regime)
        if not can_open_check['can_open']:
            return {
                'success': False,
                'reasons': can_open_check['reasons'],
                'limits': can_open_check['limits']
            }
        
        # Create position
        position = Position(
            symbol=symbol,
            quantity=shares,
            entry_price=entry_price,
            entry_date=datetime.now(),
            stop_loss=stop_loss,
            profit_target=profit_target,
            signal_strength=signal_strength,
            regime_at_entry=regime,
            current_price=entry_price
        )
        
        # Update portfolio
        self.positions[symbol] = position
        self.cash -= position_value
        self.total_trades += 1
        
        # Update sector exposure
        sector = self._get_symbol_sector(symbol)
        if sector:
            self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + (position_value / self.current_capital)
        
        logger.info(f"Opened position: {symbol} x{shares} @ ${entry_price:.2f}, Stop: ${stop_loss:.2f}, Target: ${profit_target:.2f}")
        
        return {
            'success': True,
            'position': position,
            'cash_remaining': self.cash,
            'portfolio_value': self.get_portfolio_value()
        }
    
    def close_position(self, symbol: str, exit_price: float, reason: str = 'Manual') -> Dict[str, Any]:
        """Close an existing position"""
        
        if symbol not in self.positions:
            return {'success': False, 'reason': f'No position found for {symbol}'}
        
        position = self.positions[symbol]
        
        # Calculate P&L
        position_value = position.quantity * exit_price
        pnl = (exit_price - position.entry_price) * position.quantity
        pnl_pct = pnl / (position.entry_price * position.quantity)
        
        # Update cash
        self.cash += position_value
        
        # Track winning/losing trades
        if pnl > 0:
            self.winning_trades += 1
        
        # Update sector exposure
        sector = self._get_symbol_sector(symbol)
        if sector:
            position_pct = (position.quantity * position.entry_price) / self.current_capital
            self.sector_exposure[sector] = max(0, self.sector_exposure.get(sector, 0) - position_pct)
        
        # Remove position
        del self.positions[symbol]
        
        trade_result = {
            'success': True,
            'symbol': symbol,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'quantity': position.quantity,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'days_held': position.days_held,
            'reason': reason,
            'cash_after': self.cash,
            'portfolio_value': self.get_portfolio_value()
        }
        
        logger.info(f"Closed position: {symbol} @ ${exit_price:.2f}, P&L: ${pnl:.2f} ({pnl_pct:.1%}), Reason: {reason}")
        
        return trade_result
    
    def update_positions(self, price_data: Dict[str, float]) -> List[Dict[str, Any]]:
        """Update all positions with current prices and check exit conditions"""
        
        updates = []
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol in price_data:
                current_price = price_data[symbol]
                position.update_price(current_price)
                
                # Check exit conditions
                exit_reason = self._check_exit_conditions(position)
                if exit_reason:
                    positions_to_close.append((symbol, current_price, exit_reason))
                
                updates.append({
                    'symbol': symbol,
                    'current_price': current_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'days_held': position.days_held,
                    'exit_signal': exit_reason
                })
        
        # Close positions that hit exit conditions
        for symbol, exit_price, reason in positions_to_close:
            close_result = self.close_position(symbol, exit_price, reason)
            updates.append({
                'symbol': symbol,
                'action': 'CLOSED',
                'result': close_result
            })
        
        # Update portfolio tracking
        self._update_portfolio_metrics()
        
        return updates
    
    def _check_exit_conditions(self, position: Position) -> Optional[str]:
        """Check if position should be closed based on exit rules"""
        
        current_price = position.current_price
        
        # Stop loss hit
        if current_price <= position.stop_loss:
            return 'Stop Loss'
        
        # Profit target hit
        if current_price >= position.profit_target:
            return 'Profit Target'
        
        # Maximum holding period
        if position.days_held >= self.config.max_holding_days:
            return 'Max Holding Period'
        
        # Minimum holding period check (don't exit too early unless stop hit)
        if position.days_held < self.config.min_holding_days:
            return None
        
        # Trailing stop (optional - based on ATR)
        # This could be enhanced with more sophisticated trailing logic
        
        return None
    
    def _update_portfolio_metrics(self) -> None:
        """Update portfolio performance metrics"""
        
        current_value = self.get_portfolio_value()
        self.equity_curve.append(current_value)
        
        # Update peak and drawdown
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        current_drawdown = (self.peak_value - current_value) / self.peak_value
        self.drawdown_history.append(current_drawdown)
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Update daily returns
        if len(self.equity_curve) > 1:
            daily_return = (current_value - self.equity_curve[-2]) / self.equity_curve[-2]
            self.daily_returns.append(daily_return)
        
        # Update current capital
        self.current_capital = current_value
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(
            pos.current_price * pos.quantity 
            for pos in self.positions.values()
        )
        return self.cash + positions_value
    
    def get_current_drawdown(self) -> float:
        """Get current drawdown percentage"""
        current_value = self.get_portfolio_value()
        if self.peak_value <= 0:
            return 0.0
        return (self.peak_value - current_value) / self.peak_value
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        current_value = self.get_portfolio_value()
        total_exposure = sum(pos.current_price * pos.quantity for pos in self.positions.values())
        
        # Calculate VaR and Expected Shortfall
        if len(self.daily_returns) > 30:
            returns_array = np.array(self.daily_returns)
            var_95 = np.percentile(returns_array, 5) * current_value
            es_95 = returns_array[returns_array <= np.percentile(returns_array, 5)].mean() * current_value
        else:
            var_95 = 0.0
            es_95 = 0.0
        
        # Calculate Sharpe ratio
        if len(self.daily_returns) > 30:
            returns_array = np.array(self.daily_returns)
            excess_returns = returns_array - 0.0001  # Assume 2.5% annual risk-free rate
            sharpe = excess_returns.mean() / returns_array.std() * np.sqrt(252) if returns_array.std() > 0 else 0
            volatility = returns_array.std() * np.sqrt(252)
        else:
            sharpe = 0.0
            volatility = 0.0
        
        # Concentration risk
        if len(self.positions) > 0:
            position_weights = [
                (pos.current_price * pos.quantity) / current_value 
                for pos in self.positions.values()
            ]
            concentration_risk = max(position_weights) if position_weights else 0
        else:
            concentration_risk = 0.0
        
        # Correlation risk (simplified)
        correlation_risk = len(self.positions) / 10.0 if len(self.positions) > 0 else 0  # Placeholder
        
        return RiskMetrics(
            total_value=current_value,
            total_exposure=total_exposure,
            max_drawdown=self.max_drawdown,
            current_drawdown=self.get_current_drawdown(),
            var_95=var_95,
            expected_shortfall=es_95,
            sharpe_ratio=sharpe,
            volatility=volatility,
            correlation_risk=correlation_risk,
            concentration_risk=concentration_risk
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        current_value = self.get_portfolio_value()
        total_return = (current_value - self.initial_capital) / self.initial_capital
        win_rate = self.winning_trades / max(self.total_trades, 1)
        
        risk_metrics = self.get_risk_metrics()
        
        return {
            'portfolio_value': current_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.get_current_drawdown(),
            'sharpe_ratio': risk_metrics.sharpe_ratio,
            'volatility': risk_metrics.volatility,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'current_positions': len(self.positions),
            'cash': self.cash,
            'total_exposure': risk_metrics.total_exposure,
            'var_95': risk_metrics.var_95,
            'concentration_risk': risk_metrics.concentration_risk,
            'sector_exposure': dict(self.sector_exposure)
        }
    
    def _get_symbol_sector(self, symbol: str) -> Optional[str]:
        """Get sector for a symbol (simplified mapping)"""
        # This would typically come from a database or API
        # For now, using a simplified mapping
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
        healthcare_stocks = ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO']
        finance_stocks = ['JPM', 'BAC', 'WFC', 'GS', 'MS']
        
        if symbol in tech_stocks:
            return 'Technology'
        elif symbol in healthcare_stocks:
            return 'Healthcare'
        elif symbol in finance_stocks:
            return 'Finance'
        else:
            return 'Other'
    
    def emergency_stop(self) -> Dict[str, Any]:
        """Emergency stop - close all positions"""
        logger.warning("EMERGENCY STOP ACTIVATED - Closing all positions")
        
        closed_positions = []
        for symbol, position in list(self.positions.items()):
            # Use current price for emergency close
            result = self.close_position(symbol, position.current_price, 'Emergency Stop')
            closed_positions.append(result)
        
        return {
            'positions_closed': len(closed_positions),
            'final_cash': self.cash,
            'final_value': self.get_portfolio_value(),
            'closed_positions': closed_positions
        }
    
    def should_reduce_exposure(self) -> Dict[str, Any]:
        """Check if exposure should be reduced based on risk metrics"""
        
        risk_metrics = self.get_risk_metrics()
        current_drawdown = self.get_current_drawdown()
        
        reduce_exposure = False
        reasons = []
        
        # Check drawdown
        if current_drawdown > self.config.max_drawdown * 0.8:  # 80% of max allowed
            reduce_exposure = True
            reasons.append(f"High drawdown: {current_drawdown:.2%}")
        
        # Check concentration
        if risk_metrics.concentration_risk > 0.3:  # 30% in single position
            reduce_exposure = True
            reasons.append(f"High concentration: {risk_metrics.concentration_risk:.2%}")
        
        # Check volatility
        if risk_metrics.volatility > self.config.volatility_target * 1.5:  # 150% of target vol
            reduce_exposure = True
            reasons.append(f"High volatility: {risk_metrics.volatility:.2%}")
        
        return {
            'should_reduce': reduce_exposure,
            'reasons': reasons,
            'current_metrics': {
                'drawdown': current_drawdown,
                'concentration': risk_metrics.concentration_risk,
                'volatility': risk_metrics.volatility,
                'var_95': risk_metrics.var_95
            }
        }

# Example usage
if __name__ == "__main__":
    # Test the risk manager
    risk_manager = AdaptiveRiskManager(initial_capital=100000)
    
    # Test position sizing
    position_info = risk_manager.calculate_position_size(
        symbol='AAPL',
        signal_strength=0.8,
        confidence=0.75,
        current_price=150.0,
        atr=3.0,
        regime='Low_Volatility'
    )
    
    print(f"Position sizing result: {position_info}")
    
    # Test opening a position
    if position_info['shares'] > 0:
        open_result = risk_manager.open_position(
            symbol='AAPL',
            shares=position_info['shares'],
            entry_price=150.0,
            stop_loss=position_info['stop_loss'],
            profit_target=position_info['profit_target'],
            signal_strength=0.8,
            regime='Low_Volatility'
        )
        
        print(f"Open position result: {open_result}")
        
        # Test position updates
        price_updates = {'AAPL': 155.0}  # Price moved up
        updates = risk_manager.update_positions(price_updates)
        print(f"Position updates: {updates}")
        
        # Get performance summary
        performance = risk_manager.get_performance_summary()
        print(f"Performance summary: {performance}")
        
        # Check risk metrics
        risk_metrics = risk_manager.get_risk_metrics()
        print(f"Risk metrics: {risk_metrics}")
    
    else:
        print("Position size too small to open")