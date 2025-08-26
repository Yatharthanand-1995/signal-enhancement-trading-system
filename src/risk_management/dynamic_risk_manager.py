"""
Dynamic Risk Management System
Advanced risk management with regime-adaptive parameters and real-time portfolio monitoring.

Based on academic research:
- Kelly Criterion for optimal position sizing (Kelly 1956)
- Value at Risk (VaR) models (Jorion 2007)
- Regime-based risk management (Ang & Bekaert 2002)
- Dynamic hedging strategies (Hull 2018)
- Risk parity and volatility targeting (Qian et al. 2011)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from scipy import stats
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level enumeration"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for a position or portfolio"""
    symbol: str
    timestamp: datetime
    
    # Position metrics
    position_size: float
    position_value: float
    current_price: float
    unrealized_pnl: float
    
    # Risk measures
    value_at_risk_1d: float  # 1-day VaR at 95% confidence
    value_at_risk_5d: float  # 5-day VaR at 95% confidence
    expected_shortfall: float  # Expected loss beyond VaR
    maximum_drawdown: float
    
    # Volatility measures
    realized_volatility: float
    volatility_percentile: float
    volatility_adjusted_size: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Risk limits and alerts
    risk_level: RiskLevel
    stop_loss_price: float
    take_profit_price: float
    position_limit_utilization: float
    margin_utilization: float
    
    # Regime-specific adjustments
    regime: str
    regime_confidence: float
    regime_risk_multiplier: float
    
    # Alert flags
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class RiskLimits:
    """Dynamic risk limits that adjust based on market conditions"""
    max_position_size: float = 0.10  # 10% of portfolio
    max_portfolio_risk: float = 0.02  # 2% daily VaR
    max_sector_exposure: float = 0.20  # 20% per sector
    max_correlation_exposure: float = 0.30  # 30% in correlated positions
    max_drawdown_limit: float = 0.15  # 15% maximum drawdown
    max_leverage: float = 2.0  # 2:1 leverage
    
    # Dynamic multipliers (adjusted by regime)
    volatility_multiplier: float = 1.0
    regime_multiplier: float = 1.0
    correlation_multiplier: float = 1.0

class DynamicRiskManager:
    """
    Advanced risk management system that provides:
    1. Real-time risk monitoring and alerts
    2. Regime-adaptive risk limits
    3. Portfolio-level risk optimization
    4. Dynamic position sizing
    5. Advanced VaR and stress testing
    6. Automated risk responses
    """
    
    def __init__(self, 
                 base_capital: float = 1000000,
                 confidence_level: float = 0.95,
                 lookback_period: int = 252):
        """
        Initialize dynamic risk manager
        
        Args:
            base_capital: Base portfolio capital
            confidence_level: VaR confidence level (default: 95%)
            lookback_period: Historical lookback for risk calculations
        """
        self.base_capital = base_capital
        self.confidence_level = confidence_level
        self.lookback_period = lookback_period
        
        # Portfolio tracking
        self.positions = {}
        self.portfolio_history = deque(maxlen=1000)
        self.risk_events = deque(maxlen=500)
        
        # Risk limits (dynamic)
        self.risk_limits = RiskLimits()
        
        # Research-backed risk parameters
        self.risk_parameters = self._initialize_risk_parameters()
        
        # Regime-specific risk adjustments
        self.regime_risk_adjustments = self._initialize_regime_adjustments()
        
        # Performance tracking
        self.risk_metrics_history = defaultdict(list)
        self.alert_history = deque(maxlen=1000)
        
        logger.info(f"Dynamic Risk Manager initialized with ${base_capital:,.0f} base capital")
    
    def _initialize_risk_parameters(self) -> Dict[str, Any]:
        """Initialize research-backed risk parameters"""
        return {
            # Kelly Criterion parameters (Kelly 1956)
            'kelly_max_bet': 0.25,  # Maximum Kelly bet size
            'kelly_scaling': 0.25,  # Conservative Kelly scaling
            
            # VaR parameters (Jorion 2007)
            'var_confidence_levels': [0.90, 0.95, 0.99],
            'var_holding_periods': [1, 5, 22],  # 1-day, 1-week, 1-month
            
            # Volatility targeting (Qian et al. 2011)
            'target_volatility': 0.15,  # 15% annual target
            'vol_lookback_short': 22,   # 1-month volatility
            'vol_lookback_long': 66,    # 3-month volatility
            
            # Risk-adjusted returns
            'min_sharpe_ratio': 0.5,
            'min_sortino_ratio': 0.7,
            
            # Stop loss parameters (research-optimized)
            'stop_loss_atr_multiple': 2.0,  # ATR-based stops
            'trailing_stop_factor': 0.5,    # Trailing stop adjustment
            
            # Correlation limits (diversification)
            'max_correlation_threshold': 0.7,
            'correlation_lookback': 60
        }
    
    def _initialize_regime_adjustments(self) -> Dict[str, Dict[str, float]]:
        """Initialize regime-specific risk adjustments (Ang & Bekaert 2002)"""
        return {
            'bull_market': {
                'position_size_multiplier': 1.2,    # Larger positions in bull markets
                'stop_loss_multiplier': 1.3,        # Wider stops (trends persist)
                'var_multiplier': 0.8,              # Lower VaR estimates
                'correlation_multiplier': 1.2,      # Allow higher correlation
                'leverage_multiplier': 1.1          # Slightly higher leverage
            },
            
            'bear_market': {
                'position_size_multiplier': 0.6,    # Smaller positions
                'stop_loss_multiplier': 0.8,        # Tighter stops
                'var_multiplier': 1.4,              # Higher VaR (tail risk)
                'correlation_multiplier': 0.7,      # Lower correlation tolerance
                'leverage_multiplier': 0.7          # Reduced leverage
            },
            
            'sideways_market': {
                'position_size_multiplier': 0.8,    # Conservative sizing
                'stop_loss_multiplier': 0.9,        # Moderately tight stops
                'var_multiplier': 1.0,              # Standard VaR
                'correlation_multiplier': 1.0,      # Standard correlation
                'leverage_multiplier': 0.9          # Slightly reduced leverage
            },
            
            'volatile_market': {
                'position_size_multiplier': 0.5,    # Very small positions
                'stop_loss_multiplier': 0.7,        # Very tight stops
                'var_multiplier': 1.6,              # Much higher VaR
                'correlation_multiplier': 0.6,      # Low correlation tolerance
                'leverage_multiplier': 0.6          # Significantly reduced leverage
            }
        }
    
    def calculate_position_risk(self, 
                               symbol: str,
                               position_size: float,
                               entry_price: float,
                               current_price: float,
                               market_data: pd.DataFrame,
                               regime_info: Optional[Dict[str, Any]] = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a position
        
        Args:
            symbol: Trading symbol
            position_size: Position size (number of shares/units)
            entry_price: Entry price
            current_price: Current market price
            market_data: Historical price data
            regime_info: Current market regime information
            
        Returns:
            RiskMetrics with comprehensive risk assessment
        """
        
        timestamp = datetime.now()
        position_value = position_size * current_price
        unrealized_pnl = position_size * (current_price - entry_price)
        
        # Calculate returns for risk calculations
        returns = market_data['close'].pct_change().dropna()
        
        # Get regime information early
        regime = regime_info.get('regime', 'sideways_market') if regime_info else 'sideways_market'
        regime_confidence = regime_info.get('confidence', 0.5) if regime_info else 0.5
        regime_adjustments = self.regime_risk_adjustments.get(regime, {})
        regime_var_multiplier = regime_adjustments.get('var_multiplier', 1.0)
        
        # Calculate volatility metrics
        realized_vol = self._calculate_realized_volatility(returns)
        vol_percentile = self._calculate_volatility_percentile(returns, realized_vol)
        
        # Calculate VaR measures with regime adjustment
        var_1d = self._calculate_var(returns, position_value, horizon=1) * regime_var_multiplier
        var_5d = self._calculate_var(returns, position_value, horizon=5) * regime_var_multiplier
        expected_shortfall = self._calculate_expected_shortfall(returns, position_value)
        
        # Calculate maximum drawdown
        max_drawdown = self._calculate_maximum_drawdown(market_data, entry_price, position_size)
        
        # Calculate risk-adjusted returns
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns, max_drawdown)
        
        # Regime risk multiplier already calculated above
        regime_risk_multiplier = regime_var_multiplier
        
        # Calculate dynamic stop loss and take profit
        stop_loss_price, take_profit_price = self._calculate_dynamic_stops(
            current_price, returns, regime_adjustments, position_size > 0
        )
        
        # Assess risk level
        risk_level = self._assess_risk_level(
            var_1d, position_value, realized_vol, vol_percentile, regime
        )
        
        # Calculate position limits utilization
        position_limit_util = abs(position_value) / (self.base_capital * self.risk_limits.max_position_size)
        margin_util = self._calculate_margin_utilization(position_value)
        
        # Volatility-adjusted position size
        vol_adjusted_size = self._calculate_volatility_adjusted_size(
            position_size, realized_vol, regime_adjustments
        )
        
        # Generate alerts and recommendations
        alerts, recommendations = self._generate_risk_alerts_and_recommendations(
            symbol, position_value, var_1d, risk_level, position_limit_util, realized_vol
        )
        
        # Create risk metrics
        risk_metrics = RiskMetrics(
            symbol=symbol,
            timestamp=timestamp,
            position_size=position_size,
            position_value=position_value,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            value_at_risk_1d=var_1d,
            value_at_risk_5d=var_5d,
            expected_shortfall=expected_shortfall,
            maximum_drawdown=max_drawdown,
            realized_volatility=realized_vol,
            volatility_percentile=vol_percentile,
            volatility_adjusted_size=vol_adjusted_size,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            risk_level=risk_level,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            position_limit_utilization=position_limit_util,
            margin_utilization=margin_util,
            regime=regime,
            regime_confidence=regime_confidence,
            regime_risk_multiplier=regime_risk_multiplier,
            alerts=alerts,
            recommendations=recommendations
        )
        
        # Store metrics for tracking
        self.risk_metrics_history[symbol].append(risk_metrics)
        
        # Log alerts
        for alert in alerts:
            self.alert_history.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'alert': alert,
                'risk_level': risk_level.name
            })
        
        logger.info(f"Risk metrics calculated for {symbol}: {risk_level.name} risk, "
                   f"VaR: ${var_1d:,.0f}, Volatility: {realized_vol:.1%}")
        
        return risk_metrics
    
    def _calculate_realized_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized realized volatility"""
        if len(returns) < 2:
            return 0.20  # Default 20% volatility
        
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252)
        return annualized_vol
    
    def _calculate_volatility_percentile(self, returns: pd.Series, current_vol: float) -> float:
        """Calculate current volatility percentile over lookback period"""
        if len(returns) < 30:
            return 0.5
        
        # Rolling volatility calculation
        rolling_vol = returns.rolling(window=22).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()
        
        if len(rolling_vol) == 0:
            return 0.5
        
        # Calculate percentile
        percentile = (rolling_vol < current_vol).mean()
        return percentile
    
    def _calculate_var(self, returns: pd.Series, position_value: float, 
                      horizon: int = 1, confidence: float = None) -> float:
        """Calculate Value at Risk using historical simulation"""
        if confidence is None:
            confidence = self.confidence_level
        
        if len(returns) < 30:
            # Fallback to parametric VaR
            vol = returns.std() if len(returns) > 1 else 0.02
            z_score = stats.norm.ppf(1 - confidence)
            var = abs(position_value * vol * np.sqrt(horizon) * z_score)
            return var
        
        # Historical simulation VaR
        scaled_returns = returns * np.sqrt(horizon)
        var_percentile = (1 - confidence) * 100
        var_return = np.percentile(scaled_returns, var_percentile)
        var = abs(position_value * var_return)
        
        return var
    
    def _calculate_expected_shortfall(self, returns: pd.Series, position_value: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        if len(returns) < 30:
            return self._calculate_var(returns, position_value) * 1.3  # Rough approximation
        
        var_percentile = (1 - self.confidence_level) * 100
        var_threshold = np.percentile(returns, var_percentile)
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return self._calculate_var(returns, position_value)
        
        expected_shortfall = abs(position_value * tail_returns.mean())
        return expected_shortfall
    
    def _calculate_maximum_drawdown(self, market_data: pd.DataFrame, 
                                   entry_price: float, position_size: float) -> float:
        """Calculate maximum drawdown for the position"""
        if len(market_data) < 2:
            return 0.0
        
        prices = market_data['close']
        
        # Calculate running maximum for long positions, minimum for short
        if position_size > 0:  # Long position
            running_max = prices.expanding().max()
            drawdowns = (prices - running_max) / running_max
        else:  # Short position
            running_min = prices.expanding().min()
            drawdowns = (running_min - prices) / running_min
        
        max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0
        return max_drawdown
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) < 5:
            return 0.0
        
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        
        if volatility == 0:
            return 0.0
        
        sharpe = excess_returns / volatility
        return sharpe
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) < 5:
            return 0.0
        
        excess_returns = returns.mean() * 252 - risk_free_rate
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if excess_returns > 0 else 0.0
        
        downside_deviation = downside_returns.std() * np.sqrt(252)
        
        if downside_deviation == 0:
            return 0.0
        
        sortino = excess_returns / downside_deviation
        return sortino
    
    def _calculate_calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if max_drawdown == 0 or len(returns) < 5:
            return 0.0
        
        annual_return = returns.mean() * 252
        calmar = annual_return / max_drawdown
        return calmar
    
    def _calculate_dynamic_stops(self, current_price: float, returns: pd.Series,
                                regime_adjustments: Dict[str, float], 
                                is_long: bool) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit levels"""
        
        if len(returns) < 10:
            # Fallback to simple percentage stops
            stop_pct = 0.05 * regime_adjustments.get('stop_loss_multiplier', 1.0)
            profit_pct = 0.10
        else:
            # ATR-based stops
            atr = self._calculate_atr(returns, current_price)
            stop_multiplier = self.risk_parameters['stop_loss_atr_multiple']
            stop_multiplier *= regime_adjustments.get('stop_loss_multiplier', 1.0)
            
            stop_distance = atr * stop_multiplier
            stop_pct = stop_distance / current_price
            profit_pct = stop_pct * 2.0  # 2:1 profit target
        
        if is_long:
            stop_loss_price = current_price * (1 - stop_pct)
            take_profit_price = current_price * (1 + profit_pct)
        else:
            stop_loss_price = current_price * (1 + stop_pct)
            take_profit_price = current_price * (1 - profit_pct)
        
        return stop_loss_price, take_profit_price
    
    def _calculate_atr(self, returns: pd.Series, current_price: float, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(returns) < period:
            return current_price * 0.02  # 2% fallback
        
        # Estimate ATR from returns (simplified)
        high_low = returns.rolling(period).std() * current_price * np.sqrt(252/period)
        atr = high_low.iloc[-1] if not pd.isna(high_low.iloc[-1]) else current_price * 0.02
        
        return atr
    
    def _assess_risk_level(self, var_1d: float, position_value: float, 
                          volatility: float, vol_percentile: float, regime: str) -> RiskLevel:
        """Assess overall risk level for the position"""
        
        risk_score = 0
        
        # VaR relative to position
        var_ratio = var_1d / abs(position_value) if position_value != 0 else 0
        if var_ratio > 0.05:  # 5% daily VaR
            risk_score += 2
        elif var_ratio > 0.03:  # 3% daily VaR
            risk_score += 1
        
        # Volatility assessment
        if volatility > 0.40:  # 40% annual volatility
            risk_score += 2
        elif volatility > 0.25:  # 25% annual volatility
            risk_score += 1
        
        # Volatility percentile
        if vol_percentile > 0.8:  # High volatility regime
            risk_score += 1
        
        # Regime-specific adjustments
        regime_risk_addition = {
            'volatile_market': 2,
            'bear_market': 1,
            'sideways_market': 0,
            'bull_market': 0
        }
        risk_score += regime_risk_addition.get(regime, 0)
        
        # Convert to risk level
        if risk_score >= 5:
            return RiskLevel.VERY_HIGH
        elif risk_score >= 4:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        elif risk_score >= 1:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW
    
    def _calculate_margin_utilization(self, position_value: float) -> float:
        """Calculate margin utilization percentage"""
        total_exposure = sum(abs(pos.get('value', 0)) for pos in self.positions.values())
        margin_utilization = total_exposure / (self.base_capital * self.risk_limits.max_leverage)
        return min(1.0, margin_utilization)
    
    def _calculate_volatility_adjusted_size(self, position_size: float, volatility: float,
                                          regime_adjustments: Dict[str, float]) -> float:
        """Calculate volatility-adjusted position size"""
        target_vol = self.risk_parameters['target_volatility']
        vol_adjustment = target_vol / max(volatility, 0.05)  # Prevent division by zero
        
        # Apply regime adjustments
        regime_multiplier = regime_adjustments.get('position_size_multiplier', 1.0)
        
        adjusted_size = position_size * vol_adjustment * regime_multiplier
        return adjusted_size
    
    def _generate_risk_alerts_and_recommendations(self, symbol: str, position_value: float,
                                                var_1d: float, risk_level: RiskLevel,
                                                position_limit_util: float, 
                                                volatility: float) -> Tuple[List[str], List[str]]:
        """Generate risk alerts and recommendations"""
        alerts = []
        recommendations = []
        
        # Position size alerts
        if position_limit_util > 1.0:
            alerts.append(f"Position exceeds size limit by {(position_limit_util-1)*100:.1f}%")
            recommendations.append("Reduce position size to comply with limits")
        elif position_limit_util > 0.8:
            alerts.append("Position approaching size limit")
            
        # VaR alerts
        var_ratio = var_1d / abs(position_value) if position_value != 0 else 0
        if var_ratio > 0.05:
            alerts.append(f"High daily VaR: {var_ratio:.1%} of position value")
            recommendations.append("Consider reducing position or adding hedges")
        
        # Volatility alerts
        if volatility > 0.40:
            alerts.append(f"Very high volatility: {volatility:.1%}")
            recommendations.append("Use tighter stops and smaller position sizes")
        
        # Risk level recommendations
        if risk_level == RiskLevel.VERY_HIGH:
            recommendations.append("URGENT: Reduce exposure immediately")
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("Consider reducing position size")
        elif risk_level == RiskLevel.VERY_LOW:
            recommendations.append("Position size could potentially be increased")
        
        return alerts, recommendations
    
    def calculate_optimal_position_size(self, symbol: str, entry_price: float,
                                       expected_return: float, win_rate: float,
                                       market_data: pd.DataFrame,
                                       regime_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Calculate optimal position size using Kelly Criterion and risk constraints
        
        Args:
            symbol: Trading symbol
            entry_price: Proposed entry price
            expected_return: Expected return per trade
            win_rate: Historical win rate (0-1)
            market_data: Historical market data
            regime_info: Current regime information
            
        Returns:
            Dict with optimal position size and reasoning
        """
        
        returns = market_data['close'].pct_change().dropna()
        volatility = self._calculate_realized_volatility(returns)
        
        # Kelly Criterion calculation
        kelly_fraction = self._calculate_kelly_fraction(expected_return, win_rate, volatility)
        
        # Apply Kelly scaling (conservative)
        scaled_kelly = kelly_fraction * self.risk_parameters['kelly_scaling']
        
        # Volatility targeting adjustment
        vol_target_fraction = self._calculate_volatility_target_fraction(volatility)
        
        # Regime adjustments
        regime = regime_info.get('regime', 'sideways_market') if regime_info else 'sideways_market'
        regime_adjustments = self.regime_risk_adjustments.get(regime, {})
        regime_multiplier = regime_adjustments.get('position_size_multiplier', 1.0)
        
        # Calculate various position sizing methods
        methods = {
            'kelly_scaled': scaled_kelly,
            'volatility_target': vol_target_fraction,
            'fixed_fractional': self.risk_limits.max_position_size * 0.5,  # 50% of limit
            'equal_weight': 1.0 / 20  # Assume 20 position portfolio
        }
        
        # Take the minimum for conservative approach, but ensure it's not too small
        base_fraction = max(0.01, min(methods.values()))  # At least 1%
        
        # Apply regime adjustment
        optimal_fraction = base_fraction * regime_multiplier
        
        # Calculate actual position size
        available_capital = self.base_capital * (1 - self._calculate_total_exposure())
        max_position_value = available_capital * optimal_fraction
        optimal_shares = max_position_value / entry_price
        
        # Risk assessment
        test_risk = self.calculate_position_risk(
            symbol, optimal_shares, entry_price, entry_price, market_data, regime_info
        )
        
        return {
            'optimal_shares': optimal_shares,
            'optimal_fraction': optimal_fraction,
            'position_value': max_position_value,
            'methods_considered': methods,
            'regime_adjustment': regime_multiplier,
            'kelly_fraction': kelly_fraction,
            'volatility_target': vol_target_fraction,
            'risk_assessment': test_risk,
            'reasoning': self._generate_sizing_reasoning(methods, regime, optimal_fraction)
        }
    
    def _calculate_kelly_fraction(self, expected_return: float, win_rate: float, 
                                 volatility: float) -> float:
        """Calculate Kelly fraction for position sizing"""
        if win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        # Estimate average win and loss
        avg_win = expected_return / win_rate if expected_return > 0 else 0
        avg_loss = expected_return / (win_rate - 1) if expected_return < 0 else volatility
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        if avg_loss <= 0:
            return 0.0
        
        b = abs(avg_win / avg_loss)
        if b <= 0:  # Prevent division by zero
            return 0.0
            
        p = win_rate
        q = 1 - win_rate
        
        kelly = (b * p - q) / b
        
        # Cap at maximum Kelly bet
        kelly = max(0, min(kelly, self.risk_parameters['kelly_max_bet']))
        
        return kelly
    
    def _calculate_volatility_target_fraction(self, position_volatility: float) -> float:
        """Calculate position size for volatility targeting"""
        target_vol = self.risk_parameters['target_volatility']
        
        if position_volatility <= 0:
            return 0.0
        
        # Fraction to achieve target volatility
        fraction = target_vol / position_volatility
        
        # Cap to reasonable limits
        fraction = min(fraction, self.risk_limits.max_position_size)
        
        return fraction
    
    def _calculate_total_exposure(self) -> float:
        """Calculate current total portfolio exposure"""
        total_exposure = sum(abs(pos.get('value', 0)) for pos in self.positions.values())
        exposure_fraction = total_exposure / self.base_capital
        return exposure_fraction
    
    def _generate_sizing_reasoning(self, methods: Dict[str, float], regime: str, 
                                  chosen_fraction: float) -> str:
        """Generate explanation for position sizing decision"""
        reasoning_parts = []
        
        # Method comparison
        min_method = min(methods.items(), key=lambda x: x[1])
        reasoning_parts.append(f"Conservative approach used {min_method[0]} method ({min_method[1]:.1%})")
        
        # Regime impact
        if regime != 'sideways_market':
            regime_adj = self.regime_risk_adjustments.get(regime, {}).get('position_size_multiplier', 1.0)
            if regime_adj != 1.0:
                reasoning_parts.append(f"{regime.replace('_', ' ').title()} regime adjustment: {regime_adj:.1f}x")
        
        # Final fraction
        reasoning_parts.append(f"Final allocation: {chosen_fraction:.1%} of capital")
        
        return " | ".join(reasoning_parts)
    
    def get_portfolio_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio risk summary"""
        
        if not self.positions:
            return {'status': 'No positions to analyze'}
        
        # Aggregate metrics
        total_value = sum(pos.get('value', 0) for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.get('unrealized_pnl', 0) for pos in self.positions.values())
        
        # Risk distribution
        risk_levels = defaultdict(int)
        total_var = 0
        
        for symbol in self.positions.keys():
            if symbol in self.risk_metrics_history:
                latest_metrics = self.risk_metrics_history[symbol][-1]
                risk_levels[latest_metrics.risk_level.name] += 1
                total_var += latest_metrics.value_at_risk_1d
        
        # Portfolio-level metrics
        portfolio_var_ratio = total_var / abs(total_value) if total_value != 0 else 0
        exposure_ratio = abs(total_value) / self.base_capital
        
        # Recent alerts
        recent_alerts = list(self.alert_history)[-20:] if self.alert_history else []
        
        return {
            'portfolio_value': total_value,
            'unrealized_pnl': total_unrealized_pnl,
            'exposure_ratio': exposure_ratio,
            'portfolio_var_1d': total_var,
            'portfolio_var_ratio': portfolio_var_ratio,
            'position_count': len(self.positions),
            'risk_distribution': dict(risk_levels),
            'recent_alerts_count': len(recent_alerts),
            'margin_utilization': self._calculate_margin_utilization(0),  # Portfolio level
            'risk_limits_status': {
                'max_position_size': self.risk_limits.max_position_size,
                'max_portfolio_risk': self.risk_limits.max_portfolio_risk,
                'current_portfolio_risk': portfolio_var_ratio,
                'within_limits': portfolio_var_ratio <= self.risk_limits.max_portfolio_risk
            }
        }
    
    def export_risk_report(self, filepath: str) -> None:
        """Export comprehensive risk report to JSON"""
        
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'portfolio_summary': self.get_portfolio_risk_summary(),
            'risk_limits': {
                'max_position_size': self.risk_limits.max_position_size,
                'max_portfolio_risk': self.risk_limits.max_portfolio_risk,
                'max_drawdown_limit': self.risk_limits.max_drawdown_limit,
                'max_leverage': self.risk_limits.max_leverage
            },
            'recent_risk_events': [
                {
                    'timestamp': event['timestamp'].isoformat(),
                    'symbol': event['symbol'],
                    'alert': event['alert'],
                    'risk_level': event['risk_level']
                }
                for event in list(self.alert_history)[-50:]
            ]
        }
        
        # Add individual position metrics
        position_metrics = {}
        for symbol, metrics_list in self.risk_metrics_history.items():
            if metrics_list:
                latest = metrics_list[-1]
                position_metrics[symbol] = {
                    'risk_level': latest.risk_level.name,
                    'value_at_risk_1d': latest.value_at_risk_1d,
                    'realized_volatility': latest.realized_volatility,
                    'unrealized_pnl': latest.unrealized_pnl,
                    'position_limit_utilization': latest.position_limit_utilization
                }
        
        report_data['position_metrics'] = position_metrics
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Risk report exported to {filepath}")

if __name__ == "__main__":
    # Example usage and testing
    risk_manager = DynamicRiskManager(base_capital=1000000)
    
    # Sample market data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
    
    market_data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    
    # Test risk calculation
    print("Testing Dynamic Risk Management System...")
    
    risk_metrics = risk_manager.calculate_position_risk(
        symbol='TEST',
        position_size=1000,
        entry_price=100.0,
        current_price=105.0,
        market_data=market_data,
        regime_info={'regime': 'bull_market', 'confidence': 0.8}
    )
    
    print(f"\nRisk Metrics for TEST:")
    print(f"  Risk Level: {risk_metrics.risk_level.name}")
    print(f"  VaR (1-day): ${risk_metrics.value_at_risk_1d:,.0f}")
    print(f"  Realized Volatility: {risk_metrics.realized_volatility:.1%}")
    print(f"  Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
    print(f"  Stop Loss: ${risk_metrics.stop_loss_price:.2f}")
    print(f"  Alerts: {len(risk_metrics.alerts)}")
    
    # Test optimal position sizing
    sizing_result = risk_manager.calculate_optimal_position_size(
        symbol='TEST',
        entry_price=100.0,
        expected_return=0.05,
        win_rate=0.60,
        market_data=market_data,
        regime_info={'regime': 'bull_market', 'confidence': 0.8}
    )
    
    print(f"\nOptimal Position Sizing:")
    print(f"  Recommended Shares: {sizing_result['optimal_shares']:,.0f}")
    print(f"  Portfolio Fraction: {sizing_result['optimal_fraction']:.1%}")
    print(f"  Kelly Fraction: {sizing_result['kelly_fraction']:.1%}")
    print(f"  Reasoning: {sizing_result['reasoning']}")
    
    # Portfolio summary
    portfolio_summary = risk_manager.get_portfolio_risk_summary()
    print(f"\nPortfolio Summary: {portfolio_summary['status'] if 'status' in portfolio_summary else 'Ready for trading'}")