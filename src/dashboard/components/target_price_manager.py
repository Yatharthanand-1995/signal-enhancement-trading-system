"""
Target Price Manager
Handles target price calculation, monitoring, and automated selling for paper trading
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class TargetPriceManager:
    """
    Manages target prices and automated exit conditions for paper trading positions
    """
    
    def __init__(self, paper_engine):
        self.paper_engine = paper_engine
        self.last_price_check = datetime.now()
        self.price_check_interval = timedelta(minutes=15)  # Check prices every 15 minutes
        self.alert_threshold = 0.02  # 2% price move alert
        
    def calculate_dynamic_target_price(self, position, market_data: Dict = None) -> float:
        """
        Calculate dynamic target price based on multiple factors
        
        Args:
            position: PaperPosition instance
            market_data: Optional market context data
            
        Returns:
            float: Calculated target price
        """
        try:
            # Base target from position
            base_target = position.entry_price * (1 + position.profit_target_pct)
            
            # Adjust based on signal strength (0.5 to 2.0 multiplier)
            strength_multiplier = max(0.5, min(2.0, position.signal_strength))
            
            # Adjust based on confidence (0.8 to 1.5 multiplier)  
            confidence_multiplier = max(0.8, min(1.5, position.confidence))
            
            # Volatility adjustment
            volatility_adj = 1.0
            if market_data and 'volatility' in market_data:
                # Higher volatility = higher target (more room to run)
                volatility = market_data['volatility']
                volatility_adj = 1.0 + (volatility * 0.5)  # Max 50% boost for high vol
            
            # Market regime adjustment
            regime_adj = 1.0
            if market_data and 'market_regime' in market_data:
                regime = market_data['market_regime']
                if regime == 'BULL_MARKET':
                    regime_adj = 1.2  # 20% higher targets in bull market
                elif regime == 'BEAR_MARKET':
                    regime_adj = 0.8  # 20% lower targets in bear market
            
            # Calculate final target
            target_price = (base_target * strength_multiplier * confidence_multiplier * 
                          volatility_adj * regime_adj)
            
            # Ensure reasonable bounds (5% to 50% above entry)
            min_target = position.entry_price * 1.05
            max_target = position.entry_price * 1.50
            target_price = max(min_target, min(max_target, target_price))
            
            return target_price
            
        except Exception as e:
            logger.error(f"Error calculating dynamic target for {position.symbol}: {e}")
            return position.entry_price * (1 + position.profit_target_pct)
    
    def update_all_target_prices(self, market_data: Dict = None):
        """Update target prices for all positions"""
        for symbol, position in self.paper_engine.positions.items():
            try:
                new_target = self.calculate_dynamic_target_price(position, market_data)
                
                # Only update if significantly different (>1% change)
                if abs(new_target - position.target_price) / position.target_price > 0.01:
                    old_target = position.target_price
                    position.target_price = new_target
                    logger.info(f"Updated target price for {symbol}: ${old_target:.2f} -> ${new_target:.2f}")
                    
            except Exception as e:
                logger.error(f"Error updating target price for {symbol}: {e}")
    
    def check_all_exit_conditions(self) -> List[Dict]:
        """
        Check exit conditions for all positions
        
        Returns:
            List[Dict]: List of positions that should be exited with reasons
        """
        positions_to_exit = []
        
        if not self.paper_engine.positions:
            return positions_to_exit
        
        # Update prices first
        self.paper_engine.update_positions()
        
        for symbol, position in self.paper_engine.positions.items():
            try:
                should_exit, reason = position.check_exit_conditions()
                
                if should_exit:
                    positions_to_exit.append({
                        'symbol': symbol,
                        'position': position,
                        'reason': reason,
                        'current_price': position.current_price,
                        'entry_price': position.entry_price,
                        'target_price': position.target_price,
                        'pnl_estimate': (position.current_price - position.entry_price) * position.shares
                    })
                    
            except Exception as e:
                logger.error(f"Error checking exit conditions for {symbol}: {e}")
        
        return positions_to_exit
    
    def execute_automated_exits(self, dry_run: bool = False) -> Dict:
        """
        Execute automated exits for positions that meet exit conditions
        
        Args:
            dry_run: If True, only return what would be executed
            
        Returns:
            Dict: Results of exit execution
        """
        results = {
            'positions_checked': len(self.paper_engine.positions),
            'positions_exited': 0,
            'exits_executed': [],
            'errors': [],
            'total_pnl': 0.0,
            'dry_run': dry_run
        }
        
        try:
            positions_to_exit = self.check_all_exit_conditions()
            
            for exit_info in positions_to_exit:
                symbol = exit_info['symbol']
                reason = exit_info['reason']
                
                try:
                    if not dry_run:
                        success = self.paper_engine.execute_sell_order(symbol, reason)
                        if success:
                            results['positions_exited'] += 1
                            results['total_pnl'] += exit_info['pnl_estimate']
                            results['exits_executed'].append({
                                'symbol': symbol,
                                'reason': reason,
                                'price': exit_info['current_price'],
                                'pnl': exit_info['pnl_estimate']
                            })
                            logger.info(f"Automated exit executed: {symbol} at ${exit_info['current_price']:.2f} ({reason})")
                        else:
                            results['errors'].append(f"Failed to execute sell order for {symbol}")
                    else:
                        # Dry run - just log what would happen
                        results['positions_exited'] += 1
                        results['exits_executed'].append({
                            'symbol': symbol,
                            'reason': reason,
                            'price': exit_info['current_price'],
                            'pnl': exit_info['pnl_estimate'],
                            'would_execute': True
                        })
                        logger.info(f"Would exit {symbol} at ${exit_info['current_price']:.2f} ({reason}) - DRY RUN")
                        
                except Exception as e:
                    error_msg = f"Error executing exit for {symbol}: {e}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in automated exits execution: {e}")
            results['errors'].append(str(e))
            return results
    
    def monitor_price_alerts(self) -> List[Dict]:
        """
        Monitor positions for significant price moves and generate alerts
        
        Returns:
            List[Dict]: List of price alerts
        """
        alerts = []
        
        for symbol, position in self.paper_engine.positions.items():
            try:
                # Check for significant moves since entry
                price_change_pct = (position.current_price - position.entry_price) / position.entry_price
                
                # Alert for moves > 2%
                if abs(price_change_pct) >= self.alert_threshold:
                    direction = "UP" if price_change_pct > 0 else "DOWN"
                    alerts.append({
                        'symbol': symbol,
                        'type': 'PRICE_MOVE',
                        'direction': direction,
                        'change_pct': price_change_pct,
                        'current_price': position.current_price,
                        'entry_price': position.entry_price,
                        'target_price': position.target_price,
                        'message': f"{symbol} moved {price_change_pct:+.1%} to ${position.current_price:.2f}",
                        'timestamp': datetime.now()
                    })
                
                # Alert for approaching target (within 5%)
                if position.target_price > 0:
                    target_progress = position.get_target_progress()
                    if target_progress >= 0.95:  # Within 5% of target
                        alerts.append({
                            'symbol': symbol,
                            'type': 'NEAR_TARGET',
                            'target_progress': target_progress,
                            'current_price': position.current_price,
                            'target_price': position.target_price,
                            'message': f"{symbol} is {target_progress:.1%} to target price ${position.target_price:.2f}",
                            'timestamp': datetime.now()
                        })
                
                # Alert for stop loss approaching (within 10%)
                if position.stop_loss > 0:
                    stop_distance = (position.current_price - position.stop_loss) / position.current_price
                    if stop_distance <= 0.10:  # Within 10% of stop loss
                        alerts.append({
                            'symbol': symbol,
                            'type': 'NEAR_STOP_LOSS',
                            'stop_distance': stop_distance,
                            'current_price': position.current_price,
                            'stop_loss': position.stop_loss,
                            'message': f"{symbol} is approaching stop loss ${position.stop_loss:.2f}",
                            'timestamp': datetime.now()
                        })
                        
            except Exception as e:
                logger.error(f"Error monitoring alerts for {symbol}: {e}")
        
        return alerts
    
    def run_monitoring_cycle(self, execute_exits: bool = True, dry_run: bool = False) -> Dict:
        """
        Run complete monitoring cycle: update prices, check exits, generate alerts
        
        Args:
            execute_exits: Whether to execute automated exits
            dry_run: If True, don't actually execute trades
            
        Returns:
            Dict: Complete monitoring results
        """
        results = {
            'timestamp': datetime.now(),
            'positions_monitored': len(self.paper_engine.positions),
            'price_updates': 0,
            'alerts': [],
            'exits': None,
            'errors': []
        }
        
        try:
            # Update all position prices
            self.paper_engine.update_positions()
            results['price_updates'] = len(self.paper_engine.positions)
            
            # Check for price alerts
            alerts = self.monitor_price_alerts()
            results['alerts'] = alerts
            
            # Execute automated exits if enabled
            if execute_exits:
                exit_results = self.execute_automated_exits(dry_run=dry_run)
                results['exits'] = exit_results
            
            self.last_price_check = datetime.now()
            
            return results
            
        except Exception as e:
            error_msg = f"Error in monitoring cycle: {e}"
            results['errors'].append(error_msg)
            logger.error(error_msg)
            return results
    
    def get_target_price_summary(self) -> pd.DataFrame:
        """Get summary of all positions with target price information"""
        if not self.paper_engine.positions:
            return pd.DataFrame()
        
        data = []
        for symbol, position in self.paper_engine.positions.items():
            metrics = position.get_price_metrics()
            data.append({
                'Symbol': symbol,
                'Entry Price': position.entry_price,
                'Current Price': position.current_price,
                'Target Price': position.target_price,
                'Stop Loss': position.stop_loss,
                'Return %': metrics['return_pct'],
                'Target Progress': metrics['target_progress'],
                'Days Held': position.days_held,
                'Confidence': position.confidence,
                'Signal Strength': position.signal_strength,
                'Trailing Stop': position.trailing_stop_price if position.trailing_stop_price > 0 else None,
                'Intraday High': position.intraday_high,
                'Intraday Low': position.intraday_low
            })
        
        return pd.DataFrame(data)
    
    def calculate_portfolio_risk_metrics(self) -> Dict:
        """Calculate portfolio-level risk metrics"""
        if not self.paper_engine.positions:
            return {}
        
        positions_df = self.get_target_price_summary()
        portfolio_value = self.paper_engine.get_current_portfolio_value()
        
        # Calculate position sizes
        position_values = []
        for position in self.paper_engine.positions.values():
            position_values.append(position.get_position_value())
        
        total_position_value = sum(position_values)
        
        return {
            'total_positions': len(self.paper_engine.positions),
            'total_position_value': total_position_value,
            'cash_percentage': self.paper_engine.cash / portfolio_value,
            'largest_position_pct': max(position_values) / portfolio_value if position_values else 0,
            'avg_position_size': total_position_value / len(position_values) if position_values else 0,
            'positions_at_target': len([p for p in self.paper_engine.positions.values() 
                                      if p.get_target_progress() >= 1.0]),
            'positions_near_stop': len([p for p in self.paper_engine.positions.values()
                                      if p.current_price <= p.stop_loss * 1.05]),  # Within 5% of stop
            'avg_days_held': np.mean([p.days_held for p in self.paper_engine.positions.values()]),
            'portfolio_unrealized_pnl': sum([p.unrealized_pnl for p in self.paper_engine.positions.values()])
        }