"""
Historical Signal Trading Engine
Executes paper trades based on previous day's signals for realistic validation
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from .paper_trading import PaperTradingEngine
import os
import json

logger = logging.getLogger(__name__)

class HistoricalSignalTrader:
    """
    Executes paper trades based on yesterday's signals to simulate realistic trading
    """
    
    def __init__(self, paper_engine: PaperTradingEngine):
        self.paper_engine = paper_engine
        self.signal_history_file = self._get_signal_history_path()
        self.signal_history = self._load_signal_history()
        
    def _get_signal_history_path(self) -> str:
        """Get path for storing signal history"""
        dashboard_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(dashboard_dir, '..', 'signal_history.json')
    
    def _load_signal_history(self) -> Dict:
        """Load historical signals from file"""
        try:
            if os.path.exists(self.signal_history_file):
                with open(self.signal_history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading signal history: {e}")
        
        return {}
    
    def _save_signal_history(self):
        """Save signal history to file"""
        try:
            with open(self.signal_history_file, 'w') as f:
                json.dump(self.signal_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving signal history: {e}")
    
    def store_daily_signals(self, signals_df: pd.DataFrame, date: str = None) -> bool:
        """
        Store today's signals for tomorrow's trading
        
        Args:
            signals_df: DataFrame with current signals
            date: Date string (YYYY-MM-DD), defaults to today
        
        Returns:
            bool: Success status
        """
        if signals_df is None or signals_df.empty:
            logger.warning(f"Cannot store signals: DataFrame is None or empty")
            return False
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Convert signals to storable format with flexible column mapping
            signals_data = {}
            
            # Create column mapping to handle both raw and display column names
            column_mapping = {
                'symbol': ['Symbol', '📊 Symbol', 'symbol'],
                'signal': ['Signal', '📈 Signal', 'signal', 'signal_direction'],
                'confidence': ['Confidence', '🎯 Confidence', 'confidence', 'Conf', 'CONFIDENCE', 'signal_confidence'],
                'strength': ['Strength', '⚡ Strength', 'strength', 'STRENGTH', 'signal_strength'],
                'close': ['Close', '💰 Price', 'close', 'Close_Price', 'Price', 'current_price'],
                'volume': ['Volume', '🔊 Volume', 'volume', 'VOLUME'],
                'market_cap': ['Market_Cap', 'market_cap', 'MarketCap'],
                'volatility': ['Volatility_20d', 'Volatility', 'volatility', 'Vol_20d', 'volatility_20d'],
                'rsi': ['RSI_14', 'RSI', '📊 RSI', 'rsi', 'rsi_14'],
                'macd': ['MACD', '📈 MACD', 'macd', 'macd_histogram']
            }
            
            def get_column_value(row, field_name):
                """Get value from row using flexible column mapping"""
                possible_cols = column_mapping.get(field_name, [field_name])
                for col in possible_cols:
                    if col in row and pd.notna(row[col]):
                        return row[col]
                return None
            
            logger.info(f"Processing {len(signals_df)} rows for signal storage on {date}")
            logger.info(f"Available columns: {list(signals_df.columns)}")
            
            # Debug: Show first few rows to understand data structure
            if len(signals_df) > 0:
                logger.info(f"First row sample: {dict(signals_df.iloc[0])}")
            
            signals_stored = 0
            skipped_no_symbol = 0
            for idx, row in signals_df.iterrows():
                symbol = get_column_value(row, 'symbol')
                if not symbol:
                    skipped_no_symbol += 1
                    logger.debug(f"Row {idx}: No symbol found in {dict(row)}")
                    continue
                
                # Extract signal data with fallbacks
                signal = get_column_value(row, 'signal') or 'NEUTRAL'
                confidence = get_column_value(row, 'confidence') or 0.0
                strength = get_column_value(row, 'strength') or 0.0
                close_price = get_column_value(row, 'close') or 0.0
                volume = get_column_value(row, 'volume') or 0
                market_cap = get_column_value(row, 'market_cap') or 0.0
                volatility = get_column_value(row, 'volatility') or 0.0
                rsi = get_column_value(row, 'rsi') or 0.0
                macd = get_column_value(row, 'macd') or 0.0
                
                try:
                    signals_data[symbol] = {
                        'signal': str(signal),
                        'confidence': float(confidence),
                        'strength': float(strength),
                        'close_price': float(close_price),
                        'volume': int(volume),
                        'market_cap': float(market_cap),
                        'volatility': float(volatility),
                        'rsi': float(rsi),
                        'macd': float(macd),
                        'timestamp': datetime.now().isoformat()
                    }
                    signals_stored += 1
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to convert data for {symbol}: {e}")
                    continue
            
            # Store signals for this date
            self.signal_history[date] = signals_data
            
            # Keep only last 30 days of history
            dates = sorted(self.signal_history.keys())
            if len(dates) > 30:
                for old_date in dates[:-30]:
                    del self.signal_history[old_date]
            
            self._save_signal_history()
            logger.info(f"Stored {signals_stored} signals for date {date}")
            if skipped_no_symbol > 0:
                logger.warning(f"Skipped {skipped_no_symbol} rows due to missing symbol")
            if signals_stored == 0:
                logger.error(f"No signals stored! Check column mapping. Available columns: {list(signals_df.columns)}")
            return signals_stored > 0
            
        except Exception as e:
            logger.error(f"Error storing daily signals: {e}")
            return False
    
    def get_yesterdays_signals(self, target_date: str = None) -> Dict:
        """
        Get yesterday's signals for today's trading
        
        Args:
            target_date: Specific date to get signals for (YYYY-MM-DD)
        
        Returns:
            Dict: Yesterday's signals
        """
        if target_date is None:
            yesterday = datetime.now() - timedelta(days=1)
            target_date = yesterday.strftime('%Y-%m-%d')
        
        return self.signal_history.get(target_date, {})
    
    def execute_yesterdays_signals(self, target_date: str = None, 
                                 dry_run: bool = False) -> Dict[str, any]:
        """
        Execute trades based on yesterday's signals
        
        Args:
            target_date: Date to get signals from (defaults to yesterday)
            dry_run: If True, don't actually execute trades
        
        Returns:
            Dict with execution results
        """
        try:
            # Get yesterday's signals
            yesterdays_signals = self.get_yesterdays_signals(target_date)
            
            if not yesterdays_signals:
                return {
                    'status': 'no_signals',
                    'message': 'No signals found for yesterday',
                    'trades_executed': 0
                }
            
            trades_executed = 0
            trade_results = []
            errors = []
            
            # Get current market prices for execution
            symbols = list(yesterdays_signals.keys())
            current_prices = self._get_current_prices(symbols)
            
            if not current_prices:
                return {
                    'status': 'price_error',
                    'message': 'Could not fetch current prices',
                    'trades_executed': 0
                }
            
            # Process each signal
            for symbol, signal_data in yesterdays_signals.items():
                try:
                    result = self._process_historical_signal(
                        symbol, signal_data, current_prices.get(symbol), dry_run
                    )
                    
                    if result['action_taken']:
                        trades_executed += 1
                        trade_results.append(result)
                        
                except Exception as e:
                    error_msg = f"Error processing {symbol}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            # Update all positions after trading
            if not dry_run:
                self.paper_engine.update_positions()
            
            return {
                'status': 'success',
                'trades_executed': trades_executed,
                'signals_processed': len(yesterdays_signals),
                'results': trade_results,
                'errors': errors,
                'dry_run': dry_run
            }
            
        except Exception as e:
            logger.error(f"Error executing yesterday's signals: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'trades_executed': 0
            }
    
    def _get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current market prices for symbols"""
        try:
            prices = {}
            
            # Batch fetch prices
            if len(symbols) == 1:
                ticker = yf.Ticker(symbols[0])
                hist = ticker.history(period='1d')
                if not hist.empty:
                    prices[symbols[0]] = hist['Close'].iloc[-1]
            else:
                tickers = yf.Tickers(' '.join(symbols))
                for symbol in symbols:
                    try:
                        hist = tickers.tickers[symbol].history(period='1d')
                        if not hist.empty:
                            prices[symbol] = hist['Close'].iloc[-1]
                    except Exception as e:
                        logger.warning(f"Could not get price for {symbol}: {e}")
            
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching current prices: {e}")
            return {}
    
    def _process_historical_signal(self, symbol: str, signal_data: Dict, 
                                 current_price: float, dry_run: bool) -> Dict:
        """Process individual historical signal"""
        
        signal_direction = self._normalize_signal(signal_data.get('signal', 'NEUTRAL'))
        confidence = signal_data.get('confidence', 0.0)
        
        result = {
            'symbol': symbol,
            'signal_direction': signal_direction,
            'confidence': confidence,
            'current_price': current_price,
            'action_taken': False,
            'action_type': None,
            'reason': '',
            'dry_run': dry_run
        }
        
        if current_price is None or current_price <= 0:
            result['reason'] = 'Invalid current price'
            return result
        
        # Check for buy signals
        if signal_direction in ['BUY', 'STRONG_BUY']:
            if symbol in self.paper_engine.positions:
                result['reason'] = 'Already holding position'
            elif confidence < self.paper_engine.min_confidence:
                result['reason'] = f'Confidence {confidence:.1%} below minimum {self.paper_engine.min_confidence:.1%}'
            else:
                # Execute buy order
                if not dry_run:
                    # Create signal data in expected format
                    formatted_signal = {
                        'direction': signal_direction,
                        'confidence': confidence,
                        'strength': signal_data.get('strength', 0.0),
                        'source': 'historical_signal',
                        'original_date': signal_data.get('timestamp', ''),
                        'execution_date': datetime.now().isoformat()
                    }
                    
                    success = self.paper_engine.execute_buy_order(symbol, formatted_signal)
                    if success:
                        result['action_taken'] = True
                        result['action_type'] = 'BUY'
                        result['reason'] = f'Executed buy order based on yesterday\'s signal'
                    else:
                        result['reason'] = 'Buy order failed (insufficient funds or max positions)'
                else:
                    result['action_taken'] = True
                    result['action_type'] = 'BUY'
                    result['reason'] = f'Would execute buy order (dry run)'
        
        # Check for sell signals
        elif signal_direction in ['SELL', 'STRONG_SELL']:
            if symbol not in self.paper_engine.positions:
                result['reason'] = 'No position to sell'
            else:
                # Execute sell order
                if not dry_run:
                    success = self.paper_engine.execute_sell_order(symbol, "HISTORICAL_SELL_SIGNAL")
                    if success:
                        result['action_taken'] = True
                        result['action_type'] = 'SELL'
                        result['reason'] = f'Executed sell order based on yesterday\'s signal'
                    else:
                        result['reason'] = 'Sell order failed'
                else:
                    result['action_taken'] = True
                    result['action_type'] = 'SELL'
                    result['reason'] = f'Would execute sell order (dry run)'
        
        else:
            result['reason'] = 'Neutral signal - no action'
        
        return result
    
    def _normalize_signal(self, signal_str: str) -> str:
        """Normalize signal string format"""
        if pd.isna(signal_str):
            return 'NEUTRAL'
        
        signal_upper = str(signal_str).upper()
        
        if 'STRONG BUY' in signal_upper or signal_upper == 'STRONG_BUY':
            return 'STRONG_BUY'
        elif 'BUY' in signal_upper:
            return 'BUY'
        elif 'STRONG SELL' in signal_upper or signal_upper == 'STRONG_SELL':
            return 'STRONG_SELL'
        elif 'SELL' in signal_upper:
            return 'SELL'
        elif 'HOLD' in signal_upper:
            return 'NEUTRAL'
        else:
            return 'NEUTRAL'
    
    def get_signal_history_summary(self) -> Dict[str, any]:
        """Get summary of stored signal history"""
        if not self.signal_history:
            return {
                'total_days': 0,
                'total_signals': 0,
                'date_range': None,
                'symbols_tracked': []
            }
        
        dates = sorted(self.signal_history.keys())
        all_symbols = set()
        total_signals = 0
        
        for date_signals in self.signal_history.values():
            all_symbols.update(date_signals.keys())
            total_signals += len(date_signals)
        
        return {
            'total_days': len(dates),
            'total_signals': total_signals,
            'date_range': f"{dates[0]} to {dates[-1]}" if dates else None,
            'symbols_tracked': sorted(list(all_symbols)),
            'avg_signals_per_day': total_signals / len(dates) if dates else 0
        }
    
    def get_available_dates(self) -> List[str]:
        """Get list of dates with stored signals"""
        return sorted(self.signal_history.keys())
    
    def simulate_historical_trading(self, start_date: str, end_date: str = None) -> Dict:
        """
        Simulate historical trading over a date range
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to latest available
        
        Returns:
            Dict with simulation results
        """
        available_dates = self.get_available_dates()
        
        if not available_dates:
            return {'status': 'no_data', 'message': 'No historical signals available'}
        
        if end_date is None:
            end_date = available_dates[-1]
        
        # Filter dates in range
        simulation_dates = [d for d in available_dates if start_date <= d <= end_date]
        
        if not simulation_dates:
            return {'status': 'no_dates', 'message': 'No signals in specified date range'}
        
        # Save current state
        original_state = {
            'cash': self.paper_engine.cash,
            'positions': self.paper_engine.positions.copy(),
            'trade_history': self.paper_engine.trade_history.copy()
        }
        
        simulation_results = {
            'start_date': start_date,
            'end_date': end_date,
            'dates_simulated': len(simulation_dates),
            'total_trades': 0,
            'daily_results': []
        }
        
        try:
            for date in simulation_dates:
                # Execute signals for this date
                next_day = (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                result = self.execute_yesterdays_signals(target_date=date, dry_run=False)
                
                simulation_results['total_trades'] += result.get('trades_executed', 0)
                simulation_results['daily_results'].append({
                    'date': next_day,
                    'signal_date': date,
                    'trades': result.get('trades_executed', 0),
                    'portfolio_value': self.paper_engine.get_current_portfolio_value()
                })
            
            # Calculate final performance
            final_value = self.paper_engine.get_current_portfolio_value()
            initial_value = self.paper_engine.initial_capital
            total_return = ((final_value - initial_value) / initial_value) * 100
            
            simulation_results.update({
                'status': 'success',
                'initial_value': initial_value,
                'final_value': final_value,
                'total_return': total_return,
                'total_pnl': final_value - initial_value
            })
            
        except Exception as e:
            logger.error(f"Error in historical simulation: {e}")
            # Restore original state
            self.paper_engine.cash = original_state['cash']
            self.paper_engine.positions = original_state['positions']
            self.paper_engine.trade_history = original_state['trade_history']
            
            simulation_results.update({
                'status': 'error',
                'error': str(e)
            })
        
        return simulation_results