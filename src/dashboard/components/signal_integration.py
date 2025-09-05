"""
Signal Integration for Paper Trading
Connects the main signal system with paper trading engine
"""

import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
from .paper_trading import PaperTradingEngine

logger = logging.getLogger(__name__)

class SignalTradingIntegrator:
    """
    Integrates signals from the main dashboard with paper trading
    """
    
    def __init__(self, paper_engine: PaperTradingEngine):
        self.paper_engine = paper_engine
        self.last_processed_signals = {}
        
    def process_signal_dataframe(self, signals_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process signals DataFrame and execute paper trades
        
        Args:
            signals_df: DataFrame with signal data from main dashboard
            
        Returns:
            Dict with processing results
        """
        if signals_df is None or signals_df.empty:
            return {"status": "no_data", "trades_executed": 0}
        
        trades_executed = 0
        processing_results = []
        
        try:
            # Process each signal
            for _, row in signals_df.iterrows():
                symbol = row.get('Symbol', '')
                signal_direction = row.get('Signal', 'NEUTRAL')
                confidence = row.get('Confidence', 0.0)
                strength = row.get('Strength', 0.0)
                
                if not symbol or pd.isna(confidence):
                    continue
                
                # Convert signal format to paper trading format with enhanced signal support
                signal_data = {
                    'direction': self._convert_signal_direction(signal_direction),
                    'confidence': confidence,
                    'strength': strength,
                    'source': 'dashboard_signals',
                    'timestamp': datetime.now(),
                    'additional_data': {
                        'price': row.get('Close', 0.0),
                        'volume': row.get('Volume', 0),
                        'market_cap': row.get('Market_Cap', 0),
                        'volatility': row.get('Volatility_20d', 0.0)
                    },
                    # Enhanced signal fields
                    'position_size': row.get('Position_Size', None),
                    'market_regime': row.get('Market_Regime', None),
                    'should_trade': row.get('Should_Trade', True),
                    'trade_rationale': row.get('Trade_Rationale', None)
                }
                
                # Check if this signal is different from last processed
                last_signal = self.last_processed_signals.get(symbol, {})
                if (last_signal.get('direction') != signal_data['direction'] or 
                    abs(last_signal.get('confidence', 0) - confidence) > 0.1):
                    
                    # Process the signal
                    result = self._process_individual_signal(symbol, signal_data)
                    if result['action_taken']:
                        trades_executed += 1
                        processing_results.append(result)
                    
                    # Update last processed
                    self.last_processed_signals[symbol] = signal_data
            
            # Update all positions with current prices
            self.paper_engine.update_positions()
            
            return {
                "status": "success",
                "trades_executed": trades_executed,
                "signals_processed": len(signals_df),
                "results": processing_results
            }
            
        except Exception as e:
            logger.error(f"Error processing signals: {e}")
            return {
                "status": "error",
                "error": str(e),
                "trades_executed": trades_executed
            }
    
    def _convert_signal_direction(self, signal_str: str) -> str:
        """Convert dashboard signal format to paper trading format"""
        if pd.isna(signal_str):
            return 'NEUTRAL'
        
        signal_upper = str(signal_str).upper()
        
        # Map various signal formats
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
    
    def _process_individual_signal(self, symbol: str, signal_data: Dict) -> Dict[str, Any]:
        """Process individual signal and execute trade if needed"""
        try:
            direction = signal_data['direction']
            confidence = signal_data['confidence']
            
            result = {
                'symbol': symbol,
                'signal_direction': direction,
                'confidence': confidence,
                'action_taken': False,
                'action_type': None,
                'reason': ''
            }
            
            # Check enhanced signal should_trade flag first
            should_trade = signal_data.get('should_trade', True)
            if not should_trade:
                result['reason'] = f'Enhanced signal suggests not to trade due to: {signal_data.get("trade_rationale", "market conditions")}'
                return result
            
            # Check if we should buy
            if direction in ['BUY', 'STRONG_BUY'] and symbol not in self.paper_engine.positions:
                if confidence >= self.paper_engine.min_confidence:
                    success = self.paper_engine.execute_buy_order(symbol, signal_data)
                    if success:
                        result['action_taken'] = True
                        result['action_type'] = 'BUY'
                        position_info = f"confidence: {confidence:.1%}"
                        if signal_data.get('position_size'):
                            position_info += f", Kelly position: {signal_data['position_size']:.1%}"
                        if signal_data.get('market_regime'):
                            position_info += f", regime: {signal_data['market_regime']}"
                        result['reason'] = f'Buy signal with {position_info}'
                        logger.info(f"Paper trade executed: BUY {symbol} ({position_info})")
                    else:
                        result['reason'] = 'Buy signal but order failed (insufficient funds or max positions)'
                else:
                    result['reason'] = f'Buy signal but confidence ({confidence:.1%}) below minimum ({self.paper_engine.min_confidence:.1%})'
            
            # Check if we should sell
            elif direction in ['SELL', 'STRONG_SELL'] and symbol in self.paper_engine.positions:
                success = self.paper_engine.execute_sell_order(symbol, "SELL_SIGNAL")
                if success:
                    result['action_taken'] = True
                    result['action_type'] = 'SELL'
                    result['reason'] = f'Sell signal with {confidence:.1%} confidence'
                    logger.info(f"Paper trade executed: SELL {symbol} (confidence: {confidence:.1%})")
                else:
                    result['reason'] = 'Sell signal but order failed'
            
            # Handle neutral signals
            elif direction == 'NEUTRAL':
                result['reason'] = 'Neutral signal - no action'
            
            # Handle cases where we already have position but get buy signal, or vice versa
            elif direction in ['BUY', 'STRONG_BUY'] and symbol in self.paper_engine.positions:
                result['reason'] = 'Buy signal but already holding position'
            elif direction in ['SELL', 'STRONG_SELL'] and symbol not in self.paper_engine.positions:
                result['reason'] = 'Sell signal but no position to sell'
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing signal for {symbol}: {e}")
            return {
                'symbol': symbol,
                'action_taken': False,
                'error': str(e)
            }
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of signal processing"""
        engine_metrics = self.paper_engine.get_performance_metrics()
        
        return {
            'paper_trading_metrics': engine_metrics,
            'signals_tracked': len(self.last_processed_signals),
            'last_update': datetime.now(),
            'active_positions': len(self.paper_engine.positions),
            'total_trades': self.paper_engine.total_trades
        }

def integrate_signals_with_paper_trading(signals_df: pd.DataFrame, 
                                       paper_engine: Optional[PaperTradingEngine] = None) -> Dict[str, Any]:
    """
    Convenience function to integrate signals with paper trading
    
    Args:
        signals_df: Signals DataFrame from main dashboard
        paper_engine: Optional paper trading engine (will create if None)
    
    Returns:
        Processing results dictionary
    """
    if paper_engine is None:
        paper_engine = PaperTradingEngine()
    
    integrator = SignalTradingIntegrator(paper_engine)
    return integrator.process_signal_dataframe(signals_df)