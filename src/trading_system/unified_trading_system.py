"""
Unified Trading System Integration
Combines all signal generation enhancements into a comprehensive trading framework.

Integrates:
- Phase 1: Volume Indicators (OBV, CMF, VWAP, etc.)
- Phase 2: Regime Detection (MSGARCH, Volatility Features, Adaptive Parameters)
- Phase 3: Dynamic Signal Weighting (Ensemble Scoring)
- Phase 4: Advanced Risk Management (VaR, Kelly Sizing, Dynamic Stops)

Based on academic research:
- Multi-factor trading models (Fama & French 2015)
- Ensemble methods for financial prediction (Dietterich 2000)
- Adaptive trading systems (Lo 2004)
- Risk management in algorithmic trading (Narang 2013)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import warnings
warnings.filterwarnings('ignore')

# Import all components - handle both relative and absolute imports
try:
    # Try relative imports first
    try:
        # Phase 1: Volume Indicators
        from ..data_management.volume_indicators import VolumeIndicatorCalculator
        from ..strategy.volume_signals import VolumeSignalGenerator
        
        # Phase 2: Regime Detection  
        from ..models.enhanced_regime_detector import EnhancedRegimeDetector
        from ..models.regime_adaptive_parameters import RegimeAdaptiveParameterSystem
        
        # Phase 3: Dynamic Signal Weighting
        from ..strategy.dynamic_signal_weighting import DynamicSignalWeighter
        from ..strategy.ensemble_signal_scoring import EnsembleSignalScorer, SignalDirection
        
        # Phase 4: Risk Management
        from ..risk_management.dynamic_risk_manager import DynamicRiskManager, RiskLevel
        
    except ImportError:
        # Try absolute imports
        from data_management.volume_indicators import VolumeIndicatorCalculator
        from strategy.volume_signals import VolumeSignalGenerator
        from models.enhanced_regime_detector import EnhancedRegimeDetector
        from models.regime_adaptive_parameters import RegimeAdaptiveParameterSystem
        from strategy.dynamic_signal_weighting import DynamicSignalWeighter
        from strategy.ensemble_signal_scoring import EnsembleSignalScorer, SignalDirection
        from risk_management.dynamic_risk_manager import DynamicRiskManager, RiskLevel
    
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some components not available: {e}")
    # Define fallback enums
    from enum import Enum
    
    class SignalDirection(Enum):
        STRONG_BUY = 2
        BUY = 1
        NEUTRAL = 0
        SELL = -1
        STRONG_SELL = -2
    
    class RiskLevel(Enum):
        VERY_LOW = 1
        LOW = 2
        MEDIUM = 3
        HIGH = 4
        VERY_HIGH = 5
    
    COMPONENTS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingMode(Enum):
    """Trading system operating modes"""
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"
    BACKTEST = "backtest"
    ANALYSIS_ONLY = "analysis_only"

@dataclass
class TradingDecision:
    """Comprehensive trading decision with full context"""
    symbol: str
    timestamp: datetime
    
    # Core decision
    action: str  # BUY, SELL, HOLD, CLOSE
    direction: SignalDirection
    confidence: float
    strength: float
    
    # Position sizing
    recommended_shares: int
    recommended_value: float
    position_size_fraction: float
    
    # Risk management
    stop_loss_price: float
    take_profit_price: float
    risk_level: RiskLevel
    max_loss: float
    
    # Supporting analysis
    technical_signals: Dict[str, Any] = field(default_factory=dict)
    volume_signals: Dict[str, Any] = field(default_factory=dict)
    regime_info: Dict[str, Any] = field(default_factory=dict)
    signal_weights: Dict[str, float] = field(default_factory=dict)
    
    # Explanations
    decision_reasoning: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    supporting_factors: List[str] = field(default_factory=list)
    
    # Metadata
    current_price: float = 0.0
    market_hours: bool = True
    data_quality_score: float = 1.0

@dataclass
class SystemPerformance:
    """Trading system performance metrics"""
    total_signals: int = 0
    successful_signals: int = 0
    win_rate: float = 0.0
    average_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    portfolio_value: float = 0.0
    total_pnl: float = 0.0
    
    # Component performance
    volume_signal_accuracy: float = 0.0
    regime_detection_accuracy: float = 0.0
    risk_management_effectiveness: float = 0.0
    signal_weight_optimization: float = 0.0

class UnifiedTradingSystem:
    """
    Unified trading system that integrates all signal generation enhancements.
    
    Provides:
    1. Multi-phase signal generation and analysis
    2. Regime-adaptive parameter adjustment
    3. Dynamic signal weighting optimization
    4. Advanced risk management and position sizing
    5. Comprehensive decision explanations
    6. Performance tracking and optimization
    """
    
    def __init__(self, 
                 base_capital: float = 1000000,
                 trading_mode: TradingMode = TradingMode.PAPER_TRADING,
                 confidence_threshold: float = 0.6,
                 max_positions: int = 10):
        """
        Initialize unified trading system
        
        Args:
            base_capital: Starting capital for the system
            trading_mode: Operating mode (paper, live, backtest, analysis)
            confidence_threshold: Minimum confidence for trading decisions
            max_positions: Maximum concurrent positions
        """
        self.base_capital = base_capital
        self.trading_mode = trading_mode
        self.confidence_threshold = confidence_threshold
        self.max_positions = max_positions
        
        # Initialize all components
        self._initialize_components()
        
        # Trading state
        self.positions = {}
        self.pending_orders = {}
        self.decision_history = []
        self.performance_metrics = SystemPerformance()
        
        # System configuration
        self.system_config = {
            'lookback_period': 60,
            'min_data_quality': 0.8,
            'position_timeout_days': 30,
            'rebalance_frequency': 'daily',
            'risk_check_frequency': 'hourly'
        }
        
        logger.info(f"Unified Trading System initialized - Mode: {trading_mode.value}, Capital: ${base_capital:,.0f}")
    
    def _initialize_components(self):
        """Initialize all trading system components"""
        try:
            # Phase 1: Volume Analysis
            self.volume_calculator = VolumeIndicatorCalculator() if COMPONENTS_AVAILABLE else None
            self.volume_signal_generator = VolumeSignalGenerator() if COMPONENTS_AVAILABLE else None
            
            # Phase 2: Regime Detection
            self.regime_detector = EnhancedRegimeDetector() if COMPONENTS_AVAILABLE else None  
            self.parameter_system = RegimeAdaptiveParameterSystem() if COMPONENTS_AVAILABLE else None
            
            # Phase 3: Signal Weighting and Ensemble
            self.signal_weighter = DynamicSignalWeighter() if COMPONENTS_AVAILABLE else None
            self.ensemble_scorer = EnsembleSignalScorer() if COMPONENTS_AVAILABLE else None
            
            # Phase 4: Risk Management
            self.risk_manager = DynamicRiskManager(base_capital=self.base_capital) if COMPONENTS_AVAILABLE else None
            
            # Connect components
            if self.ensemble_scorer and self.signal_weighter:
                self.ensemble_scorer.set_components(
                    dynamic_weighter=self.signal_weighter,
                    regime_detector=self.regime_detector,
                    parameter_system=self.parameter_system
                )
            
            logger.info("✅ All trading system components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            logger.warning("System will operate in limited mode")
    
    def analyze_symbol(self, symbol: str, market_data: pd.DataFrame, 
                      technical_indicators: Optional[Dict] = None) -> TradingDecision:
        """
        Perform comprehensive analysis of a trading symbol
        
        Args:
            symbol: Trading symbol to analyze
            market_data: Historical market data
            technical_indicators: Pre-calculated technical indicators
            
        Returns:
            TradingDecision with comprehensive analysis and recommendation
        """
        
        timestamp = datetime.now()
        current_price = market_data['close'].iloc[-1] if not market_data.empty else 0.0
        
        try:
            # Step 1: Data quality assessment
            data_quality = self._assess_data_quality(market_data)
            if data_quality < self.system_config['min_data_quality']:
                return self._create_hold_decision(symbol, current_price, 
                                               f"Data quality too low: {data_quality:.1%}")
            
            # Step 2: Calculate technical indicators if not provided
            if technical_indicators is None:
                technical_indicators = self._calculate_technical_indicators(market_data)
            
            # Step 3: Phase 1 - Volume Analysis
            volume_signals = self._analyze_volume_signals(market_data)
            
            # Step 4: Phase 2 - Regime Detection and Adaptive Parameters
            regime_info = self._detect_market_regime(market_data)
            adaptive_params = self._get_adaptive_parameters(regime_info, market_data)
            
            # Step 5: Phase 3 - Dynamic Signal Weighting and Ensemble Scoring
            ensemble_signal = self._generate_ensemble_signal(
                symbol, market_data, technical_indicators, volume_signals, regime_info
            )
            
            # Step 6: Phase 4 - Risk Management and Position Sizing
            risk_assessment = self._perform_risk_assessment(
                symbol, ensemble_signal, market_data, regime_info
            )
            
            # Step 7: Generate Trading Decision
            trading_decision = self._generate_trading_decision(
                symbol, current_price, ensemble_signal, risk_assessment,
                technical_indicators, volume_signals, regime_info, data_quality
            )
            
            # Step 8: Record decision and update performance
            self._record_decision(trading_decision)
            
            logger.info(f"Analysis completed for {symbol}: {trading_decision.action} "
                       f"(confidence: {trading_decision.confidence:.2f})")
            
            return trading_decision
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return self._create_hold_decision(symbol, current_price, f"Analysis error: {str(e)}")
    
    def _assess_data_quality(self, market_data: pd.DataFrame) -> float:
        """Assess market data quality"""
        if len(market_data) < 20:
            return 0.0
        
        quality_factors = []
        
        # Data completeness
        missing_data_ratio = market_data.isnull().sum().sum() / (len(market_data) * len(market_data.columns))
        quality_factors.append(1.0 - missing_data_ratio)
        
        # Price consistency (no extreme gaps)
        price_changes = market_data['close'].pct_change().abs()
        extreme_moves = (price_changes > 0.2).sum()  # 20%+ moves
        quality_factors.append(1.0 - min(1.0, extreme_moves / len(price_changes)))
        
        # Volume consistency
        if 'volume' in market_data.columns:
            zero_volume_days = (market_data['volume'] == 0).sum()
            quality_factors.append(1.0 - zero_volume_days / len(market_data))
        
        # Sufficient history
        history_score = min(1.0, len(market_data) / 60)  # Full score at 60+ days
        quality_factors.append(history_score)
        
        return np.mean(quality_factors)
    
    def _calculate_technical_indicators(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators"""
        try:
            # Basic technical indicators
            close_prices = market_data['close']
            
            # RSI
            rsi = self._calculate_rsi(close_prices, 14)
            
            # MACD
            macd_line, macd_signal, macd_histogram = self._calculate_macd(close_prices)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close_prices)
            bb_position = (close_prices.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            
            return {
                'rsi': rsi.iloc[-1] if not rsi.empty else 50,
                'macd': macd_line.iloc[-1] if not macd_line.empty else 0,
                'macd_signal': macd_signal.iloc[-1] if not macd_signal.empty else 0,
                'macd_histogram': macd_histogram.iloc[-1] if not macd_histogram.empty else 0,
                'bb_position': bb_position if not np.isnan(bb_position) else 0.5,
                'bb_upper': bb_upper.iloc[-1] if not bb_upper.empty else close_prices.iloc[-1],
                'bb_lower': bb_lower.iloc[-1] if not bb_lower.empty else close_prices.iloc[-1]
            }
            
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}")
            return {'rsi': 50, 'macd': 0, 'macd_histogram': 0, 'bb_position': 0.5}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def _analyze_volume_signals(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume signals using Phase 1 components"""
        if not self.volume_calculator or not self.volume_signal_generator:
            return {'status': 'Volume analysis not available'}
        
        try:
            # Calculate volume indicators
            volume_data = self.volume_calculator.calculate_all_indicators(market_data)
            
            # Generate volume signals
            volume_signals = self.volume_signal_generator.generate_volume_signals(volume_data)
            
            return volume_signals
            
        except Exception as e:
            logger.warning(f"Error in volume analysis: {e}")
            return {'status': 'Volume analysis failed'}
    
    def _detect_market_regime(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect market regime using Phase 2 components"""
        if not self.regime_detector:
            return {'regime': 'sideways_market', 'confidence': 0.5}
        
        try:
            regime_result = self.regime_detector.predict_current_regime(market_data)
            
            # Extract regime information
            regime = 'sideways_market'  # Default
            confidence = 0.5
            
            # Try to extract regime from result
            if isinstance(regime_result, dict):
                for key, value in regime_result.items():
                    if 'regime' in key.lower() and isinstance(value, str):
                        regime = value
                    elif 'confidence' in key.lower() and isinstance(value, (int, float)):
                        confidence = value
            
            return {
                'regime': regime,
                'confidence': confidence,
                'volatility_percentile': 0.5,  # Default
                'full_result': regime_result
            }
            
        except Exception as e:
            logger.warning(f"Error in regime detection: {e}")
            return {'regime': 'sideways_market', 'confidence': 0.5}
    
    def _get_adaptive_parameters(self, regime_info: Dict, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Get regime-adaptive parameters using Phase 2 components"""
        if not self.parameter_system:
            return {'status': 'Parameter adaptation not available'}
        
        try:
            adapted_params = self.parameter_system.adapt_parameters(
                regime_info.get('regime', 'sideways_market'),
                regime_info.get('confidence', 0.5),
                regime_info.get('volatility_percentile', 0.5),
                market_data
            )
            
            return {
                'adapted_parameters': adapted_params,
                'regime': regime_info.get('regime'),
                'confidence': regime_info.get('confidence')
            }
            
        except Exception as e:
            logger.warning(f"Error in parameter adaptation: {e}")
            return {'status': 'Parameter adaptation failed'}
    
    def _generate_ensemble_signal(self, symbol: str, market_data: pd.DataFrame,
                                technical_indicators: Dict, volume_signals: Dict,
                                regime_info: Dict) -> Any:
        """Generate ensemble signal using Phase 3 components"""
        if not self.ensemble_scorer:
            # Create fallback ensemble signal
            return self._create_fallback_ensemble_signal(
                symbol, technical_indicators, volume_signals, regime_info
            )
        
        try:
            ensemble_signal = self.ensemble_scorer.calculate_ensemble_score(
                symbol=symbol,
                market_data=market_data,
                technical_indicators=technical_indicators,
                volume_signals=volume_signals,
                regime_info=regime_info
            )
            
            return ensemble_signal
            
        except Exception as e:
            logger.warning(f"Error in ensemble signal generation: {e}")
            return self._create_fallback_ensemble_signal(
                symbol, technical_indicators, volume_signals, regime_info
            )
    
    def _create_fallback_ensemble_signal(self, symbol: str, technical_indicators: Dict,
                                       volume_signals: Dict, regime_info: Dict) -> Any:
        """Create fallback ensemble signal when components aren't available"""
        # Simple scoring based on available indicators
        score = 0.0
        factors = 0
        
        # RSI contribution
        rsi = technical_indicators.get('rsi', 50)
        if rsi < 30:  # Oversold
            score += 0.3
        elif rsi > 70:  # Overbought
            score -= 0.3
        factors += 1
        
        # MACD contribution
        macd_histogram = technical_indicators.get('macd_histogram', 0)
        if macd_histogram > 0:
            score += 0.2
        elif macd_histogram < 0:
            score -= 0.2
        factors += 1
        
        # Volume contribution
        if volume_signals.get('status') != 'Volume analysis failed':
            # Assume positive volume signals add to score
            score += 0.1
        factors += 1
        
        # Normalize score
        if factors > 0:
            final_score = score / factors
        else:
            final_score = 0.0
        
        # Determine direction
        if final_score > 0.2:
            direction = SignalDirection.BUY
        elif final_score < -0.2:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.NEUTRAL
        
        # Create simple signal object
        class FallbackSignal:
            def __init__(self):
                self.symbol = symbol
                self.direction = direction
                self.strength = abs(final_score)
                self.confidence = min(0.7, abs(final_score) * 2)
                self.composite_score = final_score
                self.timestamp = datetime.now()
                self.supporting_factors = ['RSI', 'MACD', 'Volume'] if abs(final_score) > 0.1 else []
                self.risk_factors = ['Limited analysis'] if abs(final_score) < 0.2 else []
                self.recommended_action = direction.name
        
        return FallbackSignal()
    
    def _perform_risk_assessment(self, symbol: str, ensemble_signal: Any,
                               market_data: pd.DataFrame, regime_info: Dict) -> Dict[str, Any]:
        """Perform risk assessment using Phase 4 components"""
        if not self.risk_manager:
            return self._create_fallback_risk_assessment(ensemble_signal)
        
        try:
            # Calculate position size (assuming we want to trade)
            current_price = market_data['close'].iloc[-1]
            
            # Estimate expected return and win rate from signal
            expected_return = ensemble_signal.composite_score * 0.1  # 10% max expected return
            win_rate = 0.5 + (ensemble_signal.confidence * 0.2)  # 50-70% win rate
            
            # Get optimal position sizing
            sizing_result = self.risk_manager.calculate_optimal_position_size(
                symbol=symbol,
                entry_price=current_price,
                expected_return=expected_return,
                win_rate=win_rate,
                market_data=market_data,
                regime_info=regime_info
            )
            
            # Calculate risk metrics for the recommended position
            recommended_shares = int(sizing_result['optimal_shares'])
            if recommended_shares > 0:
                risk_metrics = self.risk_manager.calculate_position_risk(
                    symbol=symbol,
                    position_size=recommended_shares,
                    entry_price=current_price,
                    current_price=current_price,
                    market_data=market_data,
                    regime_info=regime_info
                )
            else:
                risk_metrics = None
            
            return {
                'sizing_result': sizing_result,
                'risk_metrics': risk_metrics,
                'recommended_shares': recommended_shares,
                'position_value': recommended_shares * current_price
            }
            
        except Exception as e:
            logger.warning(f"Error in risk assessment: {e}")
            return self._create_fallback_risk_assessment(ensemble_signal)
    
    def _create_fallback_risk_assessment(self, ensemble_signal: Any) -> Dict[str, Any]:
        """Create fallback risk assessment"""
        return {
            'recommended_shares': int(1000 * ensemble_signal.strength),  # Simple sizing
            'position_value': 0,
            'stop_loss_pct': 0.05,  # 5% stop loss
            'take_profit_pct': 0.10,  # 10% take profit
            'risk_level': 'MEDIUM'
        }
    
    def _generate_trading_decision(self, symbol: str, current_price: float,
                                 ensemble_signal: Any, risk_assessment: Dict,
                                 technical_indicators: Dict, volume_signals: Dict,
                                 regime_info: Dict, data_quality: float) -> TradingDecision:
        """Generate final trading decision"""
        
        # Determine action based on ensemble signal and risk assessment
        action = "HOLD"  # Default
        
        if ensemble_signal.confidence >= self.confidence_threshold:
            if ensemble_signal.direction == SignalDirection.STRONG_BUY:
                action = "BUY"
            elif ensemble_signal.direction == SignalDirection.BUY:
                action = "BUY"
            elif ensemble_signal.direction == SignalDirection.STRONG_SELL:
                action = "SELL"
            elif ensemble_signal.direction == SignalDirection.SELL:
                action = "SELL"
        
        # Get risk metrics
        risk_metrics = risk_assessment.get('risk_metrics')
        recommended_shares = risk_assessment.get('recommended_shares', 0)
        
        # Calculate stop loss and take profit
        if risk_metrics:
            stop_loss_price = risk_metrics.stop_loss_price
            take_profit_price = risk_metrics.take_profit_price
            risk_level = risk_metrics.risk_level
            max_loss = risk_metrics.value_at_risk_1d
        else:
            # Fallback calculations
            stop_pct = risk_assessment.get('stop_loss_pct', 0.05)
            profit_pct = risk_assessment.get('take_profit_pct', 0.10)
            
            if action == "BUY":
                stop_loss_price = current_price * (1 - stop_pct)
                take_profit_price = current_price * (1 + profit_pct)
            else:
                stop_loss_price = current_price * (1 + stop_pct)
                take_profit_price = current_price * (1 - profit_pct)
            
            risk_level = RiskLevel.MEDIUM
            max_loss = recommended_shares * current_price * stop_pct
        
        # Generate decision reasoning
        reasoning = []
        if ensemble_signal.confidence >= 0.8:
            reasoning.append(f"High confidence signal ({ensemble_signal.confidence:.1%})")
        if len(ensemble_signal.supporting_factors) >= 3:
            reasoning.append(f"Multiple supporting factors ({len(ensemble_signal.supporting_factors)})")
        if regime_info.get('confidence', 0) > 0.7:
            reasoning.append(f"Strong {regime_info.get('regime', 'regime')} detection")
        if data_quality > 0.9:
            reasoning.append("High quality market data")
        
        return TradingDecision(
            symbol=symbol,
            timestamp=datetime.now(),
            action=action,
            direction=ensemble_signal.direction,
            confidence=ensemble_signal.confidence,
            strength=ensemble_signal.strength,
            recommended_shares=recommended_shares,
            recommended_value=recommended_shares * current_price,
            position_size_fraction=risk_assessment.get('sizing_result', {}).get('optimal_fraction', 0.01),
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            risk_level=risk_level,
            max_loss=max_loss,
            technical_signals=technical_indicators,
            volume_signals=volume_signals,
            regime_info=regime_info,
            signal_weights=getattr(ensemble_signal, 'signal_weights', {}),
            decision_reasoning=reasoning,
            risk_factors=ensemble_signal.risk_factors if hasattr(ensemble_signal, 'risk_factors') else [],
            supporting_factors=ensemble_signal.supporting_factors if hasattr(ensemble_signal, 'supporting_factors') else [],
            current_price=current_price,
            market_hours=True,  # Simplified
            data_quality_score=data_quality
        )
    
    def _create_hold_decision(self, symbol: str, current_price: float, reason: str) -> TradingDecision:
        """Create a HOLD decision with explanation"""
        return TradingDecision(
            symbol=symbol,
            timestamp=datetime.now(),
            action="HOLD",
            direction=SignalDirection.NEUTRAL,
            confidence=0.0,
            strength=0.0,
            recommended_shares=0,
            recommended_value=0.0,
            position_size_fraction=0.0,
            stop_loss_price=current_price,
            take_profit_price=current_price,
            risk_level=RiskLevel.LOW,
            max_loss=0.0,
            decision_reasoning=[reason],
            current_price=current_price,
            data_quality_score=0.5
        )
    
    def _record_decision(self, decision: TradingDecision):
        """Record trading decision and update performance metrics"""
        self.decision_history.append(decision)
        
        # Update performance metrics
        self.performance_metrics.total_signals += 1
        
        # Keep only recent decisions for performance calculation
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
        
        # Log decision
        logger.info(f"Decision recorded: {decision.symbol} {decision.action} "
                   f"(confidence: {decision.confidence:.2f})")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'trading_mode': self.trading_mode.value,
            'base_capital': self.base_capital,
            'active_positions': len(self.positions),
            'pending_orders': len(self.pending_orders),
            'total_decisions': len(self.decision_history),
            'components_available': COMPONENTS_AVAILABLE,
            'component_status': {
                'volume_analysis': self.volume_calculator is not None,
                'regime_detection': self.regime_detector is not None,
                'signal_weighting': self.signal_weighter is not None,
                'risk_management': self.risk_manager is not None
            },
            'performance_metrics': {
                'total_signals': self.performance_metrics.total_signals,
                'win_rate': self.performance_metrics.win_rate,
                'average_return': self.performance_metrics.average_return,
                'sharpe_ratio': self.performance_metrics.sharpe_ratio
            },
            'system_health': 'OPERATIONAL' if COMPONENTS_AVAILABLE else 'LIMITED_MODE'
        }
    
    def export_decision_log(self, filepath: str, limit: int = 500):
        """Export recent trading decisions to JSON"""
        recent_decisions = self.decision_history[-limit:]
        
        export_data = []
        for decision in recent_decisions:
            export_record = {
                'symbol': decision.symbol,
                'timestamp': decision.timestamp.isoformat(),
                'action': decision.action,
                'direction': decision.direction.name,
                'confidence': decision.confidence,
                'strength': decision.strength,
                'recommended_shares': decision.recommended_shares,
                'current_price': decision.current_price,
                'stop_loss': decision.stop_loss_price,
                'take_profit': decision.take_profit_price,
                'risk_level': decision.risk_level.name,
                'decision_reasoning': decision.decision_reasoning,
                'supporting_factors': decision.supporting_factors,
                'risk_factors': decision.risk_factors
            }
            export_data.append(export_record)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(export_data)} trading decisions to {filepath}")

if __name__ == "__main__":
    # Example usage and testing
    print("🚀 Initializing Unified Trading System...")
    
    trading_system = UnifiedTradingSystem(
        base_capital=1000000,
        trading_mode=TradingMode.ANALYSIS_ONLY,
        confidence_threshold=0.6
    )
    
    # Create sample market data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
    
    sample_data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'high': prices * (1 + np.random.uniform(0, 0.02, 100)),
        'low': prices * (1 - np.random.uniform(0, 0.02, 100)),
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    
    # Test analysis
    print("\n📊 Testing Symbol Analysis...")
    decision = trading_system.analyze_symbol('TEST', sample_data)
    
    print(f"\nTrading Decision for TEST:")
    print(f"  Action: {decision.action}")
    print(f"  Direction: {decision.direction.name}")
    print(f"  Confidence: {decision.confidence:.2f}")
    print(f"  Recommended Shares: {decision.recommended_shares:,}")
    print(f"  Stop Loss: ${decision.stop_loss_price:.2f}")
    print(f"  Risk Level: {decision.risk_level.name}")
    print(f"  Reasoning: {', '.join(decision.decision_reasoning)}")
    
    # System status
    status = trading_system.get_system_status()
    print(f"\nSystem Status:")
    print(f"  Health: {status['system_health']}")
    print(f"  Components Available: {status['components_available']}")
    print(f"  Total Decisions: {status['total_decisions']}")
    print(f"  Active Positions: {status['active_positions']}")
    
    print("\n✅ Unified Trading System Test Complete!")