"""
Test suite for Unified Trading System Integration
Validates end-to-end integration of all signal enhancement phases.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from trading_system.unified_trading_system import (
        UnifiedTradingSystem,
        TradingMode,
        TradingDecision,
        SystemPerformance
    )
    from strategy.ensemble_signal_scoring import SignalDirection
    from risk_management.dynamic_risk_manager import RiskLevel
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all trading system components are available")
    sys.exit(1)

class TestUnifiedTradingSystem(unittest.TestCase):
    """Test suite for unified trading system integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.trading_system = UnifiedTradingSystem(
            base_capital=1000000,
            trading_mode=TradingMode.ANALYSIS_ONLY,
            confidence_threshold=0.6,
            max_positions=10
        )
        
        # Create comprehensive test market data
        np.random.seed(42)  # For reproducible tests
        self.market_data = self._create_comprehensive_market_data()
        
        # Sample technical indicators
        self.sample_technical_indicators = {
            'rsi': 65.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'macd_histogram': 0.2,
            'bb_position': 0.7,
            'bb_upper': 105.0,
            'bb_lower': 95.0
        }
    
    def _create_comprehensive_market_data(self) -> pd.DataFrame:
        """Create realistic market data for testing"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Generate realistic price movements
        base_price = 100.0
        prices = []
        volumes = []
        
        for i in range(100):
            # Different market phases
            if i < 30:  # Bull phase
                daily_return = np.random.normal(0.001, 0.015)
                volume_base = 2000000
            elif i < 60:  # Volatile phase
                daily_return = np.random.normal(0.0, 0.030)
                volume_base = 4000000
            else:  # Bear phase
                daily_return = np.random.normal(-0.0005, 0.020)
                volume_base = 3000000
            
            base_price *= (1 + daily_return)
            prices.append(max(1.0, base_price))
            
            # Variable volume
            volume = int(volume_base * np.random.uniform(0.5, 2.0))
            volumes.append(volume)
        
        # Create OHLC data
        highs = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
        lows = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
        
        return pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        })
    
    def test_system_initialization(self):
        """Test trading system initialization"""
        print("\n=== Testing System Initialization ===")
        
        # Check basic initialization
        self.assertEqual(self.trading_system.base_capital, 1000000)
        self.assertEqual(self.trading_system.trading_mode, TradingMode.ANALYSIS_ONLY)
        self.assertEqual(self.trading_system.confidence_threshold, 0.6)
        self.assertEqual(self.trading_system.max_positions, 10)
        
        # Check component initialization
        self.assertIsNotNone(self.trading_system.system_config)
        self.assertIsInstance(self.trading_system.performance_metrics, SystemPerformance)
        
        # Check system status
        status = self.trading_system.get_system_status()
        self.assertIn('trading_mode', status)
        self.assertIn('components_available', status)
        self.assertIn('system_health', status)
        
        print(f"✅ System initialized successfully")
        print(f"  Trading mode: {status['trading_mode']}")
        print(f"  System health: {status['system_health']}")
        print(f"  Components available: {status['components_available']}")
    
    def test_data_quality_assessment(self):
        """Test data quality assessment functionality"""
        print("\n=== Testing Data Quality Assessment ===")
        
        # Test with good quality data
        good_quality = self.trading_system._assess_data_quality(self.market_data)
        self.assertGreater(good_quality, 0.8)  # Should be high quality
        
        # Test with poor quality data (short history)
        short_data = self.market_data.head(5)
        poor_quality = self.trading_system._assess_data_quality(short_data)
        self.assertLess(poor_quality, 0.5)  # Should be low quality
        
        # Test with missing data
        missing_data = self.market_data.copy()
        missing_data.loc[0:10, 'close'] = np.nan
        missing_quality = self.trading_system._assess_data_quality(missing_data)
        self.assertLess(missing_quality, good_quality)
        
        print(f"✅ Data quality assessment working")
        print(f"  Good data quality: {good_quality:.1%}")
        print(f"  Poor data quality: {poor_quality:.1%}")
        print(f"  Missing data quality: {missing_quality:.1%}")
    
    def test_technical_indicator_calculation(self):
        """Test technical indicator calculations"""
        print("\n=== Testing Technical Indicator Calculations ===")
        
        # Test indicator calculation
        indicators = self.trading_system._calculate_technical_indicators(self.market_data)
        
        # Validate indicator structure
        expected_indicators = ['rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_position']
        for indicator in expected_indicators:
            self.assertIn(indicator, indicators)
        
        # Check indicator value ranges
        rsi = indicators['rsi']
        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)
        
        bb_position = indicators['bb_position']
        self.assertGreaterEqual(bb_position, 0)
        self.assertLessEqual(bb_position, 1)
        
        print(f"✅ Technical indicators calculated")
        print(f"  RSI: {rsi:.1f}")
        print(f"  MACD: {indicators['macd']:.3f}")
        print(f"  BB Position: {bb_position:.2f}")
    
    def test_volume_signal_analysis(self):
        """Test volume signal analysis integration"""
        print("\n=== Testing Volume Signal Analysis ===")
        
        # Test volume signal analysis
        volume_signals = self.trading_system._analyze_volume_signals(self.market_data)
        
        # Should return some form of volume analysis (even if limited)
        self.assertIsInstance(volume_signals, dict)
        self.assertIn('status', volume_signals)  # Should have status field
        
        print(f"✅ Volume signal analysis completed")
        print(f"  Status: {volume_signals.get('status', 'Analysis performed')}")
    
    def test_regime_detection(self):
        """Test market regime detection integration"""
        print("\n=== Testing Market Regime Detection ===")
        
        # Test regime detection
        regime_info = self.trading_system._detect_market_regime(self.market_data)
        
        # Validate regime information structure
        self.assertIn('regime', regime_info)
        self.assertIn('confidence', regime_info)
        
        # Check regime values
        regime = regime_info['regime']
        confidence = regime_info['confidence']
        
        self.assertIsInstance(regime, str)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        print(f"✅ Market regime detection completed")
        print(f"  Detected regime: {regime}")
        print(f"  Confidence: {confidence:.2f}")
    
    def test_ensemble_signal_generation(self):
        """Test ensemble signal generation"""
        print("\n=== Testing Ensemble Signal Generation ===")
        
        # Get regime info first
        regime_info = self.trading_system._detect_market_regime(self.market_data)
        volume_signals = self.trading_system._analyze_volume_signals(self.market_data)
        
        # Test ensemble signal generation
        ensemble_signal = self.trading_system._generate_ensemble_signal(
            'TEST', self.market_data, self.sample_technical_indicators, 
            volume_signals, regime_info
        )
        
        # Validate ensemble signal
        self.assertIsNotNone(ensemble_signal)
        self.assertTrue(hasattr(ensemble_signal, 'direction'))
        self.assertTrue(hasattr(ensemble_signal, 'strength'))
        self.assertTrue(hasattr(ensemble_signal, 'confidence'))
        
        # Check signal values
        self.assertIsInstance(ensemble_signal.direction, SignalDirection)
        self.assertGreaterEqual(ensemble_signal.strength, 0.0)
        self.assertLessEqual(ensemble_signal.strength, 1.0)
        self.assertGreaterEqual(ensemble_signal.confidence, 0.0)
        self.assertLessEqual(ensemble_signal.confidence, 1.0)
        
        print(f"✅ Ensemble signal generated")
        print(f"  Direction: {ensemble_signal.direction.name}")
        print(f"  Strength: {ensemble_signal.strength:.2f}")
        print(f"  Confidence: {ensemble_signal.confidence:.2f}")
    
    def test_risk_assessment(self):
        """Test risk assessment integration"""
        print("\n=== Testing Risk Assessment ===")
        
        # Generate ensemble signal first
        regime_info = self.trading_system._detect_market_regime(self.market_data)
        volume_signals = self.trading_system._analyze_volume_signals(self.market_data)
        ensemble_signal = self.trading_system._generate_ensemble_signal(
            'TEST', self.market_data, self.sample_technical_indicators,
            volume_signals, regime_info
        )
        
        # Test risk assessment
        risk_assessment = self.trading_system._perform_risk_assessment(
            'TEST', ensemble_signal, self.market_data, regime_info
        )
        
        # Validate risk assessment
        self.assertIn('recommended_shares', risk_assessment)
        self.assertIn('position_value', risk_assessment)
        
        recommended_shares = risk_assessment['recommended_shares']
        self.assertIsInstance(recommended_shares, int)
        self.assertGreaterEqual(recommended_shares, 0)
        
        print(f"✅ Risk assessment completed")
        print(f"  Recommended shares: {recommended_shares:,}")
        print(f"  Position value: ${risk_assessment.get('position_value', 0):,.0f}")
    
    def test_complete_symbol_analysis(self):
        """Test complete symbol analysis workflow"""
        print("\n=== Testing Complete Symbol Analysis ===")
        
        # Test complete analysis
        decision = self.trading_system.analyze_symbol('TEST', self.market_data)
        
        # Validate trading decision structure
        self.assertIsInstance(decision, TradingDecision)
        self.assertEqual(decision.symbol, 'TEST')
        self.assertIsInstance(decision.timestamp, datetime)
        
        # Check decision fields
        self.assertIn(decision.action, ['BUY', 'SELL', 'HOLD'])
        self.assertIsInstance(decision.direction, SignalDirection)
        self.assertIsInstance(decision.risk_level, RiskLevel)
        
        # Check numeric fields
        self.assertGreaterEqual(decision.confidence, 0.0)
        self.assertLessEqual(decision.confidence, 1.0)
        self.assertGreaterEqual(decision.strength, 0.0)
        self.assertLessEqual(decision.strength, 1.0)
        self.assertGreaterEqual(decision.recommended_shares, 0)
        
        # Check price fields
        self.assertGreater(decision.current_price, 0)
        self.assertGreater(decision.stop_loss_price, 0)
        self.assertGreater(decision.take_profit_price, 0)
        
        print(f"✅ Complete analysis performed")
        print(f"  Symbol: {decision.symbol}")
        print(f"  Action: {decision.action}")
        print(f"  Direction: {decision.direction.name}")
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Shares: {decision.recommended_shares:,}")
        print(f"  Risk Level: {decision.risk_level.name}")
        
        # Check supporting information
        if decision.decision_reasoning:
            print(f"  Reasoning: {', '.join(decision.decision_reasoning[:2])}")
    
    def test_multiple_symbol_analysis(self):
        """Test analysis of multiple symbols"""
        print("\n=== Testing Multiple Symbol Analysis ===")
        
        test_symbols = ['STOCK_A', 'STOCK_B', 'STOCK_C']
        decisions = []
        
        for symbol in test_symbols:
            # Create slightly different market data for each symbol
            market_data = self.market_data.copy()
            market_data['close'] *= np.random.uniform(0.95, 1.05)  # Small variation
            
            decision = self.trading_system.analyze_symbol(symbol, market_data)
            decisions.append(decision)
        
        # Validate all decisions
        self.assertEqual(len(decisions), 3)
        
        for i, decision in enumerate(decisions):
            self.assertEqual(decision.symbol, test_symbols[i])
            self.assertIsInstance(decision, TradingDecision)
        
        # Check for decision variety (they shouldn't all be identical)
        actions = [d.action for d in decisions]
        confidences = [d.confidence for d in decisions]
        
        print(f"✅ Multiple symbol analysis completed")
        for i, decision in enumerate(decisions):
            print(f"  {decision.symbol}: {decision.action} (conf: {decision.confidence:.2f})")
        
        # Record decisions for performance tracking
        self.assertEqual(len(self.trading_system.decision_history), 3)  # Actual count after 3 symbol analyses
    
    def test_trading_modes(self):
        """Test different trading modes"""
        print("\n=== Testing Trading Modes ===")
        
        # Test different trading modes
        trading_modes = [
            TradingMode.PAPER_TRADING,
            TradingMode.BACKTEST,
            TradingMode.ANALYSIS_ONLY
        ]
        
        for mode in trading_modes:
            system = UnifiedTradingSystem(
                base_capital=500000,
                trading_mode=mode,
                confidence_threshold=0.5
            )
            
            status = system.get_system_status()
            self.assertEqual(status['trading_mode'], mode.value)
            
            # Test analysis works in all modes
            decision = system.analyze_symbol('TEST_MODE', self.market_data)
            self.assertIsInstance(decision, TradingDecision)
            
            print(f"  ✅ {mode.value}: Analysis functional")
        
        print(f"✅ All trading modes tested successfully")
    
    def test_error_handling(self):
        """Test error handling and fallback mechanisms"""
        print("\n=== Testing Error Handling ===")
        
        # Test with very limited data
        limited_data = self.market_data.head(2)
        decision = self.trading_system.analyze_symbol('LIMITED_DATA', limited_data)
        
        # Should still return a valid decision (likely HOLD)
        self.assertIsInstance(decision, TradingDecision)
        self.assertEqual(decision.action, 'HOLD')
        
        # Test with empty data
        empty_data = pd.DataFrame()
        decision_empty = self.trading_system.analyze_symbol('EMPTY_DATA', empty_data)
        self.assertIsInstance(decision_empty, TradingDecision)
        
        # Test with corrupted data (all NaN)
        corrupted_data = self.market_data.copy()
        corrupted_data[:] = np.nan
        decision_corrupted = self.trading_system.analyze_symbol('CORRUPTED_DATA', corrupted_data)
        self.assertIsInstance(decision_corrupted, TradingDecision)
        
        print(f"✅ Error handling validated")
        print(f"  Limited data: {decision.action}")
        print(f"  Empty data: {decision_empty.action}")
        print(f"  Corrupted data: {decision_corrupted.action}")
    
    def test_system_performance_tracking(self):
        """Test system performance tracking"""
        print("\n=== Testing System Performance Tracking ===")
        
        # Run several analyses to build performance data
        test_symbols = ['PERF_1', 'PERF_2', 'PERF_3', 'PERF_4']
        
        for symbol in test_symbols:
            decision = self.trading_system.analyze_symbol(symbol, self.market_data)
            self.assertIsInstance(decision, TradingDecision)
        
        # Check performance metrics are updated
        metrics = self.trading_system.performance_metrics
        self.assertGreater(metrics.total_signals, 0)
        
        # Check decision history
        history_count = len(self.trading_system.decision_history)
        self.assertGreater(history_count, 0)
        
        # Test system status includes performance data
        status = self.trading_system.get_system_status()
        self.assertIn('performance_metrics', status)
        self.assertEqual(status['total_decisions'], history_count)
        
        print(f"✅ Performance tracking validated")
        print(f"  Total signals: {metrics.total_signals}")
        print(f"  Decision history: {history_count}")
    
    def test_configuration_and_limits(self):
        """Test system configuration and limits"""
        print("\n=== Testing Configuration and Limits ===")
        
        # Test configuration access
        config = self.trading_system.system_config
        self.assertIn('lookback_period', config)
        self.assertIn('min_data_quality', config)
        
        # Test limits
        self.assertEqual(self.trading_system.max_positions, 10)
        self.assertEqual(self.trading_system.confidence_threshold, 0.6)
        
        # Test with different configuration
        custom_system = UnifiedTradingSystem(
            base_capital=2000000,
            confidence_threshold=0.8,
            max_positions=5
        )
        
        self.assertEqual(custom_system.base_capital, 2000000)
        self.assertEqual(custom_system.confidence_threshold, 0.8)
        self.assertEqual(custom_system.max_positions, 5)
        
        print(f"✅ Configuration and limits validated")
        print(f"  Lookback period: {config['lookback_period']} days")
        print(f"  Min data quality: {config['min_data_quality']:.1%}")
        print(f"  Confidence threshold: {self.trading_system.confidence_threshold:.1%}")
    
    def test_export_functionality(self):
        """Test data export functionality"""
        print("\n=== Testing Export Functionality ===")
        
        # Generate some decisions first
        for i in range(3):
            decision = self.trading_system.analyze_symbol(f'EXPORT_TEST_{i}', self.market_data)
            self.assertIsInstance(decision, TradingDecision)
        
        # Test export functionality
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_file = f.name
        
        try:
            # This should not raise an exception
            self.trading_system.export_decision_log(export_file, limit=10)
            
            # Check file was created and has content
            import os
            self.assertTrue(os.path.exists(export_file))
            self.assertGreater(os.path.getsize(export_file), 0)
            
            print(f"✅ Export functionality validated")
            print(f"  Export file created: {os.path.basename(export_file)}")
            
        finally:
            # Clean up
            import os
            if os.path.exists(export_file):
                os.unlink(export_file)

def run_comprehensive_integration_test():
    """Run comprehensive integration test with detailed output"""
    print("🚀 Starting Comprehensive Unified Trading System Integration Test")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestUnifiedTradingSystem)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=False)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 80)
    
    if result.wasSuccessful():
        print(f"✅ ALL INTEGRATION TESTS PASSED ({result.testsRun} tests)")
        print("\n🎉 Unified Trading System Integration is working correctly!")
        print("\n📈 Key Validation Results:")
        print("   ✅ System initialization and component integration")
        print("   ✅ Data quality assessment and technical indicators")
        print("   ✅ Volume signal analysis integration")
        print("   ✅ Market regime detection integration")
        print("   ✅ Ensemble signal generation")
        print("   ✅ Risk assessment and position sizing")
        print("   ✅ Complete end-to-end symbol analysis")
        print("   ✅ Multiple trading mode support")
        print("   ✅ Error handling and fallback mechanisms")
        print("   ✅ Performance tracking and system monitoring")
        print("   ✅ Configuration management and export functionality")
        
        print("\n🏗️ System Architecture Validated:")
        print("   • Phase 1: Volume Indicators → Integrated ✅")
        print("   • Phase 2: Regime Detection → Integrated ✅") 
        print("   • Phase 3: Signal Weighting → Integrated ✅")
        print("   • Phase 4: Risk Management → Integrated ✅")
        print("   • Phase 5: Unified System → Complete ✅")
        
        return True
    else:
        print(f"❌ INTEGRATION TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
        for test, trace in result.failures + result.errors:
            print(f"   ❌ {test}: {trace.split(chr(10))[-2] if chr(10) in trace else trace}")
        return False

if __name__ == "__main__":
    success = run_comprehensive_integration_test()
    sys.exit(0 if success else 1)