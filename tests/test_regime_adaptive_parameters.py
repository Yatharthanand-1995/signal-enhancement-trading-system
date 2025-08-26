"""
Test suite for Regime-Adaptive Parameter System
Validates parameter adaptation, signal enhancement, and regime-specific adjustments.
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
    from models.regime_adaptive_parameters import (
        RegimeAdaptiveParameterSystem, 
        RegimeParameters, 
        AdaptiveSignal
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the regime_adaptive_parameters.py file is in the correct location")
    sys.exit(1)

class TestRegimeAdaptiveParameters(unittest.TestCase):
    """Test suite for regime-adaptive parameter system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.system = RegimeAdaptiveParameterSystem()
        
        # Create realistic test market data
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Simulate different market phases
        prices = []
        base_price = 100.0
        
        # Bull market phase (30 days)
        for i in range(30):
            base_price += np.random.normal(0.5, 1.0)  # Upward drift
            prices.append(max(1.0, base_price))
        
        # Volatile sideways phase (40 days)  
        for i in range(40):
            base_price += np.random.normal(0.0, 2.0)  # High volatility
            prices.append(max(1.0, base_price))
            
        # Bear market phase (30 days)
        for i in range(30):
            base_price += np.random.normal(-0.3, 1.5)  # Downward drift
            prices.append(max(1.0, base_price))
        
        self.market_data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'high': [p * (1 + np.random.uniform(0.001, 0.02)) for p in prices],
            'low': [p * (1 - np.random.uniform(0.001, 0.02)) for p in prices],
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        
        # Sample trading signal for testing
        self.sample_signal = {
            'type': 'RSI_OVERSOLD',
            'symbol': 'TEST',
            'price': 100.0,
            'strength': 0.7,
            'timestamp': datetime.now(),
            'indicators': {'rsi': 25, 'volume_ratio': 1.5}
        }
    
    def test_system_initialization(self):
        """Test that system initializes with correct research-based parameters"""
        print("\n=== Testing System Initialization ===")
        
        # Check that all regime parameters are initialized
        expected_regimes = ['bull_market', 'bear_market', 'sideways_market', 'volatile_market']
        for regime in expected_regimes:
            self.assertIn(regime, self.system.regime_parameters)
            params = self.system.regime_parameters[regime]
            self.assertIsInstance(params, RegimeParameters)
            
            # Validate parameter ranges based on research
            self.assertGreaterEqual(params.rsi_overbought, 65)
            self.assertLessEqual(params.rsi_overbought, 85)
            self.assertGreaterEqual(params.rsi_oversold, 15)
            self.assertLessEqual(params.rsi_oversold, 35)
            
            print(f"✅ {regime}: RSI({params.rsi_oversold}-{params.rsi_overbought}), "
                  f"Stop: {params.stop_loss_pct:.1%}")
        
        print(f"✅ All {len(expected_regimes)} regime parameter sets initialized correctly")
    
    def test_parameter_adaptation_bull_market(self):
        """Test parameter adaptation for bull market conditions"""
        print("\n=== Testing Bull Market Parameter Adaptation ===")
        
        regime = 'bull_market'
        confidence = 0.8
        volatility = 0.3  # Low volatility
        
        adapted_params = self.system.adapt_parameters(regime, confidence, volatility, self.market_data)
        base_params = self.system.regime_parameters[regime]
        
        # Verify adaptation logic
        self.assertEqual(adapted_params.regime_name, regime)
        
        # In low volatility, RSI levels should be closer to base levels
        self.assertLessEqual(abs(adapted_params.rsi_overbought - base_params.rsi_overbought), 5.0)
        
        # Position size should be adjusted for high confidence
        self.assertGreater(adapted_params.position_size_multiplier, 0.5)
        
        # Signal confirmation should be reduced for high confidence
        self.assertLessEqual(adapted_params.signal_confirmation_required, base_params.signal_confirmation_required)
        
        print(f"✅ Bull market adaptation: RSI {adapted_params.rsi_oversold:.1f}-{adapted_params.rsi_overbought:.1f}")
        print(f"✅ Position multiplier: {adapted_params.position_size_multiplier:.2f}")
        print(f"✅ Confirmations required: {adapted_params.signal_confirmation_required}")
    
    def test_parameter_adaptation_volatile_market(self):
        """Test parameter adaptation for high volatility conditions"""
        print("\n=== Testing Volatile Market Parameter Adaptation ===")
        
        regime = 'volatile_market'
        confidence = 0.6  # Lower confidence
        volatility = 0.9  # High volatility
        
        adapted_params = self.system.adapt_parameters(regime, confidence, volatility, self.market_data)
        base_params = self.system.regime_parameters[regime]
        
        # High volatility should expand RSI levels
        self.assertGreater(adapted_params.rsi_overbought, base_params.rsi_overbought)
        self.assertLess(adapted_params.rsi_oversold, base_params.rsi_oversold)
        
        # MACD threshold should be higher for noise filtering
        self.assertGreater(adapted_params.macd_signal_threshold, base_params.macd_signal_threshold)
        
        # Volume multiplier should be higher
        self.assertGreater(adapted_params.volume_breakout_multiplier, base_params.volume_breakout_multiplier)
        
        # Position size should be smaller due to low confidence and high volatility
        self.assertLess(adapted_params.position_size_multiplier, base_params.position_size_multiplier)
        
        print(f"✅ Volatile market RSI levels: {adapted_params.rsi_oversold:.1f}-{adapted_params.rsi_overbought:.1f}")
        print(f"✅ MACD threshold increased: {adapted_params.macd_signal_threshold:.4f} vs {base_params.macd_signal_threshold:.4f}")
        print(f"✅ Volume multiplier: {adapted_params.volume_breakout_multiplier:.1f}")
        print(f"✅ Conservative position size: {adapted_params.position_size_multiplier:.2f}")
    
    def test_adaptive_signal_creation(self):
        """Test creation of adaptive signals with regime-specific enhancements"""
        print("\n=== Testing Adaptive Signal Creation ===")
        
        regime = 'bull_market'
        confidence = 0.8
        volatility = 0.4
        
        # Get adapted parameters
        adapted_params = self.system.adapt_parameters(regime, confidence, volatility, self.market_data)
        
        # Create adaptive signal
        adaptive_signal = self.system.create_adaptive_signal(
            self.sample_signal, regime, confidence, adapted_params, self.market_data
        )
        
        # Verify adaptive signal structure
        self.assertIsInstance(adaptive_signal, AdaptiveSignal)
        self.assertEqual(adaptive_signal.regime, regime)
        self.assertEqual(adaptive_signal.regime_confidence, confidence)
        
        # Signal strength should be adjusted (bull market gets boost)
        original_strength = self.sample_signal['strength']
        self.assertGreater(adaptive_signal.adjusted_strength, original_strength * 0.9)  # At least some boost
        
        # Risk multiplier should be reasonable for bull market
        self.assertLess(adaptive_signal.risk_multiplier, 2.0)  # Not too risky
        
        # Position size should be within reasonable bounds
        self.assertGreaterEqual(adaptive_signal.recommended_position_size, 0.01)
        self.assertLessEqual(adaptive_signal.recommended_position_size, 0.20)
        
        print(f"✅ Signal strength: {original_strength:.2f} → {adaptive_signal.adjusted_strength:.2f}")
        print(f"✅ Risk multiplier: {adaptive_signal.risk_multiplier:.2f}")
        print(f"✅ Position size: {adaptive_signal.recommended_position_size:.1%}")
        print(f"✅ Timestamp: {adaptive_signal.timestamp}")
    
    def test_volatility_percentile_calculation(self):
        """Test volatility percentile calculation accuracy"""
        print("\n=== Testing Volatility Percentile Calculation ===")
        
        # Test with sufficient data
        volatility_pct = self.system._calculate_current_volatility_percentile(self.market_data)
        
        self.assertGreaterEqual(volatility_pct, 0.0)
        self.assertLessEqual(volatility_pct, 1.0)
        
        print(f"✅ Volatility percentile: {volatility_pct:.2f}")
        
        # Test with insufficient data
        small_data = self.market_data.head(10)
        volatility_pct_small = self.system._calculate_current_volatility_percentile(small_data)
        self.assertEqual(volatility_pct_small, 0.5)  # Should default to 0.5
        
        print(f"✅ Small data default: {volatility_pct_small:.2f}")
    
    def test_regime_strength_adjustments(self):
        """Test regime-specific strength adjustments"""
        print("\n=== Testing Regime Strength Adjustments ===")
        
        regimes_and_expected = {
            'bull_market': (0.05, 0.25),      # Positive adjustment
            'bear_market': (-0.20, 0.0),      # Negative adjustment
            'sideways_market': (0.0, 0.15),   # Small positive
            'volatile_market': (-0.30, -0.10)  # Negative adjustment
        }
        
        for regime, (min_adj, max_adj) in regimes_and_expected.items():
            adjustment = self.system._calculate_regime_strength_adjustment(regime, self.market_data)
            self.assertGreaterEqual(adjustment, min_adj)
            self.assertLessEqual(adjustment, max_adj)
            print(f"✅ {regime}: {adjustment:+.2f} strength adjustment")
    
    def test_risk_multiplier_calculation(self):
        """Test risk multiplier calculations for different scenarios"""
        print("\n=== Testing Risk Multiplier Calculations ===")
        
        test_scenarios = [
            ('bull_market', 0.3, 0.8, 'Low vol, high confidence'),
            ('bear_market', 0.7, 0.6, 'High vol, medium confidence'),
            ('volatile_market', 0.9, 0.5, 'Very high vol, low confidence'),
            ('sideways_market', 0.4, 0.7, 'Medium vol, good confidence')
        ]
        
        for regime, volatility, confidence, description in test_scenarios:
            risk_multiplier = self.system._calculate_risk_multiplier(regime, volatility, confidence)
            
            # Risk multiplier should be positive and reasonable
            self.assertGreater(risk_multiplier, 0.5)
            self.assertLess(risk_multiplier, 4.0)
            
            print(f"✅ {regime} ({description}): {risk_multiplier:.2f} risk multiplier")
    
    def test_position_size_calculation(self):
        """Test Kelly-inspired position size calculations"""
        print("\n=== Testing Position Size Calculations ===")
        
        test_cases = [
            (0.8, 1.0, 1.2, 'High strength, low risk, bull market'),
            (0.4, 2.0, 0.5, 'Medium strength, high risk, volatile market'),
            (0.9, 0.8, 1.0, 'Very high strength, low risk, normal conditions'),
            (0.2, 3.0, 0.7, 'Low strength, very high risk, bear market')
        ]
        
        for strength, risk_mult, base_mult, description in test_cases:
            position_size = self.system._calculate_position_size(strength, risk_mult, base_mult)
            
            # Position size should be within reasonable bounds
            self.assertGreaterEqual(position_size, 0.01)  # At least 1%
            self.assertLessEqual(position_size, 0.20)     # At most 20%
            
            print(f"✅ {description}: {position_size:.1%} position size")
    
    def test_parameter_history_tracking(self):
        """Test that parameter adaptations are properly tracked"""
        print("\n=== Testing Parameter History Tracking ===")
        
        initial_count = len(self.system.parameter_history)
        
        # Perform several adaptations
        regimes = ['bull_market', 'bear_market', 'sideways_market']
        for regime in regimes:
            self.system.adapt_parameters(regime, 0.7, 0.5, self.market_data)
        
        # Verify history tracking
        self.assertEqual(len(self.system.parameter_history), initial_count + 3)
        
        # Check statistics
        stats = self.system.get_regime_statistics()
        self.assertIn('total_adaptations', stats)
        self.assertIn('recent_regime_distribution', stats)
        self.assertGreater(stats['total_adaptations'], 0)
        
        print(f"✅ Parameter history tracked: {len(self.system.parameter_history)} adaptations")
        print(f"✅ Statistics generated: {len(stats)} metrics")
        
        # Test regime distribution in stats
        if 'recent_regime_distribution' in stats and stats['recent_regime_distribution']:
            print(f"✅ Regime distribution: {stats['recent_regime_distribution']}")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n=== Testing Edge Cases ===")
        
        # Test unknown regime
        adapted_params = self.system.adapt_parameters('unknown_regime', 0.5, 0.5, self.market_data)
        self.assertEqual(adapted_params.regime_name, 'sideways_market')  # Should fall back
        
        print("✅ Unknown regime handled with fallback")
        
        # Test extreme confidence values
        extreme_high = self.system.adapt_parameters('bull_market', 1.0, 0.5, self.market_data)
        extreme_low = self.system.adapt_parameters('bull_market', 0.0, 0.5, self.market_data)
        
        self.assertGreater(extreme_high.position_size_multiplier, extreme_low.position_size_multiplier)
        print("✅ Extreme confidence values handled correctly")
        
        # Test extreme volatility values
        extreme_vol_high = self.system.adapt_parameters('bull_market', 0.7, 1.0, self.market_data)
        extreme_vol_low = self.system.adapt_parameters('bull_market', 0.7, 0.0, self.market_data)
        
        self.assertGreater(extreme_vol_high.macd_signal_threshold, extreme_vol_low.macd_signal_threshold)
        print("✅ Extreme volatility values handled correctly")
        
        # Test empty market data
        empty_data = pd.DataFrame({'close': [], 'volume': []})
        vol_pct = self.system._calculate_current_volatility_percentile(empty_data)
        self.assertEqual(vol_pct, 0.5)  # Should default
        
        print("✅ Empty market data handled with defaults")
    
    def test_signal_comparison_across_regimes(self):
        """Test how the same signal is adapted differently across regimes"""
        print("\n=== Testing Signal Adaptation Across Regimes ===")
        
        regimes = ['bull_market', 'bear_market', 'sideways_market', 'volatile_market']
        confidence = 0.7
        volatility = 0.5
        
        adaptive_signals = {}
        
        for regime in regimes:
            adapted_params = self.system.adapt_parameters(regime, confidence, volatility, self.market_data)
            adaptive_signal = self.system.create_adaptive_signal(
                self.sample_signal, regime, confidence, adapted_params, self.market_data
            )
            adaptive_signals[regime] = adaptive_signal
            
            print(f"  {regime:15s}: strength={adaptive_signal.adjusted_strength:.2f}, "
                  f"risk={adaptive_signal.risk_multiplier:.2f}, "
                  f"pos_size={adaptive_signal.recommended_position_size:.1%}")
        
        # Verify logical relationships
        # Bull market should have higher position sizes than bear market
        self.assertGreater(
            adaptive_signals['bull_market'].recommended_position_size,
            adaptive_signals['bear_market'].recommended_position_size
        )
        
        # Volatile market should have higher risk multiplier
        self.assertGreater(
            adaptive_signals['volatile_market'].risk_multiplier,
            adaptive_signals['bull_market'].risk_multiplier
        )
        
        print("✅ Signal adaptation varies appropriately across regimes")

def run_comprehensive_test():
    """Run comprehensive test with detailed output"""
    print("🚀 Starting Comprehensive Regime-Adaptive Parameter System Test")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestRegimeAdaptiveParameters)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=False)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    if result.wasSuccessful():
        print(f"✅ ALL TESTS PASSED ({result.testsRun} tests)")
        print("\n🎉 Regime-Adaptive Parameter System is working correctly!")
        print("\n📈 Key Validation Results:")
        print("   ✅ Research-based parameter initialization")
        print("   ✅ Dynamic parameter adaptation based on regime and volatility")
        print("   ✅ Signal strength adjustments per regime")
        print("   ✅ Kelly-inspired position sizing")
        print("   ✅ Risk multiplier calculations")
        print("   ✅ Parameter history tracking")
        print("   ✅ Edge case handling")
        
        return True
    else:
        print(f"❌ TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
        for test, trace in result.failures + result.errors:
            print(f"   ❌ {test}: {trace.split(chr(10))[-2] if chr(10) in trace else trace}")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)