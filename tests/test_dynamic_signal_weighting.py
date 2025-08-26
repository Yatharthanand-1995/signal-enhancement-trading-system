"""
Test suite for Dynamic Signal Weighting Framework
Validates dynamic weight allocation, performance tracking, and regime-based adjustments.
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
    from strategy.dynamic_signal_weighting import (
        DynamicSignalWeighter,
        SignalWeight,
        WeightingResult
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the dynamic_signal_weighting.py file is in the correct location")
    sys.exit(1)

class TestDynamicSignalWeighting(unittest.TestCase):
    """Test suite for dynamic signal weighting system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.weighter = DynamicSignalWeighter(
            lookback_period=30,
            min_weight=0.05,
            max_weight=0.35
        )
        
        # Register test signals
        self.test_signals = ['rsi', 'macd', 'bollinger_bands', 'volume_breakout', 'obv_divergence']
        for signal in self.test_signals:
            self.weighter.register_signal(signal)
        
        # Sample performance data
        self.sample_performance = {
            'rsi': 0.65,
            'macd': 0.58,
            'bollinger_bands': 0.72,
            'volume_breakout': 0.45,
            'obv_divergence': 0.69
        }
        
        # Sample current signals with metadata
        self.sample_current_signals = {
            'rsi': {
                'strength': 0.7,
                'supporting_indicators': {'volume': True, 'trend': True},
                'volume_confirmed': True,
                'signal_age': 2
            },
            'macd': {
                'strength': 0.6,
                'volume_confirmed': True,
                'signal_age': 1
            },
            'bollinger_bands': {
                'strength': 0.8,
                'signal_age': 3,
                'supporting_indicators': {'rsi': True, 'volume': False}
            },
            'volume_breakout': {
                'strength': 0.5,
                'volume_confirmed': True
            },
            'obv_divergence': {
                'strength': 0.7,
                'supporting_indicators': {'price': True},
                'signal_age': 4
            }
        }
    
    def test_initialization_and_base_weights(self):
        """Test system initialization and research-based base weights"""
        print("\n=== Testing Initialization and Base Weights ===")
        
        # Check base weights are research-based and sum appropriately
        base_weights = self.weighter.base_weights
        total_base_weight = sum(base_weights.values())
        
        # Verify key research-backed signals have appropriate weights
        self.assertGreater(base_weights.get('rsi', 0), 0.15)  # RSI should be high-weighted
        self.assertGreater(base_weights.get('macd', 0), 0.12)  # MACD important
        self.assertGreater(base_weights.get('volume_breakout', 0), 0.10)  # Volume important
        
        # Check total weight is reasonable (should be close to 1.0)
        self.assertGreater(total_base_weight, 0.8)
        self.assertLess(total_base_weight, 1.2)
        
        print(f"✅ Base weights initialized: Total = {total_base_weight:.3f}")
        for signal, weight in sorted(base_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {signal}: {weight:.3f}")
    
    def test_signal_registration(self):
        """Test signal registration and weight initialization"""
        print("\n=== Testing Signal Registration ===")
        
        # Test registering new signal
        new_signal = 'test_signal'
        self.weighter.register_signal(new_signal, base_weight=0.12)
        
        # Verify registration
        self.assertIn(new_signal, self.weighter.signals)
        signal_obj = self.weighter.signals[new_signal]
        
        self.assertEqual(signal_obj.signal_type, new_signal)
        self.assertEqual(signal_obj.base_weight, 0.12)
        self.assertEqual(signal_obj.current_weight, 0.12)
        self.assertEqual(signal_obj.performance_multiplier, 1.0)
        self.assertIsInstance(signal_obj.last_updated, datetime)
        
        print(f"✅ Signal '{new_signal}' registered with weight: {signal_obj.current_weight:.3f}")
        
        # Test auto-weight assignment
        auto_signal = 'auto_weight_signal'
        self.weighter.register_signal(auto_signal)
        auto_signal_obj = self.weighter.signals[auto_signal]
        self.assertGreater(auto_signal_obj.base_weight, 0.05)  # Should get reasonable default
        
        print(f"✅ Signal '{auto_signal}' auto-assigned weight: {auto_signal_obj.base_weight:.3f}")
    
    def test_bull_market_weight_adjustments(self):
        """Test weight adjustments for bull market conditions"""
        print("\n=== Testing Bull Market Weight Adjustments ===")
        
        # Bull market conditions
        result = self.weighter.calculate_dynamic_weights(
            market_regime='bull_market',
            regime_confidence=0.8,
            volatility_percentile=0.3,  # Low volatility
            signal_performance=self.sample_performance,
            current_signals=self.sample_current_signals
        )
        
        # Validate result structure
        self.assertIsInstance(result, WeightingResult)
        self.assertEqual(len(result.signal_weights), len(self.test_signals))
        
        # Check weight normalization (should sum to 1.0)
        total_weight = sum(result.signal_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=3)
        
        # In bull markets, trend-following signals should be emphasized
        # MACD and moving averages should get boost, RSI might be reduced
        weights = result.signal_weights
        
        # Check bounds are respected
        for signal_type, weight in weights.items():
            self.assertGreaterEqual(weight, self.weighter.min_weight * 0.9)  # Allow slight tolerance
            self.assertLessEqual(weight, self.weighter.max_weight * 1.1)
        
        # Bull market specific checks
        if 'macd' in weights:
            macd_signal = self.weighter.signals['macd']
            self.assertGreater(macd_signal.regime_multiplier, 1.0)  # Should get boost
        
        print(f"✅ Bull market weights calculated (total: {total_weight:.3f})")
        print(f"✅ Regime influence: {result.regime_influence:.3f}")
        print(f"✅ Top weighted: {sorted(weights.items(), key=lambda x: x[1], reverse=True)[0]}")
        
        # Verify explanation is generated
        self.assertIn('bull', result.explanation.lower())
        self.assertIsInstance(result.timestamp, datetime)
    
    def test_bear_market_weight_adjustments(self):
        """Test weight adjustments for bear market conditions"""
        print("\n=== Testing Bear Market Weight Adjustments ===")
        
        # Bear market conditions
        result = self.weighter.calculate_dynamic_weights(
            market_regime='bear_market',
            regime_confidence=0.9,
            volatility_percentile=0.7,  # Higher volatility
            signal_performance=self.sample_performance,
            current_signals=self.sample_current_signals
        )
        
        weights = result.signal_weights
        
        # Check weight normalization
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=3)
        
        # In bear markets, mean reversion and volume should be emphasized
        if 'bollinger_bands' in weights and 'obv_divergence' in weights:
            bb_signal = self.weighter.signals['bollinger_bands']
            obv_signal = self.weighter.signals['obv_divergence']
            
            # These should get boosts in bear markets
            self.assertGreater(bb_signal.regime_multiplier, 1.0)
            self.assertGreater(obv_signal.regime_multiplier, 1.0)
        
        print(f"✅ Bear market weights calculated")
        print(f"✅ Performance influence: {result.performance_influence:.3f}")
        print(f"✅ Volatility influence: {result.volatility_influence:.3f}")
        
        # Verify bear market emphasis in explanation
        self.assertIn('bear', result.explanation.lower())
    
    def test_volatile_market_weight_adjustments(self):
        """Test weight adjustments for volatile market conditions"""
        print("\n=== Testing Volatile Market Weight Adjustments ===")
        
        # Volatile market conditions
        result = self.weighter.calculate_dynamic_weights(
            market_regime='volatile_market',
            regime_confidence=0.6,  # Lower confidence due to volatility
            volatility_percentile=0.9,  # Very high volatility
            signal_performance=self.sample_performance,
            current_signals=self.sample_current_signals
        )
        
        weights = result.signal_weights
        
        # Volume signals should be emphasized in volatile markets
        if 'volume_breakout' in weights:
            vol_signal = self.weighter.signals['volume_breakout']
            self.assertGreater(vol_signal.regime_multiplier, 1.0)  # Should get significant boost
        
        # Traditional technical indicators should be de-emphasized
        if 'macd' in weights:
            macd_signal = self.weighter.signals['macd']
            self.assertLess(macd_signal.regime_multiplier, 1.0)  # Should be reduced
        
        print(f"✅ Volatile market weights calculated")
        print(f"✅ Confidence score: {result.confidence_score:.3f}")
        
        # Check that volatility influence is significant
        self.assertGreater(result.volatility_influence, 0.1)
        
        # Verify volatile market handling in explanation
        self.assertIn('volatile', result.explanation.lower())
    
    def test_performance_based_adjustments(self):
        """Test performance-based weight adjustments"""
        print("\n=== Testing Performance-Based Adjustments ===")
        
        # Test with varying performance data
        high_performance = {
            'rsi': 0.85,  # Very high performance
            'macd': 0.35,  # Poor performance
            'bollinger_bands': 0.65,  # Good performance
            'volume_breakout': 0.25,  # Very poor performance
            'obv_divergence': 0.75   # High performance
        }
        
        result = self.weighter.calculate_dynamic_weights(
            market_regime='sideways_market',
            regime_confidence=0.7,
            volatility_percentile=0.5,
            signal_performance=high_performance,
            current_signals=self.sample_current_signals
        )
        
        # Check that performance multipliers are applied correctly
        rsi_signal = self.weighter.signals['rsi']
        macd_signal = self.weighter.signals['macd']
        
        # High performance should lead to higher multiplier
        self.assertGreater(rsi_signal.performance_multiplier, 1.0)
        
        # Poor performance should lead to lower multiplier
        self.assertLess(macd_signal.performance_multiplier, 1.0)
        
        print(f"✅ Performance adjustments applied")
        print(f"✅ RSI performance multiplier: {rsi_signal.performance_multiplier:.3f}")
        print(f"✅ MACD performance multiplier: {macd_signal.performance_multiplier:.3f}")
        
        # Run multiple iterations to build performance history
        for i in range(10):
            self.weighter.calculate_dynamic_weights(
                'sideways_market', 0.7, 0.5, high_performance, self.sample_current_signals
            )
        
        # Check that performance history is being tracked
        self.assertGreater(len(rsi_signal.performance_history), 5)
    
    def test_confidence_scoring(self):
        """Test signal confidence calculation"""
        print("\n=== Testing Signal Confidence Scoring ===")
        
        # Test with high-confidence signal data
        high_confidence_signals = {
            'rsi': {
                'strength': 0.9,
                'supporting_indicators': {'volume': True, 'trend': True, 'momentum': True},
                'volume_confirmed': True,
                'signal_age': 5
            },
            'macd': {
                'strength': 0.3,  # Low strength
                'volume_confirmed': False,
                'signal_age': 1
            }
        }
        
        result = self.weighter.calculate_dynamic_weights(
            market_regime='bull_market',
            regime_confidence=0.8,
            volatility_percentile=0.4,
            signal_performance=self.sample_performance,
            current_signals=high_confidence_signals
        )
        
        # Check confidence scores
        rsi_signal = self.weighter.signals['rsi']
        macd_signal = self.weighter.signals['macd']
        
        # RSI should have higher confidence due to supporting factors
        self.assertGreater(rsi_signal.confidence_score, macd_signal.confidence_score)
        
        print(f"✅ RSI confidence: {rsi_signal.confidence_score:.3f}")
        print(f"✅ MACD confidence: {macd_signal.confidence_score:.3f}")
        
        # Overall confidence should be reasonable
        self.assertGreater(result.confidence_score, 0.3)
        self.assertLess(result.confidence_score, 1.0)
    
    def test_weight_bounds_enforcement(self):
        """Test that weight bounds are properly enforced"""
        print("\n=== Testing Weight Bounds Enforcement ===")
        
        # Test with extreme conditions that might push weights out of bounds
        extreme_performance = {signal: 0.95 for signal in self.test_signals}  # All very high
        
        result = self.weighter.calculate_dynamic_weights(
            market_regime='bull_market',
            regime_confidence=1.0,  # Maximum confidence
            volatility_percentile=0.1,  # Very low volatility
            signal_performance=extreme_performance,
            current_signals=self.sample_current_signals
        )
        
        # Check all weights respect bounds
        for signal_type, weight in result.signal_weights.items():
            self.assertGreaterEqual(weight, self.weighter.min_weight * 0.95)  # Small tolerance
            self.assertLessEqual(weight, self.weighter.max_weight * 1.05)
            
        print(f"✅ All weights within bounds [{self.weighter.min_weight:.3f}, {self.weighter.max_weight:.3f}]")
        
        # Test with extreme low performance
        low_performance = {signal: 0.1 for signal in self.test_signals}  # All very low
        
        result_low = self.weighter.calculate_dynamic_weights(
            market_regime='bear_market',
            regime_confidence=0.9,
            volatility_percentile=0.8,
            signal_performance=low_performance,
            current_signals=self.sample_current_signals
        )
        
        # Even with poor performance, weights should not go below minimum
        for signal_type, weight in result_low.signal_weights.items():
            self.assertGreaterEqual(weight, self.weighter.min_weight * 0.95)
            
        print(f"✅ Minimum weight bounds enforced even with poor performance")
    
    def test_weight_normalization(self):
        """Test that weights are properly normalized to sum to 1.0"""
        print("\n=== Testing Weight Normalization ===")
        
        test_scenarios = [
            ('bull_market', 0.8, 0.3),
            ('bear_market', 0.7, 0.6),
            ('volatile_market', 0.6, 0.9),
            ('sideways_market', 0.9, 0.2)
        ]
        
        for regime, confidence, volatility in test_scenarios:
            result = self.weighter.calculate_dynamic_weights(
                regime, confidence, volatility, 
                self.sample_performance, self.sample_current_signals
            )
            
            total_weight = sum(result.signal_weights.values())
            self.assertAlmostEqual(total_weight, 1.0, places=3)
            
        print(f"✅ Weight normalization verified for {len(test_scenarios)} scenarios")
    
    def test_historical_optimization(self):
        """Test historical weight optimization"""
        print("\n=== Testing Historical Optimization ===")
        
        # Create synthetic historical data
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        
        # Synthetic signal data (correlated with returns)
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 60)  # Daily returns
        
        historical_data = pd.DataFrame({
            'date': dates,
            'rsi': returns * 0.7 + np.random.normal(0, 0.01, 60),  # Positive correlation
            'macd': returns * 0.5 + np.random.normal(0, 0.015, 60),
            'bollinger_bands': returns * -0.3 + np.random.normal(0, 0.01, 60),  # Negative correlation
            'volume_breakout': returns * 0.8 + np.random.normal(0, 0.02, 60),
            'obv_divergence': returns * 0.4 + np.random.normal(0, 0.01, 60)
        })
        
        target_returns = pd.Series(returns)
        
        # Test optimization
        optimized_weights = self.weighter.optimize_weights_historical(historical_data, target_returns)
        
        # Verify optimization results
        self.assertIsInstance(optimized_weights, dict)
        self.assertGreater(len(optimized_weights), 0)
        
        # Check that weights sum to approximately 1.0
        if optimized_weights:
            total_optimized = sum(optimized_weights.values())
            self.assertAlmostEqual(total_optimized, 1.0, places=2)
            
            # Signals with higher correlation should get higher weights
            if 'volume_breakout' in optimized_weights and 'bollinger_bands' in optimized_weights:
                self.assertGreater(optimized_weights['volume_breakout'], 
                                 optimized_weights['bollinger_bands'] * 0.8)  # Allow some tolerance
        
        print(f"✅ Historical optimization completed")
        print(f"✅ Optimized {len(optimized_weights)} signals")
        if optimized_weights:
            top_signal = max(optimized_weights.items(), key=lambda x: x[1])
            print(f"✅ Top weighted signal: {top_signal[0]} ({top_signal[1]:.3f})")
    
    def test_system_statistics_and_tracking(self):
        """Test system statistics and performance tracking"""
        print("\n=== Testing System Statistics and Tracking ===")
        
        # Generate multiple weight calculations to build history
        test_scenarios = [
            ('bull_market', 0.8, 0.3),
            ('bear_market', 0.7, 0.6),
            ('volatile_market', 0.6, 0.9),
            ('sideways_market', 0.9, 0.2),
            ('bull_market', 0.9, 0.4)
        ]
        
        for regime, confidence, volatility in test_scenarios:
            self.weighter.calculate_dynamic_weights(
                regime, confidence, volatility,
                self.sample_performance, self.sample_current_signals
            )
        
        # Get statistics
        stats = self.weighter.get_weight_statistics()
        
        # Verify statistics structure
        self.assertIn('total_adjustments', stats)
        self.assertIn('average_influences', stats)
        self.assertIn('weight_statistics', stats)
        self.assertIn('registered_signals', stats)
        
        # Check values are reasonable
        self.assertEqual(stats['total_adjustments'], len(test_scenarios))
        self.assertEqual(stats['registered_signals'], len(self.test_signals))
        self.assertIsInstance(stats['average_influences'], dict)
        
        print(f"✅ System statistics generated:")
        print(f"  Total adjustments: {stats['total_adjustments']}")
        print(f"  Registered signals: {stats['registered_signals']}")
        print(f"  Average confidence: {stats.get('average_confidence', 'N/A'):.3f}")
        
        # Test weight history tracking
        self.assertEqual(len(self.weighter.weight_history), len(test_scenarios))
        
        # Test current weights retrieval
        current_weights = self.weighter.get_current_weights()
        self.assertEqual(len(current_weights), len(self.test_signals))
        
        print(f"✅ Weight history tracked: {len(self.weighter.weight_history)} entries")

def run_comprehensive_test():
    """Run comprehensive test with detailed output"""
    print("🚀 Starting Comprehensive Dynamic Signal Weighting Test")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDynamicSignalWeighting)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=False)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    if result.wasSuccessful():
        print(f"✅ ALL TESTS PASSED ({result.testsRun} tests)")
        print("\n🎉 Dynamic Signal Weighting Framework is working correctly!")
        print("\n📈 Key Validation Results:")
        print("   ✅ Research-based base weight initialization")
        print("   ✅ Regime-specific weight adjustments")
        print("   ✅ Performance-based dynamic weighting")
        print("   ✅ Signal confidence scoring")
        print("   ✅ Weight bounds enforcement")
        print("   ✅ Weight normalization (sum to 1.0)")
        print("   ✅ Historical optimization capabilities")
        print("   ✅ Comprehensive statistics and tracking")
        
        return True
    else:
        print(f"❌ TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
        for test, trace in result.failures + result.errors:
            print(f"   ❌ {test}: {trace.split(chr(10))[-2] if chr(10) in trace else trace}")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)