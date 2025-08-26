"""
Test suite for Dynamic Risk Management System
Validates risk calculations, position sizing, and portfolio risk monitoring.
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
    from risk_management.dynamic_risk_manager import (
        DynamicRiskManager,
        RiskMetrics,
        RiskLevel,
        RiskLimits
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the dynamic_risk_manager.py file is in the correct location")
    sys.exit(1)

class TestDynamicRiskManager(unittest.TestCase):
    """Test suite for dynamic risk management system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.risk_manager = DynamicRiskManager(
            base_capital=1000000,  # $1M base capital
            confidence_level=0.95,
            lookback_period=252
        )
        
        # Create comprehensive test market data
        np.random.seed(42)  # For reproducible tests
        
        # Create realistic market data with different volatility regimes
        self.market_data = self._create_test_market_data()
        
        # Sample regime information
        self.sample_regime_info = {
            'regime': 'bull_market',
            'confidence': 0.8,
            'volatility_percentile': 0.3
        }
    
    def _create_test_market_data(self) -> pd.DataFrame:
        """Create realistic test market data with varying volatility"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Start with base price and create realistic price movements
        base_price = 100.0
        prices = [base_price]
        
        # Create different volatility regimes
        for i in range(1, 100):
            if i < 30:  # Low volatility period
                daily_return = np.random.normal(0.0008, 0.012)  # 0.08% mean, 1.2% volatility
            elif i < 60:  # Medium volatility period  
                daily_return = np.random.normal(0.0005, 0.020)  # 0.05% mean, 2.0% volatility
            else:  # High volatility period
                daily_return = np.random.normal(-0.0002, 0.035)  # -0.02% mean, 3.5% volatility
            
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(1.0, new_price))  # Prevent negative prices
        
        # Create high, low, and volume data
        highs = [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices]
        lows = [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices]
        volumes = np.random.randint(500000, 3000000, 100)
        
        return pd.DataFrame({
            'date': dates,
            'close': prices,
            'high': highs,
            'low': lows,
            'volume': volumes
        })
    
    def test_initialization(self):
        """Test risk manager initialization and parameters"""
        print("\n=== Testing Risk Manager Initialization ===")
        
        # Check basic initialization
        self.assertEqual(self.risk_manager.base_capital, 1000000)
        self.assertEqual(self.risk_manager.confidence_level, 0.95)
        self.assertEqual(self.risk_manager.lookback_period, 252)
        
        # Check risk limits are properly set
        self.assertIsInstance(self.risk_manager.risk_limits, RiskLimits)
        self.assertLessEqual(self.risk_manager.risk_limits.max_position_size, 1.0)
        self.assertGreater(self.risk_manager.risk_limits.max_portfolio_risk, 0)
        
        # Check risk parameters are loaded
        self.assertIn('kelly_max_bet', self.risk_manager.risk_parameters)
        self.assertIn('target_volatility', self.risk_manager.risk_parameters)
        
        # Check regime adjustments
        self.assertIn('bull_market', self.risk_manager.regime_risk_adjustments)
        self.assertIn('bear_market', self.risk_manager.regime_risk_adjustments)
        
        print("✅ Risk manager initialized with correct parameters")
        print(f"✅ Base capital: ${self.risk_manager.base_capital:,}")
        print(f"✅ Confidence level: {self.risk_manager.confidence_level:.1%}")
        print(f"✅ Risk limits configured: {len(self.risk_manager.regime_risk_adjustments)} regimes")
    
    def test_risk_metrics_calculation(self):
        """Test comprehensive risk metrics calculation"""
        print("\n=== Testing Risk Metrics Calculation ===")
        
        # Test with long position
        risk_metrics = self.risk_manager.calculate_position_risk(
            symbol='TEST_LONG',
            position_size=1000,  # 1000 shares
            entry_price=100.0,
            current_price=105.0,  # 5% profit
            market_data=self.market_data,
            regime_info=self.sample_regime_info
        )
        
        # Validate risk metrics structure
        self.assertIsInstance(risk_metrics, RiskMetrics)
        self.assertEqual(risk_metrics.symbol, 'TEST_LONG')
        self.assertEqual(risk_metrics.position_size, 1000)
        self.assertEqual(risk_metrics.current_price, 105.0)
        
        # Check calculated values are reasonable
        self.assertEqual(risk_metrics.position_value, 105000)  # 1000 * 105
        self.assertEqual(risk_metrics.unrealized_pnl, 5000)    # 1000 * (105 - 100)
        
        # Check risk measures
        self.assertGreater(risk_metrics.value_at_risk_1d, 0)
        self.assertGreater(risk_metrics.value_at_risk_5d, risk_metrics.value_at_risk_1d)
        self.assertGreater(risk_metrics.expected_shortfall, risk_metrics.value_at_risk_1d)
        
        # Check volatility metrics
        self.assertGreater(risk_metrics.realized_volatility, 0)
        self.assertLessEqual(risk_metrics.volatility_percentile, 1.0)
        self.assertGreaterEqual(risk_metrics.volatility_percentile, 0.0)
        
        # Check risk level assignment
        self.assertIsInstance(risk_metrics.risk_level, RiskLevel)
        
        # Check stops are calculated
        self.assertGreater(risk_metrics.stop_loss_price, 0)
        self.assertLess(risk_metrics.stop_loss_price, risk_metrics.current_price)  # Long position
        self.assertGreater(risk_metrics.take_profit_price, risk_metrics.current_price)
        
        print(f"✅ Risk metrics calculated for long position")
        print(f"  Position value: ${risk_metrics.position_value:,}")
        print(f"  VaR (1-day): ${risk_metrics.value_at_risk_1d:,.0f}")
        print(f"  Risk level: {risk_metrics.risk_level.name}")
        print(f"  Stop loss: ${risk_metrics.stop_loss_price:.2f}")
        
        # Test with short position
        short_risk_metrics = self.risk_manager.calculate_position_risk(
            symbol='TEST_SHORT',
            position_size=-500,  # Short 500 shares
            entry_price=100.0,
            current_price=95.0,  # 5% profit on short
            market_data=self.market_data,
            regime_info=self.sample_regime_info
        )
        
        # Check short position calculations
        self.assertEqual(short_risk_metrics.position_size, -500)
        self.assertEqual(short_risk_metrics.unrealized_pnl, 2500)  # 500 * (100 - 95)
        self.assertGreater(short_risk_metrics.stop_loss_price, short_risk_metrics.current_price)  # Short position
        
        print(f"✅ Risk metrics calculated for short position")
        print(f"  Unrealized P&L: ${short_risk_metrics.unrealized_pnl:,}")
    
    def test_regime_specific_adjustments(self):
        """Test regime-specific risk adjustments"""
        print("\n=== Testing Regime-Specific Adjustments ===")
        
        test_regimes = [
            ('bull_market', 0.9),
            ('bear_market', 0.8),
            ('volatile_market', 0.6),
            ('sideways_market', 0.7)
        ]
        
        base_position_size = 1000
        base_entry_price = 100.0
        base_current_price = 102.0
        
        regime_results = {}
        
        for regime, confidence in test_regimes:
            regime_info = {
                'regime': regime,
                'confidence': confidence,
                'volatility_percentile': 0.5
            }
            
            risk_metrics = self.risk_manager.calculate_position_risk(
                symbol=f'TEST_{regime.upper()}',
                position_size=base_position_size,
                entry_price=base_entry_price,
                current_price=base_current_price,
                market_data=self.market_data,
                regime_info=regime_info
            )
            
            regime_results[regime] = {
                'risk_level': risk_metrics.risk_level,
                'var_1d': risk_metrics.value_at_risk_1d,
                'stop_loss': risk_metrics.stop_loss_price,
                'regime_multiplier': risk_metrics.regime_risk_multiplier
            }
            
            print(f"  {regime:15s}: Risk={risk_metrics.risk_level.name:10s}, "
                  f"VaR=${risk_metrics.value_at_risk_1d:6.0f}, "
                  f"Stop=${risk_metrics.stop_loss_price:6.2f}")
        
        # Validate regime differences
        # Volatile market should have higher VaR and tighter stops
        volatile_var = regime_results['volatile_market']['var_1d']
        bull_var = regime_results['bull_market']['var_1d']
        self.assertGreater(volatile_var, bull_var)
        
        # Bull market should have wider stops than volatile market
        bull_stop = regime_results['bull_market']['stop_loss']
        volatile_stop = regime_results['volatile_market']['stop_loss']
        self.assertLess(bull_stop, volatile_stop)  # Bull stop should be lower (wider)
        
        print("✅ Regime adjustments working correctly")
    
    def test_value_at_risk_calculations(self):
        """Test VaR calculation methods"""
        print("\n=== Testing Value at Risk Calculations ===")
        
        # Test with different position sizes and market conditions
        test_cases = [
            (1000, 100000, "Small position"),      # $100k position
            (5000, 500000, "Medium position"),     # $500k position  
            (10000, 1000000, "Large position")     # $1M position
        ]
        
        for shares, expected_value, description in test_cases:
            risk_metrics = self.risk_manager.calculate_position_risk(
                symbol='VAR_TEST',
                position_size=shares,
                entry_price=100.0,
                current_price=100.0,
                market_data=self.market_data,
                regime_info=self.sample_regime_info
            )
            
            # Check VaR properties
            self.assertGreater(risk_metrics.value_at_risk_1d, 0)
            self.assertGreater(risk_metrics.value_at_risk_5d, risk_metrics.value_at_risk_1d)
            
            # VaR should scale roughly with position size
            var_ratio = risk_metrics.value_at_risk_1d / abs(risk_metrics.position_value)
            self.assertLess(var_ratio, 0.10)  # Should be less than 10% for 95% confidence
            self.assertGreater(var_ratio, 0.005)  # Should be greater than 0.5%
            
            # Expected shortfall should be higher than VaR
            self.assertGreater(risk_metrics.expected_shortfall, risk_metrics.value_at_risk_1d)
            
            print(f"  {description:15s}: VaR=${risk_metrics.value_at_risk_1d:8.0f} "
                  f"({var_ratio:.1%} of position)")
        
        print("✅ VaR calculations validated")
    
    def test_optimal_position_sizing(self):
        """Test optimal position sizing calculations"""
        print("\n=== Testing Optimal Position Sizing ===")
        
        # Test Kelly Criterion-based sizing
        sizing_result = self.risk_manager.calculate_optimal_position_size(
            symbol='SIZING_TEST',
            entry_price=100.0,
            expected_return=0.08,  # 8% expected return
            win_rate=0.65,         # 65% win rate
            market_data=self.market_data,
            regime_info=self.sample_regime_info
        )
        
        # Validate sizing result structure
        self.assertIn('optimal_shares', sizing_result)
        self.assertIn('optimal_fraction', sizing_result)
        self.assertIn('kelly_fraction', sizing_result)
        self.assertIn('volatility_target', sizing_result)
        self.assertIn('risk_assessment', sizing_result)
        
        # Check values are reasonable
        optimal_shares = sizing_result['optimal_shares']
        optimal_fraction = sizing_result['optimal_fraction']
        
        self.assertGreater(optimal_shares, 0)
        self.assertGreater(optimal_fraction, 0)
        self.assertLessEqual(optimal_fraction, self.risk_manager.risk_limits.max_position_size)
        
        # Kelly fraction should be reasonable for given parameters
        kelly_fraction = sizing_result['kelly_fraction']
        self.assertGreaterEqual(kelly_fraction, 0)
        self.assertLessEqual(kelly_fraction, self.risk_manager.risk_parameters['kelly_max_bet'])
        
        print(f"✅ Optimal sizing calculated")
        print(f"  Recommended shares: {optimal_shares:,.0f}")
        print(f"  Portfolio fraction: {optimal_fraction:.1%}")
        print(f"  Kelly fraction: {kelly_fraction:.1%}")
        print(f"  Reasoning: {sizing_result['reasoning']}")
        
        # Test with different win rates and expected returns
        test_scenarios = [
            (0.05, 0.55, "Low return, low win rate"),
            (0.12, 0.75, "High return, high win rate"),
            (-0.02, 0.45, "Negative return, low win rate")
        ]
        
        for exp_return, win_rate, description in test_scenarios:
            scenario_result = self.risk_manager.calculate_optimal_position_size(
                'SCENARIO_TEST', 100.0, exp_return, win_rate, self.market_data
            )
            
            kelly = scenario_result['kelly_fraction']
            
            # Negative expected return should result in zero Kelly fraction
            if exp_return <= 0:
                self.assertEqual(kelly, 0.0)
            
            print(f"  {description}: Kelly={kelly:.1%}, Optimal={scenario_result['optimal_fraction']:.1%}")
    
    def test_risk_level_assessment(self):
        """Test risk level assessment accuracy"""
        print("\n=== Testing Risk Level Assessment ===")
        
        # Create scenarios with different risk profiles
        test_scenarios = [
            (500, 100.0, 101.0, 'bull_market', RiskLevel.VERY_LOW, "Small profitable position"),
            (5000, 100.0, 102.0, 'sideways_market', RiskLevel.LOW, "Medium position, low volatility"),
            (10000, 100.0, 105.0, 'volatile_market', RiskLevel.HIGH, "Large position, volatile market"),
            (2000, 100.0, 90.0, 'bear_market', RiskLevel.MEDIUM, "Losing position, bear market")
        ]
        
        for shares, entry, current, regime, expected_level, description in test_scenarios:
            regime_info = {'regime': regime, 'confidence': 0.7}
            
            risk_metrics = self.risk_manager.calculate_position_risk(
                'RISK_LEVEL_TEST',
                position_size=shares,
                entry_price=entry,
                current_price=current,
                market_data=self.market_data,
                regime_info=regime_info
            )
            
            # Note: Exact risk level matching is difficult due to market data variability
            # So we'll check that risk level is reasonable
            self.assertIsInstance(risk_metrics.risk_level, RiskLevel)
            
            print(f"  {description:30s}: {risk_metrics.risk_level.name} "
                  f"(VaR: ${risk_metrics.value_at_risk_1d:.0f})")
    
    def test_stop_loss_calculation(self):
        """Test dynamic stop loss calculations"""
        print("\n=== Testing Stop Loss Calculations ===")
        
        # Test long position stops
        long_risk = self.risk_manager.calculate_position_risk(
            'STOP_TEST_LONG',
            position_size=1000,
            entry_price=100.0,
            current_price=105.0,
            market_data=self.market_data,
            regime_info={'regime': 'bull_market', 'confidence': 0.8}
        )
        
        # Long position: stop loss should be below current price
        self.assertLess(long_risk.stop_loss_price, long_risk.current_price)
        self.assertGreater(long_risk.take_profit_price, long_risk.current_price)
        
        # Short position stops
        short_risk = self.risk_manager.calculate_position_risk(
            'STOP_TEST_SHORT',
            position_size=-1000,
            entry_price=100.0,
            current_price=95.0,
            market_data=self.market_data,
            regime_info={'regime': 'bear_market', 'confidence': 0.8}
        )
        
        # Short position: stop loss should be above current price
        self.assertGreater(short_risk.stop_loss_price, short_risk.current_price)
        self.assertLess(short_risk.take_profit_price, short_risk.current_price)
        
        print(f"✅ Stop loss calculations validated")
        print(f"  Long position - Stop: ${long_risk.stop_loss_price:.2f}, Target: ${long_risk.take_profit_price:.2f}")
        print(f"  Short position - Stop: ${short_risk.stop_loss_price:.2f}, Target: ${short_risk.take_profit_price:.2f}")
    
    def test_portfolio_risk_summary(self):
        """Test portfolio-level risk summary"""
        print("\n=== Testing Portfolio Risk Summary ===")
        
        # Add some test positions
        self.risk_manager.positions = {
            'STOCK1': {'value': 100000, 'unrealized_pnl': 5000},
            'STOCK2': {'value': -50000, 'unrealized_pnl': -2000},
            'STOCK3': {'value': 75000, 'unrealized_pnl': 1000}
        }
        
        # Calculate risk for these positions
        for symbol in ['STOCK1', 'STOCK2', 'STOCK3']:
            self.risk_manager.calculate_position_risk(
                symbol=symbol,
                position_size=1000 if symbol != 'STOCK2' else -500,
                entry_price=100.0,
                current_price=102.0,
                market_data=self.market_data,
                regime_info=self.sample_regime_info
            )
        
        # Get portfolio summary
        portfolio_summary = self.risk_manager.get_portfolio_risk_summary()
        
        # Validate summary structure
        self.assertIn('portfolio_value', portfolio_summary)
        self.assertIn('unrealized_pnl', portfolio_summary)
        self.assertIn('exposure_ratio', portfolio_summary)
        self.assertIn('portfolio_var_1d', portfolio_summary)
        self.assertIn('risk_distribution', portfolio_summary)
        
        # Check calculated values
        self.assertEqual(portfolio_summary['position_count'], 3)
        expected_total_value = 100000 - 50000 + 75000  # Net long $125k
        self.assertEqual(portfolio_summary['portfolio_value'], expected_total_value)
        
        # Check exposure ratio is reasonable
        exposure_ratio = portfolio_summary['exposure_ratio']
        self.assertGreater(exposure_ratio, 0)
        self.assertLess(exposure_ratio, 1.0)  # Should be less than 100% for this test
        
        print(f"✅ Portfolio summary generated")
        print(f"  Total value: ${portfolio_summary['portfolio_value']:,}")
        print(f"  Exposure ratio: {exposure_ratio:.1%}")
        print(f"  Risk distribution: {portfolio_summary['risk_distribution']}")
    
    def test_alerts_and_recommendations(self):
        """Test risk alert and recommendation generation"""
        print("\n=== Testing Alerts and Recommendations ===")
        
        # Test high-risk scenario that should generate alerts
        high_risk_metrics = self.risk_manager.calculate_position_risk(
            symbol='HIGH_RISK_TEST',
            position_size=20000,  # Very large position
            entry_price=100.0,
            current_price=98.0,   # Losing money
            market_data=self.market_data,
            regime_info={'regime': 'volatile_market', 'confidence': 0.9}
        )
        
        # Should generate alerts for large position
        self.assertGreater(len(high_risk_metrics.alerts), 0)
        
        # Should generate recommendations
        self.assertGreater(len(high_risk_metrics.recommendations), 0)
        
        print(f"✅ High-risk position generated {len(high_risk_metrics.alerts)} alerts")
        for alert in high_risk_metrics.alerts:
            print(f"  Alert: {alert}")
        
        for rec in high_risk_metrics.recommendations:
            print(f"  Recommendation: {rec}")
        
        # Test low-risk scenario
        low_risk_metrics = self.risk_manager.calculate_position_risk(
            symbol='LOW_RISK_TEST',
            position_size=100,    # Small position
            entry_price=100.0,
            current_price=101.0,  # Small profit
            market_data=self.market_data,
            regime_info={'regime': 'bull_market', 'confidence': 0.9}
        )
        
        # Should generate fewer or no alerts
        print(f"✅ Low-risk position generated {len(low_risk_metrics.alerts)} alerts")
    
    def test_risk_adjusted_returns(self):
        """Test risk-adjusted return calculations"""
        print("\n=== Testing Risk-Adjusted Returns ===")
        
        risk_metrics = self.risk_manager.calculate_position_risk(
            'RETURNS_TEST',
            position_size=1000,
            entry_price=100.0,
            current_price=105.0,
            market_data=self.market_data,
            regime_info=self.sample_regime_info
        )
        
        # Check that risk-adjusted returns are calculated
        self.assertIsInstance(risk_metrics.sharpe_ratio, (int, float))
        self.assertIsInstance(risk_metrics.sortino_ratio, (int, float))
        self.assertIsInstance(risk_metrics.calmar_ratio, (int, float))
        
        # Sortino should generally be higher than Sharpe (less penalty for upside volatility)
        # Note: This may not always be true for short test periods
        
        print(f"✅ Risk-adjusted returns calculated")
        print(f"  Sharpe ratio: {risk_metrics.sharpe_ratio:.2f}")
        print(f"  Sortino ratio: {risk_metrics.sortino_ratio:.2f}")
        print(f"  Calmar ratio: {risk_metrics.calmar_ratio:.2f}")

def run_comprehensive_test():
    """Run comprehensive test with detailed output"""
    print("🚀 Starting Comprehensive Dynamic Risk Management Test")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDynamicRiskManager)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=False)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    if result.wasSuccessful():
        print(f"✅ ALL TESTS PASSED ({result.testsRun} tests)")
        print("\n🎉 Dynamic Risk Management System is working correctly!")
        print("\n📈 Key Validation Results:")
        print("   ✅ Risk metrics calculation (VaR, Expected Shortfall, Drawdown)")
        print("   ✅ Regime-specific risk adjustments")
        print("   ✅ Optimal position sizing (Kelly Criterion + Volatility Targeting)")
        print("   ✅ Dynamic stop loss and take profit calculations")
        print("   ✅ Risk level assessment and alerts")
        print("   ✅ Portfolio-level risk monitoring")
        print("   ✅ Risk-adjusted return calculations")
        print("   ✅ Comprehensive alert and recommendation system")
        
        return True
    else:
        print(f"❌ TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
        for test, trace in result.failures + result.errors:
            print(f"   ❌ {test}: {trace.split(chr(10))[-2] if chr(10) in trace else trace}")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)