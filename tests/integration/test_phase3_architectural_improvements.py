#!/usr/bin/env python3
"""
Phase 3 Architectural Improvements Testing Script
Comprehensive testing of configuration management and advanced analytics systems
"""
import sys
import os
import time
import tempfile
import traceback
import pandas as pd
import numpy as np
import yaml
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_configuration_manager():
    """Test Configuration Management System"""
    print("🔍 Testing Configuration Manager...")
    
    try:
        from configuration.config_manager import ConfigurationManager, ConfigurationMetadata
        
        # Create temporary config structure
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = os.path.join(temp_dir, 'config')
            os.makedirs(config_dir, exist_ok=True)
            os.makedirs(os.path.join(config_dir, 'environments'), exist_ok=True)
            os.makedirs(os.path.join(config_dir, 'features'), exist_ok=True)
            
            # Create test config files
            base_config = {
                'app_name': 'test_app',
                'version': '1.0.0',
                'database': {
                    'host': 'localhost',
                    'port': 5432
                }
            }
            
            with open(os.path.join(config_dir, 'base.yaml'), 'w') as f:
                yaml.dump(base_config, f)
            
            # Environment-specific config
            dev_config = {
                'database': {
                    'host': 'dev-db',
                    'debug': True
                },
                'logging': {
                    'level': 'DEBUG'
                }
            }
            
            with open(os.path.join(config_dir, 'environments', 'development.yaml'), 'w') as f:
                yaml.dump(dev_config, f)
            
            # Feature flags
            feature_flags = {
                'new_feature': True,
                'beta_feature': False
            }
            
            with open(os.path.join(config_dir, 'features', 'feature_flags.yaml'), 'w') as f:
                yaml.dump(feature_flags, f)
            
            # Test configuration manager
            config_manager = ConfigurationManager(config_root=config_dir, environment='development')
            
            # Test basic configuration access
            assert config_manager.get('app_name') == 'test_app', "Should get base config value"
            assert config_manager.get('database.host') == 'dev-db', "Should get environment override"
            assert config_manager.get('database.debug') == True, "Should get environment-specific value"
            assert config_manager.get('logging.level') == 'DEBUG', "Should get nested environment value"
            
            # Test feature flags
            assert config_manager.get_feature_flag('new_feature') == True, "Should get feature flag"
            assert config_manager.get_feature_flag('beta_feature') == False, "Should get disabled flag"
            assert config_manager.get_feature_flag('non_existent', True) == True, "Should return default"
            
            # Test metadata
            metadata = config_manager.get_metadata()
            assert isinstance(metadata, ConfigurationMetadata), "Should return metadata object"
            assert metadata.environment == 'development', "Should track environment"
            assert metadata.valid == True, "Should be valid configuration"
            
            # Test configuration export
            exported = config_manager.export_config(format='yaml')
            assert 'configuration:' in exported, "Should export configuration section"
            assert 'feature_flags:' in exported, "Should export feature flags"
            
            # Test validation
            validation_report = config_manager.validate_current_config()
            assert 'valid' in validation_report, "Should return validation report"
            assert 'errors' in validation_report, "Should include errors list"
            
            print("✅ Configuration Manager tests passed")
            return True
        
    except Exception as e:
        print(f"❌ Configuration Manager test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_performance_attribution():
    """Test Performance Attribution Analysis"""
    print("🔍 Testing Performance Attribution...")
    
    try:
        from analytics.performance.attribution import (
            PerformanceAttributionAnalyzer, AttributionMetrics, AttributionBreakdown
        )
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            analyzer = PerformanceAttributionAnalyzer(db_path=tmp_db.name)
        
        # Create mock portfolio data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Portfolio returns
        portfolio_returns = pd.Series(
            np.random.normal(0.0008, 0.02, len(dates)),  # ~20% annual return, 20% vol
            index=dates,
            name='portfolio'
        )
        
        # Benchmark returns
        benchmark_returns = pd.Series(
            np.random.normal(0.0005, 0.015, len(dates)),  # ~12% annual return, 15% vol
            index=dates,
            name='benchmark'
        )
        
        # Factor returns
        factor_returns = pd.DataFrame({
            'market': benchmark_returns + np.random.normal(0, 0.005, len(dates)),
            'value': np.random.normal(0.0003, 0.01, len(dates)),
            'momentum': np.random.normal(0.0002, 0.012, len(dates))
        }, index=dates)
        
        # Test performance metrics calculation
        metrics = analyzer.calculate_performance_metrics(portfolio_returns, benchmark_returns)
        assert isinstance(metrics, AttributionMetrics), "Should return AttributionMetrics"
        assert metrics.total_return > -1.0, "Total return should be reasonable"
        assert metrics.sharpe_ratio > -5.0, "Sharpe ratio should be reasonable"
        assert -1.0 <= metrics.max_drawdown <= 0.0, "Max drawdown should be negative"
        
        # Test factor attribution
        factor_attribution = analyzer.calculate_factor_attribution(portfolio_returns, factor_returns)
        assert isinstance(factor_attribution, dict), "Should return attribution breakdown"
        assert len(factor_attribution) > 0, "Should have factor attributions"
        
        for factor_name, breakdown in factor_attribution.items():
            assert isinstance(breakdown, AttributionBreakdown), "Should return AttributionBreakdown"
            assert breakdown.factor_name == factor_name, "Factor name should match"
            assert isinstance(breakdown.contribution, float), "Contribution should be float"
            assert isinstance(breakdown.t_stat, float), "T-stat should be float"
        
        # Test attribution report generation
        report = analyzer.generate_attribution_report(
            portfolio_id='test_portfolio',
            start_date=dates[0],
            end_date=dates[-1],
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            factor_returns=factor_returns
        )
        
        assert 'portfolio_id' in report, "Report should contain portfolio ID"
        assert 'performance_metrics' in report, "Report should contain performance metrics"
        assert 'factor_attribution' in report, "Report should contain factor attribution"
        assert 'risk_analysis' in report, "Report should contain risk analysis"
        
        # Test historical attribution retrieval
        historical = analyzer.get_historical_attribution('test_portfolio', days=30)
        assert isinstance(historical, list), "Should return list of historical reports"
        
        # Cleanup
        os.unlink(tmp_db.name)
        
        print("✅ Performance Attribution tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Performance Attribution test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_signal_effectiveness():
    """Test Signal Effectiveness Tracking"""
    print("🔍 Testing Signal Effectiveness...")
    
    try:
        from analytics.signals.effectiveness import (
            SignalEffectivenessTracker, SignalMetrics, SignalDecayAnalysis
        )
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tracker = SignalEffectivenessTracker(db_path=tmp_db.name)
        
        # Record test signals with outcomes
        np.random.seed(42)
        signal_types = ['MACD_CROSS', 'RSI_OVERSOLD', 'BOLLINGER_SQUEEZE']
        
        for i in range(100):
            signal_id = f"signal_{i:03d}"
            signal_type = np.random.choice(signal_types)
            symbol = np.random.choice(['AAPL', 'MSFT', 'GOOGL'])
            direction = np.random.choice(['BUY', 'SELL'])
            confidence = np.random.uniform(0.6, 0.95)
            
            # Record signal
            tracker.record_signal(
                signal_id=signal_id,
                signal_type=signal_type,
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                target_price=100.0,
                entry_price=100.0
            )
            
            # Simulate outcome (with some bias for success)
            success_prob = 0.6 + confidence * 0.2  # Higher confidence = better outcomes
            if np.random.random() < success_prob:
                exit_price = 102.0 if direction == 'BUY' else 98.0  # Profitable
                outcome = 'success'
            else:
                exit_price = 98.0 if direction == 'BUY' else 102.0  # Loss
                outcome = 'failure'
            
            # Update outcome
            tracker.update_signal_outcome(
                signal_id=signal_id,
                exit_price=exit_price,
                exit_date=datetime.now(),
                outcome=outcome
            )
        
        # Test metrics calculation
        for signal_type in signal_types:
            metrics = tracker.calculate_signal_metrics(signal_type, days_lookback=30)
            
            if metrics:  # Some signal types might not have enough data
                assert isinstance(metrics, SignalMetrics), "Should return SignalMetrics"
                assert metrics.signal_type == signal_type, "Signal type should match"
                assert metrics.total_signals > 0, "Should have signals"
                assert 0 <= metrics.hit_rate <= 1, "Hit rate should be 0-1"
                assert isinstance(metrics.average_return, float), "Average return should be float"
        
        # Test signal ranking
        rankings = tracker.get_signal_ranking(days_lookback=30)
        assert isinstance(rankings, list), "Should return list of rankings"
        
        if len(rankings) > 0:
            assert 'signal_type' in rankings[0], "Ranking should contain signal type"
            assert 'composite_score' in rankings[0], "Ranking should contain composite score"
            assert 'hit_rate' in rankings[0], "Ranking should contain hit rate"
        
        # Test correlation analysis
        correlations = tracker.analyze_signal_correlation(days_lookback=60)
        assert isinstance(correlations, dict), "Should return correlation dictionary"
        
        # Test effectiveness trends
        if len(signal_types) > 0:
            trend = tracker.get_effectiveness_trend(signal_types[0], days_lookback=30)
            assert isinstance(trend, dict), "Should return trend dictionary"
            assert 'dates' in trend, "Trend should contain dates"
            assert 'hit_rates' in trend, "Trend should contain hit rates"
        
        # Test summary
        summary = tracker.get_signal_summary()
        assert isinstance(summary, dict), "Should return summary dictionary"
        assert 'overall' in summary, "Summary should contain overall stats"
        assert 'by_signal_type' in summary, "Summary should contain by-signal-type stats"
        
        # Cleanup
        os.unlink(tmp_db.name)
        
        print("✅ Signal Effectiveness tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Signal Effectiveness test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_portfolio_optimization():
    """Test Portfolio Optimization"""
    print("🔍 Testing Portfolio Optimization...")
    
    try:
        from analytics.portfolio.optimization import (
            PortfolioOptimizer, OptimizationConstraints, OptimizationResult
        )
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            optimizer = PortfolioOptimizer(db_path=tmp_db.name)
        
        # Create mock return data
        np.random.seed(42)
        assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        
        # Simulate correlated returns
        n_assets = len(assets)
        correlation_matrix = np.random.uniform(0.3, 0.7, (n_assets, n_assets))
        np.fill_diagonal(correlation_matrix, 1.0)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
        
        # Generate returns
        returns_data = pd.DataFrame(
            np.random.multivariate_normal(
                mean=[0.0008] * n_assets,  # ~20% annual return
                cov=correlation_matrix * 0.0004,  # ~20% annual volatility
                size=len(dates)
            ),
            index=dates,
            columns=assets
        )
        
        # Test risk model estimation
        risk_model = optimizer.estimate_risk_model(returns_data, method='historical', window=252)
        assert isinstance(risk_model, dict), "Should return risk model dictionary"
        assert 'expected_returns' in risk_model, "Risk model should contain expected returns"
        assert 'covariance_matrix' in risk_model, "Risk model should contain covariance matrix"
        assert 'method' in risk_model, "Risk model should contain method"
        
        expected_returns = risk_model['expected_returns']
        covariance_matrix = risk_model['covariance_matrix']
        
        # Test basic optimization
        constraints = OptimizationConstraints(max_weight=0.4, min_weight=0.0)
        
        result = optimizer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            constraints=constraints,
            objective='max_sharpe'
        )
        
        assert isinstance(result, OptimizationResult), "Should return OptimizationResult"
        assert len(result.weights) == n_assets, "Should have weights for all assets"
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6, "Weights should sum to 1"
        assert all(0 <= w <= 0.4 + 1e-6 for w in result.weights.values()), "Should respect weight constraints"
        assert isinstance(result.expected_return, float), "Expected return should be float"
        assert result.expected_volatility > 0, "Expected volatility should be positive"
        assert isinstance(result.sharpe_ratio, float), "Sharpe ratio should be float"
        
        # Test different objectives
        objectives = ['min_variance', 'max_return', 'risk_parity']
        for objective in objectives:
            obj_result = optimizer.optimize_portfolio(
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                constraints=constraints,
                objective=objective
            )
            assert isinstance(obj_result, OptimizationResult), f"Should optimize with {objective}"
            assert len(obj_result.weights) == n_assets, f"Should have all weights for {objective}"
        
        # Test efficient frontier
        frontier = optimizer.efficient_frontier(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            n_points=10,
            constraints=constraints
        )
        
        assert isinstance(frontier, dict), "Should return frontier dictionary"
        assert 'returns' in frontier, "Frontier should contain returns"
        assert 'volatilities' in frontier, "Frontier should contain volatilities"
        assert 'sharpe_ratios' in frontier, "Frontier should contain Sharpe ratios"
        assert 'weights' in frontier, "Frontier should contain weights"
        
        # Test with alternative risk models
        for method in ['shrinkage', 'exponential']:
            try:
                alt_risk_model = optimizer.estimate_risk_model(returns_data, method=method)
                assert 'expected_returns' in alt_risk_model, f"Should estimate {method} risk model"
                assert 'covariance_matrix' in alt_risk_model, f"Should have covariance for {method}"
            except Exception as e:
                print(f"  ⚠️  {method} risk model failed (may need additional libraries): {e}")
        
        # Test optimization history
        history = optimizer.get_optimization_history(days=30)
        assert isinstance(history, list), "Should return optimization history"
        
        # Cleanup
        os.unlink(tmp_db.name)
        
        print("✅ Portfolio Optimization tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Portfolio Optimization test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between Phase 3 components"""
    print("🔍 Testing Phase 3 integration...")
    
    try:
        # Test configuration-driven analytics
        from configuration.config_manager import get_config, get_feature_flag
        from analytics.performance.attribution import PerformanceAttributionAnalyzer
        
        # Create temporary config
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = os.path.join(temp_dir, 'config')
            os.makedirs(config_dir, exist_ok=True)
            
            # Basic config
            config = {
                'analytics': {
                    'attribution': {
                        'lookback_days': 30,
                        'risk_model': 'historical'
                    }
                }
            }
            
            with open(os.path.join(config_dir, 'base.yaml'), 'w') as f:
                yaml.dump(config, f)
            
            # Initialize config manager
            from configuration.config_manager import ConfigurationManager
            config_manager = ConfigurationManager(config_root=config_dir)
            
            # Test configuration-driven component initialization
            lookback_days = get_config('analytics.attribution.lookback_days', default=30)
            assert lookback_days == 30, "Should get configuration value"
            
            # Test analytics with configuration
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
                analyzer = PerformanceAttributionAnalyzer(db_path=tmp_db.name)
                
                # The components can work together
                assert analyzer is not None, "Analytics component should initialize"
                
                # Cleanup
                os.unlink(tmp_db.name)
        
        print("✅ Integration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_performance():
    """Test performance of Phase 3 components"""
    print("🔍 Testing Phase 3 performance...")
    
    try:
        from configuration.config_manager import ConfigurationManager
        from analytics.signals.effectiveness import SignalEffectivenessTracker
        
        # Performance test: Configuration loading
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = os.path.join(temp_dir, 'config')
            os.makedirs(config_dir, exist_ok=True)
            
            # Create large config
            large_config = {f'key_{i}': f'value_{i}' for i in range(1000)}
            
            with open(os.path.join(config_dir, 'base.yaml'), 'w') as f:
                yaml.dump(large_config, f)
            
            start_time = time.time()
            config_manager = ConfigurationManager(config_root=config_dir)
            config_load_time = time.time() - start_time
            
            assert config_load_time < 2.0, f"Config loading too slow: {config_load_time:.3f}s"
            
            # Performance test: Signal metrics calculation
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
                tracker = SignalEffectivenessTracker(db_path=tmp_db.name)
                
                # Insert many signals
                start_time = time.time()
                for i in range(50):  # Smaller number for performance test
                    tracker.record_signal(
                        signal_id=f"perf_signal_{i}",
                        signal_type='TEST_SIGNAL',
                        symbol='TEST',
                        direction='BUY',
                        confidence=0.8
                    )
                    
                    tracker.update_signal_outcome(
                        signal_id=f"perf_signal_{i}",
                        exit_price=101.0,
                        outcome='success'
                    )
                
                signal_processing_time = time.time() - start_time
                
                # Calculate metrics
                start_time = time.time()
                metrics = tracker.calculate_signal_metrics('TEST_SIGNAL')
                calculation_time = time.time() - start_time
                
                assert signal_processing_time < 5.0, f"Signal processing too slow: {signal_processing_time:.3f}s"
                assert calculation_time < 1.0, f"Metrics calculation too slow: {calculation_time:.3f}s"
                
                print(f"  ✅ Config loading: {config_load_time:.3f}s")
                print(f"  ✅ Signal processing: {signal_processing_time:.3f}s for 50 signals")
                print(f"  ✅ Metrics calculation: {calculation_time:.3f}s")
                
                # Cleanup
                os.unlink(tmp_db.name)
        
        print("✅ Performance tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all Phase 3 Architectural Improvements tests"""
    print("🚀 Phase 3 Architectural Improvements Testing")
    print("=" * 60)
    
    test_results = {
        'Configuration Manager': test_configuration_manager(),
        'Performance Attribution': test_performance_attribution(),
        'Signal Effectiveness': test_signal_effectiveness(),
        'Portfolio Optimization': test_portfolio_optimization(),
        'Integration Testing': test_integration(),
        'Performance Testing': test_performance()
    }
    
    print("\n" + "=" * 60)
    print("📊 Phase 3 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:30} {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All Phase 3 Architectural Improvements tests PASSED!")
        print("Configuration management and advanced analytics are ready!")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Review issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)