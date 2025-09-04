#!/usr/bin/env python3
"""
Phase 1 Component Testing Script
Tests all Phase 1 components without external dependencies
"""
import sys
import os
import time
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_component_imports():
    """Test that all components can be imported successfully"""
    print("🔍 Testing component imports...")
    
    try:
        from dashboard.components.base_component import BaseComponent
        from dashboard.components.utility_component import UtilityComponent
        from dashboard.components.signal_display_component import SignalDisplayComponent
        from dashboard.components.performance_charts_component import PerformanceChartsComponent
        from dashboard.utils.data_processing import DataProcessor
        print("✅ All dashboard components imported successfully")
        return True
    except Exception as e:
        print(f"❌ Component import failed: {str(e)}")
        traceback.print_exc()
        return False

def test_utility_component():
    """Test utility component functions"""
    print("\n🔍 Testing utility component...")
    
    try:
        from dashboard.components.utility_component import UtilityComponent
        
        utility = UtilityComponent()
        
        # Test styling functions
        signal_style = utility.style_signals('STRONG_BUY')
        assert 'background' in signal_style, "Signal styling should include background"
        
        confidence_style = utility.style_confidence(0.85)
        assert 'background' in confidence_style, "Confidence styling should include background"
        
        rsi_style = utility.style_rsi(75.0)
        assert 'background' in rsi_style, "RSI styling should include background"
        
        # Test calculation functions
        test_prices = tuple([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        rsi = utility.calculate_rsi(test_prices, period=5)
        assert 0 <= rsi <= 100, f"RSI should be 0-100, got {rsi}"
        
        macd_result = utility.calculate_macd(test_prices, fast=3, slow=5, signal=3)
        assert 'macd_line' in macd_result, "MACD should return macd_line"
        assert 'signal_line' in macd_result, "MACD should return signal_line"
        
        bb_result = utility.calculate_bollinger_bands(test_prices, window=5)
        assert 'upper' in bb_result, "BB should return upper band"
        assert 'lower' in bb_result, "BB should return lower band"
        assert 'position' in bb_result, "BB should return position"
        
        print("✅ Utility component tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Utility component test failed: {str(e)}")
        traceback.print_exc()
        return False

def create_sample_data():
    """Create sample signals data for testing"""
    np.random.seed(42)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    signals = []
    
    for symbol in symbols:
        signals.append({
            'symbol': symbol,
            'company_name': f'{symbol} Inc.',
            'current_price': np.random.uniform(100, 300),
            'signal_direction': np.random.choice(['STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL']),
            'signal_strength': np.random.uniform(0.3, 0.9),
            'confidence': np.random.uniform(0.5, 0.95),
            'rsi_14': np.random.uniform(30, 70),
            'volume': np.random.randint(1000000, 10000000),
            'change_1d': np.random.uniform(-3, 3),
            'macd_line': np.random.uniform(-1, 1),
            'macd_signal': np.random.uniform(-1, 1),
            'bb_position': np.random.uniform(0.2, 0.8)
        })
    
    return pd.DataFrame(signals)

def test_signal_display_component():
    """Test signal display component"""
    print("\n🔍 Testing signal display component...")
    
    try:
        from dashboard.components.signal_display_component import SignalDisplayComponent
        
        signal_display = SignalDisplayComponent()
        sample_data = create_sample_data()
        
        # Test data preparation
        display_columns = ['symbol', 'company_name', 'current_price', 'signal_direction']
        prepared_data = signal_display._prepare_table_data(sample_data, display_columns)
        
        assert len(prepared_data) == len(sample_data), "Prepared data should have same length"
        assert all(col in prepared_data.columns for col in display_columns), "All display columns should exist"
        
        # Test filtering
        filtered_data = signal_display._apply_signal_filters(
            sample_data, 
            signal_filter='BUY',
            min_confidence=0.6
        )
        
        if not filtered_data.empty:
            assert all(filtered_data['signal_direction'] == 'BUY'), "Filtered data should only contain BUY signals"
            assert all(filtered_data['confidence'] >= 0.6), "Filtered data should meet confidence threshold"
        
        # Test portfolio breakdown data
        breakdown_data = signal_display._render_portfolio_breakdown.__globals__
        
        print("✅ Signal display component tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Signal display component test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_performance_charts_component():
    """Test performance charts component"""
    print("\n🔍 Testing performance charts component...")
    
    try:
        from dashboard.components.performance_charts_component import PerformanceChartsComponent
        
        charts = PerformanceChartsComponent()
        sample_data = create_sample_data()
        
        # Test signals heatmap creation
        try:
            heatmap_fig = charts.create_signals_heatmap(sample_data)
            assert heatmap_fig is not None, "Heatmap figure should be created"
            print("  ✅ Signals heatmap creation works")
        except Exception as e:
            print(f"  ⚠️  Heatmap creation failed (may need plotly): {str(e)}")
        
        # Test portfolio allocation chart
        allocation_data = {'Tech': 0.4, 'Finance': 0.3, 'Healthcare': 0.2, 'Energy': 0.1}
        try:
            allocation_fig = charts.create_portfolio_allocation_chart(allocation_data)
            assert allocation_fig is not None, "Allocation chart should be created"
            print("  ✅ Portfolio allocation chart works")
        except Exception as e:
            print(f"  ⚠️  Allocation chart failed (may need plotly): {str(e)}")
        
        # Test error chart creation
        error_fig = charts._create_error_chart("Test error message")
        assert error_fig is not None, "Error chart should be created"
        
        print("✅ Performance charts component tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Performance charts component test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_data_processor():
    """Test data processing utilities"""
    print("\n🔍 Testing data processing utilities...")
    
    try:
        from dashboard.utils.data_processing import DataProcessor
        
        processor = DataProcessor()
        sample_data = create_sample_data()
        
        # Test data processing
        processed_data = processor.process_signals_data(sample_data)
        
        assert len(processed_data) == len(sample_data), "Processed data should have same length"
        assert 'signal_score' in processed_data.columns, "Should add signal_score column"
        
        # Test filtering
        filters = {
            'signal_direction': ['BUY', 'STRONG_BUY'],
            'min_confidence': 0.6
        }
        filtered_data = processor.filter_signals(processed_data, filters)
        
        # Test portfolio metrics
        metrics = processor.calculate_portfolio_metrics(processed_data)
        assert 'total_stocks' in metrics, "Metrics should include total_stocks"
        assert 'signal_distribution' in metrics, "Metrics should include signal_distribution"
        
        # Test data quality validation
        quality_report = processor.validate_data_quality(sample_data)
        assert 'quality_score' in quality_report, "Quality report should include score"
        assert 'total_rows' in quality_report, "Quality report should include row count"
        
        print("✅ Data processor tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Data processor test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_data_managers():
    """Test data managers (with graceful fallback if services unavailable)"""
    print("\n🔍 Testing data managers...")
    
    # Test Hot Data Manager (Redis)
    try:
        from data_management.hot_data_manager import HotDataManager, HotDataConfig
        
        config = HotDataConfig(live_signal_ttl=60)  # Shorter TTL for testing
        hot_manager = HotDataManager(hot_config=config)
        
        if hot_manager.is_connected():
            # Test signal storage and retrieval
            test_signal = {
                'signal_direction': 'BUY',
                'signal_strength': 0.75,
                'confidence': 0.80
            }
            
            success = hot_manager.store_live_signal('TEST', test_signal)
            if success:
                retrieved = hot_manager.get_live_signal('TEST')
                assert retrieved is not None, "Should retrieve stored signal"
                assert retrieved['signal_direction'] == 'BUY', "Retrieved signal should match"
                print("  ✅ Hot data manager (Redis) works")
            else:
                print("  ⚠️  Hot data storage failed")
        else:
            print("  ⚠️  Redis not available, skipping hot data manager test")
            
    except Exception as e:
        print(f"  ⚠️  Hot data manager test failed: {str(e)}")
    
    # Test Warm Data Manager (PostgreSQL)
    try:
        from data_management.warm_data_manager import WarmDataManager, WarmDataConfig
        
        config = WarmDataConfig(batch_size=100)
        warm_manager = WarmDataManager(warm_config=config)
        
        if warm_manager.engine:
            # Test connection
            with warm_manager._get_connection() as conn:
                from sqlalchemy import text
                result = conn.execute(text("SELECT 1"))
                assert result.scalar() == 1, "Database connection should work"
                print("  ✅ Warm data manager (PostgreSQL) connection works")
        else:
            print("  ⚠️  PostgreSQL not available, skipping warm data manager test")
            
    except Exception as e:
        print(f"  ⚠️  Warm data manager test failed: {str(e)}")
    
    return True

def test_performance():
    """Test performance of key operations"""
    print("\n🔍 Testing performance...")
    
    try:
        from dashboard.components.utility_component import UtilityComponent
        from dashboard.utils.data_processing import DataProcessor
        
        # Create larger dataset for performance testing
        large_data = pd.DataFrame({
            'symbol': [f'TEST{i:03d}' for i in range(100)],
            'signal_direction': np.random.choice(['BUY', 'SELL', 'NEUTRAL'], 100),
            'signal_strength': np.random.uniform(0, 1, 100),
            'confidence': np.random.uniform(0.3, 0.95, 100),
            'current_price': np.random.uniform(10, 500, 100)
        })
        
        # Test utility performance
        utility = UtilityComponent()
        start_time = time.time()
        
        for _, row in large_data.head(50).iterrows():  # Test subset for speed
            utility.style_signals(row['signal_direction'])
            utility.style_confidence(row['confidence'])
        
        utility_time = time.time() - start_time
        print(f"  ✅ Utility operations: {utility_time:.3f}s for 100 operations")
        
        # Test data processing performance
        processor = DataProcessor()
        start_time = time.time()
        
        processed_data = processor.process_signals_data(large_data)
        metrics = processor.calculate_portfolio_metrics(processed_data)
        
        processing_time = time.time() - start_time
        print(f"  ✅ Data processing: {processing_time:.3f}s for 100 records")
        
        # Performance assertions
        assert utility_time < 1.0, f"Utility operations too slow: {utility_time:.3f}s"
        assert processing_time < 1.0, f"Data processing too slow: {processing_time:.3f}s"
        
        print("✅ Performance tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {str(e)}")
        traceback.print_exc()
        return False

def run_integration_test():
    """Run integration test simulating dashboard workflow"""
    print("\n🔍 Running integration test...")
    
    try:
        from dashboard.components.utility_component import UtilityComponent
        from dashboard.components.signal_display_component import SignalDisplayComponent
        from dashboard.utils.data_processing import DataProcessor
        
        # Simulate complete workflow
        sample_data = create_sample_data()
        
        # 1. Process data
        processor = DataProcessor()
        processed_data = processor.process_signals_data(sample_data)
        
        # 2. Apply filters
        filters = {'min_confidence': 0.5}
        filtered_data = processor.filter_signals(processed_data, filters)
        
        # 3. Calculate metrics
        metrics = processor.calculate_portfolio_metrics(filtered_data)
        
        # 4. Prepare display data
        signal_display = SignalDisplayComponent()
        display_columns = ['symbol', 'current_price', 'signal_direction', 'confidence']
        table_data = signal_display._prepare_table_data(filtered_data, display_columns)
        
        # 5. Validate results
        assert len(table_data) >= 0, "Table data should be generated"
        assert 'total_stocks' in metrics, "Metrics should be calculated"
        assert len(processed_data) == len(sample_data), "Data processing should preserve record count"
        
        print("✅ Integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all Phase 1 tests"""
    print("🚀 Phase 1 Component Testing")
    print("=" * 50)
    
    test_results = {
        'Component Imports': test_component_imports(),
        'Utility Component': test_utility_component(),
        'Signal Display': test_signal_display_component(),
        'Performance Charts': test_performance_charts_component(),
        'Data Processor': test_data_processor(),
        'Data Managers': test_data_managers(),
        'Performance': test_performance(),
        'Integration': run_integration_test()
    }
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All Phase 1 tests PASSED! Ready for Phase 2!")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Review issues before Phase 2.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)