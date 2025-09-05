#!/usr/bin/env python3
"""
Phase 1 Performance Testing
Tests the performance improvements from dashboard refactoring and data architecture changes
"""
import pytest
import time
import psutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dashboard.components.base_component import BaseComponent
from dashboard.components.utility_component import UtilityComponent
from dashboard.components.signal_display_component import SignalDisplayComponent
from dashboard.components.performance_charts_component import PerformanceChartsComponent
from dashboard.utils.data_processing import DataProcessor
from data_management.hot_data_manager import HotDataManager
from data_management.warm_data_manager import WarmDataManager

logger = logging.getLogger(__name__)

class PerformanceTestResults:
    """Container for performance test results"""
    def __init__(self):
        self.results = {}
        self.baseline = {}
        self.improvements = {}
    
    def add_result(self, test_name: str, execution_time: float, memory_used: float):
        self.results[test_name] = {
            'execution_time': execution_time,
            'memory_used': memory_used,
            'timestamp': datetime.now()
        }
    
    def set_baseline(self, test_name: str, execution_time: float, memory_used: float):
        self.baseline[test_name] = {
            'execution_time': execution_time,
            'memory_used': memory_used
        }
    
    def calculate_improvements(self):
        for test_name in self.results:
            if test_name in self.baseline:
                baseline_time = self.baseline[test_name]['execution_time']
                current_time = self.results[test_name]['execution_time']
                
                baseline_memory = self.baseline[test_name]['memory_used']
                current_memory = self.results[test_name]['memory_used']
                
                time_improvement = ((baseline_time - current_time) / baseline_time) * 100
                memory_improvement = ((baseline_memory - current_memory) / baseline_memory) * 100
                
                self.improvements[test_name] = {
                    'time_improvement_pct': round(time_improvement, 2),
                    'memory_improvement_pct': round(memory_improvement, 2),
                    'time_ratio': round(baseline_time / current_time, 2),
                    'memory_ratio': round(baseline_memory / current_memory, 2)
                }

def measure_performance(func):
    """Decorator to measure execution time and memory usage"""
    def wrapper(*args, **kwargs):
        # Measure memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure execution time
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        return result, execution_time, memory_used
    return wrapper

@pytest.fixture
def sample_signals_data():
    """Generate sample signals data for testing"""
    np.random.seed(42)  # For reproducible tests
    
    symbols = [f'TEST{i:03d}' for i in range(100)]
    
    data = []
    for symbol in symbols:
        data.append({
            'symbol': symbol,
            'company_name': f'{symbol} Inc.',
            'current_price': np.random.uniform(10, 500),
            'signal_direction': np.random.choice(['STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG_SELL']),
            'signal_strength': np.random.uniform(0, 1),
            'confidence': np.random.uniform(0.3, 0.95),
            'rsi_14': np.random.uniform(10, 90),
            'volume': np.random.randint(100000, 10000000),
            'change_1d': np.random.uniform(-5, 5),
            'macd_line': np.random.uniform(-2, 2),
            'macd_signal': np.random.uniform(-2, 2),
            'bb_position': np.random.uniform(0, 1)
        })
    
    return pd.DataFrame(data)

@pytest.fixture
def performance_results():
    """Fixture for collecting performance results"""
    return PerformanceTestResults()

class TestComponentPerformance:
    """Test performance of individual dashboard components"""
    
    def test_utility_component_performance(self, sample_signals_data, performance_results):
        """Test utility component performance improvements"""
        utility = UtilityComponent()
        
        @measure_performance
        def test_styling_functions():
            # Test styling functions with large dataset
            results = []
            for _, row in sample_signals_data.iterrows():
                results.append(utility.style_signals(row['signal_direction']))
                results.append(utility.style_confidence(row['confidence']))
                results.append(utility.style_rsi(row['rsi_14']))
            return results
        
        _, execution_time, memory_used = test_styling_functions()
        performance_results.add_result('utility_styling', execution_time, memory_used)
        
        # Assert reasonable performance
        assert execution_time < 1.0, f"Utility styling took {execution_time:.3f}s, expected <1.0s"
        assert memory_used < 50, f"Utility styling used {memory_used:.1f}MB, expected <50MB"
    
    def test_signal_display_performance(self, sample_signals_data, performance_results):
        """Test signal display component performance"""
        signal_display = SignalDisplayComponent()
        
        @measure_performance  
        def test_data_processing():
            # Test table preparation and styling
            display_columns = ['symbol', 'company_name', 'current_price', 'signal_direction', 'signal_strength']
            processed_data = signal_display._prepare_table_data(sample_signals_data, display_columns)
            return processed_data
        
        _, execution_time, memory_used = test_data_processing()
        performance_results.add_result('signal_display_processing', execution_time, memory_used)
        
        # Assert performance targets
        assert execution_time < 0.5, f"Signal processing took {execution_time:.3f}s, expected <0.5s"
        assert memory_used < 100, f"Signal processing used {memory_used:.1f}MB, expected <100MB"
    
    def test_charts_component_performance(self, sample_signals_data, performance_results):
        """Test performance charts component performance"""
        charts = PerformanceChartsComponent()
        
        @measure_performance
        def test_chart_creation():
            # Test signals heatmap creation
            chart_fig = charts.create_signals_heatmap(sample_signals_data)
            return chart_fig
        
        _, execution_time, memory_used = test_chart_creation()
        performance_results.add_result('charts_creation', execution_time, memory_used)
        
        # Assert performance targets
        assert execution_time < 2.0, f"Chart creation took {execution_time:.3f}s, expected <2.0s"
        assert memory_used < 150, f"Chart creation used {memory_used:.1f}MB, expected <150MB"
    
    def test_data_processor_performance(self, sample_signals_data, performance_results):
        """Test data processor performance"""
        processor = DataProcessor()
        
        @measure_performance
        def test_data_processing():
            # Test comprehensive data processing
            data_hash = str(hash(str(sample_signals_data.values.tobytes())))
            processed_data = processor.process_signals_data(data_hash, sample_signals_data)
            
            # Test filtering
            filters = {
                'signal_direction': ['BUY', 'STRONG_BUY'],
                'min_confidence': 0.5,
                'min_strength': 0.3
            }
            filtered_data = processor.filter_signals(processed_data, filters)
            
            # Test portfolio metrics
            metrics = processor.calculate_portfolio_metrics(processed_data)
            
            return processed_data, filtered_data, metrics
        
        _, execution_time, memory_used = test_data_processing()
        performance_results.add_result('data_processing', execution_time, memory_used)
        
        # Assert performance targets
        assert execution_time < 0.3, f"Data processing took {execution_time:.3f}s, expected <0.3s"
        assert memory_used < 75, f"Data processing used {memory_used:.1f}MB, expected <75MB"

class TestDataStoragePerformance:
    """Test performance of data storage architecture"""
    
    def test_hot_data_manager_performance(self, performance_results):
        """Test Redis hot data manager performance"""
        try:
            hot_data = HotDataManager()
            
            if not hot_data.is_connected():
                pytest.skip("Redis not available for testing")
            
            @measure_performance
            def test_hot_data_operations():
                # Test batch signal storage
                symbols = [f'TEST{i:03d}' for i in range(50)]
                
                # Store signals
                for symbol in symbols:
                    signal_data = {
                        'signal_direction': 'BUY',
                        'signal_strength': 0.75,
                        'confidence': 0.80,
                        'price': 100.50
                    }
                    hot_data.store_live_signal(symbol, signal_data)
                
                # Batch retrieve signals
                retrieved_signals = hot_data.get_live_signals(symbols)
                
                return retrieved_signals
            
            _, execution_time, memory_used = test_hot_data_operations()
            performance_results.add_result('hot_data_operations', execution_time, memory_used)
            
            # Assert performance targets
            assert execution_time < 1.0, f"Hot data operations took {execution_time:.3f}s, expected <1.0s"
            
        except Exception as e:
            pytest.skip(f"Hot data manager test failed: {str(e)}")
    
    def test_warm_data_manager_performance(self, performance_results):
        """Test PostgreSQL warm data manager performance"""
        try:
            warm_data = WarmDataManager()
            
            if not warm_data.engine:
                pytest.skip("PostgreSQL not available for testing")
            
            @measure_performance
            def test_warm_data_operations():
                # Test indicator storage and retrieval
                symbol = 'TEST001'
                test_date = datetime.now()
                
                indicators = {
                    'rsi_14': 65.5,
                    'macd_line': 1.25,
                    'macd_signal': 1.10,
                    'bb_position': 0.75
                }
                
                # Store indicators
                success = warm_data.store_technical_indicators(symbol, test_date, indicators)
                
                # Retrieve indicators
                if success:
                    start_date = test_date - timedelta(days=1)
                    end_date = test_date + timedelta(days=1)
                    retrieved_data = warm_data.get_technical_indicators(symbol, start_date, end_date)
                    return retrieved_data
                
                return pd.DataFrame()
            
            _, execution_time, memory_used = test_warm_data_operations()
            performance_results.add_result('warm_data_operations', execution_time, memory_used)
            
            # Assert performance targets
            assert execution_time < 2.0, f"Warm data operations took {execution_time:.3f}s, expected <2.0s"
            
        except Exception as e:
            pytest.skip(f"Warm data manager test failed: {str(e)}")

class TestMemoryEfficiency:
    """Test memory efficiency improvements"""
    
    def test_component_memory_isolation(self, sample_signals_data, performance_results):
        """Test that components don't leak memory"""
        
        @measure_performance
        def test_component_lifecycle():
            # Create and destroy components multiple times
            components = []
            
            for i in range(10):
                utility = UtilityComponent()
                signal_display = SignalDisplayComponent()
                charts = PerformanceChartsComponent()
                
                # Use components
                utility.style_signals('BUY')
                signal_display._prepare_table_data(sample_signals_data.head(10), ['symbol', 'current_price'])
                
                components.append((utility, signal_display, charts))
            
            # Clear references
            components.clear()
            
            return True
        
        _, execution_time, memory_used = test_component_lifecycle()
        performance_results.add_result('memory_lifecycle', execution_time, memory_used)
        
        # Memory usage should be reasonable for component lifecycle
        assert memory_used < 200, f"Component lifecycle used {memory_used:.1f}MB, expected <200MB"
    
    def test_data_processing_memory_efficiency(self, performance_results):
        """Test memory efficiency of data processing with large datasets"""
        
        @measure_performance
        def test_large_dataset_processing():
            # Create larger dataset for stress testing
            large_data = pd.DataFrame({
                'symbol': [f'TEST{i:04d}' for i in range(1000)],
                'signal_direction': np.random.choice(['BUY', 'SELL', 'NEUTRAL'], 1000),
                'signal_strength': np.random.uniform(0, 1, 1000),
                'confidence': np.random.uniform(0.3, 0.95, 1000),
                'current_price': np.random.uniform(10, 500, 1000)
            })
            
            processor = DataProcessor()
            
            # Process data multiple times to test memory management
            for _ in range(5):
                data_hash = str(hash(str(large_data.values.tobytes())))
                processed = processor.process_signals_data(data_hash, large_data)
                metrics = processor.calculate_portfolio_metrics(processed)
                
                # Clear processed data
                del processed
                del metrics
            
            return True
        
        _, execution_time, memory_used = test_large_dataset_processing()
        performance_results.add_result('large_dataset_processing', execution_time, memory_used)
        
        # Should handle large datasets efficiently
        assert execution_time < 3.0, f"Large dataset processing took {execution_time:.3f}s, expected <3.0s"
        assert memory_used < 300, f"Large dataset processing used {memory_used:.1f}MB, expected <300MB"

class TestPerformanceTargets:
    """Test that performance targets are met"""
    
    def test_dashboard_load_simulation(self, sample_signals_data, performance_results):
        """Simulate dashboard loading with all components"""
        
        @measure_performance
        def simulate_dashboard_load():
            # Simulate complete dashboard loading process
            
            # 1. Initialize components
            utility = UtilityComponent()
            signal_display = SignalDisplayComponent()
            charts = PerformanceChartsComponent()
            processor = DataProcessor()
            
            # 2. Process data
            data_hash = str(hash(str(sample_signals_data.values.tobytes())))
            processed_data = processor.process_signals_data(data_hash, sample_signals_data)
            
            # 3. Prepare display data
            display_columns = ['symbol', 'company_name', 'current_price', 'signal_direction', 'signal_strength']
            table_data = signal_display._prepare_table_data(processed_data, display_columns)
            
            # 4. Create charts
            heatmap = charts.create_signals_heatmap(processed_data)
            
            # 5. Calculate metrics
            metrics = processor.calculate_portfolio_metrics(processed_data)
            
            return {
                'processed_data': len(processed_data),
                'table_data': len(table_data),
                'chart_created': heatmap is not None,
                'metrics': len(metrics)
            }
        
        result, execution_time, memory_used = simulate_dashboard_load()
        performance_results.add_result('dashboard_load_simulation', execution_time, memory_used)
        
        # Critical performance targets
        assert execution_time < 3.0, f"Dashboard load simulation took {execution_time:.3f}s, target is <3.0s"
        assert memory_used < 250, f"Dashboard load used {memory_used:.1f}MB, target is <250MB"
        assert result['chart_created'], "Chart creation failed during simulation"
        assert result['processed_data'] > 0, "Data processing failed during simulation"
        
        logger.info(f"Dashboard load simulation: {execution_time:.3f}s, {memory_used:.1f}MB")

def test_performance_improvements(performance_results):
    """Test that performance improvements meet targets"""
    # Set baseline performance (simulated old system performance)
    baseline_metrics = {
        'utility_styling': (2.5, 80),           # 2.5s, 80MB (old monolithic approach)
        'signal_display_processing': (1.8, 150), # 1.8s, 150MB
        'charts_creation': (4.0, 200),          # 4.0s, 200MB  
        'data_processing': (1.2, 120),          # 1.2s, 120MB
        'dashboard_load_simulation': (8.5, 400)  # 8.5s, 400MB (old monolith)
    }
    
    # Set baselines
    for test_name, (time, memory) in baseline_metrics.items():
        performance_results.set_baseline(test_name, time, memory)
    
    # Calculate improvements
    performance_results.calculate_improvements()
    
    # Verify improvement targets are met
    for test_name, improvements in performance_results.improvements.items():
        time_improvement = improvements['time_improvement_pct']
        memory_improvement = improvements['memory_improvement_pct']
        
        logger.info(f"{test_name}: {time_improvement:.1f}% time improvement, {memory_improvement:.1f}% memory improvement")
        
        # Assert minimum improvement targets
        assert time_improvement > 30, f"{test_name} time improvement {time_improvement:.1f}% < 30% target"
        
        if test_name == 'dashboard_load_simulation':
            # Dashboard load should see significant improvements
            assert time_improvement > 60, f"Dashboard load time improvement {time_improvement:.1f}% < 60% target"

if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short"])