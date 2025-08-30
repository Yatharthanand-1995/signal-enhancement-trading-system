#!/usr/bin/env python3
"""
Performance Optimization and Batch Processing System
High-performance ML signal generation for 106-stock universe
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import asyncio
import aiohttp
from pathlib import Path
import pickle
import threading
from queue import Queue, Empty
import logging

sys.path.append('src')

@dataclass
class OptimizedSignalResult:
    """Optimized signal result structure"""
    symbol: str
    timestamp: datetime
    signal_strength: float
    confidence: float
    ml_contribution: float
    position_size: float
    predicted_volatility: float
    generation_time: float
    cache_hit: bool = False

class PerformanceOptimizer:
    """High-performance ML signal generation system"""
    
    def __init__(self, cache_enabled=True, parallel_workers=8):
        self.cache_enabled = cache_enabled
        self.parallel_workers = parallel_workers
        self.cache_dir = Path('cache/signals')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance metrics
        self.metrics = {
            'total_generated': 0,
            'cache_hits': 0,
            'avg_generation_time': 0,
            'throughput': 0,
            'batch_times': []
        }
        
        # Preloaded data cache
        self.data_cache = {}
        self.ml_system = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def initialize_system(self):
        """Initialize ML system and preload data"""
        
        print("🚀 INITIALIZING PERFORMANCE-OPTIMIZED SYSTEM")
        print("=" * 50)
        
        start_time = time.time()
        
        # Initialize ML system
        try:
            from strategy.enhanced_signal_integration import get_enhanced_signal, initialize_enhanced_signal_integration
            
            self.ml_integrator = initialize_enhanced_signal_integration()
            self.get_enhanced_signal = get_enhanced_signal
            
            print(f"✅ ML System: {self.ml_integrator.name}")
            
        except Exception as e:
            print(f"❌ ML system initialization failed: {e}")
            return False
        
        # Preload data
        data_path = 'data/full_market/validation_data.csv'
        if os.path.exists(data_path):
            self.full_data = pd.read_csv(data_path)
            self.full_data['date'] = pd.to_datetime(self.full_data['date'])
            
            print(f"✅ Dataset: {len(self.full_data):,} records")
            
            # Preprocess data by symbol for faster access
            for symbol in self.full_data['symbol'].unique():
                symbol_data = self.full_data[self.full_data['symbol'] == symbol].sort_values('date')
                self.data_cache[symbol] = symbol_data.tail(200)  # Keep last 200 records
            
            print(f"✅ Preprocessed: {len(self.data_cache)} symbols")
        else:
            print("❌ Dataset not found")
            return False
        
        init_time = time.time() - start_time
        print(f"✅ Initialization completed in {init_time:.2f}s")
        
        return True
    
    def _cache_key(self, symbol: str, regime: str = 'normal') -> str:
        """Generate cache key for signal"""
        # Use current date as cache key component
        date_str = datetime.now().strftime('%Y-%m-%d')
        return f"signal_{symbol}_{regime}_{date_str}"
    
    def _load_from_cache(self, cache_key: str) -> Optional[OptimizedSignalResult]:
        """Load signal from cache"""
        if not self.cache_enabled:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            if cache_file.exists():
                # Check if cache is still valid (within 1 hour)
                cache_age = time.time() - cache_file.stat().st_mtime
                if cache_age < 3600:  # 1 hour
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                    result.cache_hit = True
                    return result
        except Exception:
            pass  # Cache miss or error
        
        return None
    
    def _save_to_cache(self, cache_key: str, result: OptimizedSignalResult):
        """Save signal to cache"""
        if not self.cache_enabled:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            self.logger.warning(f"Cache save failed: {e}")
    
    def generate_single_optimized_signal(self, symbol: str, regime: str = 'normal') -> Optional[OptimizedSignalResult]:
        """Generate optimized signal for single symbol"""
        
        start_time = time.time()
        
        # Check cache first
        cache_key = self._cache_key(symbol, regime)
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            self.metrics['cache_hits'] += 1
            return cached_result
        
        try:
            # Get preprocessed data
            if symbol not in self.data_cache:
                return None
            
            symbol_data = self.data_cache[symbol]
            if len(symbol_data) < 100:
                return None
            
            test_data = symbol_data.tail(100)
            current_price = test_data['close'].iloc[-1]
            
            # Generate signal
            ml_signal = self.get_enhanced_signal(
                symbol=symbol,
                data=test_data,
                current_price=current_price,
                current_regime=regime
            )
            
            generation_time = time.time() - start_time
            
            if ml_signal:
                result = OptimizedSignalResult(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    signal_strength=float(ml_signal.signal_strength),
                    confidence=float(ml_signal.confidence),
                    ml_contribution=float(ml_signal.ml_contribution),
                    position_size=float(ml_signal.recommended_position_size),
                    predicted_volatility=float(ml_signal.predicted_volatility),
                    generation_time=generation_time,
                    cache_hit=False
                )
                
                # Save to cache
                self._save_to_cache(cache_key, result)
                
                self.metrics['total_generated'] += 1
                return result
        
        except Exception as e:
            self.logger.error(f"Signal generation failed for {symbol}: {e}")
        
        return None
    
    def batch_process_symbols(self, symbols: List[str], regime: str = 'normal') -> List[OptimizedSignalResult]:
        """Process multiple symbols in parallel batch"""
        
        print(f"⚡ BATCH PROCESSING {len(symbols)} SYMBOLS")
        print("-" * 40)
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.generate_single_optimized_signal, symbol, regime): symbol
                for symbol in symbols
            }
            
            # Collect results
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch processing failed for {symbol}: {e}")
        
        batch_time = time.time() - start_time
        self.metrics['batch_times'].append(batch_time)
        
        # Update metrics
        throughput = len(results) / batch_time if batch_time > 0 else 0
        
        print(f"✅ Processed {len(results)}/{len(symbols)} symbols in {batch_time:.2f}s")
        print(f"   Throughput: {throughput:.1f} signals/second")
        print(f"   Cache hits: {sum(1 for r in results if r.cache_hit)}/{len(results)}")
        
        return results
    
    def streaming_signal_generator(self, symbols: List[str], update_interval: float = 60.0):
        """Streaming signal generator for real-time processing"""
        
        print(f"📡 STARTING STREAMING MODE")
        print(f"   Symbols: {len(symbols)}")
        print(f"   Update interval: {update_interval}s")
        print("   Press Ctrl+C to stop")
        print()
        
        try:
            while True:
                start_time = time.time()
                
                # Generate signals for all symbols
                results = self.batch_process_symbols(symbols)
                
                # Display results
                timestamp = datetime.now().strftime('%H:%M:%S')
                print(f"[{timestamp}] Generated {len(results)} signals")
                
                # Show top signals
                if results:
                    # Sort by absolute signal strength
                    sorted_results = sorted(results, key=lambda r: abs(r.signal_strength), reverse=True)
                    
                    print("Top 5 strongest signals:")
                    for i, result in enumerate(sorted_results[:5]):
                        print(f"  {i+1}. {result.symbol}: {result.signal_strength:+.3f} "
                              f"(Conf: {result.confidence:.3f})")
                
                # Wait for next update
                elapsed = time.time() - start_time
                sleep_time = max(0, update_interval - elapsed)
                
                if sleep_time > 0:
                    print(f"Sleeping {sleep_time:.1f}s until next update...\n")
                    time.sleep(sleep_time)
                else:
                    print("Warning: Processing took longer than update interval\n")
        
        except KeyboardInterrupt:
            print("\n🛑 Streaming mode stopped")
    
    def benchmark_performance(self, test_symbols: int = 50, iterations: int = 3):
        """Benchmark system performance"""
        
        print(f"🏁 PERFORMANCE BENCHMARK")
        print("=" * 30)
        print(f"Testing with {test_symbols} symbols, {iterations} iterations")
        print()
        
        # Select test symbols
        available_symbols = list(self.data_cache.keys())
        test_list = available_symbols[:test_symbols]
        
        benchmark_results = []
        
        for i in range(iterations):
            print(f"📊 Iteration {i+1}/{iterations}")
            
            # Clear cache for fair testing
            if self.cache_enabled:
                cache_files = list(self.cache_dir.glob("*.pkl"))
                for cache_file in cache_files:
                    try:
                        cache_file.unlink()
                    except:
                        pass
            
            # Run benchmark
            start_time = time.time()
            results = self.batch_process_symbols(test_list)
            total_time = time.time() - start_time
            
            throughput = len(results) / total_time if total_time > 0 else 0
            avg_time = np.mean([r.generation_time for r in results]) if results else 0
            
            benchmark_results.append({
                'iteration': i + 1,
                'total_time': total_time,
                'successful_signals': len(results),
                'throughput': throughput,
                'avg_generation_time': avg_time
            })
            
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Throughput: {throughput:.1f} signals/sec")
            print(f"   Avg gen time: {avg_time:.3f}s")
            print()
        
        # Summary
        avg_throughput = np.mean([b['throughput'] for b in benchmark_results])
        avg_gen_time = np.mean([b['avg_generation_time'] for b in benchmark_results])
        
        print(f"🎯 BENCHMARK SUMMARY")
        print("-" * 20)
        print(f"Average throughput: {avg_throughput:.1f} signals/second")
        print(f"Average generation time: {avg_gen_time:.3f}s per signal")
        print(f"Peak throughput: {max(b['throughput'] for b in benchmark_results):.1f} signals/second")
        
        # Performance rating
        if avg_throughput > 100:
            rating = "🚀 EXCELLENT"
        elif avg_throughput > 50:
            rating = "✅ GOOD"
        elif avg_throughput > 25:
            rating = "⚠️ ACCEPTABLE"
        else:
            rating = "❌ NEEDS OPTIMIZATION"
        
        print(f"Performance Rating: {rating}")
        
        return benchmark_results
    
    def optimize_memory_usage(self):
        """Optimize memory usage for large-scale processing"""
        
        print(f"💾 OPTIMIZING MEMORY USAGE")
        print("-" * 25)
        
        # Trim data cache to essential records only
        for symbol in self.data_cache:
            if len(self.data_cache[symbol]) > 150:
                self.data_cache[symbol] = self.data_cache[symbol].tail(150)
        
        # Clean old cache files
        if self.cache_enabled:
            cutoff_time = time.time() - 86400  # 24 hours
            cleaned = 0
            
            for cache_file in self.cache_dir.glob("*.pkl"):
                if cache_file.stat().st_mtime < cutoff_time:
                    try:
                        cache_file.unlink()
                        cleaned += 1
                    except:
                        pass
            
            print(f"✅ Cleaned {cleaned} old cache files")
        
        print(f"✅ Memory optimization completed")
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        
        cache_hit_rate = (self.metrics['cache_hits'] / max(self.metrics['total_generated'], 1)) * 100
        avg_batch_time = np.mean(self.metrics['batch_times']) if self.metrics['batch_times'] else 0
        
        return {
            'total_signals_generated': self.metrics['total_generated'],
            'cache_hit_rate': cache_hit_rate,
            'average_batch_time': avg_batch_time,
            'symbols_cached': len(self.data_cache),
            'cache_enabled': self.cache_enabled,
            'parallel_workers': self.parallel_workers
        }

def main():
    """Main performance optimization demo"""
    
    print("⚡ PERFORMANCE OPTIMIZATION & BATCH PROCESSING")
    print("=" * 60)
    print("High-performance ML signal generation for 106-stock universe")
    print()
    
    # Initialize optimizer
    optimizer = PerformanceOptimizer(cache_enabled=True, parallel_workers=8)
    
    if not optimizer.initialize_system():
        print("❌ System initialization failed")
        return
    
    # Get available symbols
    available_symbols = list(optimizer.data_cache.keys())
    print(f"✅ Available symbols: {len(available_symbols)}")
    
    # Demo menu
    while True:
        print(f"\n🎛️ PERFORMANCE OPTIMIZATION MENU")
        print("-" * 35)
        print("1. Benchmark Performance")
        print("2. Batch Process All Symbols")  
        print("3. Streaming Mode (Real-time)")
        print("4. Memory Optimization")
        print("5. Performance Report")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            optimizer.benchmark_performance(test_symbols=50, iterations=3)
        
        elif choice == '2':
            results = optimizer.batch_process_symbols(available_symbols)
            print(f"\n📊 Batch Results: {len(results)} signals generated")
            
            if results:
                # Show distribution
                buy_signals = sum(1 for r in results if r.signal_strength > 0.05)
                sell_signals = sum(1 for r in results if r.signal_strength < -0.05)
                print(f"Buy: {buy_signals}, Sell: {sell_signals}, Neutral: {len(results) - buy_signals - sell_signals}")
        
        elif choice == '3':
            update_interval = float(input("Update interval (seconds, default 60): ") or "60")
            optimizer.streaming_signal_generator(available_symbols[:20], update_interval)  # Limit to 20 for demo
        
        elif choice == '4':
            optimizer.optimize_memory_usage()
        
        elif choice == '5':
            report = optimizer.get_performance_report()
            print(f"\n📊 PERFORMANCE REPORT")
            print("-" * 22)
            for key, value in report.items():
                if isinstance(value, float):
                    print(f"{key.replace('_', ' ').title()}: {value:.2f}")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value}")
        
        elif choice == '6':
            print("👋 Goodbye!")
            break
        
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()