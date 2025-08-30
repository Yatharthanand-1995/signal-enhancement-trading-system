#!/usr/bin/env python3
"""
Comprehensive Testing Framework for All 106 Stocks
Complete validation of ML system across the full stock universe
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path

sys.path.append('src')

class ComprehensiveMLTester:
    """Comprehensive testing framework for all 106 stocks"""
    
    def __init__(self):
        self.data_dir = 'data/full_market'
        self.results_dir = 'testing_results'
        Path(self.results_dir).mkdir(exist_ok=True)
        
        # Stock categorization for analysis
        self.stock_categories = {
            'Mega Cap Tech': ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA'],
            'Large Cap Tech': ['ORCL', 'ADBE', 'CRM', 'AMD', 'INTC', 'CSCO', 'QCOM'],
            'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'C', 'AXP', 'BLK', 'SCHW', 'USB'],
            'Healthcare': ['UNH', 'JNJ', 'PFE', 'ABBV', 'LLY', 'TMO', 'ABT', 'AMGN', 'SYK', 'BSX'],
            'Consumer': ['WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'TJX', 'NKE', 'MCD', 'SBUX'],
            'Industrial': ['CAT', 'BA', 'RTX', 'HON', 'UPS', 'DE', 'GE', 'MMM', 'LMT'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'KMI', 'OKE', 'WMB'],
            'Utilities': ['NEE', 'DUK', 'SO', 'AEP', 'EXC', 'XEL', 'ES', 'AWK', 'PEG'],
            'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'EQR', 'O', 'WELL', 'SPG']
        }
        
    def load_data(self):
        """Load the full market dataset"""
        
        print("📊 LOADING COMPREHENSIVE DATASET")
        print("-" * 40)
        
        val_path = os.path.join(self.data_dir, 'validation_data.csv')
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Dataset not found: {val_path}")
        
        self.data = pd.read_csv(val_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        self.all_symbols = sorted(self.data['symbol'].unique())
        
        print(f"✅ Dataset loaded: {len(self.data):,} records")
        print(f"✅ Symbols: {len(self.all_symbols)}")
        print(f"✅ Period: {self.data['date'].min().date()} to {self.data['date'].max().date()}")
        
        return True
    
    def initialize_ml_system(self):
        """Initialize the ML signal integration system"""
        
        print(f"\n🤖 INITIALIZING ML SYSTEM")
        print("-" * 30)
        
        try:
            from strategy.enhanced_signal_integration import get_enhanced_signal, initialize_enhanced_signal_integration
            
            self.integrator = initialize_enhanced_signal_integration()
            self.get_enhanced_signal = get_enhanced_signal
            
            print(f"✅ ML System: {self.integrator.name}")
            print(f"✅ Correlations: {len(self.integrator.signal_correlations)} proven features")
            
            return True
        except Exception as e:
            print(f"❌ ML initialization failed: {e}")
            return False
    
    def test_single_stock(self, symbol):
        """Test ML signal generation for a single stock"""
        
        try:
            symbol_data = self.data[self.data['symbol'] == symbol].sort_values('date')
            
            if len(symbol_data) < 100:
                return {
                    'symbol': symbol,
                    'status': 'insufficient_data',
                    'records': len(symbol_data),
                    'error': f'Only {len(symbol_data)} records (need 100+)'
                }
            
            # Use last 100 records for testing
            test_data = symbol_data.tail(100)
            current_price = test_data['close'].iloc[-1]
            
            # Generate ML-enhanced signal
            start_time = time.time()
            ml_signal = self.get_enhanced_signal(
                symbol=symbol,
                data=test_data,
                current_price=current_price,
                current_regime='normal'
            )
            generation_time = time.time() - start_time
            
            if ml_signal:
                return {
                    'symbol': symbol,
                    'status': 'success',
                    'signal_strength': float(ml_signal.signal_strength),
                    'confidence': float(ml_signal.confidence),
                    'ml_contribution': float(ml_signal.ml_contribution),
                    'technical_contribution': float(ml_signal.technical_contribution),
                    'predicted_volatility': float(ml_signal.predicted_volatility),
                    'position_size': float(ml_signal.recommended_position_size),
                    'stop_loss_pct': float(ml_signal.stop_loss_pct),
                    'quality': ml_signal.quality.value,
                    'generation_time': generation_time,
                    'records': len(symbol_data),
                    'ml_explanation': getattr(ml_signal, 'ml_explanation', 'N/A')
                }
            else:
                return {
                    'symbol': symbol,
                    'status': 'no_signal',
                    'records': len(symbol_data),
                    'generation_time': generation_time,
                    'error': 'No signal generated'
                }
                
        except Exception as e:
            return {
                'symbol': symbol,
                'status': 'error',
                'error': str(e),
                'generation_time': 0
            }
    
    def run_parallel_testing(self, max_workers=10):
        """Run testing across all stocks in parallel"""
        
        print(f"\n🚀 RUNNING COMPREHENSIVE PARALLEL TESTING")
        print("=" * 50)
        print(f"Testing {len(self.all_symbols)} stocks with {max_workers} parallel workers")
        print()
        
        all_results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.test_single_stock, symbol): symbol 
                for symbol in self.all_symbols
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                result = future.result()
                all_results.append(result)
                
                completed += 1
                
                # Progress indicator
                if completed % 10 == 0 or completed == len(self.all_symbols):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(self.all_symbols) - completed) / rate if rate > 0 else 0
                    
                    print(f"Progress: {completed:3d}/{len(self.all_symbols)} "
                          f"({completed/len(self.all_symbols)*100:.1f}%) "
                          f"- Rate: {rate:.1f}/sec - ETA: {eta:.0f}s")
                
                # Show some results
                if result['status'] == 'success':
                    print(f"  ✅ {symbol}: Signal {result['signal_strength']:+.3f}, "
                          f"Conf {result['confidence']:.3f}, "
                          f"ML {result['ml_contribution']:+.3f}")
                elif result['status'] == 'error':
                    print(f"  ❌ {symbol}: {result['error']}")
                elif result['status'] == 'no_signal':
                    print(f"  ⚠️ {symbol}: No signal generated")
        
        total_time = time.time() - start_time
        print(f"\n⏱️ Testing completed in {total_time:.1f} seconds")
        print(f"   Average: {total_time/len(self.all_symbols):.2f} seconds per stock")
        
        return all_results
    
    def analyze_results(self, results):
        """Comprehensive analysis of testing results"""
        
        print(f"\n📊 COMPREHENSIVE RESULTS ANALYSIS")
        print("=" * 45)
        
        # Basic statistics
        total_tests = len(results)
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        no_signal = [r for r in results if r['status'] == 'no_signal']
        insufficient_data = [r for r in results if r['status'] == 'insufficient_data']
        
        print(f"📈 OVERALL RESULTS")
        print("-" * 20)
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {len(successful)} ({len(successful)/total_tests*100:.1f}%)")
        print(f"No Signal: {len(no_signal)} ({len(no_signal)/total_tests*100:.1f}%)")
        print(f"Failed: {len(failed)} ({len(failed)/total_tests*100:.1f}%)")
        print(f"Insufficient Data: {len(insufficient_data)} ({len(insufficient_data)/total_tests*100:.1f}%)")
        
        if successful:
            # Signal analysis
            signals = [r['signal_strength'] for r in successful]
            confidences = [r['confidence'] for r in successful]
            ml_contribs = [abs(r['ml_contribution']) for r in successful]
            volatilities = [r['predicted_volatility'] for r in successful]
            position_sizes = [r['position_size'] for r in successful]
            
            print(f"\n📊 SIGNAL QUALITY METRICS")
            print("-" * 25)
            print(f"Average Signal Strength: {np.mean(signals):+.3f}")
            print(f"Signal Range: {min(signals):+.3f} to {max(signals):+.3f}")
            print(f"Average Confidence: {np.mean(confidences):.3f}")
            print(f"Confidence Range: {min(confidences):.3f} to {max(confidences):.3f}")
            print(f"Average ML Contribution: {np.mean(ml_contribs):.3f}")
            print(f"ML Contribution Range: {min(ml_contribs):.3f} to {max(ml_contribs):.3f}")
            
            print(f"\n🛡️ RISK MANAGEMENT")
            print("-" * 18)
            print(f"Average Position Size: {np.mean(position_sizes):.1%}")
            print(f"Position Range: {min(position_sizes):.1%} to {max(position_sizes):.1%}")
            print(f"Average Volatility: {np.mean(volatilities):.1%}")
            print(f"Volatility Range: {min(volatilities):.1%} to {max(volatilities):.1%}")
            
            # Signal distribution
            buy_signals = sum(1 for s in signals if s > 0.05)
            sell_signals = sum(1 for s in signals if s < -0.05)
            neutral_signals = len(signals) - buy_signals - sell_signals
            
            print(f"\n📊 SIGNAL DISTRIBUTION")
            print("-" * 22)
            print(f"Buy Signals: {buy_signals} ({buy_signals/len(signals)*100:.1f}%)")
            print(f"Sell Signals: {sell_signals} ({sell_signals/len(signals)*100:.1f}%)")
            print(f"Neutral Signals: {neutral_signals} ({neutral_signals/len(signals)*100:.1f}%)")
            
            # ML effectiveness
            significant_ml = sum(1 for ml in ml_contribs if ml > 0.01)
            print(f"\n🤖 ML EFFECTIVENESS")
            print("-" * 18)
            print(f"Signals with ML Input: {significant_ml}/{len(successful)} ({significant_ml/len(successful)*100:.1f}%)")
        
        # Category analysis
        self.analyze_by_category(successful)
        
        # Performance analysis
        generation_times = [r.get('generation_time', 0) for r in successful]
        if generation_times:
            print(f"\n⚡ PERFORMANCE METRICS")
            print("-" * 22)
            print(f"Average Generation Time: {np.mean(generation_times):.3f}s")
            print(f"Fastest: {min(generation_times):.3f}s")
            print(f"Slowest: {max(generation_times):.3f}s")
            print(f"Throughput: {1/np.mean(generation_times):.1f} signals/second")
        
        return {
            'total_tests': total_tests,
            'successful': len(successful),
            'success_rate': len(successful)/total_tests,
            'average_confidence': np.mean(confidences) if successful else 0,
            'average_signal': np.mean(signals) if successful else 0,
            'ml_effectiveness': significant_ml/len(successful) if successful else 0,
            'throughput': 1/np.mean(generation_times) if generation_times else 0
        }
    
    def analyze_by_category(self, successful_results):
        """Analyze results by stock category"""
        
        print(f"\n📈 CATEGORY ANALYSIS")
        print("-" * 22)
        
        # Create reverse mapping
        symbol_to_category = {}
        for category, symbols in self.stock_categories.items():
            for symbol in symbols:
                symbol_to_category[symbol] = category
        
        category_results = {}
        for result in successful_results:
            symbol = result['symbol']
            category = symbol_to_category.get(symbol, 'Other')
            
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(result)
        
        for category, results in category_results.items():
            if results:
                signals = [r['signal_strength'] for r in results]
                confidences = [r['confidence'] for r in results]
                ml_contribs = [abs(r['ml_contribution']) for r in results]
                
                print(f"\n{category}:")
                print(f"  Count: {len(results)} stocks")
                print(f"  Avg Signal: {np.mean(signals):+.3f}")
                print(f"  Avg Confidence: {np.mean(confidences):.3f}")
                print(f"  Avg ML Contribution: {np.mean(ml_contribs):.3f}")
    
    def save_results(self, results, analysis):
        """Save detailed results to files"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw results
        results_file = os.path.join(self.results_dir, f'comprehensive_test_results_{timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save analysis summary
        analysis_file = os.path.join(self.results_dir, f'analysis_summary_{timestamp}.json')
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save successful signals as CSV for further analysis
        successful = [r for r in results if r['status'] == 'success']
        if successful:
            df = pd.DataFrame(successful)
            csv_file = os.path.join(self.results_dir, f'successful_signals_{timestamp}.csv')
            df.to_csv(csv_file, index=False)
        
        print(f"\n💾 RESULTS SAVED")
        print("-" * 16)
        print(f"Raw Results: {results_file}")
        print(f"Analysis: {analysis_file}")
        if successful:
            print(f"Signals CSV: {csv_file}")
        
        return results_file, analysis_file
    
    def run_comprehensive_test(self, max_workers=8):
        """Run the complete comprehensive testing framework"""
        
        print("🧪 COMPREHENSIVE ML TESTING FRAMEWORK")
        print("=" * 60)
        print(f"Testing ML system across all {len(self.all_symbols) if hasattr(self, 'all_symbols') else '106'} stocks")
        print()
        
        try:
            # Initialize
            self.load_data()
            if not self.initialize_ml_system():
                return False
            
            # Run tests
            results = self.run_parallel_testing(max_workers=max_workers)
            
            # Analyze
            analysis = self.analyze_results(results)
            
            # Save
            self.save_results(results, analysis)
            
            # Final summary
            print(f"\n🎯 COMPREHENSIVE TEST SUMMARY")
            print("=" * 35)
            print(f"✅ Tested: {analysis['total_tests']} stocks")
            print(f"✅ Success Rate: {analysis['success_rate']*100:.1f}%")
            print(f"✅ Average Confidence: {analysis['average_confidence']:.3f}")
            print(f"✅ ML Effectiveness: {analysis['ml_effectiveness']*100:.1f}%")
            print(f"✅ Throughput: {analysis['throughput']:.1f} signals/second")
            
            success_threshold = 0.80  # 80% success rate
            if analysis['success_rate'] >= success_threshold:
                print(f"\n🎉 COMPREHENSIVE TESTING PASSED!")
                print("ML system validated across full stock universe")
                return True
            else:
                print(f"\n⚠️ TESTING NEEDS IMPROVEMENT")
                print(f"Success rate {analysis['success_rate']*100:.1f}% below {success_threshold*100:.0f}% target")
                return False
                
        except Exception as e:
            print(f"❌ Comprehensive testing failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Run comprehensive testing"""
    
    tester = ComprehensiveMLTester()
    success = tester.run_comprehensive_test(max_workers=6)  # Conservative parallel count
    
    if success:
        print(f"\n🚀 READY FOR PRODUCTION")
        print("ML system validated across entire stock universe")
    else:
        print(f"\n🔧 NEEDS OPTIMIZATION")
        print("Review results and optimize system components")

if __name__ == "__main__":
    main()