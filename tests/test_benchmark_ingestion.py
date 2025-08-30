"""
Test script for benchmark data ingestion
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.utils.benchmark_data_ingestion import ingest_all_benchmark_data, benchmark_ingestion
    from src.utils.logging_setup import get_logger
    
    logger = get_logger(__name__)
    
    def test_benchmark_ingestion():
        """Test the benchmark data ingestion process"""
        try:
            print("🚀 Starting benchmark data ingestion...")
            print(f"Symbols: {', '.join(benchmark_ingestion.benchmark_symbols)}")
            print(f"Date range: {benchmark_ingestion.start_date} to {benchmark_ingestion.end_date}")
            print(f"Database: {benchmark_ingestion.db_path}")
            
            # Run the ingestion process
            success = ingest_all_benchmark_data()
            
            if success:
                print("\n✅ Benchmark data ingestion completed successfully!")
                
                # Show validation results
                validation_results = benchmark_ingestion.validate_benchmark_data()
                
                print("\n📊 Benchmark Performance Summary (5-Year Period):")
                print("=" * 80)
                print(f"{'Symbol':<6} {'Records':<8} {'Ann. Return':<12} {'Volatility':<11} {'Max DD':<10} {'Total Return':<12}")
                print("=" * 80)
                
                for symbol, stats in validation_results.items():
                    if stats.get('status') == 'valid':
                        print(f"{symbol:<6} {stats['record_count']:<8} "
                              f"{stats['annualized_return']:>10.2f}% "
                              f"{stats['avg_volatility']:>9.2f}% "
                              f"{stats['max_drawdown']:>8.2f}% "
                              f"{stats['total_return']:>10.2f}%")
                    else:
                        print(f"{symbol:<6} {'ERROR':<8} {stats.get('status', 'unknown')}")
                
                print("=" * 80)
                print("\nBenchmark data is ready for backtesting comparisons!")
                
            else:
                print("❌ Benchmark data ingestion failed or incomplete")
                
            return success
            
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            logger.error(f"Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    if __name__ == "__main__":
        success = test_benchmark_ingestion()
        if success:
            print("\n🎉 Ready for Phase 2: Historical Signal Reconstruction!")
        else:
            print("\n💥 Please fix benchmark ingestion issues before proceeding")
            sys.exit(1)

except ImportError as e:
    print(f"❌ Import error: {str(e)}")
    print("Please ensure all required modules are installed and paths are correct")
    sys.exit(1)