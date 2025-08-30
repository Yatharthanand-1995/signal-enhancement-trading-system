"""
Test script to initialize the backtesting database schema
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.utils.backtesting_schema_sqlite import initialize_sqlite_backtesting_database, sqlite_backtesting_schema
    from src.utils.logging_setup import get_logger
    
    logger = get_logger(__name__)
    
    def test_schema_initialization():
        """Test the SQLite backtesting schema initialization"""
        try:
            print("🚀 Initializing SQLite backtesting database schema...")
            print(f"Database path: {sqlite_backtesting_schema.db_path}")
            
            # Initialize the schema
            initialize_sqlite_backtesting_database()
            
            # Get schema information
            schema_info = sqlite_backtesting_schema.get_schema_info()
            
            print("✅ Backtesting database schema initialized successfully!")
            print(f"\nDatabase: {schema_info['db_path']}")
            print(f"Tables created: {schema_info['tables_created']}")
            print(f"Schema version: {schema_info['schema_version']}")
            
            print("\nCreated tables with row counts:")
            for table_name, count in schema_info.get('table_counts', {}).items():
                print(f"- {table_name}: {count} records")
            
            print("\nCore backtesting infrastructure:")
            print("- market_regimes: 7 distinct market periods from 2019-2024")
            print("- backtest_configs: Strategy configuration templates")
            print("- backtest_results: Comprehensive performance metrics")
            print("- backtest_trades: Individual trade-level details")
            print("- backtest_portfolio_values: Daily portfolio tracking")
            print("- benchmark_performance: Multi-benchmark comparison data")
            print("- signal_performance_analysis: Signal component attribution")
            print("- regime_performance: Regime-specific performance analysis")
            print("- benchmark_reference: Benchmark metadata and characteristics")
            
            return True
            
        except Exception as e:
            print(f"❌ Schema initialization failed: {str(e)}")
            logger.error(f"Schema initialization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    if __name__ == "__main__":
        success = test_schema_initialization()
        if success:
            print("\n🎉 Ready to start backtesting implementation!")
        else:
            print("\n💥 Please fix schema issues before proceeding")
            sys.exit(1)

except ImportError as e:
    print(f"❌ Import error: {str(e)}")
    print("Please ensure all required modules are installed and paths are correct")
    sys.exit(1)