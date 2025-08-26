#!/usr/bin/env python3
"""
System Initialization Script
Sets up database, fetches initial data, and validates the system
"""
import os
import sys
import time
import logging
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_management.stock_data_manager import Top100StocksDataManager
from data_management.technical_indicators import TechnicalIndicatorCalculator
from config.config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system_init.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_logs_directory():
    """Create logs directory if it doesn't exist"""
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    logger.info("Created logs directory")

def test_database_connection():
    """Test database connectivity"""
    try:
        data_manager = Top100StocksDataManager()
        
        # Test basic query
        data_manager.cursor.execute("SELECT version();")
        version = data_manager.cursor.fetchone()
        logger.info(f"Database connected successfully: {version[0]}")
        
        # Test tables exist
        data_manager.cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        """)
        
        tables = [row[0] for row in data_manager.cursor.fetchall()]
        expected_tables = ['securities', 'daily_prices_2024_01', 'technical_indicators', 
                          'trading_signals', 'market_regimes']
        
        missing_tables = [table for table in expected_tables if table not in tables]
        if missing_tables:
            logger.warning(f"Missing tables: {missing_tables}")
        else:
            logger.info("All required tables exist")
            
        data_manager.close()
        return True
        
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return False

def initialize_top_stocks():
    """Initialize top 100 stocks data"""
    try:
        logger.info("Initializing top 100 stocks...")
        data_manager = Top100StocksDataManager()
        
        # Get top 100 stocks
        top_stocks = data_manager.get_top_100_stocks()
        logger.info(f"Retrieved {len(top_stocks)} top stocks")
        
        # Sample for initial testing (first 10 stocks)
        sample_stocks = top_stocks[:10]
        logger.info(f"Starting with sample of {len(sample_stocks)} stocks: {sample_stocks}")
        
        # Fetch historical data (last 2 years)
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        logger.info(f"Fetching historical data from {start_date}")
        
        data_manager.fetch_historical_data(sample_stocks, start_date=start_date)
        
        # Get data quality report
        quality_report = data_manager.get_data_quality_report()
        logger.info(f"Data quality report: {quality_report['statistics']}")
        
        # Optimize database
        data_manager.optimize_database()
        
        data_manager.close()
        return sample_stocks
        
    except Exception as e:
        logger.error(f"Error initializing stocks: {str(e)}")
        return []

def calculate_initial_indicators(symbols):
    """Calculate technical indicators for initial stocks"""
    try:
        logger.info("Calculating technical indicators...")
        calculator = TechnicalIndicatorCalculator()
        
        # Calculate indicators for sample stocks
        calculator.calculate_and_store_indicators(symbols=symbols, lookback_days=200)
        
        # Get indicator summary
        summary = calculator.get_indicator_summary()
        logger.info(f"Indicators calculated: {summary}")
        
        # Test signal features for first stock
        if symbols:
            test_symbol = symbols[0]
            features = calculator.generate_signal_features(test_symbol)
            logger.info(f"Sample signal features for {test_symbol}: {features}")
            
        calculator.close()
        return True
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return False

def validate_system():
    """Validate system is working correctly"""
    logger.info("Validating system...")
    
    try:
        data_manager = Top100StocksDataManager()
        
        # Check data completeness
        data_manager.cursor.execute("""
            SELECT 
                s.symbol,
                COUNT(DISTINCT dp.trade_date) as price_days,
                COUNT(DISTINCT ti.trade_date) as indicator_days
            FROM securities s
            LEFT JOIN daily_prices dp ON s.id = dp.symbol_id 
                AND dp.trade_date >= CURRENT_DATE - INTERVAL '30 days'
            LEFT JOIN technical_indicators ti ON s.id = ti.symbol_id 
                AND ti.trade_date >= CURRENT_DATE - INTERVAL '30 days'
            WHERE s.is_active = true
            GROUP BY s.symbol
            ORDER BY price_days DESC
            LIMIT 10
        """)
        
        results = data_manager.cursor.fetchall()
        
        validation_passed = True
        for symbol, price_days, indicator_days in results:
            if price_days < 10:  # Expect at least 10 days of data
                logger.warning(f"{symbol}: Insufficient price data ({price_days} days)")
                validation_passed = False
            elif indicator_days < 5:  # Expect some indicator data
                logger.warning(f"{symbol}: Insufficient indicator data ({indicator_days} days)")
                validation_passed = False
            else:
                logger.info(f"{symbol}: OK ({price_days} price days, {indicator_days} indicator days)")
        
        data_manager.close()
        
        if validation_passed:
            logger.info("System validation PASSED")
        else:
            logger.warning("System validation FAILED - some issues detected")
            
        return validation_passed
        
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        return False

def create_sample_cron_jobs():
    """Create sample cron job scripts"""
    try:
        # Daily data update script
        daily_update_script = """#!/bin/bash
# Daily data update script
cd "$(dirname "$0")/.."
python3 scripts/daily_update.py >> logs/daily_update.log 2>&1
"""
        
        with open('scripts/daily_update.sh', 'w') as f:
            f.write(daily_update_script)
        
        os.chmod('scripts/daily_update.sh', 0o755)
        
        # Create daily update Python script
        daily_update_py = """#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_management.stock_data_manager import Top100StocksDataManager
from data_management.technical_indicators import TechnicalIndicatorCalculator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Update stock data
        data_manager = Top100StocksDataManager()
        top_stocks = data_manager.get_top_100_stocks()
        data_manager.update_daily_data(top_stocks[:10])  # Start with sample
        data_manager.optimize_database()
        data_manager.close()
        
        # Update indicators
        calculator = TechnicalIndicatorCalculator()
        calculator.calculate_and_store_indicators(symbols=top_stocks[:10])
        calculator.close()
        
        logger.info("Daily update completed successfully")
        
    except Exception as e:
        logger.error(f"Daily update failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
        
        with open('scripts/daily_update.py', 'w') as f:
            f.write(daily_update_py)
        
        os.chmod('scripts/daily_update.py', 0o755)
        
        logger.info("Created sample cron job scripts")
        
    except Exception as e:
        logger.error(f"Error creating cron jobs: {str(e)}")

def main():
    """Main initialization function"""
    logger.info("="*50)
    logger.info("STARTING TRADING SYSTEM INITIALIZATION")
    logger.info("="*50)
    
    # Step 1: Create logs directory
    create_logs_directory()
    
    # Step 2: Test database connection
    logger.info("Step 1: Testing database connection...")
    if not test_database_connection():
        logger.error("Database connection failed. Please check your database setup.")
        logger.info("Make sure to run: docker-compose up -d postgres redis")
        sys.exit(1)
    
    # Step 3: Initialize stock data
    logger.info("Step 2: Initializing stock data...")
    sample_stocks = initialize_top_stocks()
    if not sample_stocks:
        logger.error("Failed to initialize stock data")
        sys.exit(1)
    
    # Step 4: Calculate indicators
    logger.info("Step 3: Calculating technical indicators...")
    if not calculate_initial_indicators(sample_stocks):
        logger.error("Failed to calculate indicators")
        sys.exit(1)
    
    # Step 5: Validate system
    logger.info("Step 4: Validating system...")
    if not validate_system():
        logger.warning("System validation completed with warnings")
    
    # Step 6: Create maintenance scripts
    logger.info("Step 5: Creating maintenance scripts...")
    create_sample_cron_jobs()
    
    logger.info("="*50)
    logger.info("SYSTEM INITIALIZATION COMPLETED SUCCESSFULLY")
    logger.info("="*50)
    
    logger.info("Next steps:")
    logger.info("1. Set up cron job: 0 7 * * * /path/to/scripts/daily_update.sh")
    logger.info("2. Run the ML model training (coming next)")
    logger.info("3. Start the dashboard: streamlit run src/dashboard/main.py")
    
    print("\n" + "="*50)
    print("TRADING SYSTEM READY FOR NEXT PHASE")
    print("="*50)

if __name__ == "__main__":
    main()