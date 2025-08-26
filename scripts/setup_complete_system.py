#!/usr/bin/env python3
"""
Complete System Setup Script
Sets up the entire trading system from scratch
"""
import os
import sys
import subprocess
import time
import logging
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/complete_setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_command(command, description, ignore_error=False):
    """Run a command and handle errors"""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        logger.info(f"Success: {description}")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        if ignore_error:
            logger.warning(f"Command failed (ignored): {description}")
            logger.warning(f"Error: {e.stderr}")
            return False
        else:
            logger.error(f"Command failed: {description}")
            logger.error(f"Error: {e.stderr}")
            raise

def create_directories():
    """Create necessary directories"""
    directories = [
        'logs',
        'models',
        'data',
        'backtest_results',
        'reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created/verified directory: {directory}")

def check_prerequisites():
    """Check if prerequisites are installed"""
    logger.info("Checking prerequisites...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or python_version.minor < 8:
        raise RuntimeError("Python 3.8+ is required")
    
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check Docker
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        logger.info(f"Docker: {result.stdout.strip()}")
    except FileNotFoundError:
        logger.warning("Docker not found - you'll need to install PostgreSQL manually")
    
    # Check Docker Compose
    try:
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        logger.info(f"Docker Compose: {result.stdout.strip()}")
    except FileNotFoundError:
        logger.warning("Docker Compose not found")

def setup_python_environment():
    """Set up Python environment and install dependencies"""
    logger.info("Setting up Python environment...")
    
    # Install requirements
    run_command("pip install --upgrade pip", "Upgrading pip")
    run_command("pip install -r requirements.txt", "Installing Python dependencies")
    
    logger.info("Python environment setup complete")

def start_database_services():
    """Start PostgreSQL and Redis services"""
    logger.info("Starting database services...")
    
    # Start services
    run_command("docker-compose up -d", "Starting PostgreSQL and Redis")
    
    # Wait for services to be ready
    logger.info("Waiting for services to be ready...")
    time.sleep(10)
    
    # Check PostgreSQL
    max_retries = 30
    retries = 0
    while retries < max_retries:
        try:
            result = subprocess.run(
                ['docker', 'exec', 'trading_postgres', 'pg_isready', '-U', 'trading_user'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                logger.info("PostgreSQL is ready")
                break
        except subprocess.CalledProcessError:
            pass
        
        retries += 1
        time.sleep(2)
        logger.info(f"Waiting for PostgreSQL... ({retries}/{max_retries})")
    
    if retries >= max_retries:
        raise RuntimeError("PostgreSQL failed to start")
    
    # Check Redis
    try:
        result = subprocess.run(
            ['docker', 'exec', 'trading_redis', 'redis-cli', 'ping'],
            capture_output=True, text=True
        )
        if 'PONG' in result.stdout:
            logger.info("Redis is ready")
    except subprocess.CalledProcessError:
        logger.warning("Redis check failed")

def initialize_database_and_data():
    """Initialize database and fetch initial data"""
    logger.info("Initializing database and fetching data...")
    
    # Run system initialization
    run_command("python scripts/initialize_system.py", "System initialization")
    
    logger.info("Database and data initialization complete")

def train_ml_models():
    """Train machine learning models"""
    logger.info("Training ML models...")
    
    try:
        from data_management.stock_data_manager import Top100StocksDataManager
        from models.ml_ensemble import LSTMXGBoostEnsemble
        from models.regime_detection import MarketRegimeDetector
        
        # Get sample data for training
        data_manager = Top100StocksDataManager()
        top_stocks = data_manager.get_top_100_stocks()[:5]  # Start with 5 stocks
        
        # Fetch comprehensive data for training
        data_manager.cursor.execute("""
            SELECT dp.trade_date, dp.open, dp.high, dp.low, dp.close, dp.volume,
                   ti.rsi_9, ti.rsi_14, ti.macd_value, ti.macd_signal, ti.macd_histogram,
                   ti.bb_upper, ti.bb_middle, ti.bb_lower, ti.sma_20, ti.sma_50,
                   ti.ema_12, ti.ema_26, ti.atr_14, ti.volume_sma_20,
                   s.symbol
            FROM daily_prices dp
            LEFT JOIN technical_indicators ti ON dp.symbol_id = ti.symbol_id 
                AND dp.trade_date = ti.trade_date
            JOIN securities s ON dp.symbol_id = s.id
            WHERE s.symbol IN ('AAPL', 'MSFT', 'GOOGL')
              AND dp.trade_date >= '2022-01-01'
            ORDER BY dp.trade_date
        """)
        
        columns = [desc[0] for desc in data_manager.cursor.description]
        data = data_manager.cursor.fetchall()
        
        if data:
            import pandas as pd
            df = pd.DataFrame(data, columns=columns)
            
            logger.info(f"Training data shape: {df.shape}")
            
            # Train ensemble model
            logger.info("Training LSTM-XGBoost ensemble...")
            ensemble = LSTMXGBoostEnsemble()
            
            # Group by symbol and train on each
            for symbol in ['AAPL']:  # Start with one symbol
                symbol_data = df[df['symbol'] == symbol].copy()
                if len(symbol_data) > 200:
                    logger.info(f"Training on {symbol} with {len(symbol_data)} samples")
                    results = ensemble.train(symbol_data)
                    logger.info(f"Training results for {symbol}: {results.get('success', False)}")
            
            # Train regime detection
            logger.info("Training regime detection model...")
            detector = MarketRegimeDetector(n_regimes=2)
            market_data = detector.prepare_market_data(lookback_days=500)
            
            if len(market_data) > 50:
                regime_results = detector.fit(market_data)
                logger.info(f"Regime detection results: {regime_results.get('success', False)}")
                
                # Save model
                detector.save_model('models/regime_detector.pkl')
            
        data_manager.close()
        
    except Exception as e:
        logger.error(f"ML model training failed: {str(e)}")
        logger.info("Continuing setup without trained models...")

def create_systemd_service():
    """Create systemd service for automated updates (Linux only)"""
    if os.name != 'posix':
        logger.info("Skipping systemd service creation (not on Linux)")
        return
    
    service_content = f"""[Unit]
Description=Trading System Daily Update
After=network.target

[Service]
Type=oneshot
User={os.getenv('USER', 'trading')}
WorkingDirectory={os.path.abspath('.')}
ExecStart=/usr/bin/python3 scripts/daily_update.py
Environment=PATH={os.environ.get('PATH')}

[Install]
WantedBy=multi-user.target
"""

    timer_content = """[Unit]
Description=Run Trading System Daily Update
Requires=trading-system-update.service

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
"""
    
    try:
        # Write service files
        with open('/tmp/trading-system-update.service', 'w') as f:
            f.write(service_content)
        
        with open('/tmp/trading-system-update.timer', 'w') as f:
            f.write(timer_content)
        
        logger.info("Systemd service files created in /tmp/")
        logger.info("To install: sudo cp /tmp/trading-system-update.* /etc/systemd/system/")
        logger.info("Then run: sudo systemctl enable trading-system-update.timer")
        
    except Exception as e:
        logger.warning(f"Could not create systemd service: {str(e)}")

def create_startup_scripts():
    """Create startup scripts for different platforms"""
    
    # Create startup script
    startup_script = f"""#!/bin/bash
# Trading System Startup Script

echo "Starting Trading System..."

# Navigate to project directory
cd "{os.path.abspath('.')}"

# Activate virtual environment if it exists
if [ -d "trading_env" ]; then
    source trading_env/bin/activate
fi

# Start database services
echo "Starting database services..."
docker-compose up -d

# Wait for services
sleep 10

# Run system health check
echo "Running health check..."
python scripts/system_health_check.py

# Start dashboard
echo "Starting dashboard on http://localhost:8501"
streamlit run src/dashboard/main.py

echo "Trading System startup complete!"
"""
    
    with open('start_system.sh', 'w') as f:
        f.write(startup_script)
    
    os.chmod('start_system.sh', 0o755)
    
    # Create Windows batch file
    windows_script = f"""@echo off
echo Starting Trading System...

cd /d "{os.path.abspath('.')}"

REM Activate virtual environment if it exists
if exist trading_env\\Scripts\\activate.bat (
    call trading_env\\Scripts\\activate.bat
)

REM Start database services
echo Starting database services...
docker-compose up -d

REM Wait for services
timeout /t 10 /nobreak

REM Run system health check
echo Running health check...
python scripts\\system_health_check.py

REM Start dashboard
echo Starting dashboard on http://localhost:8501
streamlit run src\\dashboard\\main.py

echo Trading System startup complete!
pause
"""
    
    with open('start_system.bat', 'w') as f:
        f.write(windows_script)
    
    logger.info("Created startup scripts: start_system.sh and start_system.bat")

def create_system_health_check():
    """Create system health check script"""
    health_check_script = """#!/usr/bin/env python3
import sys
import os
import psycopg2
import redis
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def check_database():
    try:
        from config.config import config
        conn = psycopg2.connect(
            host=config.db.host,
            port=config.db.port,
            database=config.db.database,
            user=config.db.user,
            password=config.db.password
        )
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM securities WHERE is_active = true")
        count = cursor.fetchone()[0]
        conn.close()
        print(f"✅ Database: {count} active securities")
        return True
    except Exception as e:
        print(f"❌ Database: {str(e)}")
        return False

def check_redis():
    try:
        from config.config import config
        r = redis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            db=config.redis.db
        )
        r.ping()
        print("✅ Redis: Connected")
        return True
    except Exception as e:
        print(f"❌ Redis: {str(e)}")
        return False

def check_data_freshness():
    try:
        from config.config import config
        conn = psycopg2.connect(
            host=config.db.host,
            port=config.db.port,
            database=config.db.database,
            user=config.db.user,
            password=config.db.password
        )
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(trade_date) FROM daily_prices")
        last_date = cursor.fetchone()[0]
        conn.close()
        
        if last_date:
            days_old = (datetime.now().date() - last_date).days
            if days_old <= 3:
                print(f"✅ Data freshness: {days_old} days old")
                return True
            else:
                print(f"⚠️ Data freshness: {days_old} days old (consider updating)")
                return True
        else:
            print("❌ Data freshness: No data found")
            return False
    except Exception as e:
        print(f"❌ Data freshness: {str(e)}")
        return False

def main():
    print("🔍 Trading System Health Check")
    print("=" * 40)
    
    checks = [
        check_database(),
        check_redis(),
        check_data_freshness()
    ]
    
    passed = sum(checks)
    total = len(checks)
    
    print("=" * 40)
    print(f"Health Check Complete: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 System is healthy!")
        sys.exit(0)
    else:
        print("⚠️ Some issues detected")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
    
    with open('scripts/system_health_check.py', 'w') as f:
        f.write(health_check_script)
    
    os.chmod('scripts/system_health_check.py', 0o755)
    logger.info("Created system health check script")

def run_final_validation():
    """Run final system validation"""
    logger.info("Running final system validation...")
    
    # Run health check
    run_command("python scripts/system_health_check.py", "System health check")
    
    # Test dashboard startup (just check if it can import)
    try:
        import streamlit
        logger.info("✅ Streamlit dashboard ready")
    except ImportError:
        logger.warning("⚠️ Streamlit not installed - dashboard may not work")
    
    # Test core components
    try:
        from data_management.stock_data_manager import Top100StocksDataManager
        from models.regime_detection import MarketRegimeDetector
        from risk_management.risk_manager import AdaptiveRiskManager
        logger.info("✅ Core components imported successfully")
    except ImportError as e:
        logger.error(f"❌ Core component import failed: {str(e)}")
    
    logger.info("Final validation complete")

def main():
    """Main setup function"""
    start_time = time.time()
    
    print("="*60)
    print("🚀 ADVANCED TRADING SYSTEM COMPLETE SETUP")
    print("="*60)
    
    try:
        # Step 1: Check prerequisites
        logger.info("Step 1/10: Checking prerequisites...")
        check_prerequisites()
        
        # Step 2: Create directories
        logger.info("Step 2/10: Creating directories...")
        create_directories()
        
        # Step 3: Setup Python environment
        logger.info("Step 3/10: Setting up Python environment...")
        setup_python_environment()
        
        # Step 4: Start database services
        logger.info("Step 4/10: Starting database services...")
        start_database_services()
        
        # Step 5: Initialize database and data
        logger.info("Step 5/10: Initializing database and data...")
        initialize_database_and_data()
        
        # Step 6: Train ML models
        logger.info("Step 6/10: Training ML models...")
        train_ml_models()
        
        # Step 7: Create system scripts
        logger.info("Step 7/10: Creating system scripts...")
        create_startup_scripts()
        create_system_health_check()
        
        # Step 8: Create systemd service (Linux only)
        logger.info("Step 8/10: Creating system services...")
        create_systemd_service()
        
        # Step 9: Run final validation
        logger.info("Step 9/10: Running final validation...")
        run_final_validation()
        
        # Step 10: Complete
        logger.info("Step 10/10: Setup complete!")
        
        setup_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"⏱️ Setup time: {setup_time/60:.1f} minutes")
        print()
        print("📋 NEXT STEPS:")
        print("1. Start the system: ./start_system.sh (Linux/Mac) or start_system.bat (Windows)")
        print("2. Open dashboard: http://localhost:8501")
        print("3. Check system health: python scripts/system_health_check.py")
        print("4. View logs: tail -f logs/complete_setup.log")
        print()
        print("📚 DOCUMENTATION:")
        print("- README.md: Comprehensive documentation")
        print("- config/config.py: System configuration")
        print("- logs/: System logs")
        print()
        print("⚠️ IMPORTANT:")
        print("- This system is for educational purposes only")
        print("- Not financial advice - use at your own risk")
        print("- Test thoroughly before any live trading")
        print()
        print("🤖 TRADING SYSTEM READY!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        print(f"\n❌ Setup failed: {str(e)}")
        print("Check logs/complete_setup.log for details")
        sys.exit(1)

if __name__ == "__main__":
    main()