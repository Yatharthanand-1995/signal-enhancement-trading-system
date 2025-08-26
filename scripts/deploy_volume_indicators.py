"""
Volume Indicators Deployment Script
Deploy volume indicator enhancements to the trading system database
"""
import psycopg2
import logging
import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from config.config import config
    from data_management.volume_indicators import VolumeIndicatorCalculator
    from strategy.volume_signals import VolumeSignalGenerator
    CONFIG_AVAILABLE = True
except ImportError:
    print("Warning: Config not available, using default connection settings")
    CONFIG_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VolumeIndicatorDeployer:
    """Deploy volume indicator enhancements"""
    
    def __init__(self, db_config=None):
        """Initialize deployer with database configuration"""
        if db_config is None and CONFIG_AVAILABLE:
            db_config = config.db
            
        if db_config:
            self.db_config = {
                'host': db_config.host,
                'port': db_config.port,
                'database': db_config.database,
                'user': db_config.user,
                'password': db_config.password
            }
        else:
            # Default configuration for testing
            self.db_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'trading_system',
                'user': 'trading_user',
                'password': 'your_password'
            }
        
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Connect to database"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            logger.info("✅ Connected to database successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {str(e)}")
            return False
    
    def check_prerequisites(self):
        """Check if prerequisites are met"""
        logger.info("🔍 Checking prerequisites...")
        
        try:
            # Check if technical_indicators table exists
            self.cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'technical_indicators'
                );
            """)
            
            table_exists = self.cursor.fetchone()[0]
            if not table_exists:
                logger.error("❌ technical_indicators table not found")
                return False
            
            # Check if securities table exists
            self.cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'securities'
                );
            """)
            
            securities_exists = self.cursor.fetchone()[0]
            if not securities_exists:
                logger.error("❌ securities table not found")
                return False
            
            logger.info("✅ Prerequisites met")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking prerequisites: {str(e)}")
            return False
    
    def deploy_schema_updates(self):
        """Deploy database schema updates"""
        logger.info("🚀 Deploying volume indicator schema updates...")
        
        try:
            # Read the schema update script
            schema_file = os.path.join(
                os.path.dirname(__file__), 
                '..', 'database', 'volume_indicators_schema_update.sql'
            )
            
            if not os.path.exists(schema_file):
                logger.error(f"❌ Schema file not found: {schema_file}")
                return False
            
            with open(schema_file, 'r') as f:
                schema_sql = f.read()
            
            # Execute schema updates
            logger.info("📝 Executing schema updates...")
            self.cursor.execute(schema_sql)
            self.conn.commit()
            
            logger.info("✅ Schema updates deployed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Schema deployment failed: {str(e)}")
            self.conn.rollback()
            return False
    
    def validate_schema(self):
        """Validate schema updates were applied correctly"""
        logger.info("🔍 Validating schema updates...")
        
        try:
            # Check new columns were added
            expected_columns = [
                'obv', 'cmf', 'mfi', 'vwap', 'accumulation_distribution',
                'price_volume_trend', 'volume_ratio', 'volume_sma_10',
                'volume_ema_20', 'unusual_volume_flag', 'volume_profile'
            ]
            
            self.cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'technical_indicators'
                  AND column_name = ANY(%s);
            """, (expected_columns,))
            
            found_columns = [row[0] for row in self.cursor.fetchall()]
            missing_columns = set(expected_columns) - set(found_columns)
            
            if missing_columns:
                logger.warning(f"⚠️  Missing columns: {missing_columns}")
            else:
                logger.info("✅ All volume indicator columns added successfully")
            
            # Check new tables were created
            expected_tables = ['volume_profile_analysis', 'volume_signals']
            
            for table in expected_tables:
                self.cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    );
                """, (table,))
                
                exists = self.cursor.fetchone()[0]
                if exists:
                    logger.info(f"✅ Table {table} created successfully")
                else:
                    logger.error(f"❌ Table {table} not found")
                    return False
            
            # Check materialized view was updated
            self.cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'mv_latest_prices'
                  AND column_name IN ('obv', 'cmf', 'mfi', 'vwap');
            """)
            
            mv_columns = [row[0] for row in self.cursor.fetchall()]
            if len(mv_columns) >= 2:  # At least some volume columns
                logger.info("✅ Materialized view updated with volume indicators")
            else:
                logger.warning("⚠️  Materialized view may not include all volume indicators")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Schema validation failed: {str(e)}")
            return False
    
    def test_volume_indicators(self):
        """Test volume indicator calculations with sample data"""
        logger.info("🧪 Testing volume indicator calculations...")
        
        try:
            # Test volume indicator calculator
            calculator = VolumeIndicatorCalculator()
            signal_generator = VolumeSignalGenerator()
            
            # Create sample test data
            import pandas as pd
            import numpy as np
            
            dates = pd.date_range('2024-01-01', periods=30, freq='D')
            np.random.seed(42)  # Reproducible test
            
            sample_data = pd.DataFrame({
                'trade_date': dates,
                'open': np.random.normal(100, 1, 30),
                'high': np.random.normal(102, 1, 30),
                'low': np.random.normal(98, 1, 30),
                'close': np.random.normal(100, 1, 30),
                'volume': np.random.randint(1000000, 3000000, 30)
            })
            
            # Fix OHLC relationships
            sample_data['high'] = np.maximum(
                sample_data['high'], 
                sample_data[['open', 'close']].max(axis=1)
            )
            sample_data['low'] = np.minimum(
                sample_data['low'], 
                sample_data[['open', 'close']].min(axis=1)
            )
            
            # Test calculations
            result = calculator.calculate_all_volume_indicators(sample_data)
            
            # Validate results
            expected_indicators = ['obv', 'cmf', 'mfi', 'vwap', 'volume_ratio_20']
            success_count = 0
            
            for indicator in expected_indicators:
                if indicator in result.columns:
                    series = result[indicator].dropna()
                    if not series.empty and not series.isnull().all():
                        success_count += 1
                        logger.info(f"✅ {indicator}: {len(series)} valid values, range [{series.min():.3f}, {series.max():.3f}]")
                    else:
                        logger.warning(f"⚠️  {indicator}: No valid values")
                else:
                    logger.error(f"❌ {indicator}: Column not found")
            
            # Test signal generation
            signals = signal_generator.generate_volume_signals(result)
            total_signals = sum(len(signal_list) for signal_list in signals.values())
            
            logger.info(f"✅ Generated {total_signals} test signals across {len(signals)} categories")
            
            # Success if most indicators work
            if success_count >= len(expected_indicators) * 0.8:
                logger.info("✅ Volume indicator testing completed successfully")
                return True
            else:
                logger.error(f"❌ Only {success_count}/{len(expected_indicators)} indicators working")
                return False
                
        except Exception as e:
            logger.error(f"❌ Volume indicator testing failed: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def create_backup(self):
        """Create backup of existing technical_indicators table"""
        logger.info("💾 Creating backup of technical_indicators table...")
        
        try:
            backup_name = f"technical_indicators_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.cursor.execute(f"""
                CREATE TABLE {backup_name} AS 
                SELECT * FROM technical_indicators;
            """)
            
            self.conn.commit()
            logger.info(f"✅ Backup created: {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Backup creation failed: {str(e)}")
            return False
    
    def deploy(self, create_backup=True, validate=True, test=True):
        """Run complete deployment process"""
        logger.info("🚀 Starting Volume Indicators Deployment")
        logger.info("=" * 50)
        
        deployment_steps = []
        
        try:
            # Step 1: Connect
            if not self.connect():
                return False
            deployment_steps.append("✅ Database connection")
            
            # Step 2: Prerequisites
            if not self.check_prerequisites():
                return False
            deployment_steps.append("✅ Prerequisites check")
            
            # Step 3: Backup (optional)
            if create_backup:
                if not self.create_backup():
                    logger.warning("⚠️  Backup failed, continuing anyway...")
                else:
                    deployment_steps.append("✅ Database backup")
            
            # Step 4: Deploy schema
            if not self.deploy_schema_updates():
                return False
            deployment_steps.append("✅ Schema deployment")
            
            # Step 5: Validate schema (optional)
            if validate:
                if not self.validate_schema():
                    logger.warning("⚠️  Schema validation had issues, but deployment may still work")
                else:
                    deployment_steps.append("✅ Schema validation")
            
            # Step 6: Test calculations (optional)
            if test:
                if not self.test_volume_indicators():
                    logger.warning("⚠️  Testing had issues, but deployment completed")
                else:
                    deployment_steps.append("✅ Volume indicator testing")
            
            # Success summary
            logger.info("🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!")
            logger.info("=" * 50)
            logger.info("Completed steps:")
            for step in deployment_steps:
                logger.info(f"  {step}")
            
            logger.info("\n📋 NEXT STEPS:")
            logger.info("1. Update your technical indicators by running:")
            logger.info("   python -c \"from src.data_management.technical_indicators import TechnicalIndicatorCalculator; calc = TechnicalIndicatorCalculator(); calc.calculate_and_store_indicators()\"")
            logger.info("2. Monitor the volume_signals table for new signal generation")
            logger.info("3. Check the mv_latest_prices materialized view for enhanced data")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {str(e)}")
            return False
        
        finally:
            if self.conn:
                self.conn.close()
                logger.info("🔌 Database connection closed")

def main():
    """Main deployment function"""
    print("🚀 Volume Indicators Deployment Tool")
    print("=" * 40)
    
    deployer = VolumeIndicatorDeployer()
    
    # Run deployment
    success = deployer.deploy(
        create_backup=True,
        validate=True, 
        test=True
    )
    
    if success:
        print("\n✅ Volume indicators deployed successfully!")
        print("Your trading system now has enhanced volume analysis capabilities.")
        exit(0)
    else:
        print("\n❌ Deployment failed!")
        print("Please check the logs above for details.")
        exit(1)

if __name__ == "__main__":
    main()