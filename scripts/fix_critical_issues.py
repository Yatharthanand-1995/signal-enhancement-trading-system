#!/usr/bin/env python3
"""
Critical Issues Fix Script
Addresses the top priority infrastructure problems identified in the system analysis.
"""
import os
import sys
import subprocess
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/critical_fixes.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CriticalIssuesFixer:
    """Fixes critical infrastructure issues in the trading system"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.fixes_applied = []
        self.errors = []
        
    def run_command(self, command: str, description: str = "") -> Tuple[bool, str]:
        """Run shell command and return success status and output"""
        try:
            logger.info(f"Running: {description or command}")
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Success: {description or command}")
                return True, result.stdout
            else:
                logger.error(f"❌ Failed: {description or command}")
                logger.error(f"Error: {result.stderr}")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Timeout: {description or command}")
            return False, "Command timed out"
        except Exception as e:
            logger.error(f"💥 Exception: {description or command} - {str(e)}")
            return False, str(e)
    
    def check_docker_installation(self) -> bool:
        """Check and fix Docker installation"""
        logger.info("🐳 Checking Docker installation...")
        
        # Check if Docker Desktop is installed
        docker_app_path = "/Applications/Docker.app"
        if not os.path.exists(docker_app_path):
            logger.warning("Docker Desktop not found, attempting installation...")
            success, output = self.run_command(
                "brew install --cask docker",
                "Installing Docker Desktop via Homebrew"
            )
            if not success:
                self.errors.append("Failed to install Docker Desktop")
                return False
        
        # Try to start Docker if not running
        success, _ = self.run_command("docker --version", "Check Docker version")
        if not success:
            logger.info("Starting Docker Desktop...")
            self.run_command("open -a Docker", "Starting Docker Desktop")
            
            # Wait for Docker to start
            for i in range(12):  # Wait up to 2 minutes
                time.sleep(10)
                success, _ = self.run_command("docker --version", "Check Docker after start")
                if success:
                    break
                logger.info(f"Waiting for Docker to start... ({i+1}/12)")
            
            if not success:
                self.errors.append("Docker failed to start after 2 minutes")
                return False
        
        self.fixes_applied.append("Docker installation verified")
        return True
    
    def fix_python_environment(self) -> bool:
        """Fix Python environment and dependencies"""
        logger.info("🐍 Fixing Python environment...")
        
        # Check Python version
        success, output = self.run_command("python3 --version", "Check Python version")
        if not success:
            self.errors.append("Python3 not found in PATH")
            return False
            
        logger.info(f"Python version: {output.strip()}")
        
        # Create/activate virtual environment
        venv_path = self.project_root / "trading_env"
        if not venv_path.exists():
            success, _ = self.run_command(
                "python3 -m venv trading_env",
                "Creating virtual environment"
            )
            if not success:
                self.errors.append("Failed to create virtual environment")
                return False
        
        # Install compatible dependencies
        success, _ = self.run_command(
            "source trading_env/bin/activate && pip install --upgrade pip setuptools wheel",
            "Upgrading pip and build tools"
        )
        if not success:
            logger.warning("Failed to upgrade pip, continuing...")
        
        # Install core dependencies individually to avoid conflicts
        core_packages = [
            "numpy>=1.24.0",
            "pandas>=2.0.0", 
            "python-dotenv>=1.0.0",
            "loguru>=0.7.2",
            "pytest>=7.4.0"
        ]
        
        for package in core_packages:
            success, _ = self.run_command(
                f"source trading_env/bin/activate && pip install '{package}'",
                f"Installing {package}"
            )
            if success:
                logger.info(f"✅ Installed {package}")
            else:
                logger.warning(f"⚠️ Failed to install {package}, continuing...")
        
        self.fixes_applied.append("Python environment setup")
        return True
    
    def fix_database_configuration(self) -> bool:
        """Fix database configuration and security issues"""
        logger.info("🗄️ Fixing database configuration...")
        
        # Create secure environment file
        secure_env_path = self.project_root / ".env.secure"
        if not secure_env_path.exists():
            secure_env_content = """# Secure Environment Configuration
# Generated by critical issues fix script

# Environment
ENVIRONMENT=development
DEBUG=false
LOG_LEVEL=INFO

# Database Configuration (use strong passwords in production)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_system
DB_USER=trading_user
DB_PASSWORD=CHANGE_THIS_PASSWORD_IN_PRODUCTION

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=CHANGE_THIS_PASSWORD_IN_PRODUCTION

# API Keys (Set these with your actual keys)
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here
POLYGON_KEY=your_polygon_key_here

# Logging
LOG_FILE=logs/trading_system.log

# Security
SECRET_KEY=CHANGE_THIS_SECRET_KEY_IN_PRODUCTION
"""
            with open(secure_env_path, 'w') as f:
                f.write(secure_env_content)
            
            logger.info("✅ Created secure environment file (.env.secure)")
            logger.warning("⚠️ Please update passwords and API keys in .env.secure")
        
        self.fixes_applied.append("Database configuration secured")
        return True
    
    def start_database_services(self) -> bool:
        """Start PostgreSQL and Redis services"""
        logger.info("🚀 Starting database services...")
        
        # Check if docker-compose.yml exists
        compose_file = self.project_root / "docker-compose.yml"
        if not compose_file.exists():
            self.errors.append("docker-compose.yml not found")
            return False
        
        # Start services
        success, _ = self.run_command(
            "docker-compose up -d postgres redis",
            "Starting PostgreSQL and Redis containers"
        )
        if not success:
            self.errors.append("Failed to start database services")
            return False
        
        # Wait for services to be ready
        logger.info("Waiting for databases to be ready...")
        time.sleep(15)
        
        # Check PostgreSQL health
        success, _ = self.run_command(
            "docker exec trading_postgres pg_isready -U trading_user -d trading_system",
            "Checking PostgreSQL readiness"
        )
        if success:
            logger.info("✅ PostgreSQL is ready")
        else:
            logger.warning("⚠️ PostgreSQL may not be ready, check manually")
        
        # Apply 2025 partitions
        success, _ = self.run_command(
            "docker exec -i trading_postgres psql -U trading_user -d trading_system < database/add_2025_partitions.sql",
            "Adding 2025 database partitions"
        )
        if success:
            logger.info("✅ Added 2025 database partitions")
        else:
            logger.warning("⚠️ Failed to add 2025 partitions, may need manual intervention")
        
        self.fixes_applied.append("Database services started")
        return True
    
    def setup_monitoring_and_logging(self) -> bool:
        """Setup basic monitoring and logging"""
        logger.info("📊 Setting up monitoring and logging...")
        
        # Create logs directory if it doesn't exist
        logs_dir = self.project_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Create monitoring script
        monitoring_script = self.project_root / "scripts" / "health_check.py"
        monitoring_content = '''#!/usr/bin/env python3
"""
System Health Check Script
Monitors critical components of the trading system
"""
import subprocess
import sys
import json
from datetime import datetime

def check_docker():
    """Check Docker status"""
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def check_database():
    """Check PostgreSQL status"""
    try:
        result = subprocess.run([
            "docker", "exec", "trading_postgres", 
            "pg_isready", "-U", "trading_user", "-d", "trading_system"
        ], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def check_redis():
    """Check Redis status"""
    try:
        result = subprocess.run([
            "docker", "exec", "trading_redis", "redis-cli", "ping"
        ], capture_output=True, text=True)
        return "PONG" in result.stdout
    except:
        return False

def main():
    """Run health checks"""
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "docker": check_docker(),
        "postgresql": check_database(),
        "redis": check_redis()
    }
    
    print(json.dumps(health_status, indent=2))
    
    # Exit with error if any critical service is down
    if not all([health_status["docker"], health_status["postgresql"]]):
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()
'''
        
        with open(monitoring_script, 'w') as f:
            f.write(monitoring_content)
        
        # Make it executable
        os.chmod(monitoring_script, 0o755)
        
        logger.info("✅ Created health check script")
        
        self.fixes_applied.append("Monitoring and logging setup")
        return True
    
    def run_all_fixes(self) -> bool:
        """Run all critical fixes"""
        logger.info("🔧 Starting critical issues fix process...")
        
        fixes = [
            ("Docker Installation", self.check_docker_installation),
            ("Python Environment", self.fix_python_environment),
            ("Database Configuration", self.fix_database_configuration),
            ("Database Services", self.start_database_services),
            ("Monitoring Setup", self.setup_monitoring_and_logging)
        ]
        
        for fix_name, fix_function in fixes:
            try:
                logger.info(f"\\n{'='*50}")
                logger.info(f"🎯 {fix_name}")
                logger.info(f"{'='*50}")
                
                success = fix_function()
                if success:
                    logger.info(f"✅ {fix_name} completed successfully")
                else:
                    logger.error(f"❌ {fix_name} failed")
                    
            except Exception as e:
                logger.error(f"💥 {fix_name} failed with exception: {str(e)}")
                self.errors.append(f"{fix_name}: {str(e)}")
        
        # Summary
        logger.info("\\n" + "="*70)
        logger.info("📋 CRITICAL FIXES SUMMARY")
        logger.info("="*70)
        
        if self.fixes_applied:
            logger.info("✅ Fixes Applied:")
            for fix in self.fixes_applied:
                logger.info(f"   • {fix}")
        
        if self.errors:
            logger.error("❌ Errors Encountered:")
            for error in self.errors:
                logger.error(f"   • {error}")
        
        logger.info("\\n🎯 Next Steps:")
        logger.info("   1. Update .env.secure with actual passwords and API keys")
        logger.info("   2. Run: python scripts/health_check.py to verify services")
        logger.info("   3. Test basic functionality with: python -c 'import src.config.config as cfg; print(cfg.config.to_dict())'")
        logger.info("   4. Consider running the test suite: pytest tests/")
        
        return len(self.errors) == 0

def main():
    """Main entry point"""
    try:
        fixer = CriticalIssuesFixer()
        success = fixer.run_all_fixes()
        
        if success:
            logger.info("\\n🎉 All critical fixes completed successfully!")
            sys.exit(0)
        else:
            logger.error("\\n⚠️ Some fixes failed. Check logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\\n⏹️ Fix process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\\n💥 Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()