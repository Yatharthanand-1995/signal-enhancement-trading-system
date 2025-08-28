#!/usr/bin/env python3
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
