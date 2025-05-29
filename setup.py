#!/usr/bin/env python3
"""
Enhanced Reddit Mention Tracker Setup Script

This script handles:
- Dependency installation
- Database initialization
- Configuration setup
- Service verification
- Optional component setup (Redis, monitoring, etc.)
"""
import os
import sys
import subprocess
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import json
import yaml

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('setup.log')
    ]
)
logger = logging.getLogger(__name__)

class SetupManager:
    """Comprehensive setup manager for Reddit Mention Tracker."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.requirements_file = self.project_root / "requirements.txt"
        self.config_dir = self.project_root / "config"
        self.logs_dir = self.project_root / "logs"
        self.exports_dir = self.project_root / "exports"
        self.data_dir = self.project_root / "data"
        
        # Setup status tracking
        self.setup_status = {
            'python_version': False,
            'dependencies': False,
            'playwright': False,
            'directories': False,
            'database': False,
            'configuration': False,
            'redis': False,
            'monitoring': False
        }
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        logger.info("Checking Python version...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
            return False
        
        logger.info(f"Python {version.major}.{version.minor}.{version.micro} - OK")
        self.setup_status['python_version'] = True
        return True
    
    def create_directories(self) -> bool:
        """Create necessary directories."""
        logger.info("Creating project directories...")
        
        directories = [
            self.logs_dir,
            self.exports_dir,
            self.data_dir,
            self.config_dir,
            self.project_root / "database",
            self.project_root / "analytics",
            self.project_root / "scraper",
            self.project_root / "ui",
            self.project_root / "monitoring",
            self.project_root / "api"
        ]
        
        try:
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            
            self.setup_status['directories'] = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to create directories: {str(e)}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies."""
        logger.info("Installing Python dependencies...")
        
        if not self.requirements_file.exists():
            logger.error(f"Requirements file not found: {self.requirements_file}")
            return False
        
        try:
            # Upgrade pip first
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ], check=True, capture_output=True)
            
            # Install requirements
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)
            ], check=True, capture_output=True)
            
            logger.info("Dependencies installed successfully")
            self.setup_status['dependencies'] = True
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def setup_playwright(self) -> bool:
        """Setup Playwright browsers."""
        logger.info("Setting up Playwright browsers...")
        
        try:
            # Install Playwright browsers
            subprocess.run([
                sys.executable, "-m", "playwright", "install"
            ], check=True, capture_output=True)
            
            # Install system dependencies
            subprocess.run([
                sys.executable, "-m", "playwright", "install-deps"
            ], check=True, capture_output=True)
            
            logger.info("Playwright setup completed")
            self.setup_status['playwright'] = True
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Playwright setup failed: {e}")
            logger.info("You may need to run 'playwright install' manually")
            return False
    
    def setup_database(self) -> bool:
        """Initialize database."""
        logger.info("Setting up database...")
        
        try:
            # Import after dependencies are installed
            from database.models import DatabaseManager
            
            # Initialize database
            db_manager = DatabaseManager()
            db_manager.create_tables()
            
            logger.info("Database initialized successfully")
            self.setup_status['database'] = True
            return True
            
        except Exception as e:
            logger.error(f"Database setup failed: {str(e)}")
            return False
    
    def create_default_config(self) -> bool:
        """Create default configuration files."""
        logger.info("Creating default configuration...")
        
        try:
            # Create .env file
            env_file = self.project_root / ".env"
            if not env_file.exists():
                env_content = """# Reddit Mention Tracker Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Database
DATABASE_URL=sqlite:///reddit_mentions.db

# Redis (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Security
SECRET_KEY=your-secret-key-change-in-production

# Monitoring
MONITORING_ENABLED=true
EMAIL_ALERTS_ENABLED=false
WEBHOOK_ALERTS_ENABLED=false

# Features
ADVANCED_SENTIMENT=true
REAL_TIME_MONITORING=true
DATA_VALIDATION=true
CACHING=true
API_ENDPOINTS=true
"""
                env_file.write_text(env_content)
                logger.info("Created .env configuration file")
            
            # Create default settings YAML
            settings_file = self.config_dir / "settings.yaml"
            if not settings_file.exists():
                default_settings = {
                    'environment': 'development',
                    'debug': True,
                    'log_level': 'INFO',
                    'scraping': {
                        'max_pages_per_search': 5,
                        'max_concurrent_requests': 3,
                        'request_delay_min': 1.0,
                        'request_delay_max': 3.0,
                        'timeout_seconds': 30,
                        'retry_attempts': 3,
                        'headless_mode': True,
                        'quality_threshold': 0.3
                    },
                    'features': {
                        'advanced_sentiment': True,
                        'real_time_monitoring': True,
                        'data_validation': True,
                        'caching': True,
                        'api_endpoints': True
                    }
                }
                
                with open(settings_file, 'w') as f:
                    yaml.dump(default_settings, f, default_flow_style=False, indent=2)
                
                logger.info("Created default settings.yaml")
            
            self.setup_status['configuration'] = True
            return True
            
        except Exception as e:
            logger.error(f"Configuration setup failed: {str(e)}")
            return False
    
    def check_redis(self) -> bool:
        """Check Redis availability (optional)."""
        logger.info("Checking Redis availability...")
        
        try:
            import redis
            
            # Try to connect to Redis
            r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=5)
            r.ping()
            
            logger.info("Redis is available and running")
            self.setup_status['redis'] = True
            return True
            
        except Exception as e:
            logger.warning(f"Redis not available: {str(e)}")
            logger.info("Redis is optional but recommended for caching")
            return False
    
    def verify_installation(self) -> bool:
        """Verify that all components are working."""
        logger.info("Verifying installation...")
        
        try:
            # Test imports
            from config.advanced_settings import get_settings
            from database.models import DatabaseManager
            from scraper.reddit_scraper import RedditScraper
            from analytics.metrics_analyzer import MetricsAnalyzer
            from analytics.data_validator import DataValidator
            from ui.visualization import MetricsVisualizer
            
            logger.info("All core components imported successfully")
            
            # Test configuration loading
            settings = get_settings()
            logger.info(f"Configuration loaded: {settings.app_name} v{settings.app_version}")
            
            # Test database connection
            db_manager = DatabaseManager()
            logger.info("Database connection verified")
            
            self.setup_status['monitoring'] = True
            return True
            
        except Exception as e:
            logger.error(f"Installation verification failed: {str(e)}")
            return False
    
    def print_setup_summary(self):
        """Print setup summary and next steps."""
        logger.info("\n" + "="*60)
        logger.info("SETUP SUMMARY")
        logger.info("="*60)
        
        for component, status in self.setup_status.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"{status_icon} {component.replace('_', ' ').title()}: {'OK' if status else 'FAILED'}")
        
        success_count = sum(self.setup_status.values())
        total_count = len(self.setup_status)
        
        logger.info(f"\nSetup completed: {success_count}/{total_count} components successful")
        
        if success_count == total_count:
            logger.info("\n🎉 Setup completed successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Run: python app.py")
            logger.info("2. Open your browser to: http://localhost:7860")
            logger.info("3. Start tracking Reddit mentions!")
            
            if not self.setup_status['redis']:
                logger.info("\nOptional: Install and start Redis for better performance:")
                logger.info("- Ubuntu/Debian: sudo apt install redis-server")
                logger.info("- macOS: brew install redis")
                logger.info("- Windows: Download from https://redis.io/download")
        else:
            logger.error("\n❌ Setup incomplete. Please check the errors above.")
            logger.info("\nTroubleshooting:")
            logger.info("1. Check the setup.log file for detailed error messages")
            logger.info("2. Ensure you have Python 3.8+ installed")
            logger.info("3. Try running with administrator/sudo privileges")
            logger.info("4. Check your internet connection for dependency downloads")
    
    def run_interactive_setup(self):
        """Run interactive setup with user prompts."""
        print("🚀 Reddit Mention Tracker - Enhanced Setup")
        print("="*50)
        
        # Ask user about optional components
        setup_redis = input("Setup Redis for caching? (y/N): ").lower().startswith('y')
        setup_monitoring = input("Enable system monitoring? (Y/n): ").lower() != 'n'
        
        # Run setup steps
        steps = [
            ("Checking Python version", self.check_python_version),
            ("Creating directories", self.create_directories),
            ("Installing dependencies", self.install_dependencies),
            ("Setting up Playwright", self.setup_playwright),
            ("Setting up database", self.setup_database),
            ("Creating configuration", self.create_default_config),
            ("Verifying installation", self.verify_installation)
        ]
        
        if setup_redis:
            steps.append(("Checking Redis", self.check_redis))
        
        # Execute setup steps
        for step_name, step_func in steps:
            print(f"\n📋 {step_name}...")
            try:
                success = step_func()
                if success:
                    print(f"✅ {step_name} completed")
                else:
                    print(f"❌ {step_name} failed")
            except Exception as e:
                print(f"❌ {step_name} failed: {str(e)}")
                logger.error(f"{step_name} failed: {str(e)}")
        
        # Print summary
        self.print_setup_summary()
    
    def run_automated_setup(self):
        """Run automated setup without user interaction."""
        logger.info("Starting automated setup...")
        
        steps = [
            self.check_python_version,
            self.create_directories,
            self.install_dependencies,
            self.setup_playwright,
            self.setup_database,
            self.create_default_config,
            self.check_redis,
            self.verify_installation
        ]
        
        for step_func in steps:
            try:
                step_func()
            except Exception as e:
                logger.error(f"Setup step failed: {str(e)}")
        
        self.print_setup_summary()

def main():
    """Main setup function."""
    setup_manager = SetupManager()
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        setup_manager.run_automated_setup()
    else:
        setup_manager.run_interactive_setup()

if __name__ == "__main__":
    main() 