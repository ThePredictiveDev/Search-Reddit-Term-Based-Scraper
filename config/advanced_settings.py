"""
Advanced configuration management system with environment-specific settings,
validation, and dynamic updates.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import yaml
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Environment(str, Enum):
    """Application environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class ScrapingConfig:
    """Scraping configuration."""
    max_pages_per_search: int = 5
    max_concurrent_requests: int = 3
    request_delay_min: float = 1.0
    request_delay_max: float = 3.0
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay: float = 5.0
    user_agent_rotation: bool = True
    proxy_rotation: bool = False
    headless_mode: bool = True
    stealth_mode: bool = True
    quality_threshold: float = 0.3
    enable_caching: bool = True
    cache_ttl_hours: int = 24

@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = "sqlite:///reddit_mentions.db"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    backup_enabled: bool = True
    backup_interval_hours: int = 6
    cleanup_days: int = 30

@dataclass
class RedisConfig:
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    max_connections: int = 50
    retry_on_timeout: bool = True
    health_check_interval: int = 30

@dataclass
class MonitoringConfig:
    """System monitoring configuration."""
    enabled: bool = True
    check_interval: int = 30
    metrics_retention_hours: int = 168  # 7 days
    alert_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'cpu_usage': {'warning': 70, 'critical': 90},
        'memory_usage': {'warning': 80, 'critical': 95},
        'disk_usage': {'warning': 85, 'critical': 95},
        'response_time': {'warning': 5.0, 'critical': 10.0},
        'error_rate': {'warning': 0.05, 'critical': 0.1}
    })
    email_alerts: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'use_tls': True,
        'from_address': '',
        'to_addresses': [],
        'username': '',
        'password': ''
    })
    webhook_alerts: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,
        'url': '',
        'headers': {},
        'timeout': 30
    })

@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str = "your-secret-key-change-in-production"
    api_key_required: bool = False
    rate_limiting: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'requests_per_minute': 60,
        'burst_limit': 10
    })
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    encryption_enabled: bool = False
    session_timeout_minutes: int = 60

@dataclass
class UIConfig:
    """UI configuration."""
    theme: str = "default"
    max_search_history: int = 50
    auto_refresh_interval: int = 30
    export_formats: List[str] = field(default_factory=lambda: ["csv", "json", "xlsx"])
    chart_animation: bool = True
    real_time_updates: bool = True
    max_concurrent_searches: int = 3

class AdvancedSettings(BaseSettings):
    """Advanced application settings with validation and environment support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )
    
    # Environment and basic settings
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Application environment")
    debug: bool = Field(default=True, description="Enable debug mode")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    app_name: str = Field(default="Reddit Mention Tracker", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    
    # Server settings
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=7860, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    
    # Component configurations
    scraping: ScrapingConfig = Field(default_factory=ScrapingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    
    # Feature flags
    features: Dict[str, bool] = Field(default_factory=lambda: {
        'advanced_sentiment': True,
        'real_time_monitoring': True,
        'data_validation': True,
        'caching': True,
        'api_endpoints': True,
        'export_functionality': True,
        'email_notifications': False,
        'webhook_notifications': False,
        'proxy_support': False,
        'advanced_analytics': True
    })
    
    # External service configurations
    external_services: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        'reddit_api': {
            'enabled': False,
            'client_id': '',
            'client_secret': '',
            'user_agent': 'RedditMentionTracker/2.0'
        },
        'sentiment_api': {
            'enabled': False,
            'provider': 'textblob',  # textblob, vader, transformers
            'api_key': '',
            'endpoint': ''
        },
        'proxy_service': {
            'enabled': False,
            'provider': '',
            'api_key': '',
            'rotation_interval': 300
        }
    })
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment setting."""
        if v not in Environment:
            raise ValueError(f"Invalid environment: {v}")
        return v
    
    @validator('port')
    def validate_port(cls, v):
        """Validate port number."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @validator('workers')
    def validate_workers(cls, v):
        """Validate worker count."""
        if v < 1:
            raise ValueError("Workers must be at least 1")
        return v
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT
    
    def get_database_url(self) -> str:
        """Get database URL with environment-specific overrides."""
        if self.is_production():
            return os.getenv('DATABASE_URL', self.database.url)
        return self.database.url
    
    def get_redis_url(self) -> str:
        """Get Redis URL."""
        if self.redis.password:
            return f"redis://:{self.redis.password}@{self.redis.host}:{self.redis.port}/{self.redis.db}"
        return f"redis://{self.redis.host}:{self.redis.port}/{self.redis.db}"
    
    def get_log_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                },
                'detailed': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'level': self.log_level.value,
                    'class': 'logging.StreamHandler',
                    'formatter': 'standard' if self.is_production() else 'detailed'
                },
                'file': {
                    'level': 'INFO',
                    'class': 'logging.FileHandler',
                    'filename': 'logs/app.log',
                    'formatter': 'detailed'
                }
            },
            'loggers': {
                '': {
                    'handlers': ['console', 'file'] if not self.is_production() else ['file'],
                    'level': self.log_level.value,
                    'propagate': False
                }
            }
        }
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), dict):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            'environment': self.environment.value,
            'debug': self.debug,
            'log_level': self.log_level.value,
            'app_name': self.app_name,
            'app_version': self.app_version,
            'host': self.host,
            'port': self.port,
            'workers': self.workers,
            'scraping': self.scraping.__dict__,
            'database': self.database.__dict__,
            'redis': self.redis.__dict__,
            'monitoring': self.monitoring.__dict__,
            'security': self.security.__dict__,
            'ui': self.ui.__dict__,
            'features': self.features,
            'external_services': self.external_services
        }
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        file_path = Path(file_path)
        config_dict = self.to_dict()
        
        if file_path.suffix.lower() == '.yaml' or file_path.suffix.lower() == '.yml':
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'AdvancedSettings':
        """Load configuration from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            with open(file_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
        
        # Create instance with base settings
        settings = cls()
        
        # Update with loaded configuration
        settings.update_from_dict(config_dict)
        
        return settings

class ConfigManager:
    """Configuration manager with hot reloading and validation."""
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        self.config_file = Path(config_file) if config_file else None
        self.settings: Optional[AdvancedSettings] = None
        self.logger = logging.getLogger(__name__)
        self._watchers = []
        
        # Load initial configuration
        self.reload()
    
    def reload(self) -> None:
        """Reload configuration from file or environment."""
        try:
            if self.config_file and self.config_file.exists():
                self.settings = AdvancedSettings.load_from_file(self.config_file)
                self.logger.info(f"Configuration loaded from {self.config_file}")
            else:
                self.settings = AdvancedSettings()
                self.logger.info("Configuration loaded from environment variables")
            
            # Apply environment-specific overrides
            self._apply_environment_overrides()
            
            # Validate configuration
            self._validate_configuration()
            
            # Notify watchers
            self._notify_watchers()
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            if self.settings is None:
                # Fallback to default settings
                self.settings = AdvancedSettings()
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment-specific configuration overrides."""
        if not self.settings:
            return
        
        env = self.settings.environment
        
        # Production overrides
        if env == Environment.PRODUCTION:
            self.settings.debug = False
            self.settings.log_level = LogLevel.WARNING
            self.settings.scraping.headless_mode = True
            self.settings.monitoring.enabled = True
            self.settings.security.api_key_required = True
            
        # Development overrides
        elif env == Environment.DEVELOPMENT:
            self.settings.debug = True
            self.settings.log_level = LogLevel.DEBUG
            self.settings.monitoring.check_interval = 60  # Less frequent in dev
            
        # Testing overrides
        elif env == Environment.TESTING:
            self.settings.database.url = "sqlite:///:memory:"
            self.settings.scraping.max_pages_per_search = 1
            self.settings.monitoring.enabled = False
    
    def _validate_configuration(self) -> None:
        """Validate configuration settings."""
        if not self.settings:
            return
        
        # Validate required settings for production
        if self.settings.is_production():
            if self.settings.security.secret_key == "your-secret-key-change-in-production":
                raise ValueError("Secret key must be changed in production")
            
            if self.settings.monitoring.email_alerts['enabled'] and not self.settings.monitoring.email_alerts['from_address']:
                raise ValueError("Email alerts enabled but no from_address configured")
        
        # Validate database URL
        if not self.settings.database.url:
            raise ValueError("Database URL is required")
        
        # Validate feature dependencies
        if self.settings.features['caching'] and not self.settings.redis.host:
            self.logger.warning("Caching enabled but Redis not configured")
    
    def _notify_watchers(self) -> None:
        """Notify configuration change watchers."""
        for watcher in self._watchers:
            try:
                watcher(self.settings)
            except Exception as e:
                self.logger.error(f"Configuration watcher failed: {str(e)}")
    
    def add_watcher(self, callback) -> None:
        """Add configuration change watcher."""
        self._watchers.append(callback)
    
    def remove_watcher(self, callback) -> None:
        """Remove configuration change watcher."""
        if callback in self._watchers:
            self._watchers.remove(callback)
    
    def get_settings(self) -> AdvancedSettings:
        """Get current settings."""
        if self.settings is None:
            self.reload()
        return self.settings
    
    def update_setting(self, key: str, value: Any) -> None:
        """Update a specific setting."""
        if not self.settings:
            return
        
        if hasattr(self.settings, key):
            setattr(self.settings, key, value)
            self._notify_watchers()
            self.logger.info(f"Setting updated: {key} = {value}")
        else:
            self.logger.warning(f"Unknown setting: {key}")
    
    def save_current_config(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to file."""
        if not self.settings:
            return
        
        target_file = Path(file_path) if file_path else self.config_file
        if not target_file:
            target_file = Path("config/current_settings.yaml")
        
        # Ensure directory exists
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.settings.save_to_file(target_file)
        self.logger.info(f"Configuration saved to {target_file}")

# Global configuration manager
config_manager = ConfigManager()

def get_settings() -> AdvancedSettings:
    """Get application settings."""
    return config_manager.get_settings()

def reload_config() -> None:
    """Reload configuration."""
    config_manager.reload()

def update_setting(key: str, value: Any) -> None:
    """Update a configuration setting."""
    config_manager.update_setting(key, value)

def add_config_watcher(callback) -> None:
    """Add configuration change watcher."""
    config_manager.add_watcher(callback) 