"""
Centralized configuration management for Reddit Mention Tracker.
Provides environment-based configuration with validation and defaults.
"""
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str = "sqlite:///reddit_mentions.db"
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600

@dataclass
class RedisConfig:
    """Redis cache configuration settings."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    max_connections: int = 50
    
@dataclass
class ScrapingConfig:
    """Web scraping configuration settings."""
    max_pages_default: int = 5
    max_pages_limit: int = 50
    rate_limit: float = 2.0  # requests per second
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    user_agent_rotation: bool = True
    headless: bool = True
    viewport_width: int = 1920
    viewport_height: int = 1080
    
@dataclass
class SentimentConfig:
    """Sentiment analysis configuration."""
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    batch_size: int = 32
    max_length: int = 512
    use_gpu: bool = False
    fallback_to_textblob: bool = True
    confidence_threshold: float = 0.7

@dataclass
class MonitoringConfig:
    """Real-time monitoring configuration."""
    websocket_host: str = "localhost"
    websocket_port: int = 8765
    max_connections: int = 100
    heartbeat_interval: int = 30
    message_queue_size: int = 1000

@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    cors_origins: list = field(default_factory=lambda: ["*"])
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

@dataclass
class UIConfig:
    """User interface configuration."""
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False
    debug: bool = False
    auth: Optional[tuple] = None
    theme: str = "default"
    title: str = "Reddit Mention Tracker"

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
class Settings:
    """Main application settings with environment variable support."""
    
    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Load configuration
        self.database = self._load_database_config()
        self.redis = self._load_redis_config()
        self.scraping = self._load_scraping_config()
        self.sentiment = self._load_sentiment_config()
        self.monitoring = self._load_monitoring_config()
        self.api = self._load_api_config()
        self.ui = self._load_ui_config()
        self.logging = self._load_logging_config()
        
        # Feature flags
        self.features = self._load_feature_flags()
        
        # Validate configuration
        self._validate_config()
        
    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration from environment."""
        return DatabaseConfig(
            url=os.getenv("DATABASE_URL", "sqlite:///reddit_mentions.db"),
            echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
            pool_size=int(os.getenv("DATABASE_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "20")),
            pool_timeout=int(os.getenv("DATABASE_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("DATABASE_POOL_RECYCLE", "3600"))
        )
    
    def _load_redis_config(self) -> RedisConfig:
        """Load Redis configuration from environment."""
        return RedisConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            socket_timeout=int(os.getenv("REDIS_SOCKET_TIMEOUT", "5")),
            socket_connect_timeout=int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5")),
            retry_on_timeout=os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true",
            max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
        )
    
    def _load_scraping_config(self) -> ScrapingConfig:
        """Load scraping configuration from environment."""
        return ScrapingConfig(
            max_pages_default=int(os.getenv("SCRAPING_MAX_PAGES_DEFAULT", "5")),
            max_pages_limit=int(os.getenv("SCRAPING_MAX_PAGES_LIMIT", "50")),
            rate_limit=float(os.getenv("SCRAPING_RATE_LIMIT", "2.0")),
            request_timeout=int(os.getenv("SCRAPING_REQUEST_TIMEOUT", "30")),
            max_retries=int(os.getenv("SCRAPING_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("SCRAPING_RETRY_DELAY", "1.0")),
            user_agent_rotation=os.getenv("SCRAPING_USER_AGENT_ROTATION", "true").lower() == "true",
            headless=os.getenv("SCRAPING_HEADLESS", "true").lower() == "true",
            viewport_width=int(os.getenv("SCRAPING_VIEWPORT_WIDTH", "1920")),
            viewport_height=int(os.getenv("SCRAPING_VIEWPORT_HEIGHT", "1080"))
        )
    
    def _load_sentiment_config(self) -> SentimentConfig:
        """Load sentiment analysis configuration from environment."""
        return SentimentConfig(
            model_name=os.getenv("SENTIMENT_MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment-latest"),
            batch_size=int(os.getenv("SENTIMENT_BATCH_SIZE", "32")),
            max_length=int(os.getenv("SENTIMENT_MAX_LENGTH", "512")),
            use_gpu=os.getenv("SENTIMENT_USE_GPU", "false").lower() == "true",
            fallback_to_textblob=os.getenv("SENTIMENT_FALLBACK_TO_TEXTBLOB", "true").lower() == "true",
            confidence_threshold=float(os.getenv("SENTIMENT_CONFIDENCE_THRESHOLD", "0.7"))
        )
    
    def _load_monitoring_config(self) -> MonitoringConfig:
        """Load monitoring configuration from environment."""
        return MonitoringConfig(
            websocket_host=os.getenv("MONITORING_WEBSOCKET_HOST", "localhost"),
            websocket_port=int(os.getenv("MONITORING_WEBSOCKET_PORT", "8765")),
            max_connections=int(os.getenv("MONITORING_MAX_CONNECTIONS", "100")),
            heartbeat_interval=int(os.getenv("MONITORING_HEARTBEAT_INTERVAL", "30")),
            message_queue_size=int(os.getenv("MONITORING_MESSAGE_QUEUE_SIZE", "1000"))
        )
    
    def _load_api_config(self) -> APIConfig:
        """Load API configuration from environment."""
        cors_origins = os.getenv("API_CORS_ORIGINS", "*").split(",")
        return APIConfig(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            workers=int(os.getenv("API_WORKERS", "1")),
            reload=os.getenv("API_RELOAD", "false").lower() == "true",
            log_level=os.getenv("API_LOG_LEVEL", "info"),
            cors_origins=cors_origins,
            rate_limit_requests=int(os.getenv("API_RATE_LIMIT_REQUESTS", "100")),
            rate_limit_window=int(os.getenv("API_RATE_LIMIT_WINDOW", "60"))
        )
    
    def _load_ui_config(self) -> UIConfig:
        """Load UI configuration from environment."""
        auth = None
        auth_user = os.getenv("UI_AUTH_USER")
        auth_pass = os.getenv("UI_AUTH_PASSWORD")
        if auth_user and auth_pass:
            auth = (auth_user, auth_pass)
            
        return UIConfig(
            host=os.getenv("UI_HOST", "0.0.0.0"),
            port=int(os.getenv("UI_PORT", "7860")),
            share=os.getenv("UI_SHARE", "false").lower() == "true",
            debug=os.getenv("UI_DEBUG", "false").lower() == "true",
            auth=auth,
            theme=os.getenv("UI_THEME", "default"),
            title=os.getenv("UI_TITLE", "Reddit Mention Tracker")
        )
    
    def _load_logging_config(self) -> LoggingConfig:
        """Load logging configuration from environment."""
        return LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file_path=os.getenv("LOG_FILE_PATH"),
            max_file_size=int(os.getenv("LOG_MAX_FILE_SIZE", str(10 * 1024 * 1024))),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5"))
        )
    
    def _load_feature_flags(self) -> Dict[str, bool]:
        """Load feature flags from environment."""
        return {
            "cache_enabled": os.getenv("FEATURE_CACHE_ENABLED", "true").lower() == "true",
            "advanced_sentiment": os.getenv("FEATURE_ADVANCED_SENTIMENT", "true").lower() == "true",
            "realtime_monitoring": os.getenv("FEATURE_REALTIME_MONITORING", "true").lower() == "true",
            "api_enabled": os.getenv("FEATURE_API_ENABLED", "true").lower() == "true",
            "export_enabled": os.getenv("FEATURE_EXPORT_ENABLED", "true").lower() == "true",
            "metrics_collection": os.getenv("FEATURE_METRICS_COLLECTION", "true").lower() == "true"
        }
    
    def _validate_config(self):
        """Validate configuration settings."""
        errors = []
        
        # Validate database URL
        if not self.database.url:
            errors.append("Database URL cannot be empty")
        
        # Validate rate limits
        if self.scraping.rate_limit <= 0:
            errors.append("Scraping rate limit must be positive")
        
        if self.scraping.max_pages_limit < self.scraping.max_pages_default:
            errors.append("Max pages limit cannot be less than default")
        
        # Validate ports
        if not (1 <= self.api.port <= 65535):
            errors.append("API port must be between 1 and 65535")
        
        if not (1 <= self.ui.port <= 65535):
            errors.append("UI port must be between 1 and 65535")
        
        if not (1 <= self.monitoring.websocket_port <= 65535):
            errors.append("WebSocket port must be between 1 and 65535")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for serialization."""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "database": self.database.__dict__,
            "redis": self.redis.__dict__,
            "scraping": self.scraping.__dict__,
            "sentiment": self.sentiment.__dict__,
            "monitoring": self.monitoring.__dict__,
            "api": self.api.__dict__,
            "ui": self.ui.__dict__,
            "logging": self.logging.__dict__,
            "features": self.features
        }
    
    def save_to_file(self, file_path: str):
        """Save configuration to JSON file."""
        config_dict = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'Settings':
        """Load configuration from JSON file."""
        if not os.path.exists(file_path):
            return cls()
        
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        # Set environment variables from file
        for section, values in config_dict.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    env_key = f"{section.upper()}_{key.upper()}"
                    if env_key not in os.environ:
                        os.environ[env_key] = str(value)
        
        return cls()

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings

def configure_logging():
    """Configure application logging based on settings."""
    log_config = settings.logging
    
    # Set up logging format
    formatter = logging.Formatter(log_config.format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_config.level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_config.file_path:
        from logging.handlers import RotatingFileHandler
        
        # Ensure log directory exists
        log_dir = Path(log_config.file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_config.file_path,
            maxBytes=log_config.max_file_size,
            backupCount=log_config.backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("playwright").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING) 