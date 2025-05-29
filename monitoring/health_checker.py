"""
Comprehensive health monitoring and alerting system for Reddit Mention Tracker.
Provides real-time health checks, performance monitoring, and alerting capabilities.
"""
import asyncio
import logging
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthMetric:
    """Individual health metric data."""
    name: str
    value: Any
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    unit: str = ""
    description: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class HealthCheck:
    """Health check configuration."""
    name: str
    check_function: Callable
    interval: int = 60  # seconds
    timeout: int = 30   # seconds
    enabled: bool = True
    critical: bool = False
    description: str = ""

@dataclass
class Alert:
    """Alert configuration and data."""
    id: str
    metric_name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False

class HealthChecker:
    """Comprehensive health monitoring system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics: Dict[str, HealthMetric] = {}
        self.checks: Dict[str, HealthCheck] = {}
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable] = []
        self.running = False
        self.check_threads: Dict[str, threading.Thread] = {}
        
        # Performance tracking
        self.performance_history: Dict[str, List[float]] = {}
        self.max_history_size = 1000
        
        # Initialize default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default system health checks."""
        
        # System resource checks
        self.register_check(HealthCheck(
            name="cpu_usage",
            check_function=self._check_cpu_usage,
            interval=30,
            critical=True,
            description="Monitor CPU usage percentage"
        ))
        
        self.register_check(HealthCheck(
            name="memory_usage",
            check_function=self._check_memory_usage,
            interval=30,
            critical=True,
            description="Monitor memory usage percentage"
        ))
        
        self.register_check(HealthCheck(
            name="disk_usage",
            check_function=self._check_disk_usage,
            interval=60,
            critical=True,
            description="Monitor disk usage percentage"
        ))
        
        # Application-specific checks
        self.register_check(HealthCheck(
            name="database_connection",
            check_function=self._check_database_connection,
            interval=60,
            critical=True,
            description="Check database connectivity"
        ))
        
        self.register_check(HealthCheck(
            name="redis_connection",
            check_function=self._check_redis_connection,
            interval=60,
            critical=False,
            description="Check Redis cache connectivity"
        ))
        
        self.register_check(HealthCheck(
            name="scraper_health",
            check_function=self._check_scraper_health,
            interval=120,
            critical=False,
            description="Check web scraper functionality"
        ))
    
    def register_check(self, check: HealthCheck):
        """Register a new health check."""
        self.checks[check.name] = check
        self.logger.info(f"Registered health check: {check.name}")
    
    def unregister_check(self, name: str):
        """Unregister a health check."""
        if name in self.checks:
            del self.checks[name]
            if name in self.check_threads:
                # Stop the thread if running
                self.check_threads[name].join(timeout=1)
                del self.check_threads[name]
            self.logger.info(f"Unregistered health check: {name}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start health monitoring."""
        if self.running:
            self.logger.warning("Health monitoring already running")
            return
        
        self.running = True
        self.logger.info("Starting health monitoring")
        
        # Start check threads
        for name, check in self.checks.items():
            if check.enabled:
                thread = threading.Thread(
                    target=self._run_check_loop,
                    args=(name, check),
                    daemon=True
                )
                thread.start()
                self.check_threads[name] = thread
        
        self.logger.info(f"Started {len(self.check_threads)} health check threads")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping health monitoring")
        
        # Wait for threads to finish
        for thread in self.check_threads.values():
            thread.join(timeout=1)
        
        self.check_threads.clear()
        self.logger.info("Health monitoring stopped")
    
    def _run_check_loop(self, name: str, check: HealthCheck):
        """Run a health check in a loop."""
        while self.running:
            try:
                start_time = time.time()
                
                # Run the check with timeout
                result = asyncio.run(
                    asyncio.wait_for(
                        self._run_check_async(check),
                        timeout=check.timeout
                    )
                )
                
                # Record performance
                duration = time.time() - start_time
                self._record_performance(f"{name}_duration", duration)
                
                # Update metric
                if result:
                    self.metrics[name] = result
                    
                    # Check for alerts
                    self._check_alerts(result)
                
            except asyncio.TimeoutError:
                self.logger.error(f"Health check {name} timed out")
                self._create_alert(
                    name,
                    HealthStatus.CRITICAL,
                    f"Health check {name} timed out after {check.timeout}s"
                )
            except Exception as e:
                self.logger.error(f"Health check {name} failed: {str(e)}")
                self._create_alert(
                    name,
                    HealthStatus.CRITICAL,
                    f"Health check {name} failed: {str(e)}"
                )
            
            # Wait for next check
            time.sleep(check.interval)
    
    async def _run_check_async(self, check: HealthCheck) -> Optional[HealthMetric]:
        """Run a health check asynchronously."""
        try:
            if asyncio.iscoroutinefunction(check.check_function):
                return await check.check_function()
            else:
                return check.check_function()
        except Exception as e:
            self.logger.error(f"Error in health check {check.name}: {str(e)}")
            return None
    
    def _check_cpu_usage(self) -> HealthMetric:
        """Check CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if cpu_percent > 90:
            status = HealthStatus.CRITICAL
        elif cpu_percent > 70:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
        
        return HealthMetric(
            name="cpu_usage",
            value=cpu_percent,
            status=status,
            threshold_warning=70.0,
            threshold_critical=90.0,
            unit="%",
            description="Current CPU usage percentage"
        )
    
    def _check_memory_usage(self) -> HealthMetric:
        """Check memory usage."""
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        if memory_percent > 90:
            status = HealthStatus.CRITICAL
        elif memory_percent > 80:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
        
        return HealthMetric(
            name="memory_usage",
            value=memory_percent,
            status=status,
            threshold_warning=80.0,
            threshold_critical=90.0,
            unit="%",
            description="Current memory usage percentage"
        )
    
    def _check_disk_usage(self) -> HealthMetric:
        """Check disk usage."""
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        if disk_percent > 95:
            status = HealthStatus.CRITICAL
        elif disk_percent > 85:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
        
        return HealthMetric(
            name="disk_usage",
            value=disk_percent,
            status=status,
            threshold_warning=85.0,
            threshold_critical=95.0,
            unit="%",
            description="Current disk usage percentage"
        )
    
    def _check_database_connection(self) -> HealthMetric:
        """Check database connectivity."""
        try:
            from database.models import DatabaseManager
            
            db_manager = DatabaseManager()
            # Simple query to test connection
            with db_manager.get_session() as session:
                session.execute("SELECT 1")
            
            return HealthMetric(
                name="database_connection",
                value="connected",
                status=HealthStatus.HEALTHY,
                description="Database connection is healthy"
            )
        except Exception as e:
            return HealthMetric(
                name="database_connection",
                value="disconnected",
                status=HealthStatus.CRITICAL,
                description=f"Database connection failed: {str(e)}"
            )
    
    def _check_redis_connection(self) -> HealthMetric:
        """Check Redis connectivity."""
        try:
            from database.cache_manager import CacheManager
            
            cache_manager = CacheManager()
            if cache_manager.is_available():
                # Test Redis connection
                cache_manager.redis_client.ping()
                return HealthMetric(
                    name="redis_connection",
                    value="connected",
                    status=HealthStatus.HEALTHY,
                    description="Redis connection is healthy"
                )
            else:
                return HealthMetric(
                    name="redis_connection",
                    value="unavailable",
                    status=HealthStatus.WARNING,
                    description="Redis is not configured or unavailable"
                )
        except Exception as e:
            return HealthMetric(
                name="redis_connection",
                value="disconnected",
                status=HealthStatus.WARNING,
                description=f"Redis connection failed: {str(e)}"
            )
    
    def _check_scraper_health(self) -> HealthMetric:
        """Check web scraper health."""
        try:
            from scraper.reddit_scraper import RedditScraper
            from database.models import DatabaseManager
            
            db_manager = DatabaseManager()
            scraper = RedditScraper(db_manager)
            
            # Check if browser can be initialized
            # This is a lightweight check without actually scraping
            return HealthMetric(
                name="scraper_health",
                value="healthy",
                status=HealthStatus.HEALTHY,
                description="Web scraper is ready"
            )
        except Exception as e:
            return HealthMetric(
                name="scraper_health",
                value="unhealthy",
                status=HealthStatus.WARNING,
                description=f"Web scraper issue: {str(e)}"
            )
    
    def _record_performance(self, metric_name: str, value: float):
        """Record performance metric."""
        if metric_name not in self.performance_history:
            self.performance_history[metric_name] = []
        
        history = self.performance_history[metric_name]
        history.append(value)
        
        # Limit history size
        if len(history) > self.max_history_size:
            history.pop(0)
    
    def _check_alerts(self, metric: HealthMetric):
        """Check if metric should trigger an alert."""
        if metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
            # Check if we already have an active alert for this metric
            active_alerts = [
                alert for alert in self.alerts
                if alert.metric_name == metric.name and not alert.resolved
            ]
            
            if not active_alerts:
                self._create_alert(
                    metric.name,
                    metric.status,
                    f"{metric.name}: {metric.value}{metric.unit} - {metric.description}"
                )
        else:
            # Resolve any existing alerts for this metric
            for alert in self.alerts:
                if alert.metric_name == metric.name and not alert.resolved:
                    alert.resolved = True
                    self.logger.info(f"Resolved alert: {alert.id}")
    
    def _create_alert(self, metric_name: str, status: HealthStatus, message: str):
        """Create a new alert."""
        alert_id = f"{metric_name}_{int(time.time())}"
        alert = Alert(
            id=alert_id,
            metric_name=metric_name,
            status=status,
            message=message,
            timestamp=datetime.utcnow()
        )
        
        self.alerts.append(alert)
        self.logger.warning(f"Created alert: {alert_id} - {message}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {str(e)}")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        if not self.metrics:
            return {
                "overall_status": HealthStatus.UNKNOWN.value,
                "message": "No health data available",
                "metrics": {},
                "alerts": []
            }
        
        # Determine overall status
        statuses = [metric.status for metric in self.metrics.values()]
        
        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Count metrics by status
        status_counts = {}
        for status in HealthStatus:
            status_counts[status.value] = sum(
                1 for metric in self.metrics.values()
                if metric.status == status
            )
        
        # Get active alerts
        active_alerts = [
            {
                "id": alert.id,
                "metric": alert.metric_name,
                "status": alert.status.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "acknowledged": alert.acknowledged
            }
            for alert in self.alerts
            if not alert.resolved
        ]
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics_count": len(self.metrics),
            "status_counts": status_counts,
            "active_alerts": len(active_alerts),
            "metrics": {
                name: {
                    "value": metric.value,
                    "status": metric.status.value,
                    "unit": metric.unit,
                    "description": metric.description,
                    "timestamp": metric.timestamp.isoformat()
                }
                for name, metric in self.metrics.items()
            },
            "alerts": active_alerts
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        
        for metric_name, history in self.performance_history.items():
            if history:
                stats[metric_name] = {
                    "count": len(history),
                    "min": min(history),
                    "max": max(history),
                    "avg": sum(history) / len(history),
                    "recent": history[-10:] if len(history) >= 10 else history
                }
        
        return stats
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                self.logger.info(f"Acknowledged alert: {alert_id}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                self.logger.info(f"Resolved alert: {alert_id}")
                return True
        return False
    
    def cleanup_old_alerts(self, days: int = 7):
        """Clean up old resolved alerts."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        initial_count = len(self.alerts)
        self.alerts = [
            alert for alert in self.alerts
            if not alert.resolved or alert.timestamp > cutoff_date
        ]
        
        cleaned_count = initial_count - len(self.alerts)
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old alerts")

# Global health checker instance
health_checker = HealthChecker()

def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    return health_checker

def start_health_monitoring():
    """Start health monitoring."""
    health_checker.start_monitoring()

def stop_health_monitoring():
    """Stop health monitoring."""
    health_checker.stop_monitoring()

# Alert notification functions
def log_alert_callback(alert: Alert):
    """Log alert to console/file."""
    logger = logging.getLogger("health_alerts")
    level = logging.CRITICAL if alert.status == HealthStatus.CRITICAL else logging.WARNING
    logger.log(level, f"ALERT [{alert.status.value.upper()}] {alert.message}")

def email_alert_callback(alert: Alert):
    """Send alert via email (placeholder implementation)."""
    # This would integrate with an email service
    # For now, just log the intent
    logger = logging.getLogger("health_alerts")
    logger.info(f"Would send email alert: {alert.message}")

# Register default alert callbacks
health_checker.add_alert_callback(log_alert_callback) 