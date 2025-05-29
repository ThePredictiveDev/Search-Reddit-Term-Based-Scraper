"""
Comprehensive system monitoring with health checks, performance metrics, and alerting.
"""
import asyncio
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from collections import deque, defaultdict
import statistics

# Optional email imports
try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    smtplib = None
    MimeText = None
    MimeMultipart = None

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ComponentStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    """Health check configuration."""
    name: str
    check_function: Callable
    interval: int = 60  # seconds
    timeout: int = 30
    failure_threshold: int = 3
    recovery_threshold: int = 2
    enabled: bool = True
    
@dataclass
class Alert:
    """System alert."""
    id: str
    level: AlertLevel
    component: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    name: str
    value: float
    timestamp: datetime
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

class SystemMonitor:
    """Comprehensive system monitoring and alerting."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Health checks registry
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_status: Dict[str, ComponentStatus] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Performance metrics
        self.metrics_buffer: deque = deque(maxlen=10000)
        self.metric_aggregates: Dict[str, Dict] = defaultdict(dict)
        
        # Alerting
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_handlers: List[Callable] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None
        
        # Performance thresholds
        self.thresholds = {
            'cpu_usage': {'warning': 70, 'critical': 90},
            'memory_usage': {'warning': 80, 'critical': 95},
            'disk_usage': {'warning': 85, 'critical': 95},
            'response_time': {'warning': 5.0, 'critical': 10.0},
            'error_rate': {'warning': 0.05, 'critical': 0.1},
            'queue_size': {'warning': 1000, 'critical': 5000}
        }
        
        # Initialize default health checks
        self._register_default_health_checks()
        
        # Setup alert handlers
        self._setup_alert_handlers()
    
    def _register_default_health_checks(self):
        """Register default system health checks."""
        self.register_health_check(HealthCheck(
            name="system_resources",
            check_function=self._check_system_resources,
            interval=30
        ))
        
        self.register_health_check(HealthCheck(
            name="database_connection",
            check_function=self._check_database_connection,
            interval=60
        ))
        
        self.register_health_check(HealthCheck(
            name="redis_connection",
            check_function=self._check_redis_connection,
            interval=60
        ))
        
        self.register_health_check(HealthCheck(
            name="disk_space",
            check_function=self._check_disk_space,
            interval=300  # 5 minutes
        ))
        
        self.register_health_check(HealthCheck(
            name="network_connectivity",
            check_function=self._check_network_connectivity,
            interval=120
        ))
    
    def _setup_alert_handlers(self):
        """Setup alert notification handlers."""
        # Email alerts
        if self.config.get('email_alerts', {}).get('enabled', False):
            self.add_alert_handler(self._send_email_alert)
        
        # Webhook alerts
        if self.config.get('webhook_alerts', {}).get('enabled', False):
            self.add_alert_handler(self._send_webhook_alert)
        
        # Log alerts (always enabled)
        self.add_alert_handler(self._log_alert)
    
    def register_health_check(self, health_check: HealthCheck):
        """Register a new health check."""
        self.health_checks[health_check.name] = health_check
        self.health_status[health_check.name] = ComponentStatus.UNKNOWN
        self.logger.info(f"Registered health check: {health_check.name}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert notification handler."""
        self.alert_handlers.append(handler)
    
    async def start_monitoring(self):
        """Start the monitoring system."""
        if self.monitoring_active:
            self.logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("System monitoring started")
    
    async def stop_monitoring(self):
        """Stop the monitoring system."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("System monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        last_check_times = {}
        
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                # Run health checks based on their intervals
                for name, health_check in self.health_checks.items():
                    if not health_check.enabled:
                        continue
                    
                    last_check = last_check_times.get(name, 0)
                    if current_time - last_check >= health_check.interval:
                        asyncio.create_task(self._run_health_check(health_check))
                        last_check_times[name] = current_time
                
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Process metric aggregations
                self._process_metric_aggregations()
                
                # Check for anomalies
                await self._check_anomalies()
                
                await asyncio.sleep(10)  # Main loop interval
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(30)  # Longer sleep on error
    
    async def _run_health_check(self, health_check: HealthCheck):
        """Run a single health check."""
        try:
            start_time = time.time()
            
            # Run the check with timeout
            result = await asyncio.wait_for(
                self._execute_health_check(health_check),
                timeout=health_check.timeout
            )
            
            duration = time.time() - start_time
            
            # Update health status
            previous_status = self.health_status.get(health_check.name, ComponentStatus.UNKNOWN)
            new_status = ComponentStatus.HEALTHY if result else ComponentStatus.UNHEALTHY
            
            self.health_status[health_check.name] = new_status
            self.health_history[health_check.name].append({
                'timestamp': datetime.utcnow(),
                'status': new_status,
                'duration': duration,
                'result': result
            })
            
            # Generate alerts on status changes
            if previous_status != new_status and previous_status != ComponentStatus.UNKNOWN:
                await self._handle_health_status_change(health_check.name, previous_status, new_status)
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Health check {health_check.name} timed out")
            self.health_status[health_check.name] = ComponentStatus.UNHEALTHY
            await self._create_alert(
                AlertLevel.WARNING,
                health_check.name,
                f"Health check timed out after {health_check.timeout}s"
            )
        except Exception as e:
            self.logger.error(f"Health check {health_check.name} failed: {str(e)}")
            self.health_status[health_check.name] = ComponentStatus.UNHEALTHY
            await self._create_alert(
                AlertLevel.ERROR,
                health_check.name,
                f"Health check failed: {str(e)}"
            )
    
    async def _execute_health_check(self, health_check: HealthCheck) -> bool:
        """Execute a health check function."""
        if asyncio.iscoroutinefunction(health_check.check_function):
            return await health_check.check_function()
        else:
            return health_check.check_function()
    
    async def _handle_health_status_change(self, component: str, old_status: ComponentStatus, new_status: ComponentStatus):
        """Handle health status changes."""
        if new_status == ComponentStatus.UNHEALTHY:
            await self._create_alert(
                AlertLevel.ERROR,
                component,
                f"Component became unhealthy (was {old_status.value})"
            )
        elif new_status == ComponentStatus.HEALTHY and old_status == ComponentStatus.UNHEALTHY:
            await self._create_alert(
                AlertLevel.INFO,
                component,
                "Component recovered to healthy state"
            )
            # Resolve related alerts
            await self._resolve_component_alerts(component)
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics."""
        timestamp = datetime.utcnow()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self._add_metric("cpu_usage", cpu_percent, timestamp, "percent")
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self._add_metric("memory_usage", memory.percent, timestamp, "percent")
        self._add_metric("memory_available", memory.available / (1024**3), timestamp, "GB")
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self._add_metric("disk_usage", disk_percent, timestamp, "percent")
        self._add_metric("disk_free", disk.free / (1024**3), timestamp, "GB")
        
        # Network metrics
        network = psutil.net_io_counters()
        self._add_metric("network_bytes_sent", network.bytes_sent, timestamp, "bytes")
        self._add_metric("network_bytes_recv", network.bytes_recv, timestamp, "bytes")
        
        # Process metrics
        process_count = len(psutil.pids())
        self._add_metric("process_count", process_count, timestamp, "count")
        
        # Check thresholds
        await self._check_metric_thresholds(timestamp)
    
    def _add_metric(self, name: str, value: float, timestamp: datetime, unit: str = "", tags: Dict[str, str] = None):
        """Add a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=timestamp,
            unit=unit,
            tags=tags or {}
        )
        self.metrics_buffer.append(metric)
    
    async def _check_metric_thresholds(self, timestamp: datetime):
        """Check metrics against defined thresholds."""
        recent_metrics = [m for m in self.metrics_buffer if (timestamp - m.timestamp).total_seconds() < 300]
        
        for metric_name, thresholds in self.thresholds.items():
            matching_metrics = [m for m in recent_metrics if m.name == metric_name]
            if not matching_metrics:
                continue
            
            latest_value = matching_metrics[-1].value
            avg_value = statistics.mean([m.value for m in matching_metrics[-5:]])  # Last 5 values
            
            # Check critical threshold
            if latest_value >= thresholds.get('critical', float('inf')):
                await self._create_alert(
                    AlertLevel.CRITICAL,
                    "system_metrics",
                    f"{metric_name} is critical: {latest_value:.2f} (threshold: {thresholds['critical']})"
                )
            # Check warning threshold
            elif latest_value >= thresholds.get('warning', float('inf')):
                await self._create_alert(
                    AlertLevel.WARNING,
                    "system_metrics",
                    f"{metric_name} is high: {latest_value:.2f} (threshold: {thresholds['warning']})"
                )
    
    def _process_metric_aggregations(self):
        """Process metric aggregations for analysis."""
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        recent_metrics = [m for m in self.metrics_buffer if m.timestamp >= cutoff_time]
        
        # Group by metric name
        grouped_metrics = defaultdict(list)
        for metric in recent_metrics:
            grouped_metrics[metric.name].append(metric.value)
        
        # Calculate aggregates
        for metric_name, values in grouped_metrics.items():
            if values:
                self.metric_aggregates[metric_name] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'avg': statistics.mean(values),
                    'median': statistics.median(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0
                }
    
    async def _check_anomalies(self):
        """Check for performance anomalies."""
        # Simple anomaly detection based on standard deviation
        for metric_name, aggregates in self.metric_aggregates.items():
            if aggregates['count'] < 10:  # Need enough data points
                continue
            
            recent_metrics = [m for m in self.metrics_buffer 
                            if m.name == metric_name and 
                            (datetime.utcnow() - m.timestamp).total_seconds() < 300]
            
            if not recent_metrics:
                continue
            
            latest_value = recent_metrics[-1].value
            avg = aggregates['avg']
            std = aggregates['std']
            
            # Check if latest value is more than 3 standard deviations from mean
            if std > 0 and abs(latest_value - avg) > 3 * std:
                await self._create_alert(
                    AlertLevel.WARNING,
                    "anomaly_detection",
                    f"Anomaly detected in {metric_name}: {latest_value:.2f} "
                    f"(avg: {avg:.2f}, std: {std:.2f})"
                )
    
    async def _create_alert(self, level: AlertLevel, component: str, message: str, metadata: Dict[str, Any] = None):
        """Create and process a new alert."""
        alert_id = f"{component}_{level.value}_{int(time.time())}"
        
        # Check if similar alert already exists
        existing_alert = self._find_similar_alert(component, message)
        if existing_alert and not existing_alert.resolved:
            return  # Don't create duplicate alerts
        
        alert = Alert(
            id=alert_id,
            level=level,
            component=component,
            message=message,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                await self._safe_call_handler(handler, alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {str(e)}")
    
    def _find_similar_alert(self, component: str, message: str) -> Optional[Alert]:
        """Find similar existing alert."""
        for alert in self.active_alerts.values():
            if alert.component == component and alert.message == message:
                return alert
        return None
    
    async def _safe_call_handler(self, handler: Callable, alert: Alert):
        """Safely call an alert handler."""
        if asyncio.iscoroutinefunction(handler):
            await handler(alert)
        else:
            handler(alert)
    
    async def _resolve_component_alerts(self, component: str):
        """Resolve all active alerts for a component."""
        for alert in list(self.active_alerts.values()):
            if alert.component == component and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                del self.active_alerts[alert.id]
    
    # Health check implementations
    async def _check_system_resources(self) -> bool:
        """Check system resource availability."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            return cpu_percent < 95 and memory_percent < 95
        except Exception:
            return False
    
    async def _check_database_connection(self) -> bool:
        """Check database connectivity."""
        try:
            from database.models import DatabaseManager
            db_manager = DatabaseManager()
            with db_manager.get_session() as session:
                session.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    async def _check_redis_connection(self) -> bool:
        """Check Redis connectivity."""
        try:
            from database.cache_manager import CacheManager
            cache_manager = CacheManager()
            return cache_manager.is_available()
        except Exception:
            return False
    
    async def _check_disk_space(self) -> bool:
        """Check available disk space."""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            return usage_percent < 90
        except Exception:
            return False
    
    async def _check_network_connectivity(self) -> bool:
        """Check network connectivity."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get('https://www.google.com', timeout=10) as response:
                    return response.status == 200
        except Exception:
            return False
    
    # Alert handlers
    async def _log_alert(self, alert: Alert):
        """Log alert to application logs."""
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }.get(alert.level, logging.INFO)
        
        self.logger.log(log_level, f"ALERT [{alert.level.value.upper()}] {alert.component}: {alert.message}")
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert notification."""
        if not EMAIL_AVAILABLE:
            self.logger.warning("Email functionality not available, skipping email alert")
            return
            
        try:
            email_config = self.config.get('email_alerts', {})
            if not email_config.get('enabled', False):
                return
            
            # Create email message
            msg = MimeMultipart()
            msg['From'] = email_config['from_address']
            msg['To'] = ', '.join(email_config['to_addresses'])
            msg['Subject'] = f"[{alert.level.value.upper()}] System Alert: {alert.component}"
            
            body = f"""
            Alert Details:
            - Level: {alert.level.value.upper()}
            - Component: {alert.component}
            - Message: {alert.message}
            - Timestamp: {alert.timestamp}
            - Alert ID: {alert.id}
            
            System: Reddit Mention Tracker
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            if email_config.get('use_tls', True):
                server.starttls()
            if email_config.get('username') and email_config.get('password'):
                server.login(email_config['username'], email_config['password'])
            
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {str(e)}")
    
    async def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert notification."""
        try:
            webhook_config = self.config.get('webhook_alerts', {})
            if not webhook_config.get('enabled', False):
                return
            
            import aiohttp
            
            payload = {
                'alert_id': alert.id,
                'level': alert.level.value,
                'component': alert.component,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'metadata': alert.metadata
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_config['url'],
                    json=payload,
                    headers=webhook_config.get('headers', {}),
                    timeout=30
                ) as response:
                    if response.status != 200:
                        self.logger.warning(f"Webhook alert failed with status {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {str(e)}")
    
    # Public API methods
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        overall_status = ComponentStatus.HEALTHY
        
        for status in self.health_status.values():
            if status == ComponentStatus.UNHEALTHY:
                overall_status = ComponentStatus.UNHEALTHY
                break
            elif status == ComponentStatus.DEGRADED and overall_status == ComponentStatus.HEALTHY:
                overall_status = ComponentStatus.DEGRADED
        
        return {
            'overall_status': overall_status.value,
            'components': {name: status.value for name, status in self.health_status.items()},
            'active_alerts': len(self.active_alerts),
            'last_check': datetime.utcnow().isoformat()
        }
    
    def get_performance_metrics(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance metrics for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_buffer if m.timestamp >= cutoff_time]
        
        # Group by metric name
        grouped_metrics = defaultdict(list)
        for metric in recent_metrics:
            grouped_metrics[metric.name].append({
                'value': metric.value,
                'timestamp': metric.timestamp.isoformat(),
                'unit': metric.unit
            })
        
        return dict(grouped_metrics)
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return [
            {
                'id': alert.id,
                'level': alert.level.value,
                'component': alert.component,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'metadata': alert.metadata
            }
            for alert in self.active_alerts.values()
        ]

# Global monitor instance
system_monitor = SystemMonitor()

def get_system_monitor() -> SystemMonitor:
    """Get the global system monitor instance."""
    return system_monitor 