"""
Comprehensive metrics and analytics system for AngelaMCP.
Tracks performance, costs, usage patterns, and system health.
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
from contextlib import asynccontextmanager

import psutil
from config.settings import settings


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class AgentMetrics:
    """Metrics for an individual agent."""
    agent_name: str
    requests_total: int = 0
    requests_successful: int = 0
    requests_failed: int = 0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.requests_total == 0:
            return 0.0
        return (self.requests_successful / self.requests_total) * 100


@dataclass
class SystemMetrics:
    """System-wide performance metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    active_connections: int = 0
    uptime_seconds: int = 0
    
    def update_system_stats(self):
        """Update system statistics."""
        self.cpu_usage = psutil.cpu_percent()
        self.memory_usage = psutil.virtual_memory().percent
        self.disk_usage = psutil.disk_usage('/').percent


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, metrics_collector: 'MetricsCollector', operation_name: str, tags: Dict[str, str] = None):
        self.metrics = metrics_collector
        self.operation_name = operation_name
        self.tags = tags or {}
        self.start_time = None
        
    async def __aenter__(self):
        self.start_time = time.time()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        await self.metrics.record_timing(self.operation_name, duration, self.tags)
        
        if exc_type is not None:
            await self.metrics.record_error(self.operation_name, str(exc_val), self.tags)


class CostTracker:
    """Tracks API costs for different providers."""
    
    def __init__(self):
        self.daily_costs = defaultdict(float)
        self.monthly_costs = defaultdict(float)
        self.total_costs = defaultdict(float)
        self.last_reset_day = datetime.now().day
        self.last_reset_month = datetime.now().month
        
    def add_cost(self, provider: str, input_tokens: int, output_tokens: int) -> float:
        """Add cost for API usage."""
        cost = 0.0
        
        if provider == "openai":
            cost = (input_tokens * settings.openai_input_cost / 1000) + \
                   (output_tokens * settings.openai_output_cost / 1000)
        elif provider == "gemini":
            cost = (input_tokens * settings.gemini_input_cost / 1000) + \
                   (output_tokens * settings.gemini_output_cost / 1000)
                   
        # Reset daily/monthly counters if needed
        now = datetime.now()
        if now.day != self.last_reset_day:
            self.daily_costs.clear()
            self.last_reset_day = now.day
            
        if now.month != self.last_reset_month:
            self.monthly_costs.clear()
            self.last_reset_month = now.month
            
        self.daily_costs[provider] += cost
        self.monthly_costs[provider] += cost
        self.total_costs[provider] += cost
        
        return cost
    
    def get_daily_total(self) -> float:
        """Get total daily cost across all providers."""
        return sum(self.daily_costs.values())
    
    def get_monthly_total(self) -> float:
        """Get total monthly cost across all providers."""
        return sum(self.monthly_costs.values())
        
    def is_over_budget(self) -> Dict[str, bool]:
        """Check if over daily/monthly budget limits."""
        return {
            "daily": self.get_daily_total() > settings.daily_budget_limit,
            "monthly": self.get_monthly_total() > settings.monthly_budget_limit
        }
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get comprehensive budget status."""
        daily_total = self.get_daily_total()
        monthly_total = self.get_monthly_total()
        
        return {
            "daily": {
                "spent": daily_total,
                "limit": settings.daily_budget_limit,
                "remaining": max(0, settings.daily_budget_limit - daily_total),
                "percentage_used": (daily_total / settings.daily_budget_limit) * 100 if settings.daily_budget_limit > 0 else 0
            },
            "monthly": {
                "spent": monthly_total,
                "limit": settings.monthly_budget_limit,
                "remaining": max(0, settings.monthly_budget_limit - monthly_total),
                "percentage_used": (monthly_total / settings.monthly_budget_limit) * 100 if settings.monthly_budget_limit > 0 else 0
            }
        }


class MetricsCollector:
    """Main metrics collection and aggregation system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        
        # Agent metrics
        self.agent_metrics: Dict[str, AgentMetrics] = {
            "claude_code": AgentMetrics("claude_code"),
            "openai": AgentMetrics("openai"),
            "gemini": AgentMetrics("gemini")
        }
        
        # System metrics
        self.system_metrics = SystemMetrics()
        
        # Cost tracking
        self.cost_tracker = CostTracker()
        
        # Performance metrics
        self.response_times = defaultdict(lambda: deque(maxlen=100))
        self.error_counts = defaultdict(int)
        self.request_counts = defaultdict(int)
        
        # Time series data (keep last 1000 points)
        self.time_series: deque = deque(maxlen=1000)
        
        # Background metrics collection
        self._metrics_task = None
        self._running = False
        
    async def start_collection(self):
        """Start background metrics collection."""
        if self._running:
            return
            
        self._running = True
        self._metrics_task = asyncio.create_task(self._collect_system_metrics())
        self.logger.info("Started metrics collection")
    
    async def stop_collection(self):
        """Stop background metrics collection."""
        self._running = False
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped metrics collection")
    
    async def _collect_system_metrics(self):
        """Background task to collect system metrics."""
        while self._running:
            try:
                self.system_metrics.update_system_stats()
                
                # Record system metrics as time series
                now = datetime.now()
                self.time_series.append(MetricPoint(
                    "cpu_usage", self.system_metrics.cpu_usage, now, unit="%"
                ))
                self.time_series.append(MetricPoint(
                    "memory_usage", self.system_metrics.memory_usage, now, unit="%"
                ))
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def record_agent_request(
        self,
        agent_name: str,
        success: bool,
        input_tokens: int = 0,
        output_tokens: int = 0,
        response_time: float = 0.0
    ):
        """Record agent request metrics."""
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = AgentMetrics(agent_name)
            
        metrics = self.agent_metrics[agent_name]
        metrics.requests_total += 1
        metrics.last_request_time = datetime.now()
        
        if success:
            metrics.requests_successful += 1
        else:
            metrics.requests_failed += 1
            
        metrics.total_tokens_input += input_tokens
        metrics.total_tokens_output += output_tokens
        
        # Update average response time
        if response_time > 0:
            total_time = metrics.avg_response_time * (metrics.requests_total - 1) + response_time
            metrics.avg_response_time = total_time / metrics.requests_total
            
        # Track cost
        if agent_name in ["openai", "gemini"] and (input_tokens > 0 or output_tokens > 0):
            cost = self.cost_tracker.add_cost(agent_name, input_tokens, output_tokens)
            metrics.total_cost += cost
            
            # Check budget alerts
            budget_status = self.cost_tracker.is_over_budget()
            if budget_status["daily"] or budget_status["monthly"]:
                self.logger.warning(f"Budget limit exceeded: {budget_status}")
    
    async def record_timing(self, operation: str, duration: float, tags: Dict[str, str] = None):
        """Record operation timing."""
        self.response_times[operation].append(duration)
        
        # Record as time series
        self.time_series.append(MetricPoint(
            f"{operation}_duration",
            duration,
            datetime.now(),
            tags or {},
            "seconds"
        ))
    
    async def record_error(self, operation: str, error_message: str, tags: Dict[str, str] = None):
        """Record error occurrence."""
        self.error_counts[operation] += 1
        
        # Record as time series
        self.time_series.append(MetricPoint(
            f"{operation}_error",
            1,
            datetime.now(),
            {**(tags or {}), "error": error_message[:100]},
            "count"
        ))
    
    @asynccontextmanager
    async def time_operation(self, operation_name: str, tags: Dict[str, str] = None):
        """Context manager for timing operations."""
        timer = PerformanceTimer(self, operation_name, tags)
        async with timer:
            yield timer
    
    def get_agent_summary(self) -> Dict[str, Any]:
        """Get summary of all agent metrics."""
        summary = {}
        
        for agent_name, metrics in self.agent_metrics.items():
            summary[agent_name] = {
                "requests_total": metrics.requests_total,
                "success_rate": metrics.success_rate,
                "avg_response_time": metrics.avg_response_time,
                "total_tokens": metrics.total_tokens_input + metrics.total_tokens_output,
                "total_cost": metrics.total_cost,
                "last_request": metrics.last_request_time.isoformat() if metrics.last_request_time else None
            }
            
        return summary
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get system performance summary."""
        uptime = time.time() - self.start_time
        
        return {
            "uptime_seconds": int(uptime),
            "uptime_formatted": str(timedelta(seconds=int(uptime))),
            "cpu_usage": self.system_metrics.cpu_usage,
            "memory_usage": self.system_metrics.memory_usage,
            "disk_usage": self.system_metrics.disk_usage,
            "active_connections": self.system_metrics.active_connections
        }
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary."""
        return {
            "budget_status": self.cost_tracker.get_budget_status(),
            "daily_costs": dict(self.cost_tracker.daily_costs),
            "monthly_costs": dict(self.cost_tracker.monthly_costs),
            "total_costs": dict(self.cost_tracker.total_costs)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        summary = {}
        
        for operation, times in self.response_times.items():
            if times:
                summary[operation] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "request_count": len(times)
                }
                
        return {
            "response_times": summary,
            "error_counts": dict(self.error_counts),
            "request_counts": dict(self.request_counts)
        }
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get all metrics in a comprehensive report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "agents": self.get_agent_summary(),
            "system": self.get_system_summary(),
            "costs": self.get_cost_summary(),
            "performance": self.get_performance_summary(),
            "alerts": self._get_alerts()
        }
    
    def _get_alerts(self) -> List[Dict[str, Any]]:
        """Generate system alerts based on metrics."""
        alerts = []
        
        # Budget alerts
        budget_status = self.cost_tracker.is_over_budget()
        if budget_status["daily"]:
            alerts.append({
                "type": "budget",
                "severity": "warning",
                "message": "Daily budget limit exceeded",
                "details": f"Spent: ${self.cost_tracker.get_daily_total():.2f}"
            })
            
        if budget_status["monthly"]:
            alerts.append({
                "type": "budget", 
                "severity": "critical",
                "message": "Monthly budget limit exceeded",
                "details": f"Spent: ${self.cost_tracker.get_monthly_total():.2f}"
            })
        
        # System performance alerts
        if self.system_metrics.cpu_usage > 90:
            alerts.append({
                "type": "performance",
                "severity": "warning", 
                "message": "High CPU usage detected",
                "details": f"CPU: {self.system_metrics.cpu_usage:.1f}%"
            })
            
        if self.system_metrics.memory_usage > 90:
            alerts.append({
                "type": "performance",
                "severity": "warning",
                "message": "High memory usage detected", 
                "details": f"Memory: {self.system_metrics.memory_usage:.1f}%"
            })
        
        # Agent failure rate alerts
        for agent_name, metrics in self.agent_metrics.items():
            if metrics.requests_total > 10 and metrics.success_rate < 80:
                alerts.append({
                    "type": "agent",
                    "severity": "warning",
                    "message": f"Low success rate for {agent_name}",
                    "details": f"Success rate: {metrics.success_rate:.1f}%"
                })
        
        return alerts
    
    async def flush(self):
        """Flush metrics to storage/logging."""
        try:
            metrics_summary = self.get_comprehensive_metrics()
            self.logger.info(f"Metrics summary: {metrics_summary}")
            
            # Here you could also send to external monitoring systems
            # like Prometheus, Grafana, DataDog, etc.
            
        except Exception as e:
            self.logger.error(f"Error flushing metrics: {e}")
    
    async def cleanup(self):
        """Cleanup metrics collector."""
        await self.stop_collection()
        await self.flush()


# Global metrics instance
_metrics_collector: Optional[MetricsCollector] = None


async def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _metrics_collector
    
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
        await _metrics_collector.start_collection()
        
    return _metrics_collector


async def shutdown_metrics():
    """Shutdown global metrics collector."""
    global _metrics_collector
    
    if _metrics_collector:
        await _metrics_collector.cleanup()
        _metrics_collector = None