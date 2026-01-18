"""Monitoring functionality for Neuro-LLM-Server"""

import time
import threading
from typing import Dict, Optional
from collections import deque
from dataclasses import dataclass, field
from config import Config
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Metrics:
    """Metrics data structure"""
    request_count: int = 0
    total_latency: float = 0.0
    min_latency: float = float('inf')
    max_latency: float = 0.0
    error_count: int = 0
    gpu_utilization: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    last_update: float = field(default_factory=time.time)


class Monitoring:
    """Monitoring system for Neuro-LLM-Server"""
    
    def __init__(self, config: Config):
        self.config = config
        self.metrics = Metrics()
        self.latency_history = deque(maxlen=100)  # Keep last 100 latencies
        self.lock = threading.Lock()
        self.gpu_available = False
        self.pynvml_initialized = False
        
        if config.monitoring.enable_gpu_monitoring:
            self._init_gpu_monitoring()
    
    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml_initialized = True
            self.gpu_available = True
            logger.info("GPU monitoring initialized")
        except ImportError:
            logger.warning("pynvml not available, GPU monitoring disabled")
            self.gpu_available = False
        except Exception as e:
            logger.warning(f"Failed to initialize GPU monitoring: {e}")
            self.gpu_available = False
    
    def _get_gpu_stats(self) -> Optional[Dict[str, float]]:
        """Get GPU statistics"""
        if not self.gpu_available or not self.pynvml_initialized:
            return None
        
        try:
            import pynvml
            import torch
            
            # Get first visible GPU (CUDA_VISIBLE_DEVICES may be set)
            if torch.cuda.is_available():
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # Get utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                
                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = mem_info.used / (1024 ** 3)  # GB
                mem_total = mem_info.total / (1024 ** 3)  # GB
                
                return {
                    "utilization": gpu_util,
                    "memory_used": mem_used,
                    "memory_total": mem_total,
                }
        except Exception as e:
            logger.debug(f"Error getting GPU stats: {e}")
            return None
        
        return None
    
    def record_request(self, latency: float, error: bool = False):
        """Record a request"""
        with self.lock:
            self.metrics.request_count += 1
            self.metrics.total_latency += latency
            self.metrics.min_latency = min(self.metrics.min_latency, latency)
            self.metrics.max_latency = max(self.metrics.max_latency, latency)
            self.latency_history.append(latency)
            
            if error:
                self.metrics.error_count += 1
            
            self.metrics.last_update = time.time()
    
    def update_gpu_metrics(self):
        """Update GPU metrics"""
        if not self.config.monitoring.enable_gpu_monitoring:
            return
        
        gpu_stats = self._get_gpu_stats()
        if gpu_stats:
            with self.lock:
                self.metrics.gpu_utilization = gpu_stats["utilization"]
                self.metrics.gpu_memory_used = gpu_stats["memory_used"]
                self.metrics.gpu_memory_total = gpu_stats["memory_total"]
    
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        self.update_gpu_metrics()
        
        with self.lock:
            avg_latency = (
                self.metrics.total_latency / self.metrics.request_count
                if self.metrics.request_count > 0
                else 0.0
            )
            
            # Calculate throughput (requests per second)
            throughput = 0.0
            if len(self.latency_history) > 1:
                time_span = time.time() - (self.metrics.last_update - sum(self.latency_history) / len(self.latency_history))
                if time_span > 0:
                    throughput = len(self.latency_history) / time_span
            
            return {
                "request_count": self.metrics.request_count,
                "error_count": self.metrics.error_count,
                "average_latency": avg_latency,
                "min_latency": self.metrics.min_latency if self.metrics.min_latency != float('inf') else 0.0,
                "max_latency": self.metrics.max_latency,
                "throughput": throughput,
                "gpu_utilization": self.metrics.gpu_utilization,
                "gpu_memory_used_gb": self.metrics.gpu_memory_used,
                "gpu_memory_total_gb": self.metrics.gpu_memory_total,
                "gpu_memory_percent": (
                    (self.metrics.gpu_memory_used / self.metrics.gpu_memory_total * 100)
                    if self.metrics.gpu_memory_total > 0
                    else 0.0
                ),
            }
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        metrics = self.get_metrics()
        
        lines = [
            "# HELP neuro_llm_requests_total Total number of requests",
            "# TYPE neuro_llm_requests_total counter",
            f"neuro_llm_requests_total {metrics['request_count']}",
            "",
            "# HELP neuro_llm_errors_total Total number of errors",
            "# TYPE neuro_llm_errors_total counter",
            f"neuro_llm_errors_total {metrics['error_count']}",
            "",
            "# HELP neuro_llm_latency_seconds Average request latency in seconds",
            "# TYPE neuro_llm_latency_seconds gauge",
            f"neuro_llm_latency_seconds {metrics['average_latency']:.4f}",
            "",
            "# HELP neuro_llm_latency_min_seconds Minimum request latency in seconds",
            "# TYPE neuro_llm_latency_min_seconds gauge",
            f"neuro_llm_latency_min_seconds {metrics['min_latency']:.4f}",
            "",
            "# HELP neuro_llm_latency_max_seconds Maximum request latency in seconds",
            "# TYPE neuro_llm_latency_max_seconds gauge",
            f"neuro_llm_latency_max_seconds {metrics['max_latency']:.4f}",
            "",
            "# HELP neuro_llm_throughput_requests_per_second Requests per second",
            "# TYPE neuro_llm_throughput_requests_per_second gauge",
            f"neuro_llm_throughput_requests_per_second {metrics['throughput']:.4f}",
        ]
        
        if self.config.monitoring.enable_gpu_monitoring:
            lines.extend([
                "",
                "# HELP neuro_llm_gpu_utilization_percent GPU utilization percentage",
                "# TYPE neuro_llm_gpu_utilization_percent gauge",
                f"neuro_llm_gpu_utilization_percent {metrics['gpu_utilization']:.2f}",
                "",
                "# HELP neuro_llm_gpu_memory_used_gb GPU memory used in GB",
                "# TYPE neuro_llm_gpu_memory_used_gb gauge",
                f"neuro_llm_gpu_memory_used_gb {metrics['gpu_memory_used_gb']:.2f}",
                "",
                "# HELP neuro_llm_gpu_memory_total_gb GPU memory total in GB",
                "# TYPE neuro_llm_gpu_memory_total_gb gauge",
                f"neuro_llm_gpu_memory_total_gb {metrics['gpu_memory_total_gb']:.2f}",
                "",
                "# HELP neuro_llm_gpu_memory_percent GPU memory usage percentage",
                "# TYPE neuro_llm_gpu_memory_percent gauge",
                f"neuro_llm_gpu_memory_percent {metrics['gpu_memory_percent']:.2f}",
            ])
        
        return "\n".join(lines)
    
    def get_health_status(self) -> Dict:
        """Get health status"""
        metrics = self.get_metrics()
        
        # Health is OK if:
        # - No recent errors (or error rate < 10%)
        # - GPU memory not critically high (< 95%)
        error_rate = (
            metrics['error_count'] / metrics['request_count']
            if metrics['request_count'] > 0
            else 0.0
        )
        
        gpu_memory_ok = metrics['gpu_memory_percent'] < 95.0
        
        is_healthy = error_rate < 0.1 and gpu_memory_ok
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "error_rate": error_rate,
            "gpu_memory_percent": metrics['gpu_memory_percent'],
            "gpu_memory_ok": gpu_memory_ok,
        }
