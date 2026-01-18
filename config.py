"""Configuration management for Neuro-LLM-Server"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

try:
    import yaml
except ImportError:
    yaml = None

from utils.errors import ValidationError
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str = "openbmb/MiniCPM-Llama3-V-2_5-int4"
    quantization: str = "int4"
    trust_remote_code: bool = True


@dataclass
class GPUConfig:
    """GPU configuration"""
    cuda_visible_devices: str = "0"
    device_map: str = "auto"


@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = "127.0.0.1"
    port: int = 8000
    timeout: int = 30
    max_concurrent_requests: int = 4
    enable_queue: bool = True


@dataclass
class InferenceConfig:
    """Inference default parameters"""
    temperature: float = 0.7
    max_tokens: int = 200
    top_p: float = 1.0
    enable_torch_compile: bool = False


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enabled: bool = True
    enable_gpu_monitoring: bool = True
    metrics_interval: float = 5.0


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    log_file: str = ""


@dataclass
class HuggingFaceConfig:
    """Hugging Face configuration"""
    token: str = ""


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    
    @classmethod
    def from_file(cls, config_path: Optional[str] = None) -> "Config":
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to config file. If None, uses default or environment variable.
        
        Returns:
            Config instance
        """
        if config_path is None:
            config_path = os.getenv("NEURO_LLM_CONFIG", "config.yaml")
        
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()
        
        if yaml is None:
            raise ValidationError("PyYAML is required to load config file. Install with: pip install PyYAML")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            raise ValidationError(f"Failed to load config file: {e}")
        
        # Build config from dictionary
        config = cls()
        
        # Model config
        if "model" in data:
            model_data = data["model"]
            config.model = ModelConfig(
                name=model_data.get("name", config.model.name),
                quantization=model_data.get("quantization", config.model.quantization),
                trust_remote_code=model_data.get("trust_remote_code", config.model.trust_remote_code),
            )
            # Auto-detect quantization from model name if not specified
            if config.model.quantization == "int4" and "-int4" not in config.model.name:
                if "-int8" in config.model.name:
                    config.model.quantization = "int8"
                elif "-int4" in config.model.name:
                    config.model.quantization = "int4"
                elif "fp16" in config.model.name.lower():
                    config.model.quantization = "fp16"
                elif "fp32" in config.model.name.lower():
                    config.model.quantization = "fp32"
        
        # GPU config
        if "gpu" in data:
            gpu_data = data["gpu"]
            # CUDA_VISIBLE_DEVICES can be overridden by environment variable
            cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES", gpu_data.get("cuda_visible_devices", config.gpu.cuda_visible_devices))
            config.gpu = GPUConfig(
                cuda_visible_devices=cuda_devices,
                device_map=gpu_data.get("device_map", config.gpu.device_map),
            )
        else:
            # Check environment variable even if gpu section is missing
            cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES", config.gpu.cuda_visible_devices)
            config.gpu.cuda_visible_devices = cuda_devices
        
        # Server config
        if "server" in data:
            server_data = data["server"]
            config.server = ServerConfig(
                host=server_data.get("host", config.server.host),
                port=server_data.get("port", config.server.port),
                timeout=server_data.get("timeout", config.server.timeout),
                max_concurrent_requests=server_data.get("max_concurrent_requests", config.server.max_concurrent_requests),
                enable_queue=server_data.get("enable_queue", config.server.enable_queue),
            )
        
        # Inference config
        if "inference" in data:
            inference_data = data["inference"]
            config.inference = InferenceConfig(
                temperature=inference_data.get("temperature", config.inference.temperature),
                max_tokens=inference_data.get("max_tokens", config.inference.max_tokens),
                top_p=inference_data.get("top_p", config.inference.top_p),
                enable_torch_compile=inference_data.get("enable_torch_compile", config.inference.enable_torch_compile),
            )
        
        # Monitoring config
        if "monitoring" in data:
            monitoring_data = data["monitoring"]
            config.monitoring = MonitoringConfig(
                enabled=monitoring_data.get("enabled", config.monitoring.enabled),
                enable_gpu_monitoring=monitoring_data.get("enable_gpu_monitoring", config.monitoring.enable_gpu_monitoring),
                metrics_interval=monitoring_data.get("metrics_interval", config.monitoring.metrics_interval),
            )
        
        # Logging config
        if "logging" in data:
            logging_data = data["logging"]
            config.logging = LoggingConfig(
                level=logging_data.get("level", config.logging.level),
                log_file=logging_data.get("log_file", config.logging.log_file),
            )
        
        # Hugging Face config
        if "huggingface" in data:
            hf_data = data["huggingface"]
            hf_token = os.getenv("HF_TOKEN", hf_data.get("token", config.huggingface.token))
            config.huggingface = HuggingFaceConfig(token=hf_token)
        else:
            # Check environment variable even if huggingface section is missing
            hf_token = os.getenv("HF_TOKEN", config.huggingface.token)
            config.huggingface.token = hf_token
        
        # Apply GPU settings
        if config.gpu.cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu.cuda_visible_devices
            logger.info(f"Set CUDA_VISIBLE_DEVICES={config.gpu.cuda_visible_devices}")
        
        return config
    
    def validate(self) -> None:
        """Validate configuration values"""
        errors = []
        
        # Validate model name
        if not self.model.name:
            errors.append("model.name is required")
        
        # Validate quantization
        if self.model.quantization not in ["int4", "int8", "fp16", "fp32"]:
            errors.append(f"Invalid quantization type: {self.model.quantization}")
        
        # Validate server port
        if not (1 <= self.server.port <= 65535):
            errors.append(f"Invalid server port: {self.server.port}")
        
        # Validate timeout
        if self.server.timeout <= 0:
            errors.append(f"Invalid timeout: {self.server.timeout}")
        
        # Validate temperature
        if not (0.0 <= self.inference.temperature <= 2.0):
            errors.append(f"Invalid temperature: {self.inference.temperature}")
        
        # Validate max_tokens
        if self.inference.max_tokens <= 0:
            errors.append(f"Invalid max_tokens: {self.inference.max_tokens}")
        
        if errors:
            raise ValidationError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
