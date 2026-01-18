"""Model management for Neuro-LLM-Server"""

import os
import sys
import re
from pathlib import Path
from typing import Optional, Tuple
import torch
from transformers import AutoModel, AutoTokenizer
from config import Config, ModelConfig
from utils.errors import ModelLoadError
from utils.logging import get_logger

logger = get_logger(__name__)


def fix_model_cache_imports():
    """Fix import errors in Hugging Face cache files"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    
    if not cache_dir.exists():
        return
    
    # Search for MiniCPM-Llama3-V-2_5 model cache directories
    model_cache_dirs = list(cache_dir.glob("openbmb/MiniCPM-Llama3-V-2_5/*"))
    
    for model_dir in model_cache_dirs:
        resampler_file = model_dir / "resampler.py"
        if resampler_file.exists():
            try:
                content = resampler_file.read_text(encoding='utf-8')
                original_content = content
                
                # Check if List is used
                uses_list = bool(re.search(r'\bList\[', content))
                
                if not uses_list:
                    continue
                
                # Check if List is imported
                has_list_import = bool(
                    re.search(r'from typing import.*\bList\b', content) or
                    re.search(r'from typing import List', content)
                )
                
                if has_list_import:
                    continue
                
                # Find typing import statement
                typing_import_pattern = r'from typing import\s+([^#\n]+)'
                match = re.search(typing_import_pattern, content)
                
                if match:
                    # Add List to existing typing import
                    existing_imports = match.group(1).strip()
                    if "List" not in existing_imports:
                        new_imports = existing_imports.rstrip() + ", List"
                        content = content.replace(
                            match.group(0),
                            f"from typing import {new_imports}"
                        )
                else:
                    # Add typing import if it doesn't exist
                    lines = content.split('\n')
                    insert_pos = 0
                    for i, line in enumerate(lines):
                        if line.strip().startswith('import ') or line.strip().startswith('from '):
                            insert_pos = i + 1
                    if insert_pos == 0:
                        insert_pos = 0
                    lines.insert(insert_pos, "from typing import List")
                    content = '\n'.join(lines)
                
                # Write back if changed
                if content != original_content:
                    resampler_file.write_text(content, encoding='utf-8')
                    logger.info(f"Fixed cache file: {resampler_file}")
            except Exception as e:
                logger.warning(f"Error fixing cache file: {e}")


class ModelManager:
    """Manages model loading and quantization"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model: Optional[torch.nn.Module] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device: Optional[torch.device] = None
        
    def _get_torch_dtype(self, quantization: str) -> Optional[torch.dtype]:
        """Get torch dtype from quantization type"""
        if quantization == "fp32":
            return torch.float32
        elif quantization == "fp16":
            return torch.float16
        elif quantization in ["int4", "int8"]:
            # Quantization is handled by bitsandbytes, return None
            return None
        else:
            logger.warning(f"Unknown quantization type: {quantization}, using default")
            return None
    
    def _apply_quantization(self, model: torch.nn.Module, quantization: str) -> torch.nn.Module:
        """Apply quantization to model"""
        if quantization == "int4":
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("Applying int4 quantization with BitsAndBytes")
                # Note: quantization is applied during model loading, not here
                return model
            except ImportError:
                logger.warning("bitsandbytes not available, skipping int4 quantization")
                return model
        elif quantization == "int8":
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                logger.info("Applying int8 quantization with BitsAndBytes")
                # Note: quantization is applied during model loading, not here
                return model
            except ImportError:
                logger.warning("bitsandbytes not available, skipping int8 quantization")
                return model
        else:
            return model
    
    def load_model(self, retry_count: int = 3) -> Tuple[torch.nn.Module, AutoTokenizer]:
        """
        Load model and tokenizer with retry logic
        
        Args:
            retry_count: Number of retry attempts
        
        Returns:
            Tuple of (model, tokenizer)
        
        Raises:
            ModelLoadError: If model loading fails after retries
        """
        # Fix cache imports before loading
        try:
            fix_model_cache_imports()
        except Exception as e:
            logger.warning(f"Failed to fix cache imports: {e}")
        
        model_config = self.config.model
        quantization = model_config.quantization
        
        logger.info(f"Loading model: {model_config.name}")
        logger.info(f"Quantization: {quantization}")
        logger.info("Initial model download may take time on first run")
        
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.warning("CUDA not available, using CPU (very slow)")
        
        # Get torch dtype
        torch_dtype = self._get_torch_dtype(quantization)
        
        # Prepare model loading kwargs
        model_kwargs = {
            "trust_remote_code": model_config.trust_remote_code,
        }
        
        # Add quantization config if needed
        if quantization in ["int4", "int8"]:
            try:
                from transformers import BitsAndBytesConfig
                if quantization == "int4":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                else:  # int8
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = self.config.gpu.device_map
            except ImportError:
                logger.warning("bitsandbytes not available, loading without quantization")
                if torch_dtype:
                    model_kwargs["torch_dtype"] = torch_dtype
        else:
            if torch_dtype:
                model_kwargs["torch_dtype"] = torch_dtype
        
        # Load tokenizer first (simpler, less likely to fail)
        for attempt in range(1, retry_count + 1):
            try:
                logger.info(f"Loading tokenizer (attempt {attempt}/{retry_count})...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_config.name,
                    trust_remote_code=model_config.trust_remote_code,
                    token=self.config.huggingface.token if self.config.huggingface.token else None,
                )
                logger.info("Tokenizer loaded successfully")
                break
            except Exception as e:
                if attempt == retry_count:
                    raise ModelLoadError(f"Failed to load tokenizer after {retry_count} attempts: {e}")
                logger.warning(f"Tokenizer load attempt {attempt} failed: {e}, retrying...")
        
        # Load model with retry
        for attempt in range(1, retry_count + 1):
            try:
                logger.info(f"Loading model (attempt {attempt}/{retry_count})...")
                self.model = AutoModel.from_pretrained(
                    model_config.name,
                    token=self.config.huggingface.token if self.config.huggingface.token else None,
                    **model_kwargs
                )
                
                # Move to device if not using device_map
                if self.config.gpu.device_map == "single" and self.device.type == "cuda":
                    self.model = self.model.to(self.device)
                
                # Set to eval mode
                self.model.eval()
                
                # Apply torch.compile if enabled
                if self.config.inference.enable_torch_compile:
                    try:
                        logger.info("Applying torch.compile optimization...")
                        self.model = torch.compile(self.model)
                        logger.info("torch.compile applied successfully")
                    except Exception as e:
                        logger.warning(f"torch.compile failed: {e}, continuing without it")
                
                logger.info("Model loaded successfully")
                break
            except Exception as e:
                if attempt == retry_count:
                    raise ModelLoadError(f"Failed to load model after {retry_count} attempts: {e}")
                logger.warning(f"Model load attempt {attempt} failed: {e}, retrying...")
        
        return self.model, self.tokenizer
    
    def get_model(self) -> torch.nn.Module:
        """Get loaded model"""
        if self.model is None:
            raise ModelLoadError("Model not loaded. Call load_model() first.")
        return self.model
    
    def get_tokenizer(self) -> AutoTokenizer:
        """Get loaded tokenizer"""
        if self.tokenizer is None:
            raise ModelLoadError("Tokenizer not loaded. Call load_model() first.")
        return self.tokenizer
    
    def get_device(self) -> torch.device:
        """Get device"""
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device
