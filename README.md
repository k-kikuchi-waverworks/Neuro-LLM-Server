# Neuro-LLM-Server

Production-ready OpenAI API-compatible server for serving the MiniCPM-Llama3-V-2.5 multimodal LLM.

## Overview

Neuro-LLM-Server provides a production-grade implementation for serving the [MiniCPM-Llama3-V-2.5](https://github.com/OpenBMB/MiniCPM-V) model with:

- OpenAI API-compatible endpoints
- Configurable quantization (int4/int8/fp16/fp32)
- Request queueing and concurrency control
- GPU monitoring and metrics
- Comprehensive error handling
- Health check and Prometheus metrics endpoints

## Features

- **OpenAI API Compatibility**: Full support for `/v1/chat/completions` endpoint
- **Streaming Support**: Real-time streaming responses
- **Quantization Options**: int4 (~8GB VRAM), int8 (~12GB VRAM), fp16 (~16GB VRAM)
- **GPU Management**: Configurable GPU selection via `CUDA_VISIBLE_DEVICES`
- **Request Queueing**: Automatic queueing when max concurrent requests reached
- **Monitoring**: GPU utilization, latency, throughput metrics
- **Health Checks**: `/health` endpoint for service status
- **Prometheus Metrics**: `/metrics` endpoint for monitoring integration

## Requirements

- Python 3.10+
- Anaconda or Miniconda (recommended)
- NVIDIA GPU with CUDA support (for GPU inference)
- Windows 11 (PowerShell) - Native support

## Installation

### 1. Create Conda Environment

```powershell
conda create -n MiniCPM-V python=3.10 -y
conda activate MiniCPM-V
```

### 2. Install PyTorch (CUDA)

```powershell
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4 (recommended, backward compatible with 13.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install Dependencies

```powershell
cd tools\Neuro-LLM-Server
pip install -r requirements.txt
```

## Configuration

### Configuration File (`config.yaml`)

Create or edit `config.yaml` in the Neuro-LLM-Server directory:

```yaml
# Model Configuration
model:
    name: 'openbmb/MiniCPM-Llama3-V-2_5-int4' # int4, int8, or full precision
    quantization: 'int4' # int4, int8, fp16, fp32
    trust_remote_code: true

# GPU Configuration
gpu:
    cuda_visible_devices: '0' # Use GPU 0 (5090)
    device_map: 'auto'

# Server Configuration
server:
    host: '127.0.0.1'
    port: 8000
    timeout: 30
    max_concurrent_requests: 4
    enable_queue: true

# Inference Default Parameters
inference:
    temperature: 0.7
    max_tokens: 200
    top_p: 1.0
    enable_torch_compile: false

# Monitoring Configuration
monitoring:
    enabled: true
    enable_gpu_monitoring: true
    metrics_interval: 5.0

# Logging Configuration
logging:
    level: 'INFO'
    log_file: '' # Empty = console only

# Hugging Face Configuration
huggingface:
    token: '' # Set via HF_TOKEN environment variable
```

### Environment Variables

- `NEURO_LLM_CONFIG`: Path to config file (default: `config.yaml`)
- `CUDA_VISIBLE_DEVICES`: GPU selection (e.g., `"0"` for GPU 0, `"0,1"` for multiple GPUs)
- `HF_TOKEN`: Hugging Face token for gated models

## Usage

### Manual Start

```powershell
# Activate conda environment
conda activate MiniCPM-V

# Start server
cd tools\Neuro-LLM-Server
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Using start_all.ps1

The server is automatically started when using `.\start_all.ps1` from the project root.

## API Endpoints

### POST `/v1/chat/completions`

OpenAI-compatible chat completions endpoint.

**Request:**

```json
{
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,..."
                    }
                }
            ]
        }
    ],
    "temperature": 0.7,
    "max_tokens": 200,
    "stream": true
}
```

**Response (Streaming):**

```
data: {"id":"chatcmpl-...","object":"chat.completions.chunk","choices":[{"delta":{"content":"The"}}]}
data: {"id":"chatcmpl-...","object":"chat.completions.chunk","choices":[{"delta":{"content":" image"}}]}
...
```

### GET /health

Health check endpoint.

**Response:**

```json
{
    "status": "healthy",
    "error_rate": 0.0,
    "gpu_memory_percent": 45.2,
    "gpu_memory_ok": true
}
```

### GET /metrics

Prometheus metrics endpoint.

**Response:**

```
# HELP neuro_llm_requests_total Total number of requests
# TYPE neuro_llm_requests_total counter
neuro_llm_requests_total 42
...
```

## Quantization Options

### int4 (Recommended for 8GB+ VRAM)

- Model: `openbmb/MiniCPM-Llama3-V-2_5-int4`
- VRAM: ~8GB
- Quality: Good (slight quality loss)

### int8 (Recommended for 12GB+ VRAM)

- Model: `openbmb/MiniCPM-Llama3-V-2_5-int8`
- VRAM: ~12GB
- Quality: Very good (minimal quality loss)

### fp16 (Full Precision)

- Model: `openbmb/MiniCPM-Llama3-V-2_5`
- VRAM: ~16GB
- Quality: Best (no quality loss)

## GPU Configuration

### Single GPU (5090)

Set in `config.yaml`:

```yaml
gpu:
    cuda_visible_devices: '0'
```

Or via environment variable:

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
```

### Multi-GPU (Future)

For multi-GPU support, set:

```yaml
gpu:
    cuda_visible_devices: '0,1'
    device_map: 'balanced'
```

## Monitoring

### Health Check

```powershell
curl http://127.0.0.1:8000/health
```

### Metrics

```powershell
curl http://127.0.0.1:8000/metrics
```

Metrics include:

- Request count and error rate
- Average, min, max latency
- Throughput (requests/second)
- GPU utilization and memory usage

## Troubleshooting

### Model Loading Fails

1. Check GPU availability: `nvidia-smi`
2. Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check Hugging Face token (for gated models): `echo $env:HF_TOKEN`
4. Verify model name in `config.yaml`

### Out of Memory

- Use int4 quantization instead of int8/fp16
- Reduce `max_concurrent_requests` in config
- Close other GPU-intensive applications

### Import Errors

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify conda environment is activated: `conda activate MiniCPM-V`

## Architecture

```
main.py
├── Config (config.py)
├── ModelManager (model_manager.py)
│   └── Loads and manages model/tokenizer
├── InferenceEngine (inference_engine.py)
│   └── Handles inference requests
├── Monitoring (monitoring.py)
│   └── GPU metrics, latency, throughput
└── RequestQueue (request_queue.py)
    └── Manages concurrency
```

## Development

### Project Structure

```
Neuro-LLM-Server/
├── main.py              # FastAPI application
├── config.yaml          # Configuration file
├── config.py            # Configuration management
├── model_manager.py     # Model loading and quantization
├── inference_engine.py  # Inference handling
├── monitoring.py        # Monitoring and metrics
├── request_queue.py     # Request queueing
├── utils/
│   ├── errors.py        # Custom error classes
│   └── logging.py       # Logging configuration
└── requirements.txt     # Dependencies
```

## License

See LICENSE file.

## Acknowledgments

- [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) - The multimodal LLM model
- [Neuro](https://github.com/kimjammer/Neuro) - Original project inspiration
