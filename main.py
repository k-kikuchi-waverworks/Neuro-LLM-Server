"""Neuro-LLM-Server - Production implementation"""

import os
import sys
import json
import time
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

from config import Config
from model_manager import ModelManager
from inference_engine import InferenceEngine
from monitoring import Monitoring
from request_queue import RequestQueue
from utils.errors import (
    NeuroLLMError,
    ModelLoadError,
    InferenceError,
    ValidationError,
    TimeoutError as NeuroTimeoutError,
)
from utils.logging import setup_logging, get_logger

# Global instances
config: Optional[Config] = None
model_manager: Optional[ModelManager] = None
inference_engine: Optional[InferenceEngine] = None
monitoring: Optional[Monitoring] = None
request_queue: Optional[RequestQueue] = None
logger = None


# Pydantic models for API
class ImageURL(BaseModel):
    url: str = ""


class Content(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None


class Message(BaseModel):
    role: str
    content: List[Content]


class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 200
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    stream: bool = True
    mode: str = "instruct"
    skip_special_tokens: bool = False
    custom_token_bans: str = ""


class Delta(BaseModel):
    role: str = "assistant"
    content: str = ""


class Choice(BaseModel):
    index: int = 0
    finish_reason: Optional[str] = None
    delta: Delta


class ChatResponse(BaseModel):
    id: str = "chatcmpl-00000"
    object: str = "chat.completions.chunk"
    created: int = 0
    model: str = "MiniCPM-Llama3-V-2_5"
    choices: List[Choice]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global config, model_manager, inference_engine, monitoring, request_queue, logger

    # Startup
    try:
        # Load configuration
        config_path = os.getenv("NEURO_LLM_CONFIG", "config.yaml")
        config = Config.from_file(config_path)
        config.validate()

        # Setup logging
        setup_logging(
            log_level=config.logging.level,
            log_file=config.logging.log_file if config.logging.log_file else None
        )
        logger = get_logger(__name__)

        logger.info("=" * 60)
        logger.info("Neuro-LLM-Server Starting")
        logger.info("=" * 60)
        logger.info(f"Model: {config.model.name}")
        logger.info(f"Quantization: {config.model.quantization}")
        logger.info(f"Server: {config.server.host}:{config.server.port}")

        # Initialize components
        model_manager = ModelManager(config)
        logger.info("Loading model...")
        model_manager.load_model()
        logger.info("Model loaded successfully")

        inference_engine = InferenceEngine(model_manager, config)
        monitoring = Monitoring(config)
        request_queue = RequestQueue(
            max_concurrent=config.server.max_concurrent_requests,
            enable_queue=config.server.enable_queue
        )

        logger.info("=" * 60)
        logger.info("Neuro-LLM-Server Ready")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    yield

    # Shutdown
    logger.info("Shutting down Neuro-LLM-Server...")


# Create FastAPI app
app = FastAPI(
    title="Neuro-LLM-Server",
    description="Production-ready multimodal LLM server for MiniCPM-Llama3-V-2_5",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware for request timing
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """Middleware to measure request latency"""
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    # Record metrics (skip for monitoring endpoints)
    if monitoring and not request.url.path.startswith(("/health", "/metrics")):
        monitoring.record_request(latency, error=response.status_code >= 400)

    return response


# Error handlers
@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "code": "validation_error"
            }
        }
    )


@app.exception_handler(InferenceError)
async def inference_error_handler(request: Request, exc: InferenceError):
    """Handle inference errors"""
    logger.error(f"Inference error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "server_error",
                "code": "inference_error"
            }
        }
    )


@app.exception_handler(NeuroLLMError)
async def neuro_llm_error_handler(request: Request, exc: NeuroLLMError):
    """Handle general Neuro-LLM-Server errors"""
    logger.error(f"Error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "server_error",
                "code": "internal_error"
            }
        }
    )


def chat_generator(chat_request: ChatRequest):
    """Generate chat response chunks"""
    if inference_engine is None:
        raise InferenceError("Inference engine not initialized")

    try:
        # Convert Pydantic models to dicts for inference engine
        messages = []
        for msg in chat_request.messages:
            content_list = []
            for content in msg.content:
                if content.type == "text":
                    content_list.append({"type": "text", "text": content.text})
                elif content.type == "image_url" and content.image_url:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": content.image_url.url}
                    })
            messages.append({"role": msg.role, "content": content_list})

        # Generate response
        generator = inference_engine.generate(
            messages=messages,
            temperature=chat_request.temperature,
            max_tokens=chat_request.max_tokens,
            top_p=chat_request.top_p if chat_request.top_p < 1.0 else None,
            stop=chat_request.stop if chat_request.stop else [],
            stream=chat_request.stream,
        )

        index = 0
        for new_text in generator:
            delta = Delta(role="assistant", content=new_text)
            choice = Choice(index=index, finish_reason=None, delta=delta)
            model_name = config.model.name if config else "MiniCPM-Llama3-V-2_5"
            chat_response = ChatResponse(
                id=f"chatcmpl-{int(time.time())}",
                created=int(time.time()),
                model=model_name,
                choices=[choice]
            )
            index += 1
            yield chat_response.model_dump_json() + "\n"

        # Final chunk
        delta = Delta(role="assistant", content="")
        choice = Choice(index=index, finish_reason="stop", delta=delta)
        model_name = config.model.name if config else "MiniCPM-Llama3-V-2_5"
        chat_response = ChatResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=model_name,
            choices=[choice]
        )
        yield chat_response.model_dump_json() + "\n"

    except Exception as e:
        logger.error(f"Error in chat_generator: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.post("/v1/chat/completions")
async def chat_completions(chat_request: ChatRequest):
    """Chat completions endpoint"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    # Execute with queue management
    async def process_request():
        if chat_request.stream:
            return EventSourceResponse(chat_generator(chat_request))
        else:
            # Non-streaming mode
            result_text = ""
            for chunk_json in chat_generator(chat_request):
                chunk = json.loads(chunk_json.strip())
                if chunk.get("choices") and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        result_text += content

            # Return non-streaming response
            model_name = config.model.name if config else "MiniCPM-Llama3-V-2_5"
            response = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,  # TODO: Implement token counting
                    "completion_tokens": 0,  # TODO: Implement token counting
                    "total_tokens": 0  # TODO: Implement token counting
                }
            }
            return response

    if request_queue is None:
        raise HTTPException(status_code=503, detail="Request queue not initialized")

    return await request_queue.execute(process_request())


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if monitoring is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "monitoring not initialized"}
        )

    health = monitoring.get_health_status()
    status_code = 200 if health["status"] == "healthy" else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": health["status"],
            "error_rate": health["error_rate"],
            "gpu_memory_percent": health["gpu_memory_percent"],
            "gpu_memory_ok": health["gpu_memory_ok"],
        }
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    if monitoring is None:
        return JSONResponse(
            status_code=503,
            content={"error": "monitoring not initialized"}
        )

    from fastapi.responses import Response
    return Response(
        content=monitoring.get_prometheus_metrics(),
        media_type="text/plain"
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Neuro-LLM-Server",
        "version": "2.0.0",
        "model": config.model.name if config else "unknown",
        "status": "running"
    }
