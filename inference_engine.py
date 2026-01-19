"""Inference engine for Neuro-LLM-Server"""

import time
from typing import Generator, Optional, List, Dict, Any, Tuple
from PIL import Image
from io import BytesIO
import base64
from config import Config
from utils.errors import InferenceError, ValidationError
from utils.logging import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    """Handles inference requests"""

    def __init__(self, model_manager, config: Config):
        self.model_manager = model_manager
        self.config = config
        self.model = None
        self.tokenizer = None

    def _ensure_loaded(self):
        """Ensure model and tokenizer are loaded"""
        if self.model is None or self.tokenizer is None:
            self.model = self.model_manager.get_model()
            self.tokenizer = self.model_manager.get_tokenizer()

    def _base64_to_image(self, base64_string: str) -> bytes:
        """Convert base64 string to image bytes"""
        if "data:image" in base64_string:
            base64_string = base64_string.split(",")[1]
        return base64.b64decode(base64_string)

    def _create_image_from_bytes(self, image_bytes: bytes) -> Image.Image:
        """Create PIL Image from bytes"""
        image_stream = BytesIO(image_bytes)
        return Image.open(image_stream)

    def _prepare_messages(self, messages: List[Dict[str, Any]]) -> Tuple[Optional[Image.Image], List[Dict[str, str]]]:
        """
        Prepare messages and extract image

        Returns:
            Tuple of (image, text_messages)
        """
        image = None
        text_messages = []

        for message in messages:
            if isinstance(message, dict):
                role = message.get("role", "user")
                content = message.get("content", [])

                if isinstance(content, str):
                    # Simple text message
                    text_messages.append({"role": role, "content": content})
                elif isinstance(content, list):
                    # Multi-modal message
                    message_texts = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text" and item.get("text"):
                                message_texts.append(item["text"])
                            elif item.get("type") == "image_url":
                                image_url = item.get("image_url", {})
                                if isinstance(image_url, dict):
                                    url = image_url.get("url", "")
                                    if url:
                                        try:
                                            image_bytes = self._base64_to_image(url)
                                            image = self._create_image_from_bytes(image_bytes).convert('RGB')
                                        except Exception as e:
                                            logger.warning(f"Failed to process image: {e}")

                    if message_texts:
                        text_messages.append({"role": role, "content": " ".join(message_texts)})

        # If no image provided, create dummy image (model requires image input)
        if image is None:
            image = Image.new('RGB', (448, 448), color=(0, 0, 0))
            logger.debug("No image provided, using dummy image")

        return image, text_messages

    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stream: bool = True,
    ) -> Generator[str, None, None]:
        """
        Generate response from model

        Args:
            messages: List of messages (OpenAI format)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            stop: Stop sequences
            stream: Whether to stream response

        Yields:
            Generated text chunks
        """
        self._ensure_loaded()

        # Use defaults from config if not provided
        temperature = temperature if temperature is not None else self.config.inference.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.inference.max_tokens
        top_p = top_p if top_p is not None else self.config.inference.top_p
        stop = stop if stop else []

        # Validate parameters
        if not (0.0 <= temperature <= 2.0):
            raise ValidationError(f"Invalid temperature: {temperature} (must be 0.0-2.0)")
        if max_tokens <= 0:
            raise ValidationError(f"Invalid max_tokens: {max_tokens} (must be > 0)")

        try:
            # Prepare image and messages
            image, text_messages = self._prepare_messages(messages)

            # Prepare chat kwargs
            chat_kwargs = {
                "image": image,
                "msgs": text_messages,
                "tokenizer": self.tokenizer,
                "sampling": True,
                "temperature": temperature,
                "stream": stream,
            }

            # Add top_p if specified and valid
            if top_p is not None and top_p < 1.0:
                chat_kwargs["top_p"] = top_p

            # Generate response
            generated_text = ""
            token_count = 0

            try:
                response_generator = self.model.chat(**chat_kwargs)

                for new_text in response_generator:
                    # Check max_tokens limit
                    if max_tokens > 0 and token_count >= max_tokens:
                        break

                    # Check stop sequences
                    generated_text += new_text
                    should_stop = False
                    for stop_str in stop:
                        if stop_str in generated_text:
                            stop_index = generated_text.find(stop_str)
                            if stop_index >= 0:
                                generated_text = generated_text[:stop_index]
                                should_stop = True
                                break

                    if should_stop:
                        break

                    # Estimate token count (simplified)
                    token_count += len(new_text.split())

                    yield new_text

            except Exception as e:
                raise InferenceError(f"Error during model inference: {e}")

        except ValidationError:
            raise
        except Exception as e:
            raise InferenceError(f"Error preparing inference: {e}")
