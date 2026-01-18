"""Custom error classes for Neuro-LLM-Server"""


class NeuroLLMError(Exception):
    """Base exception for Neuro-LLM-Server errors"""
    pass


class ModelLoadError(NeuroLLMError):
    """Raised when model loading fails"""
    pass


class InferenceError(NeuroLLMError):
    """Raised when inference fails"""
    pass


class ValidationError(NeuroLLMError):
    """Raised when request validation fails"""
    pass


class TimeoutError(NeuroLLMError):
    """Raised when operation times out"""
    pass
