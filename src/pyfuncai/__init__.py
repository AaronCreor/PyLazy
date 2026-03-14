"""Public package interface for PyFuncAI."""

from pyfuncai.core import GeneratedFunction, connect, createFunction, create_function
from pyfuncai.exceptions import (
    ConfigurationError,
    GenerationError,
    ProviderError,
    PyFuncAIError,
    ValidationError,
)
from pyfuncai.providers import GeminiProvider, OllamaProvider, OpenAIProvider

__all__ = [
    "ConfigurationError",
    "GeminiProvider",
    "GeneratedFunction",
    "GenerationError",
    "OllamaProvider",
    "OpenAIProvider",
    "ProviderError",
    "PyFuncAIError",
    "ValidationError",
    "connect",
    "createFunction",
    "create_function",
]

__version__ = "0.1.0"
