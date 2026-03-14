"""Public package interface for PyFuncAI."""

from importlib.metadata import PackageNotFoundError, version

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

try:
    __version__ = version("PyFuncAI")
except PackageNotFoundError:
    __version__ = "0+unknown"
