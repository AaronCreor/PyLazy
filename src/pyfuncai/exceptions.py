"""Shared exception hierarchy for PyFuncAI."""


class PyFuncAIError(Exception):
    """Base exception for all library-specific errors."""


class ConfigurationError(PyFuncAIError):
    """Raised when runtime or provider configuration is incomplete or invalid."""


class ProviderError(PyFuncAIError):
    """Raised when an upstream model provider request fails."""


class GenerationError(PyFuncAIError):
    """Raised when PyFuncAI cannot turn a model response into usable source code."""


class ValidationError(PyFuncAIError):
    """Raised when generated source violates PyFuncAI's validation constraints."""
