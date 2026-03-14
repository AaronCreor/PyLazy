"""Top-level runtime and public API for PyFuncAI."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import re
from typing import Any

from pyfuncai.cache import DiskCache, make_cache_key
from pyfuncai.compiler import compile_function
from pyfuncai.exceptions import ConfigurationError, GenerationError
from pyfuncai.prompts import build_generation_prompt, extract_python_source
from pyfuncai.providers import BaseProvider, create_provider
from pyfuncai.validation import DEFAULT_ALLOWED_MODULES, validate_function_source

_DEFAULT_PROVIDER: BaseProvider | None = None
_DEFAULT_CACHE: DiskCache | None = None
_REGISTERED_FUNCTIONS: list["GeneratedFunction"] = []


def connect(
    provider: str,
    /,
    **config: Any,
) -> BaseProvider:
    """Configure the default provider used by later `create_function()` calls.

    Parameters
    ----------
    provider:
        Provider name. Supported values are `"ollama"`, `"openai"`, and `"gemini"`.
    **config:
        Provider-specific settings such as `model`, `api_key`, `base_url`,
        `cache_dir`, and request timeout values.

    Returns
    -------
    BaseProvider
        The provider instance that was created and registered as the default.
    """

    global _DEFAULT_PROVIDER
    global _DEFAULT_CACHE

    cache_dir = config.pop("cache_dir", None)
    _DEFAULT_PROVIDER = create_provider(provider, **config)
    _DEFAULT_CACHE = DiskCache(cache_dir=cache_dir)
    return _DEFAULT_PROVIDER


@dataclass
class GeneratedFunction:
    """A callable wrapper that lazily or eagerly materializes generated code."""

    prompt: str
    signature: str
    provider: BaseProvider
    cache_store: DiskCache
    cache_enabled: bool = True
    mode: str = "lazy"
    function_name: str = "generated_function"
    system_prompt: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    allow_modules: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_ALLOWED_MODULES
    )
    _built_function: Any = field(init=False, default=None, repr=False)
    _source: str | None = field(init=False, default=None, repr=False)
    __signature__: inspect.Signature = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize constructor inputs and expose function-like metadata."""

        if self.mode not in {"lazy", "eager"}:
            raise ConfigurationError("mode must be either 'lazy' or 'eager'.")

        self.function_name = _normalize_function_name(self.function_name)
        self.allow_modules = tuple(sorted({module for module in self.allow_modules}))
        self.__name__ = self.function_name
        self.__qualname__ = self.function_name
        self.__doc__ = self.prompt
        self.__signature__ = _signature_object_from_text(self.signature)

    @property
    def cache_key(self) -> str:
        """Return the deterministic cache key for this generation request."""

        payload = {
            "function_name": self.function_name,
            "signature": self.signature,
            "prompt": self.prompt,
            "provider": self.provider.cache_identity(),
            "mode": self.mode,
            "allow_modules": list(self.allow_modules),
        }
        return make_cache_key(payload)

    @property
    def source(self) -> str | None:
        """Return the generated source if it is already built or cached locally."""

        if self._source is not None:
            return self._source

        if not self.cache_enabled:
            return None

        entry = self.cache_store.get(self.cache_key)
        if entry is None:
            return None

        self._source = entry.source
        return self._source

    @property
    def is_built(self) -> bool:
        """Report whether the callable has already been compiled in-process."""

        return self._built_function is not None

    def build(self, *, force: bool = False) -> Any:
        """Generate, validate, compile, and cache the underlying callable."""

        if self._built_function is not None and not force:
            return self._built_function

        source = None if force else self.source
        if source is None:
            source = self._generate_source()
            if self.cache_enabled:
                self.cache_store.set(
                    cache_key=self.cache_key,
                    function_name=self.function_name,
                    signature=self.signature,
                    source=source,
                    provider=self.provider.cache_identity(),
                    prompt=self.prompt,
                    mode=self.mode,
                    allow_modules=list(self.allow_modules),
                )

        validate_function_source(
            source,
            function_name=self.function_name,
            signature=self.signature,
            allowed_modules=self.allow_modules,
        )
        self._built_function = compile_function(
            source,
            function_name=self.function_name,
            allowed_modules=self.allow_modules,
        )
        self._source = source
        return self._built_function

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the generated callable, building it on first use if necessary."""

        function = self.build()
        return function(*args, **kwargs)

    def _generate_source(self) -> str:
        """Request source code from the provider and extract the function body."""

        generation_prompt = build_generation_prompt(
            prompt=self.prompt,
            function_name=self.function_name,
            signature=self.signature,
            allowed_modules=self.allow_modules,
        )
        raw_response = self.provider.generate_text(
            prompt=generation_prompt,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        source = extract_python_source(raw_response)
        if not source.strip():
            raise GenerationError("The provider returned an empty response.")
        return source


def create_function(
    prompt: str,
    *,
    signature: str,
    cache: bool = True,
    mode: str = "lazy",
    function_name: str | None = None,
    provider: BaseProvider | None = None,
    cache_dir: str | None = None,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    allow_modules: tuple[str, ...] | list[str] | None = None,
) -> GeneratedFunction:
    """Create a prompt-defined Python callable.

    Parameters
    ----------
    prompt:
        Natural-language description of what the generated function should do.
    signature:
        Python function signature in the form `"(value: str) -> int"`.
    cache:
        When `True`, generated source is cached to disk and reused across runs.
    mode:
        `"lazy"` defers code generation until the first call, while `"eager"`
        generates immediately during `create_function()`.
    function_name:
        Optional explicit function name. When omitted, PyFuncAI derives a stable
        snake_case name from the prompt.
    provider:
        Optional provider instance. If omitted, PyFuncAI uses the most recent
        provider configured by `connect()`.
    cache_dir:
        Optional cache directory for this function. If omitted, the runtime cache
        directory configured by `connect()` is reused.
    system_prompt:
        Optional provider-level instruction prefix added before the generated
        prompt template.
    temperature:
        Optional provider sampling temperature.
    max_output_tokens:
        Optional provider token cap for the generated response.
    allow_modules:
        Optional iterable of standard-library modules allowed in generated code.

    Returns
    -------
    GeneratedFunction
        A callable wrapper that can be invoked like a normal function and also
        materialized ahead of time via `.build()`.
    """

    provider_instance = provider or _DEFAULT_PROVIDER
    if provider_instance is None:
        raise ConfigurationError(
            "No provider configured. Call connect(...) first or pass provider=."
        )

    cache_store = (
        DiskCache(cache_dir) if cache_dir is not None else _resolve_cache_store()
    )
    generated = GeneratedFunction(
        prompt=prompt,
        signature=signature,
        provider=provider_instance,
        cache_store=cache_store,
        cache_enabled=cache,
        mode=mode,
        function_name=function_name or _derive_function_name(prompt),
        system_prompt=system_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        allow_modules=tuple(allow_modules or DEFAULT_ALLOWED_MODULES),
    )
    if mode == "eager":
        generated.build()
    _register_generated_function(generated)
    return generated


def createFunction(*args: Any, **kwargs: Any) -> GeneratedFunction:
    """Compatibility alias for code that prefers camelCase API names."""

    return create_function(*args, **kwargs)


def _resolve_cache_store() -> DiskCache:
    """Return the active cache store, creating a default one if necessary."""

    global _DEFAULT_CACHE

    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = DiskCache()
    return _DEFAULT_CACHE


def _register_generated_function(function: GeneratedFunction) -> None:
    """Track created functions so CLI tooling can materialize them later."""

    _REGISTERED_FUNCTIONS.append(function)


def _registered_functions() -> tuple[GeneratedFunction, ...]:
    """Return the currently tracked generated functions in creation order."""

    return tuple(_REGISTERED_FUNCTIONS)


def _clear_registered_functions() -> None:
    """Reset the in-process generated-function registry."""

    _REGISTERED_FUNCTIONS.clear()


def _derive_function_name(prompt: str) -> str:
    """Create a readable snake_case function name from a prompt string."""

    words = re.findall(r"[A-Za-z0-9]+", prompt.lower())[:6]
    if not words:
        return "generated_function"

    return _normalize_function_name("_".join(words))


def _normalize_function_name(name: str) -> str:
    """Normalize arbitrary text into a valid Python identifier."""

    normalized = re.sub(r"\W+", "_", name.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        normalized = "generated_function"
    if normalized[0].isdigit():
        normalized = f"generated_{normalized}"
    return normalized


def _signature_object_from_text(signature: str) -> inspect.Signature:
    """Convert a textual signature into an inspect-compatible Signature object."""

    namespace: dict[str, Any] = {}
    source = (
        "from __future__ import annotations\n" f"def __temp{signature}:\n    pass\n"
    )
    exec(source, namespace, namespace)
    return inspect.signature(namespace["__temp"])
