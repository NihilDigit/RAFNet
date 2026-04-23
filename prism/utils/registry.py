from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class Registry:
    """Simple name -> constructor mapping."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._builders: Dict[str, Callable[..., Any]] = {}

    def register(self, name: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            key = name or func.__name__
            if key in self._builders:
                raise KeyError(f"{self._name} registry already contains '{key}'")
            self._builders[key] = func
            return func

        return decorator

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name not in self._builders:
            raise KeyError(f"{self._name} registry does not contain '{name}'")
        return self._builders[name](*args, **kwargs)

    def get(self, name: str) -> Callable[..., Any]:
        return self._builders[name]

    def __contains__(self, name: str) -> bool:  # pragma: no cover - simple delegation
        return name in self._builders


__all__ = ["Registry"]
