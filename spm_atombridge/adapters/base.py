"""Adapter protocol and capability detection."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod


class Adapter(ABC):
    """Every external engine is wrapped by a subclass of Adapter."""

    name: str = "base"
    requires: tuple[str, ...] = ()
    fallback: str | None = None
    description: str = ""

    @classmethod
    def is_available(cls) -> bool:
        for mod in cls.requires:
            try:
                importlib.import_module(mod)
            except ImportError:
                return False
        return True

    @classmethod
    def capability_report(cls) -> dict:
        return {
            "name": cls.name,
            "description": cls.description,
            "available": cls.is_available(),
            "requires": list(cls.requires),
            "fallback": cls.fallback,
        }

    @abstractmethod
    def run(self, *args, **kwargs):
        """Execute the adapter's main action."""