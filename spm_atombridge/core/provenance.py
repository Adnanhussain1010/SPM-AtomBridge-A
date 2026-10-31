"""Provenance records for reproducible analysis."""

from __future__ import annotations

import datetime
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _runtime_versions() -> dict:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for mod in ("numpy", "scipy", "ase", "pymatgen"):
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", None)
            if v is None:
                try:
                    import importlib.metadata as md
                    v = md.version(mod)
                except Exception:
                    v = "unknown"
            versions[mod] = v
        except ImportError:
            versions[mod] = "not installed"
    return versions


@dataclass
class ProvenanceRecord:
    """A single recorded action in the analysis session."""

    action: str
    adapter: str
    parameters: dict = field(default_factory=dict)
    input_hash: str = ""
    input_path: str = ""
    versions: dict = field(default_factory=_runtime_versions)
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat(timespec="seconds")
    )

    @classmethod
    def from_file(
        cls, action: str, adapter: str, path, parameters=None
    ) -> "ProvenanceRecord":
        p = Path(path)
        return cls(
            action=action,
            adapter=adapter,
            parameters=parameters or {},
            input_hash=_hash_file(p) if p.is_file() else "",
            input_path=str(p.resolve()),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path) -> None:
        Path(path).write_text(self.to_json())
