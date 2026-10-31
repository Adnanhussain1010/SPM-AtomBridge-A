"""SPM experimental file readers — wraps AFMReader into Image objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spm_atombridge.adapters.base import Adapter
from spm_atombridge.core.image import Image
from spm_atombridge.core.provenance import ProvenanceRecord


_SUFFIX_MAP = {
    ".gwy": "topography",
    ".ibw": "topography",
    ".spm": "topography",
    ".jpk": "topography",
    ".asd": "topography",
}


class SPMReaderAdapter(Adapter):
    """Reads experimental SPM files via AFMReader."""

    name = "afmreader"
    requires = ("afmreader",)
    fallback = "Use Image.from_npy() with a manually exported array"
    description = "Read Gwyddion/Bruker/Igor Pro SPM files into Image objects"

    def read(self, path: str | Path, channel: str = "height") -> Image:
        """Load an SPM file and return an Image.

        Parameters
        ----------
        path : SPM file
        channel : which data channel to extract (e.g. 'height', 'phase',
                  'z', 'current', 'potential')
        """
        p = Path(path)
        suffix = p.suffix.lower()
        kind = _SUFFIX_MAP.get(suffix, "topography")

        if suffix == ".gwy":
            data, meta = self._read_gwy(p, channel)
        elif suffix == ".ibw":
            data, meta = self._read_ibw(p)
        elif suffix == ".spm":
            data, meta = self._read_spm(p)
        else:
            raise ValueError(f"Unsupported SPM format: {suffix}")

        rec = ProvenanceRecord.from_file(
            action="read_spm",
            adapter=self.name,
            path=p,
            parameters={"channel": channel, "format": suffix},
        )
        return Image(data=data, kind=kind, units=meta.get("units", "m"),
                     metadata=meta, provenance=[rec])

    def _read_gwy(self, path: Path, channel: str):
        import afmreader
        # AFMReader API is stable but module paths vary by version
        try:
            from afmreader.gwy import load_gwy
            data, meta = load_gwy(path, channel=channel)
        except (ImportError, AttributeError):
            from afmreader.gwyddion import load_gwyddion
            data, meta = load_gwyddion(path)
        return np.asarray(data), dict(meta) if meta else {}

    def _read_ibw(self, path: Path):
        from afmreader.ibw import load_ibw
        data, meta = load_ibw(path)
        return np.asarray(data), dict(meta) if meta else {}

    def _read_spm(self, path: Path):
        from afmreader.spm import load_spm
        data, meta = load_spm(path)
        return np.asarray(data), dict(meta) if meta else {}

    def run(self, path: str | Path, **kwargs) -> Image:  # pragma: no cover
        return self.read(path, **kwargs)