"""OVITO adapter: structure visualization (free Python module)."""

from __future__ import annotations

from pathlib import Path

from spm_atombridge.adapters.base import Adapter
from spm_atombridge.core.structure import Structure


class OvitoAdapter(Adapter):
    name = "ovito"
    requires = ("ovito",)
    fallback = "py3dmol or matplotlib 3D scatter"
    description = "Structure rendering via OVITO free Python module"

    def render_png(
        self,
        structure: Structure,
        output: str | Path,
        width: int = 1200,
        height: int = 900,
    ) -> Path:
        """Render structure to PNG using OVITO free OpenGL renderer."""
        from ovito.io import import_file
        from ovito.io.ase import ase_to_ovito
        from ovito.vis import Viewport, OpenGLRenderer

        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)

        data = ase_to_ovito(structure.atoms)
        pipeline = import_file.__self__ if hasattr(import_file, "__self__") else None

        # Minimal pipeline via direct data collection
        from ovito.pipeline import Pipeline, StaticSource
        pl = Pipeline(source=StaticSource(data=data))
        pl.add_to_scene()

        vp = Viewport(type=Viewport.Type.PERSPECTIVE)
        vp.zoom_all()
        vp.render_image(
            filename=str(out),
            size=(width, height),
            renderer=OpenGLRenderer(),
        )
        return out

    def run(self, *args, **kwargs):  # pragma: no cover
        return self.render_png(*args, **kwargs)