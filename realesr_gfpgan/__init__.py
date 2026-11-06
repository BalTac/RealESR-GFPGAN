"""RealESR-GFPGAN — unified face + scene restoration."""

__version__ = "2.0.0"

try:  # v1 model needs basicsr; the v2 pipeline does not
    from .model import RealESR_GFPGAN  # noqa: F401
except Exception:  # pragma: no cover - optional heavy dependency
    RealESR_GFPGAN = None

from .pipeline import RestorePipeline, load_bundle  # noqa: F401

__all__ = ["RestorePipeline", "load_bundle", "RealESR_GFPGAN", "__version__"]
