"""
ExoLife: A toolkit for exoplanet habitability estimation.

Subpackages
-----------
- data:        I/O and data ingestion
- pipeline:    Orchestrated end-to-end workflows
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version(__name__)
except PackageNotFoundError:  # local editable install
    __version__ = "0.0.0-dev"

__all__ = [
    "data",
    "pipeline",
]

from . import data, pipeline
