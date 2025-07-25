"""
ExoLife: A toolkit for exoplanet habitability estimation.

Subpackages
-----------
- data:        I/O and data ingestion
- evaluation:  Metrics and evaluation routines
- features:    Feature engineering and transformations
- models:      ML model definitions
- pipeline:    Orchestrated end-to-end workflows
- training:    Training loops and utilities
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version(__name__)
except PackageNotFoundError:  # local editable install
    __version__ = "0.0.0-dev"

__all__ = [
    "data",
    # "evaluation",
    # "features",
    # "models",
    # "pipeline",
    # "training",
]

from . import data

# from . import evaluation
# from . import features
# from . import models
# from . import pipeline
# from . import training
