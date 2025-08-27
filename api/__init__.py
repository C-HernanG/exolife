"""
ExoLife REST API package.

This package defines a simple FastAPI application used for
demonstration purposes.  The API provides a health check endpoint
and a root endpoint.  Extend this package with additional routes
to expose model predictions or other functionality.
"""

from .server import app  # noqa: F401
