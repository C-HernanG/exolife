"""
Individual fetcher implementations for scalable data acquisition.

This package contains focused, testable fetcher implementations that follow
the single responsibility principle. Each fetcher handles a specific type
of data source with clear, maintainable logic.
"""

from .gaia_fetcher import GaiaFetcher
from .http_fetcher import HttpFetcher
from .tap_fetcher import TapFetcher

__all__ = [
    "HttpFetcher",
    "TapFetcher",
    "GaiaFetcher",
]
