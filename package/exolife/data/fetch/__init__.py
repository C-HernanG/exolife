"""
Modular and scalable data fetching system.

This package provides a clean, extensible architecture for data fetching
with individual fetchers, simplified registry, and robust management.
"""

# Core components
from .fetcher_base import BaseFetcher, DataSourceConfig, FetchResult

# Individual fetchers (auto-registered via decorators)
from .fetchers import GaiaFetcher, HttpFetcher, TapFetcher

# Management layer
from .manager import FetchManager, fetch_all_sources, fetch_manager, fetch_source

# Registry and factory
from .registry import (
    FetcherRegistry,
    create_fetcher,
    get_fetcher_info,
    list_fetcher_types,
    register_fetcher,
)

__all__ = [
    # Core
    "BaseFetcher",
    "DataSourceConfig",
    "FetchResult",
    # Registry
    "FetcherRegistry",
    "register_fetcher",
    "create_fetcher",
    "list_fetcher_types",
    "get_fetcher_info",
    # Management
    "FetchManager",
    "fetch_source",
    "fetch_all_sources",
    "fetch_manager",
    # Fetchers
    "HttpFetcher",
    "TapFetcher",
    "GaiaFetcher",
]
