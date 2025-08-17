"""Fetch module for scalable data fetching."""

from .fetcher_base import BaseFetcher, DataSourceConfig, FetchResult
from .fetcher_factory import (
    FetcherRegistry,
    get_fetcher,
    get_fetcher_info,
    list_fetcher_types,
    register_fetcher,
)
from .fetchers import HttpFetcher, TAPFetcher

__all__ = [
    "BaseFetcher",
    "DataSourceConfig",
    "FetchResult",
    "FetcherRegistry",
    "register_fetcher",
    "get_fetcher",
    "list_fetcher_types",
    "get_fetcher_info",
    "HttpFetcher",
    "TAPFetcher",
]
