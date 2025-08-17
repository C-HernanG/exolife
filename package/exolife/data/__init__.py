"""
ExoLife data package: fetching, merging, and preprocessing.
"""

from .fetch import BaseFetcher, DataSourceConfig, FetcherRegistry, register_fetcher
from .merge import BaseMergeStrategy, MergeConfig, MergeStrategyFactory
from .preprocess import BasePreprocessor, PreprocessorConfig
from .utils import list_data_sources, load_config

__all__ = [
    "BaseFetcher",
    "DataSourceConfig",
    "FetcherRegistry",
    "register_fetcher",
    "BaseMergeStrategy",
    "MergeConfig",
    "MergeStrategyFactory",
    "BasePreprocessor",
    "PreprocessorConfig",
    "list_data_sources",
    "load_config",
]
