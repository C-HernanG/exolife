"""
ExoLife data package: fetching, merging, and preprocessing.
"""

from .fetch import BaseFetcher, DataSourceConfig, FetcherRegistry, register_fetcher
from .merge import BaseMerger, MergeConfig, create_merger
from .preprocess import BasePreprocessor, PreprocessorConfig
from .utils import list_data_sources, load_config

# Backward compatibility aliases
BaseMergeStrategy = BaseMerger
MergeStrategyFactory = create_merger

__all__ = [
    "BaseFetcher",
    "DataSourceConfig",
    "FetcherRegistry",
    "register_fetcher",
    "BaseMerger",
    "MergeConfig",
    "create_merger",
    "BasePreprocessor",
    "PreprocessorConfig",
    "list_data_sources",
    "load_config",
    # Backward compatibility
    "BaseMergeStrategy",
    "MergeStrategyFactory",
]
