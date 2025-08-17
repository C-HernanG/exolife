"""Merge module for strategy-based data merging."""

from .merge_base import BaseMergeStrategy, MergeConfig, MergeResult
from .merge_factory import (
    MergeStrategyFactory,
    get_merge_strategy,
    get_merge_strategy_info,
    register_custom_strategy,
)

__all__ = [
    "BaseMergeStrategy",
    "MergeConfig",
    "MergeResult",
    "MergeStrategyFactory",
    "get_merge_strategy",
    "get_merge_strategy_info",
    "register_custom_strategy",
]
