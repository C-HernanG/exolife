"""
Modular data merging system for astronomical catalogs.

This package provides a flexible, scalable architecture for merging
astronomical datasets with support for:

- Multiple merging strategies (identifier-based, coordinate-based, catalog-specific)
- Automatic strategy discovery and registration
- Configuration-driven pipeline processing
- Quality assessment and uncertainty propagation
- Provenance tracking and data validation

The system uses a registry pattern for discovering mergers and supports
both simple programmatic usage and complex pipeline configurations.
"""

from .base_merger import BaseMerger, MergeConfig, MergeResult
from .mergers import CatalogMerger, CoordinateMerger, IdentifierMerger, PipelineMerger
from .registry import (
    create_merger,
    get_available_strategies,
    get_merger_info,
    merge_data,
    register_merger,
)

# Backward compatibility with existing code


def list_mergers():
    """List available merge strategies (backward compatibility)."""
    return get_available_strategies()


def get_merge_strategy(config: MergeConfig) -> BaseMerger:
    """Get merge strategy for configuration (backward compatibility)."""
    return create_merger(config)


__all__ = [
    # Core interfaces
    "BaseMerger",
    "MergeConfig",
    "MergeResult",
    # Registry functions
    "create_merger",
    "register_merger",
    "get_available_strategies",
    "get_merger_info",
    "merge_data",
    # Merger implementations
    "IdentifierMerger",
    "CoordinateMerger",
    "CatalogMerger",
    "PipelineMerger",
    # Backward compatibility
    "list_mergers",
    "get_merge_strategy",
]
