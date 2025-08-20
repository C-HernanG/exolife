"""
Individual merger implementations for ExoLife data processing.

This package contains specialized merger classes, each focusing on a specific
merging strategy or data source combination. Each merger implements the
BaseMerger interface for consistency and interoperability.
"""

from .catalog_merger import CatalogMerger
from .coordinate_merger import CoordinateMerger
from .identifier_merger import IdentifierMerger
from .pipeline_merger import PipelineMerger

# Backward compatibility function


def merge_data(strategy, overwrite=True):
    """Backward compatibility function for merge_data."""
    from ..registry import merge_data as new_merge_data

    # Map legacy strategy names to current ones
    strategy_mapping = {
        "unified_ingestion": "exolife_merge_v1",
        "unified": "exolife_merge_v1",
        "baseline": "exolife_merge_v1",
        "gaia_enriched": "exolife_merge_v1",
        "comprehensive": "exolife_merge_v1",
        "ingestion": "exolife_merge_v1",
    }

    actual_strategy = strategy_mapping.get(strategy, strategy)
    return new_merge_data(strategy=actual_strategy, output_name="exolife_catalog")


__all__ = [
    "IdentifierMerger",
    "CoordinateMerger",
    "CatalogMerger",
    "PipelineMerger",
    "merge_data",  # Backward compatibility
]
