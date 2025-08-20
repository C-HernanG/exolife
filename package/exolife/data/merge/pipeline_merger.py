"""
Backward compatibility module for the pipeline merger.

This module provides compatibility with existing code that imports
from the old pipeline_merger location.
"""

from .base_merger import MergeConfig
from .mergers.pipeline_merger import PipelineMerger
from .registry import create_merger

# Backward compatibility aliases
ConfigurablePipelineMerger = PipelineMerger


class ConfigurableMergeManager:
    """Backward compatibility wrapper for the merge manager."""

    def __init__(self):
        self.logger = __import__("logging").getLogger(__name__)

    def list_mergers(self):
        """List available merge strategies."""
        from .registry import get_available_strategies

        return get_available_strategies()

    def merge_data(self, method="exolife_merge_v1", overwrite=True):
        """Execute merge using the new system."""
        config = MergeConfig(strategy=method, output_name="exolife_catalog", sources=[])

        merger = create_merger(config)
        result = merger.merge()

        if not result.success:
            raise RuntimeError(f"Merge failed: {result.error_message}")

        # Load and return the result
        import pandas as pd

        return pd.read_parquet(result.output_path)


# Global instances for backward compatibility
merge_manager = ConfigurableMergeManager()


def merge_data(method="exolife_merge_v1", overwrite=True):
    """Backward compatibility function."""
    return merge_manager.merge_data(method, overwrite)


def list_mergers():
    """Backward compatibility function."""
    return merge_manager.list_mergers()
