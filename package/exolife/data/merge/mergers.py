"""
Configuration-driven merge strategy for ExoLife data processing pipeline.

This module provides a single, comprehensive merger that is fully configurable
through YAML files in the config/merges directory.
"""

import logging
from typing import List

import pandas as pd

from .pipeline_merger import ConfigurableMergeManager

logger = logging.getLogger(__name__)


class MergeManager:
    """
    Configuration-driven merge manager that delegates to the pipeline merger.

    This maintains backward compatibility while using the new configurable approach.
    """

    def __init__(self):
        self.pipeline_manager = ConfigurableMergeManager()
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    def list_mergers(self) -> List[str]:
        """
        List available merge strategies from configuration files.
        """
        return self.pipeline_manager.list_mergers()

    def merge_data(self, method: str, overwrite: bool = True) -> pd.DataFrame:
        """
        Execute merge using the configurable pipeline.

        Args:
            method: Configuration name (maps to config file in config/merges/)
            overwrite: Whether to overwrite existing results

        Returns:
            Merged DataFrame from configurable pipeline
        """
        # Map legacy method names to default configuration
        method_mapping = {
            "baseline": "exolife_pipeline",
            "gaia_enriched": "exolife_pipeline",
            "comprehensive": "exolife_pipeline",
            "unified": "exolife_pipeline",
            "unified_ingestion": "exolife_pipeline",
            "ingestion": "exolife_pipeline",
        }

        actual_method = method_mapping.get(method.lower(), method.lower())

        self.logger.info(f"Executing merge with method: {method} -> {actual_method}")

        return self.pipeline_manager.merge_data(actual_method, overwrite)


# Global instance for easy access
merge_manager = MergeManager()


def merge_data(method: str, overwrite: bool = True) -> pd.DataFrame:
    """
    Convenience function that wraps the merge manager.

    Note: All methods now use the configurable pipeline.
    """
    return merge_manager.merge_data(method, overwrite)


def list_mergers() -> List[str]:
    """
    Convenience function to list available merge strategies.
    """
    return merge_manager.list_mergers()
