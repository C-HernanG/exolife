"""
Simplified fetch manager for orchestrating data retrieval operations.

This module provides a clean interface for managing data fetching operations
across multiple sources with parallel processing capabilities.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional

from ..utils import get_data_source, list_data_sources
from .fetcher_base import DataSourceConfig
from .registry import create_fetcher

logger = logging.getLogger(__name__)


class FetchManager:
    """
    Simplified manager for orchestrating data fetch operations.

    Provides a clean interface for fetching single sources or running
    bulk operations across multiple data sources with robust error handling.
    """

    def __init__(self):
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    def fetch_source(self, source_id: str, force: bool = False) -> Path:
        """
        Fetch a single data source.

        Args:
            source_id: ID of the data source to fetch
            force: Force fresh fetch, ignore cache

        Returns:
            Path to the fetched data file

        Raises:
            KeyError: If source_id is not found
            RuntimeError: If fetch operation fails
        """
        try:
            # Get source configuration
            ds = get_data_source(source_id)
        except KeyError:
            raise KeyError(f"Source {source_id} not found in configuration")

        # Create fetcher configuration
        config = self._create_config(ds)

        # Get appropriate fetcher and execute
        fetcher = create_fetcher(config)
        result = fetcher.fetch(force=force)

        if not result.success:
            raise RuntimeError(f"Failed to fetch {source_id}: {result.error_message}")

        self.logger.info(
            f"Successfully fetched {source_id}: {result.rows_fetched} rows"
        )
        return result.path

    def fetch_all(
        self,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        force: bool = False,
    ) -> Dict[str, Path]:
        """
        Fetch all configured data sources.

        Args:
            parallel: Execute fetches in parallel
            max_workers: Maximum number of parallel workers
            force: Force fresh fetch for all sources

        Returns:
            Dictionary mapping source IDs to their data file paths
        """
        source_ids = list_data_sources()
        results: Dict[str, Path] = {}

        self.logger.info(f"Fetching {len(source_ids)} sources (parallel={parallel})...")

        if parallel:
            results = self._fetch_parallel(source_ids, max_workers, force)
        else:
            results = self._fetch_sequential(source_ids, force)

        self.logger.info(f"Completed fetching {len(results)}/{len(source_ids)} sources")
        return results

    def _fetch_parallel(
        self, source_ids: list, max_workers: Optional[int], force: bool
    ) -> Dict[str, Path]:
        """Execute parallel fetch operations."""
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all fetch tasks
            futures = {
                executor.submit(self.fetch_source, source_id, force): source_id
                for source_id in source_ids
            }

            # Collect results as they complete
            for future in as_completed(futures):
                source_id = futures[future]
                try:
                    results[source_id] = future.result()
                except Exception as e:
                    self.logger.error(f"Failed to fetch {source_id}: {e}")

        return results

    def _fetch_sequential(self, source_ids: list, force: bool) -> Dict[str, Path]:
        """Execute sequential fetch operations."""
        results = {}

        for source_id in source_ids:
            try:
                results[source_id] = self.fetch_source(source_id, force)
            except Exception as e:
                self.logger.error(f"Failed to fetch {source_id}: {e}")

        return results

    def _create_config(self, ds) -> DataSourceConfig:
        """Create DataSourceConfig from legacy DataSource object."""
        config = DataSourceConfig(
            id=ds.id,
            name=ds.name,
            description=ds.description,
            download_url=ds.download_url,
            format=getattr(ds, "format", "csv"),
        )

        # Add additional attributes
        config.adql = getattr(ds, "adql", None)
        config.columns_to_keep = getattr(ds, "columns_to_keep", [])
        config.primary_keys = getattr(ds, "primary_keys", [])
        config.join_keys = getattr(ds, "join_keys", {})

        return config


# Global instance for backward compatibility
fetch_manager = FetchManager()


def fetch_source(source_id: str, force: bool = False) -> Path:
    """Convenience function to fetch a single source."""
    return fetch_manager.fetch_source(source_id, force)


def fetch_all_sources(
    parallel: bool = True, max_workers: Optional[int] = None, force: bool = False
) -> Dict[str, Path]:
    """Convenience function to fetch all sources."""
    return fetch_manager.fetch_all(parallel, max_workers, force)
