"""
Fetcher implementations using the proven logic.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ...settings import settings
from ..utils import (
    DataSource,
    fetch_adql,
    get_data_source,
    list_data_sources,
    stream_download,
    timestamp,
    write_generic,
)
from .fetcher_base import BaseFetcher, DataSourceConfig, FetchResult
from .fetcher_factory import get_fetcher, register_fetcher

logger = logging.getLogger(__name__)


@register_fetcher("http", is_default=True)
class HttpFetcher(BaseFetcher):
    """
    Fetcher for HTTP/HTTPS downloadable data sources.
    """

    @property
    def fetcher_type(self) -> str:
        return "http"

    def can_handle(self, config: DataSourceConfig) -> bool:
        """
        Check if this fetcher can handle the given configuration.
        """
        return bool(
            config.download_url
            and config.download_url.startswith(("http://", "https://"))
        )

    def fetch(self, force: bool = False) -> FetchResult:
        """
        Fetch data using HTTP download.
        """
        try:
            # Get output path
            output_path = settings.raw_dir / f"{self.config.id}.parquet"

            # Check if file already exists and force is not set
            if output_path.exists() and not force:
                self.logger.info(
                    f"File already exists for {self.config.id}, skipping download"
                )
                return FetchResult(
                    source_id=self.config.id,
                    path=output_path,
                    success=True,
                    error_message=None,
                )

            # Use the utility for stream downloading
            self.logger.info(
                f"Downloading {self.config.id} from {self.config.download_url}"
            )

            # Download and save content
            content = stream_download(self.config.download_url)

            # Convert to DataSource for processing with utility functions
            source = self._convert_to_source(self.config)

            # Use write_generic to parse and save as parquet
            write_generic(
                content, self.config.download_url, source.columns_to_keep, output_path
            )

            return FetchResult(
                source_id=self.config.id,
                path=output_path,
                success=True,
                error_message=None,
            )

        except Exception as e:
            self.logger.error(f"Failed to fetch {self.config.id}: {str(e)}")
            return FetchResult(
                source_id=self.config.id, path=None, success=False, error_message=str(e)
            )

    def _convert_to_source(self, config: DataSourceConfig) -> DataSource:
        """
        Convert DataSourceConfig to DataSource format.
        """
        return DataSource(
            id=config.id,
            name=config.name,
            description=config.description,
            download_url=config.download_url,
            adql=getattr(config, "adql", None),
            columns_to_keep=getattr(config, "columns_to_keep", []),
            primary_keys=getattr(config, "primary_keys", []),
            join_keys=getattr(config, "join_keys", {}),
            refresh=getattr(config, "refresh", "as_needed"),
        )


@register_fetcher("tap")
class TapFetcher(BaseFetcher):
    """
    TAP/ADQL fetcher for astronomical data queries.
    Handles TAP (Table Access Protocol) endpoints with ADQL queries.
    """

    @property
    def fetcher_type(self) -> str:
        return "tap"

    def can_handle(self, config: DataSourceConfig) -> bool:
        """
        Check if this fetcher can handle TAP/ADQL queries.
        """
        return bool(getattr(config, "adql", None))

    def fetch(self, force: bool = False) -> FetchResult:
        """
        Fetch data using TAP/ADQL query.
        """
        try:
            source = self._convert_to_source(self.config)
            path = self._fetch_source(source, force)
            df = pd.read_parquet(path)

            return FetchResult(
                source_id=self.config.id,
                path=path,
                success=True,
                rows_fetched=len(df),
                size_bytes=path.stat().st_size,
            )

        except Exception as e:
            logger.error(f"TAP fetch failed for {self.config.id}: {e}")
            return FetchResult(
                source_id=self.config.id, path=None, success=False, error_message=str(e)
            )

    def _convert_to_source(self, config: DataSourceConfig) -> DataSource:
        """
        Convert DataSourceConfig to DataSource format.
        """
        return DataSource(
            id=config.id,
            name=config.name,
            description=config.description,
            download_url=config.download_url,
            adql=getattr(config, "adql", None),
            columns_to_keep=getattr(config, "columns_to_keep", []),
            primary_keys=getattr(config, "primary_keys", []),
            join_keys=getattr(config, "join_keys", {}),
            refresh=getattr(config, "refresh", "static"),
            format=config.format,
        )

    def _fetch_source(self, ds: DataSource, force: bool = False) -> Path:
        """
        Fetch or load a single data source using TAP/ADQL query.
        """
        interim = settings.raw_dir / f"{ds.id}.parquet"

        # handle on_demand ADQL
        if ds.adql and ds.refresh == "on_demand" and not force:
            if interim.exists():
                logger.info("Using cached on_demand TAP source %s", ds.id)
                return interim
            pd.DataFrame(columns=ds.columns_to_keep).to_parquet(interim, index=False)
            return interim

        # cached
        if interim.exists() and not force:
            logger.info("Using cached TAP source %s", ds.id)
            return interim

        # TAP/ADQL query
        raw = fetch_adql(ds)
        raw_path = settings.raw_dir / ds.id / f"{ds.id}_{timestamp()}.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(raw)

        # write trimmed
        write_generic(raw, ds.download_url or "", ds.columns_to_keep, interim)
        logger.info("Fetched and stored TAP data for %s", ds.id)
        return interim


class FetchManager:
    """
    Manager class that provides the fetch_all_sources functionality
    integrating with the scalable fetcher architecture.
    """

    def __init__(self):
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    def fetch_all_sources(
        self,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        force: bool = False,
    ) -> Dict[str, Path]:
        """
        Fetch or load all configured sources using the appropriate fetchers.
        """
        ids = list_data_sources()
        results: Dict[str, Path] = {}

        self.logger.info(
            "Fetching all sources using scalable fetcher system (parallel=%s)...",
            parallel,
        )

        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {
                    ex.submit(self._fetch_single_source, i, force): i for i in ids
                }
                for f in as_completed(futures):
                    source_id = futures[f]
                    try:
                        results[source_id] = f.result()
                    except Exception as e:
                        self.logger.error(f"Failed to fetch {source_id}: {e}")
        else:
            for i in ids:
                try:
                    results[i] = self._fetch_single_source(i, force)
                except Exception as e:
                    self.logger.error(f"Failed to fetch {i}: {e}")

        self.logger.info("Completed fetching %d sources", len(results))
        return results

    def _fetch_single_source(self, source_id: str, force: bool = False) -> Path:
        """
        Fetch a single source using the appropriate fetcher from the registry.
        """
        try:
            ds = get_data_source(source_id)
        except KeyError:
            raise KeyError(f"Source {source_id} not found in configuration")

        # Create configuration for the fetcher
        config = DataSourceConfig(
            id=ds.id,
            name=ds.name,
            description=ds.description,
            download_url=ds.download_url,
            format=ds.format or "csv",
        )

        # Add additional attributes for compatibility
        config.adql = ds.adql
        config.columns_to_keep = ds.columns_to_keep
        config.primary_keys = ds.primary_keys
        config.join_keys = ds.join_keys
        config.refresh = ds.refresh

        # Get the best fetcher for this configuration
        fetcher = get_fetcher(config)
        result = fetcher.fetch(force=force)

        if not result.success:
            raise RuntimeError(f"Failed to fetch {source_id}: {result.error_message}")

        return result.path


# Global instance for easy access
fetch_manager = FetchManager()


def fetch_all_sources(
    parallel: bool = True, max_workers: Optional[int] = None, force: bool = False
) -> Dict[str, Path]:
    """
    Convenience function that wraps the fetch manager.
    """
    return fetch_manager.fetch_all_sources(parallel, max_workers, force)
