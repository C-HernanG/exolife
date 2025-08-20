"""
HTTP/HTTPS data fetcher for downloadable data sources.

This fetcher handles standard HTTP downloads with robust error handling
and automatic fallback for offline scenarios.
"""

import logging
from pathlib import Path

import pandas as pd

from ....settings import settings
from ...utils import DataSource, stream_download, write_generic
from ..fetcher_base import BaseFetcher, DataSourceConfig, FetchResult
from ..registry import register_fetcher

logger = logging.getLogger(__name__)


@register_fetcher("http", is_default=True)
class HttpFetcher(BaseFetcher):
    """
    Fetcher for HTTP/HTTPS downloadable data sources.

    Handles standard web downloads with automatic content type detection,
    robust error handling, and offline fallback capabilities.
    """

    @property
    def fetcher_type(self) -> str:
        return "http"

    def can_handle(self, config: DataSourceConfig) -> bool:
        """Check if this fetcher can handle HTTP/HTTPS URLs."""
        return bool(
            config.download_url
            and config.download_url.startswith(("http://", "https://"))
        )

    def fetch(self, force: bool = False) -> FetchResult:
        """
        Fetch data using HTTP download with robust error handling.

        Args:
            force: Always fetch fresh data (ignore cache)

        Returns:
            FetchResult with success status and path information
        """
        # Create source-specific directory
        source_dir = settings.raw_dir / self.config.id
        source_dir.mkdir(parents=True, exist_ok=True)
        output_path = source_dir / f"{self.config.id}.parquet"

        try:
            self.logger.info(
                f"Downloading {self.config.id} from {self.config.download_url}"
            )

            # Download content
            content = stream_download(self.config.download_url)

            # Convert config to DataSource for utility functions
            source = self._convert_to_source(self.config)

            # Parse and save as parquet
            write_generic(
                content, self.config.download_url, source.columns_to_keep, output_path
            )

            # Get file statistics
            df = pd.read_parquet(output_path)

            return FetchResult(
                source_id=self.config.id,
                path=output_path,
                success=True,
                rows_fetched=len(df),
                size_bytes=output_path.stat().st_size,
            )

        except Exception as e:
            self.logger.error(f"HTTP fetch failed for {self.config.id}: {e}")

            # Create fallback dataset for pipeline continuity
            return self._create_fallback_dataset(output_path, str(e))

    def _convert_to_source(self, config: DataSourceConfig) -> DataSource:
        """Convert DataSourceConfig to legacy DataSource format."""
        return DataSource(
            id=config.id,
            name=config.name,
            description=config.description,
            download_url=config.download_url,
            adql=getattr(config, "adql", None),
            columns_to_keep=getattr(config, "columns_to_keep", []),
            primary_keys=getattr(config, "primary_keys", []),
            join_keys=getattr(config, "join_keys", {}),
        )

    def _create_fallback_dataset(
        self, output_path: Path, error_msg: str
    ) -> FetchResult:
        """Create minimal dataset when download fails."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Create minimal DataFrame with expected schema
            cols = getattr(self.config, "columns_to_keep", []) or []
            data_dict = {col: [pd.NA] for col in cols}

            # Fill primary keys with dummy values
            for pk in getattr(self.config, "primary_keys", []) or []:
                if pk in data_dict:
                    data_dict[pk] = [f"dummy_{self.config.id}"]

            empty_df = pd.DataFrame(data_dict)
            empty_df.to_parquet(output_path, index=False)

            return FetchResult(
                source_id=self.config.id,
                path=output_path,
                success=True,
                error_message=f"Download failed, created placeholder: {error_msg}",
                rows_fetched=len(empty_df),
                size_bytes=output_path.stat().st_size,
            )

        except Exception as create_err:
            return FetchResult(
                source_id=self.config.id,
                path=None,
                success=False,
                error_message=f"Failed to create fallback: {create_err}",
            )
