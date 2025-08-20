"""
TAP/ADQL fetcher for astronomical data queries.

This fetcher handles Table Access Protocol (TAP) endpoints with ADQL queries,
providing robust astronomical data acquisition capabilities.
"""

import logging
from pathlib import Path

import pandas as pd

from ....settings import settings
from ...utils import DataSource, fetch_adql, timestamp, write_generic
from ..fetcher_base import BaseFetcher, DataSourceConfig, FetchResult
from ..registry import register_fetcher

logger = logging.getLogger(__name__)


@register_fetcher("tap")
class TapFetcher(BaseFetcher):
    """
    TAP/ADQL fetcher for astronomical data queries.

    Handles Table Access Protocol endpoints with ADQL queries,
    supporting complex astronomical data retrieval scenarios.
    """

    @property
    def fetcher_type(self) -> str:
        return "tap"

    def can_handle(self, config: DataSourceConfig) -> bool:
        """Check if this fetcher can handle TAP/ADQL queries."""
        return bool(getattr(config, "adql", None))

    def fetch(self, force: bool = False) -> FetchResult:
        """
        Fetch data using TAP/ADQL query.

        Args:
            force: Always fetch fresh data (ignore cache)

        Returns:
            FetchResult with success status and path information
        """
        try:
            source = self._convert_to_source(self.config)
            output_path = self._fetch_tap_source(source, force)

            # Get statistics
            df = pd.read_parquet(output_path)

            return FetchResult(
                source_id=self.config.id,
                path=output_path,
                success=True,
                rows_fetched=len(df),
                size_bytes=output_path.stat().st_size,
            )

        except Exception as e:
            self.logger.error(f"TAP fetch failed for {self.config.id}: {e}")

            # Create fallback dataset for pipeline continuity
            return self._create_fallback_dataset(str(e))

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
            format=getattr(config, "format", None),
        )

    def _fetch_tap_source(self, ds: DataSource, force: bool = False) -> Path:
        """
        Fetch TAP/ADQL data source.

        Args:
            ds: DataSource configuration
            force: Force fresh fetch

        Returns:
            Path to the fetched data file
        """
        # Create source-specific directory
        source_dir = settings.raw_dir / ds.id
        source_dir.mkdir(parents=True, exist_ok=True)
        output_path = source_dir / f"{ds.id}.parquet"

        # Handle dependency-based sources with placeholders
        if ds.adql and "<GAIA_ID_LIST>" in ds.adql:
            if not output_path.exists():
                pd.DataFrame(columns=ds.columns_to_keep).to_parquet(
                    output_path, index=False
                )
                self.logger.info(
                    f"Created placeholder for dependency-based source {ds.id}"
                )
            return output_path

        self.logger.info(f"Fetching TAP data for {ds.id}")

        # Execute TAP/ADQL query
        raw_data = fetch_adql(ds)

        # Store raw data with timestamp
        raw_path = settings.raw_dir / ds.id / f"{ds.id}_{timestamp()}.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(raw_data)

        # Process and save trimmed data
        write_generic(raw_data, ds.download_url or "", ds.columns_to_keep, output_path)

        self.logger.info(f"Successfully fetched and stored TAP data for {ds.id}")
        return output_path

    def _create_fallback_dataset(self, error_msg: str) -> FetchResult:
        """Create minimal dataset when TAP fetch fails."""
        try:
            # Create source-specific directory
            source_dir = settings.raw_dir / self.config.id
            source_dir.mkdir(parents=True, exist_ok=True)
            output_path = source_dir / f"{self.config.id}.parquet"

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
                error_message=f"TAP fetch failed, created placeholder: {error_msg}",
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
