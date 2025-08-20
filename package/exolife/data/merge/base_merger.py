"""
Base merger interface and configuration classes for data merging operations.

This module provides a simplified, more intuitive base class for implementing
data mergers, replacing the previous abstract approach with a more practical
template method pattern.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field


class MergeConfig(BaseModel):
    """Configuration for merge operations with sensible defaults."""

    strategy: str = Field(..., description="Merge strategy identifier")
    output_name: str = Field(..., description="Output file base name")
    sources: List[str] = Field(default_factory=list, description="Source catalog IDs")
    join_keys: Dict[str, List[str]] = Field(
        default_factory=dict, description="Join keys per source"
    )
    join_type: str = Field(default="left", description="Join operation type")
    conflict_resolution: str = Field(
        default="left", description="Column conflict resolution strategy"
    )
    drop_duplicates: bool = Field(default=True, description="Remove duplicate rows")
    validate_results: bool = Field(default=True, description="Validate merge results")

    model_config = {"extra": "allow"}


class MergeResult(BaseModel):
    """Result container for merge operations."""

    output_name: str
    output_path: Path
    success: bool
    rows_processed: int
    sources_merged: List[str]
    execution_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    statistics: Optional[Dict[str, Any]] = None

    model_config = {"arbitrary_types_allowed": True}


class BaseMerger(ABC):
    """
    Base class for all data mergers with common functionality.

    This class provides a template method pattern where subclasses implement
    specific merging logic while inheriting common data loading, validation,
    and saving functionality.
    """

    def __init__(self, config: MergeConfig):
        self.config = config
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    @property
    @abstractmethod
    def merger_name(self) -> str:
        """Return the human-readable name of this merger."""
        pass

    @property
    @abstractmethod
    def supported_strategies(self) -> List[str]:
        """Return list of strategy names this merger can handle."""
        pass

    def can_handle(self, strategy: str) -> bool:
        """Check if this merger can handle the given strategy."""
        return strategy.lower() in [s.lower() for s in self.supported_strategies]

    @abstractmethod
    def execute_merge(self, data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Implement the core merging logic.

        Args:
            data_sources: Dictionary mapping source IDs to DataFrames

        Returns:
            Merged DataFrame
        """
        pass

    def merge(
        self, data_sources: Optional[Dict[str, pd.DataFrame]] = None
    ) -> MergeResult:
        """
        Execute the complete merge operation with error handling and logging.

        This is the main entry point that coordinates the entire merge process.
        """
        import time

        start_time = time.time()

        try:
            self.logger.info(f"Starting {self.merger_name} merge operation")

            # Load data sources if not provided
            if data_sources is None:
                data_sources = self._load_data_sources()

            # Validate inputs
            self._validate_inputs(data_sources)

            # Execute the merge
            merged_df = self.execute_merge(data_sources)

            # Post-process results
            merged_df = self._post_process(merged_df)

            # Save results
            output_path = self._save_results(merged_df)

            execution_time = time.time() - start_time

            result = MergeResult(
                output_name=self.config.output_name,
                output_path=output_path,
                success=True,
                rows_processed=len(merged_df),
                sources_merged=list(data_sources.keys()),
                execution_time_seconds=execution_time,
                statistics=self._compute_statistics(merged_df, data_sources),
            )

            self.logger.info(f"Merge completed successfully in {execution_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Merge operation failed: {e}")
            return MergeResult(
                output_name=self.config.output_name,
                output_path=Path("/tmp/failed"),
                success=False,
                rows_processed=0,
                sources_merged=[],
                error_message=str(e),
            )

    def _load_data_sources(self) -> Dict[str, pd.DataFrame]:
        """Load data sources from disk based on configuration."""
        from exolife.settings import Settings

        settings = Settings()

        data_sources = {}

        for source_id in self.config.sources:
            # Try parquet first, fallback to CSV in source subfolder
            source_dir = settings.raw_dir / source_id
            parquet_path = source_dir / f"{source_id}.parquet"
            csv_path = source_dir / f"{source_id}.csv"

            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                self.logger.debug(f"Loaded {source_id} from parquet: {len(df)} rows")
            elif csv_path.exists():
                df = pd.read_csv(csv_path)
                self.logger.debug(f"Loaded {source_id} from CSV: {len(df)} rows")
            else:
                self.logger.warning(f"Data source not found: {source_id}")
                continue

            data_sources[source_id] = df

        return data_sources

    def _validate_inputs(self, data_sources: Dict[str, pd.DataFrame]) -> None:
        """Validate input data sources."""
        if not data_sources:
            raise ValueError("No data sources provided")

        # Check for empty datasets
        empty_sources = [name for name, df in data_sources.items() if df.empty]
        if empty_sources:
            self.logger.warning(f"Empty data sources detected: {empty_sources}")

        # Validate join keys exist
        for source_id, join_keys in self.config.join_keys.items():
            if source_id in data_sources:
                df = data_sources[source_id]
                missing_keys = set(join_keys) - set(df.columns)
                if missing_keys:
                    raise ValueError(
                        f"Missing join keys in {source_id}: {missing_keys}"
                    )

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply post-processing steps to merged data."""
        if self.config.drop_duplicates:
            initial_rows = len(df)
            df = df.drop_duplicates()
            if len(df) < initial_rows:
                self.logger.info(f"Removed {initial_rows - len(df)} duplicate rows")

        return df

    def _save_results(self, df: pd.DataFrame) -> Path:
        """Save merged results to disk in interim directory."""
        from exolife.settings import Settings

        settings = Settings()

        # Save to interim directory first - this is raw merged data
        stage_dir = settings.interim_dir / "01_initial_merge"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Use a generic interim filename, not the final output name
        interim_filename = "merged_unified_ingestion"
        output_path = stage_dir / f"{interim_filename}.parquet"

        # Save parquet format only
        df.to_parquet(output_path, index=False)

        self.logger.info(f"Saved interim results to {output_path}: {len(df)} rows")

        return output_path

    def _save_final_catalog(self, df: pd.DataFrame) -> Path:
        """Save final catalog to processed directory."""
        from exolife.settings import Settings

        settings = Settings()

        # Save final catalog to processed/exolife_catalog directory
        catalog_dir = settings.processed_dir / "exolife_catalog"
        catalog_dir.mkdir(parents=True, exist_ok=True)

        output_path = catalog_dir / f"{self.config.output_name}.parquet"

        # Save parquet format only
        df.to_parquet(output_path, index=False)

        self.logger.info(f"Saved final catalog to {output_path}: {len(df)} rows")

        return output_path

    def _compute_statistics(
        self, merged_df: pd.DataFrame, data_sources: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Compute merge operation statistics."""
        return {
            "output_rows": len(merged_df),
            "output_columns": len(merged_df.columns),
            "input_sources": {name: len(df) for name, df in data_sources.items()},
            "total_input_rows": sum(len(df) for df in data_sources.values()),
            "merge_efficiency": (
                len(merged_df) / sum(len(df) for df in data_sources.values())
                if data_sources
                else 0
            ),
        }


class MergeError(Exception):
    """Base exception for merge operations."""

    pass


class ConfigurationError(MergeError):
    """Exception raised for configuration-related errors."""

    pass


class DataValidationError(MergeError):
    """Exception raised for data validation errors."""

    pass
