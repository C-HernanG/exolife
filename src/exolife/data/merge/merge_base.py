"""
Abstract base class and interfaces for data merging strategies.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MergeConfig(BaseModel):
    """
    Configuration for data merging operations.
    """

    strategy: str = Field(..., description="Merge strategy to use")
    output_name: str = Field(..., description="Name for the merged output")
    sources: List[str] = Field(..., description="List of source IDs to merge")
    join_keys: Dict[str, List[str]] = Field(
        default_factory=dict, description="Join keys per source"
    )
    join_type: str = Field(default="inner", description="Type of join operation")
    priority_order: Optional[List[str]] = Field(
        None, description="Priority order for conflict resolution"
    )
    conflict_resolution: str = Field(
        default="left", description="How to resolve column conflicts"
    )
    drop_duplicates: bool = Field(
        default=True, description="Whether to drop duplicate rows"
    )
    validate_merge: bool = Field(
        default=True, description="Whether to validate merge results"
    )

    model_config = {"extra": "allow"}


class MergeResult(BaseModel):
    """
    Result of a merge operation.
    """

    output_name: str
    output_path: Path
    success: bool
    error_message: Optional[str] = None
    input_sources: List[str]
    rows_before: Dict[str, int]
    rows_after: Optional[int] = None
    merge_statistics: Optional[Dict[str, Any]] = None

    model_config = {"arbitrary_types_allowed": True}


class MergeValidationError(Exception):
    """
    Exception raised when merge validation fails.
    """

    pass


class BaseMergeStrategy(ABC):
    """
    Abstract base class for data merge strategies.
    """

    def __init__(self, config: MergeConfig):
        self.config = config
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    @abstractmethod
    def can_handle(self, config: MergeConfig) -> bool:
        """
        Check if this strategy can handle the given merge configuration.
        """
        pass

    @abstractmethod
    def merge(self, data_sources: Dict[str, pd.DataFrame]) -> MergeResult:
        """
        Perform the merge operation on the provided data sources.
        """
        pass

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """
        Return the name identifier for this merge strategy.
        """
        pass

    def validate_config(self) -> bool:
        """
        Validate the merge configuration.
        """
        # Check if all required sources are specified
        if not self.config.sources:
            self.logger.error("No sources specified for merge")
            return False

        # Check join type is valid
        valid_join_types = ["inner", "outer", "left", "right"]
        if self.config.join_type not in valid_join_types:
            self.logger.error(f"Invalid join type: {self.config.join_type}")
            return False

        return True

    def validate_data_sources(self, data_sources: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate that all required data sources are available.
        """
        missing_sources = set(self.config.sources) - set(data_sources.keys())
        if missing_sources:
            self.logger.error(f"Missing data sources: {missing_sources}")
            return False

        # Check that data sources are not empty
        empty_sources = [
            source for source in self.config.sources if data_sources[source].empty
        ]
        if empty_sources:
            self.logger.warning(f"Empty data sources: {empty_sources}")

        return True

    def get_output_path(self, directory: Path) -> Path:
        """
        Get the output path for the merged data.
        """
        return directory / f"{self.config.output_name}.parquet"

    def save_result(self, df: pd.DataFrame, path: Path) -> None:
        """
        Save the merged DataFrame to the specified path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        self.logger.info(f"Saved merged data with {len(df)} rows to {path}")

    def get_merge_statistics(
        self, original_data: Dict[str, pd.DataFrame], merged_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate statistics about the merge operation.
        """
        stats = {
            "input_row_counts": {
                source: len(df) for source, df in original_data.items()
            },
            "output_row_count": len(merged_df),
            "total_input_rows": sum(len(df) for df in original_data.values()),
            "columns_in_output": list(merged_df.columns),
            "merge_efficiency": (
                len(merged_df) / sum(len(df) for df in original_data.values())
                if original_data
                else 0
            ),
        }

        # Check for data loss
        if stats["output_row_count"] == 0:
            stats["warnings"] = ["No rows in merged output"]
        elif stats["merge_efficiency"] < 0.1:
            stats["warnings"] = ["Significant data loss during merge (>90%)"]

        return stats

    def resolve_column_conflicts(
        self, dataframes: List[pd.DataFrame], source_names: List[str]
    ) -> List[pd.DataFrame]:
        """
        Resolve column naming conflicts between dataframes.
        """
        if self.config.conflict_resolution == "prefix":
            # Add source prefix to conflicting columns
            resolved_dfs = []
            all_columns = set()

            # Find conflicting columns
            for df in dataframes:
                all_columns.update(df.columns)

            for df, source_name in zip(dataframes, source_names):
                new_df = df.copy()
                for col in df.columns:
                    # Count how many dataframes have this column
                    count = sum(1 for other_df in dataframes if col in other_df.columns)
                    if count > 1 and col not in self.get_join_keys(source_name):
                        new_df = new_df.rename(columns={col: f"{source_name}_{col}"})
                resolved_dfs.append(new_df)
            return resolved_dfs

        elif self.config.conflict_resolution == "suffix":
            # Add source suffix to conflicting columns
            resolved_dfs = []
            for df, source_name in zip(dataframes, source_names):
                new_df = df.copy()
                for col in df.columns:
                    count = sum(1 for other_df in dataframes if col in other_df.columns)
                    if count > 1 and col not in self.get_join_keys(source_name):
                        new_df = new_df.rename(columns={col: f"{col}_{source_name}"})
                resolved_dfs.append(new_df)
            return resolved_dfs

        else:  # "left" or default - keep left side columns
            return dataframes

    def get_join_keys(self, source_name: str) -> List[str]:
        """
        Get join keys for a specific source.
        """
        return self.config.join_keys.get(source_name, [])

    def validate_join_keys(self, data_sources: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate that join keys exist in the respective data sources.
        """
        for source_name, join_keys in self.config.join_keys.items():
            if source_name not in data_sources:
                continue

            df = data_sources[source_name]
            missing_keys = set(join_keys) - set(df.columns)
            if missing_keys:
                self.logger.error(f"Missing join keys in {source_name}: {missing_keys}")
                return False

        return True

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(strategy={self.config.strategy})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
