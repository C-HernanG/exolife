"""
Simple identifier-based merger for exact matches between catalogs.

This merger handles straightforward cross-matching based on exact identifier
matches such as Gaia source IDs, TIC IDs, or normalized planet names.
"""

from typing import Dict, List

import pandas as pd

from exolife.data.utils import gaia_int, norm_name

from ..base_merger import BaseMerger


class IdentifierMerger(BaseMerger):
    """
    Merger for exact identifier-based cross-matching.

    Supports various identifier types including:
    - Gaia source IDs (with DR2/DR3 compatibility)
    - Stellar catalog IDs (TIC, KepID, etc.)
    - Normalized planet names
    - Host star names
    """

    @property
    def merger_name(self) -> str:
        return "Identifier-Based Merger"

    @property
    def supported_strategies(self) -> List[str]:
        return ["gaia_source_id", "exact_name", "stellar_id", "planet_name"]

    def execute_merge(self, data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Execute identifier-based merging."""
        if not data_sources:
            raise ValueError("No data sources provided")

        # Start with the first dataset as primary
        primary_name = list(data_sources.keys())[0]
        merged_df = data_sources[primary_name].copy()

        self.logger.info(
            f"Starting with primary dataset: {primary_name} ({len(merged_df)} rows)"
        )

        # Merge each additional dataset
        for source_name, source_df in list(data_sources.items())[1:]:
            merged_df = self._merge_single_source(merged_df, source_df, source_name)

        return merged_df

    def _merge_single_source(
        self, primary_df: pd.DataFrame, source_df: pd.DataFrame, source_name: str
    ) -> pd.DataFrame:
        """Merge a single source dataset using identifier matching."""
        strategy = self.config.strategy.lower()

        if strategy == "gaia_source_id":
            return self._merge_by_gaia_id(primary_df, source_df, source_name)
        elif strategy == "exact_name":
            return self._merge_by_exact_name(primary_df, source_df, source_name)
        elif strategy == "stellar_id":
            return self._merge_by_stellar_id(primary_df, source_df, source_name)
        elif strategy == "planet_name":
            return self._merge_by_planet_name(primary_df, source_df, source_name)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

    def _merge_by_gaia_id(
        self, primary_df: pd.DataFrame, source_df: pd.DataFrame, source_name: str
    ) -> pd.DataFrame:
        """Merge using Gaia source IDs with DR2/DR3 compatibility."""
        join_keys = self.config.join_keys.get(source_name, ["gaia_id"])
        primary_key = join_keys[0] if len(join_keys) > 0 else "gaia_id"

        # Find appropriate source key
        source_key = None
        for col in source_df.columns:
            if "gaia" in col.lower() and "id" in col.lower():
                source_key = col
                break

        if source_key is None:
            self.logger.warning(f"No Gaia ID column found in {source_name}")
            return primary_df

        # Convert to integer for consistent matching
        primary_df_copy = primary_df.copy()
        source_df_copy = source_df.copy()

        primary_df_copy["_gaia_match_key"] = primary_df_copy[primary_key].apply(
            gaia_int
        )
        source_df_copy["_gaia_match_key"] = source_df_copy[source_key].apply(gaia_int)

        # Add source prefix to avoid column conflicts
        source_columns = {
            col: f"{source_name}_{col}" if col != "_gaia_match_key" else col
            for col in source_df_copy.columns
        }
        source_df_copy = source_df_copy.rename(columns=source_columns)

        # Perform merge
        merged = primary_df_copy.merge(
            source_df_copy, on="_gaia_match_key", how=self.config.join_type
        )

        # Clean up temporary columns
        merged = merged.drop(columns=["_gaia_match_key"])

        self.logger.info(f"Merged {source_name} by Gaia ID: {len(merged)} rows")
        return merged

    def _merge_by_exact_name(
        self, primary_df: pd.DataFrame, source_df: pd.DataFrame, source_name: str
    ) -> pd.DataFrame:
        """Merge using exact name matching with normalization."""
        join_keys = self.config.join_keys.get(source_name, ["name"])
        primary_key = join_keys[0] if len(join_keys) > 0 else "name"

        # Find appropriate source key
        source_key = None
        for col in source_df.columns:
            if "name" in col.lower() and col != primary_key:
                source_key = col
                break

        if source_key is None:
            self.logger.warning(f"No name column found in {source_name}")
            return primary_df

        # Normalize names for matching
        primary_df_copy = primary_df.copy()
        source_df_copy = source_df.copy()

        primary_df_copy["_name_match_key"] = norm_name(primary_df_copy[primary_key])
        source_df_copy["_name_match_key"] = norm_name(source_df_copy[source_key])

        # Add source prefix to avoid column conflicts
        source_columns = {
            col: f"{source_name}_{col}" if col != "_name_match_key" else col
            for col in source_df_copy.columns
        }
        source_df_copy = source_df_copy.rename(columns=source_columns)

        # Perform merge
        merged = primary_df_copy.merge(
            source_df_copy, on="_name_match_key", how=self.config.join_type
        )

        # Clean up temporary columns
        merged = merged.drop(columns=["_name_match_key"])

        self.logger.info(f"Merged {source_name} by exact name: {len(merged)} rows")
        return merged

    def _merge_by_stellar_id(
        self, primary_df: pd.DataFrame, source_df: pd.DataFrame, source_name: str
    ) -> pd.DataFrame:
        """Merge using stellar catalog IDs (TIC, KepID, etc.)."""
        join_keys = self.config.join_keys.get(source_name, ["tic_id"])
        primary_key = join_keys[0] if len(join_keys) > 0 else "tic_id"

        # Find appropriate source key
        source_key = None
        for col in source_df.columns:
            if any(id_type in col.lower() for id_type in ["tic", "kep", "stellar"]):
                source_key = col
                break

        if source_key is None:
            self.logger.warning(f"No stellar ID column found in {source_name}")
            return primary_df

        # Convert to numeric for consistent matching
        primary_df_copy = primary_df.copy()
        source_df_copy = source_df.copy()

        primary_df_copy["_stellar_id_key"] = pd.to_numeric(
            primary_df_copy[primary_key], errors="coerce"
        )
        source_df_copy["_stellar_id_key"] = pd.to_numeric(
            source_df_copy[source_key], errors="coerce"
        )

        # Add source prefix to avoid column conflicts
        source_columns = {
            col: f"{source_name}_{col}" if col != "_stellar_id_key" else col
            for col in source_df_copy.columns
        }
        source_df_copy = source_df_copy.rename(columns=source_columns)

        # Perform merge
        merged = primary_df_copy.merge(
            source_df_copy, on="_stellar_id_key", how=self.config.join_type
        )

        # Clean up temporary columns
        merged = merged.drop(columns=["_stellar_id_key"])

        self.logger.info(f"Merged {source_name} by stellar ID: {len(merged)} rows")
        return merged

    def _merge_by_planet_name(
        self, primary_df: pd.DataFrame, source_df: pd.DataFrame, source_name: str
    ) -> pd.DataFrame:
        """Merge using planet names with normalization."""
        join_keys = self.config.join_keys.get(source_name, ["pl_name"])
        primary_key = join_keys[0] if len(join_keys) > 0 else "pl_name"

        # Find appropriate source key
        source_key = None
        for col in source_df.columns:
            if "pl_name" in col.lower() or "planet" in col.lower():
                source_key = col
                break

        if source_key is None:
            self.logger.warning(f"No planet name column found in {source_name}")
            return primary_df

        # Normalize planet names for matching
        primary_df_copy = primary_df.copy()
        source_df_copy = source_df.copy()

        primary_df_copy["_planet_match_key"] = norm_name(primary_df_copy[primary_key])
        source_df_copy["_planet_match_key"] = norm_name(source_df_copy[source_key])

        # Add source prefix to avoid column conflicts
        source_columns = {
            col: f"{source_name}_{col}" if col != "_planet_match_key" else col
            for col in source_df_copy.columns
        }
        source_df_copy = source_df_copy.rename(columns=source_columns)

        # Perform merge
        merged = primary_df_copy.merge(
            source_df_copy, on="_planet_match_key", how=self.config.join_type
        )

        # Clean up temporary columns
        merged = merged.drop(columns=["_planet_match_key"])

        self.logger.info(f"Merged {source_name} by planet name: {len(merged)} rows")
        return merged
