"""
Catalog-specific merger for astronomical survey data.

This merger handles domain-specific logic for different astronomical catalogs,
including candidate validation, quality assessment, and provenance tracking.
"""

from typing import Dict, List, Optional

import pandas as pd

from exolife.data.utils import gaia_int, norm_name

from ..base_merger import BaseMerger


class CatalogMerger(BaseMerger):
    """
    Merger specialized for astronomical catalog integration.

    Features:
    - Catalog-specific validation logic
    - Candidate disposition handling
    - Quality flag propagation
    - Provenance tracking
    """

    @property
    def merger_name(self) -> str:
        return "Catalog-Specific Merger"

    @property
    def supported_strategies(self) -> List[str]:
        return [
            "exoplanet_catalog",
            "candidate_validation",
            "stellar_parameters",
            "mission_data",
        ]

    def execute_merge(self, data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Execute catalog-specific merging with domain logic."""
        if not data_sources:
            raise ValueError("No data sources provided")

        # Identify primary catalog
        primary_catalog = self._identify_primary_catalog(data_sources)
        merged_df = data_sources[primary_catalog].copy()

        self.logger.info(
            f"Using {primary_catalog} as primary catalog ({len(merged_df)} rows)"
        )

        # Add record IDs for tracking
        merged_df = self._add_record_identifiers(merged_df)

        # Merge catalogs in priority order
        for catalog_name, catalog_df in data_sources.items():
            if catalog_name != primary_catalog:
                merged_df = self._merge_catalog(merged_df, catalog_df, catalog_name)

        # Apply catalog-specific post-processing
        merged_df = self._apply_catalog_logic(merged_df)

        return merged_df

    def _identify_primary_catalog(self, data_sources: Dict[str, pd.DataFrame]) -> str:
        """Identify the primary catalog based on completeness and size."""
        # Prefer NASA Exoplanet Archive as primary if available
        nasa_catalogs = [
            name
            for name in data_sources.keys()
            if "nasa" in name.lower() or "exoplanet" in name.lower()
        ]

        if nasa_catalogs:
            # Choose the largest NASA catalog
            return max(nasa_catalogs, key=lambda x: len(data_sources[x]))

        # Otherwise, choose the largest catalog
        return max(data_sources.keys(), key=lambda x: len(data_sources[x]))

    def _add_record_identifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add unique record identifiers for tracking."""
        if "record_id" not in df.columns:
            if "pl_name" in df.columns:
                df["record_id"] = (
                    df.index.astype(str) + "_" + df["pl_name"].fillna("unknown")
                )
            else:
                df["record_id"] = df.index.astype(str)

        return df

    def _merge_catalog(
        self, primary_df: pd.DataFrame, catalog_df: pd.DataFrame, catalog_name: str
    ) -> pd.DataFrame:
        """Merge a single catalog using appropriate strategy."""
        # Determine merge strategy based on catalog type
        if self._is_gaia_catalog(catalog_name):
            return self._merge_gaia_catalog(primary_df, catalog_df, catalog_name)
        elif self._is_candidate_catalog(catalog_name):
            return self._merge_candidate_catalog(primary_df, catalog_df, catalog_name)
        elif self._is_stellar_catalog(catalog_name):
            return self._merge_stellar_catalog(primary_df, catalog_df, catalog_name)
        else:
            return self._merge_generic_catalog(primary_df, catalog_df, catalog_name)

    def _is_gaia_catalog(self, catalog_name: str) -> bool:
        """Check if catalog is a Gaia catalog."""
        return "gaia" in catalog_name.lower()

    def _is_candidate_catalog(self, catalog_name: str) -> bool:
        """Check if catalog contains candidate data."""
        candidate_keywords = ["toi", "koi", "candidate", "tess", "kepler"]
        return any(keyword in catalog_name.lower() for keyword in candidate_keywords)

    def _is_stellar_catalog(self, catalog_name: str) -> bool:
        """Check if catalog is primarily stellar parameters."""
        stellar_keywords = ["sweet", "stellar", "hipparcos", "tycho"]
        return any(keyword in catalog_name.lower() for keyword in stellar_keywords)

    def _merge_gaia_catalog(
        self, primary_df: pd.DataFrame, gaia_df: pd.DataFrame, catalog_name: str
    ) -> pd.DataFrame:
        """Merge Gaia catalog using source ID matching."""
        # Find Gaia ID columns
        primary_gaia_col = self._find_gaia_column(primary_df)
        gaia_id_col = self._find_gaia_column(gaia_df)

        if not primary_gaia_col or not gaia_id_col:
            self.logger.warning(f"Cannot find Gaia ID columns for {catalog_name}")
            return primary_df

        # Convert to integers for matching
        primary_df_copy = primary_df.copy()
        gaia_df_copy = gaia_df.copy()

        primary_df_copy["_gaia_match"] = primary_df_copy[primary_gaia_col].apply(
            gaia_int
        )
        gaia_df_copy["_gaia_match"] = gaia_df_copy[gaia_id_col].apply(gaia_int)

        # Add catalog prefix to avoid conflicts
        gaia_columns = {
            col: f"{catalog_name}_{col}" if col != "_gaia_match" else col
            for col in gaia_df_copy.columns
        }
        gaia_df_copy = gaia_df_copy.rename(columns=gaia_columns)

        # Merge and clean up
        merged = primary_df_copy.merge(gaia_df_copy, on="_gaia_match", how="left")
        merged = merged.drop(columns=["_gaia_match"])

        # Add Gaia-specific quality flags
        merged = self._add_gaia_quality_flags(merged, catalog_name)

        self.logger.info(f"Merged Gaia catalog {catalog_name}: {len(merged)} rows")
        return merged

    def _merge_candidate_catalog(
        self, primary_df: pd.DataFrame, candidate_df: pd.DataFrame, catalog_name: str
    ) -> pd.DataFrame:
        """Merge candidate catalog with disposition handling."""
        # Use coordinate matching for candidates
        from .coordinate_merger import CoordinateMerger

        # Create temporary config for coordinate merger
        coord_config = self.config.model_copy()
        coord_config.strategy = "coordinate_match"

        coord_merger = CoordinateMerger(coord_config)
        merged = coord_merger._merge_by_coordinates(
            primary_df, candidate_df, catalog_name
        )

        # Add candidate-specific flags
        merged = self._add_candidate_flags(merged, catalog_name)

        self.logger.info(f"Merged candidate catalog {catalog_name}: {len(merged)} rows")
        return merged

    def _merge_stellar_catalog(
        self, primary_df: pd.DataFrame, stellar_df: pd.DataFrame, catalog_name: str
    ) -> pd.DataFrame:
        """Merge stellar catalog using hostname matching."""
        # Find hostname columns
        primary_host_col = self._find_hostname_column(primary_df)
        stellar_host_col = self._find_hostname_column(stellar_df)

        if not primary_host_col or not stellar_host_col:
            self.logger.warning(f"Cannot find hostname columns for {catalog_name}")
            return primary_df

        # Normalize hostnames for matching
        primary_df_copy = primary_df.copy()
        stellar_df_copy = stellar_df.copy()

        primary_df_copy["_host_match"] = norm_name(primary_df_copy[primary_host_col])
        stellar_df_copy["_host_match"] = norm_name(stellar_df_copy[stellar_host_col])

        # Add catalog prefix
        stellar_columns = {
            col: f"{catalog_name}_{col}" if col != "_host_match" else col
            for col in stellar_df_copy.columns
        }
        stellar_df_copy = stellar_df_copy.rename(columns=stellar_columns)

        # Merge and clean up
        merged = primary_df_copy.merge(stellar_df_copy, on="_host_match", how="left")
        merged = merged.drop(columns=["_host_match"])

        self.logger.info(f"Merged stellar catalog {catalog_name}: {len(merged)} rows")
        return merged

    def _merge_generic_catalog(
        self, primary_df: pd.DataFrame, catalog_df: pd.DataFrame, catalog_name: str
    ) -> pd.DataFrame:
        """Generic merge strategy for unknown catalog types."""
        # Try planet name matching first
        if "pl_name" in primary_df.columns and any(
            "name" in col.lower() for col in catalog_df.columns
        ):
            name_col = next(col for col in catalog_df.columns if "name" in col.lower())

            primary_df_copy = primary_df.copy()
            catalog_df_copy = catalog_df.copy()

            primary_df_copy["_name_match"] = norm_name(primary_df_copy["pl_name"])
            catalog_df_copy["_name_match"] = norm_name(catalog_df_copy[name_col])

            # Add catalog prefix
            catalog_columns = {
                col: f"{catalog_name}_{col}" if col != "_name_match" else col
                for col in catalog_df_copy.columns
            }
            catalog_df_copy = catalog_df_copy.rename(columns=catalog_columns)

            # Merge and clean up
            merged = primary_df_copy.merge(
                catalog_df_copy, on="_name_match", how="left"
            )
            merged = merged.drop(columns=["_name_match"])

            self.logger.info(f"Merged {catalog_name} by name: {len(merged)} rows")
            return merged

        self.logger.warning(f"Could not determine merge strategy for {catalog_name}")
        return primary_df

    def _find_gaia_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find Gaia ID column in DataFrame."""
        for col in df.columns:
            if "gaia" in col.lower() and "id" in col.lower():
                return col
        return None

    def _find_hostname_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find hostname column in DataFrame."""
        for col in df.columns:
            if col.lower() in ["hostname", "host", "star_name", "name"]:
                return col
        return None

    def _add_gaia_quality_flags(
        self, df: pd.DataFrame, catalog_name: str
    ) -> pd.DataFrame:
        """Add Gaia-specific quality assessment flags."""
        # RUWE quality flag
        ruwe_col = f"{catalog_name}_ruwe"
        if ruwe_col in df.columns:
            df[f"{catalog_name}_astrometry_quality"] = df[ruwe_col].apply(
                lambda x: (
                    "good"
                    if pd.notna(x) and x < 1.4
                    else "poor" if pd.notna(x) else "unknown"
                )
            )

        # Parallax quality flag
        parallax_col = f"{catalog_name}_parallax"
        parallax_err_col = f"{catalog_name}_parallax_error"
        if parallax_col in df.columns and parallax_err_col in df.columns:
            df[f"{catalog_name}_parallax_snr"] = df[parallax_col] / df[parallax_err_col]
            df[f"{catalog_name}_parallax_quality"] = df[
                f"{catalog_name}_parallax_snr"
            ].apply(
                lambda x: (
                    "good"
                    if pd.notna(x) and x > 5.0
                    else "poor" if pd.notna(x) else "unknown"
                )
            )

        return df

    def _add_candidate_flags(self, df: pd.DataFrame, catalog_name: str) -> pd.DataFrame:
        """Add candidate-specific validation flags."""
        if "toi" in catalog_name.lower():
            # TESS candidate flags
            disp_col = f"{catalog_name}_tfopwg_disp"
            if disp_col in df.columns:
                df["tess_candidate_flag"] = df[disp_col].notna()
                df["tess_disposition"] = df[disp_col]

                # Candidate confidence levels
                df["tess_candidate_confidence"] = df[disp_col].apply(
                    lambda x: (
                        "high"
                        if x in ["PC", "CP"]
                        else (
                            "medium" if x == "APC" else "low" if pd.notna(x) else "none"
                        )
                    )
                )

        elif "koi" in catalog_name.lower():
            # Kepler candidate flags
            disp_col = f"{catalog_name}_koi_disposition"
            if disp_col in df.columns:
                df["kepler_candidate_flag"] = df[disp_col] == "CANDIDATE"
                df["kepler_disposition"] = df[disp_col]

                # False positive flags
                fp_cols = [col for col in df.columns if "fpflag" in col.lower()]
                if fp_cols:
                    df["kepler_fp_flag_count"] = df[fp_cols].notna().sum(axis=1)

        return df

    def _apply_catalog_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply catalog-specific business logic and validation."""
        # Ensure required identifiers exist
        if "exolife_star_id" not in df.columns:
            df["exolife_star_id"] = df.index.astype(str)

        if "exolife_planet_id" not in df.columns:
            df["exolife_planet_id"] = df.index.astype(str)

        # Add data provenance tracking
        df = self._add_provenance_tracking(df)

        # Apply quality assessments
        df = self._compute_overall_quality_scores(df)

        return df

    def _add_provenance_tracking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add provenance information for each measurement."""
        # Track which catalogs contributed data for each row
        catalog_columns = [col for col in df.columns if "_" in col]
        catalog_names = list(set(col.split("_")[0] for col in catalog_columns))

        for catalog in catalog_names:
            catalog_cols = [col for col in df.columns if col.startswith(f"{catalog}_")]
            if catalog_cols:
                df[f"has_{catalog}_data"] = df[catalog_cols].notna().any(axis=1)

        return df

    def _compute_overall_quality_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute overall data quality scores."""
        # Count available key parameters
        key_params = [
            "pl_rade",
            "pl_masse",
            "pl_orbsmax",
            "st_teff",
            "st_lum",
            "st_mass",
        ]
        available_params = [col for col in key_params if col in df.columns]

        if available_params:
            df["data_completeness_score"] = df[available_params].notna().sum(
                axis=1
            ) / len(available_params)

        # Overall quality assessment
        quality_factors = []

        if "data_completeness_score" in df.columns:
            quality_factors.append(df["data_completeness_score"])

        # Add Gaia quality if available
        gaia_quality_cols = [
            col
            for col in df.columns
            if "gaia" in col.lower() and "quality" in col.lower()
        ]
        for col in gaia_quality_cols:
            quality_factors.append((df[col] == "good").astype(float))

        if quality_factors:
            df["overall_quality_score"] = pd.concat(quality_factors, axis=1).mean(
                axis=1
            )

        return df
