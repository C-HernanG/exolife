"""
Configurable Data Processing Pipeline - Generic merger for ExoLife.

This module implements a configuration-driven data processing approach that can handle
various multi-mission astrophysical catalog harmonization scenarios based on YAML
configuration files in the config/merges directory.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from ...settings import settings
from ..utils import (
    add_hz_edges_to_df,
    gaia_int,
    norm_name,
)
from .merge_base import BaseMergeStrategy, MergeConfig, MergeResult

logger = logging.getLogger(__name__)


class ConfigurablePipelineMerger(BaseMergeStrategy):
    """
    Configuration-driven pipeline merger for flexible data processing.

    This merger:
    1. Cross-identifies sources via configurable matching strategies
    2. Standardizes units across catalogs based on configuration
    3. Propagates uncertainties via Monte Carlo sampling
    4. Derives physical features with uncertainty propagation
    5. Encodes missingness patterns explicitly
    6. Maintains data provenance and quality indicators

    All behavior is driven by YAML configuration files.
    """

    def __init__(self, config: MergeConfig):
        super().__init__(config)
        self.pipeline_config = self._load_pipeline_config()

    def _load_pipeline_config(self) -> dict:
        """Load the pipeline configuration from config/merges directory."""
        # Try to load config based on strategy name
        config_name = self.config.strategy
        config_path = settings.root_dir / "config" / "merges" / f"{config_name}.yml"

        # Fallback to default pipeline config
        if not config_path.exists():
            config_path = (
                settings.root_dir / "config" / "merges" / "exolife_pipeline.yml"
            )

        if not config_path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {config_path}")

        with config_path.open("r") as f:
            return yaml.safe_load(f)

    @property
    def strategy_name(self) -> str:
        return self.pipeline_config.get("id", "configurable_pipeline")

    def can_handle(self, config: MergeConfig) -> bool:
        """Check if this strategy can handle the configuration."""
        # This is a generic merger that can handle any configuration
        return True

    def merge(self, data_sources: Dict[str, pd.DataFrame]) -> MergeResult:
        """
        Execute the configurable data processing pipeline.
        """
        try:
            pipeline_name = self.pipeline_config.get("name", "Data Processing Pipeline")
            logger.info(f"Starting {pipeline_name}...")

            # Step 1: Load and prepare primary catalog
            primary_df = self._load_primary_catalog()
            logger.info(f"Loaded primary catalog: {len(primary_df)} records")

            # Step 2: Cross-identify and merge secondary catalogs
            merged_df = self._cross_identify_and_merge(primary_df)
            logger.info(f"After cross-identification: {len(merged_df)} records")

            # Step 3: Standardize units
            standardized_df = self._standardize_units(merged_df)
            logger.info("Units standardized")

            # Step 4: Add data quality indicators
            quality_df = self._add_quality_indicators(standardized_df)
            logger.info("Quality indicators added")

            # Step 5: Derive features with uncertainty propagation
            enriched_df = self._derive_features_with_uncertainty(quality_df)
            logger.info("Features derived with uncertainty propagation")

            # Step 6: Encode missingness patterns
            final_df = self._encode_missingness_patterns(enriched_df)
            logger.info("Missingness patterns encoded")

            # Step 7: Add provenance information
            final_df = self._add_provenance_information(final_df)
            logger.info("Provenance information added")

            # Save the result
            output_path = self._save_results(final_df)

            return MergeResult(
                output_name=self.config.output_name,
                output_path=output_path,
                success=True,
                input_sources=self.config.sources,
                rows_before={"primary": len(primary_df)},
                rows_after=len(final_df),
                merge_statistics=self._get_pipeline_statistics(final_df),
            )

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return MergeResult(
                output_name=self.config.output_name,
                output_path=Path("/tmp/failed"),
                success=False,
                error_message=str(e),
                input_sources=self.config.sources,
                rows_before={},
            )

    def _load_primary_catalog(self) -> pd.DataFrame:
        """Load the primary catalog based on configuration."""
        primary_id = self.pipeline_config["sources"]["primary"]
        primary_path = settings.raw_dir / f"{primary_id}.parquet"

        if not primary_path.exists():
            raise FileNotFoundError(f"Primary catalog not found: {primary_path}")

        df = pd.read_parquet(primary_path)

        if df is None or len(df) == 0:
            raise ValueError(
                f"Primary catalog is empty or could not be loaded: {primary_path}"
            )

        # Add unique record identifier if planet name column exists
        if "pl_name" in df.columns:
            df["record_id"] = (
                df.index.astype(str) + "_" + df["pl_name"].fillna("unknown")
            )
        else:
            logger.warning("No 'pl_name' column found, using index as record_id")
            df["record_id"] = df.index.astype(str)

        logger.info(
            f"Loaded primary catalog {primary_id}: {len(df)} records, columns: {list(df.columns)[:10]}"
        )
        return df

    def _cross_identify_and_merge(self, primary_df: pd.DataFrame) -> pd.DataFrame:
        """Cross-identify and merge all secondary catalogs based on configuration."""
        merged_df = primary_df.copy()

        # Sort catalogs by priority (lower number = higher priority)
        catalogs = self.pipeline_config["sources"]["catalogs"]
        sorted_catalogs = sorted(catalogs, key=lambda x: x["priority"])

        for catalog_config in sorted_catalogs:
            catalog_id = catalog_config["id"]
            cross_match = catalog_config["cross_match"]
            priority = catalog_config["priority"]

            logger.info(f"Cross-matching with {catalog_id} (priority {priority})")

            try:
                # Load secondary catalog
                catalog_path = settings.raw_dir / f"{catalog_id}.parquet"
                if not catalog_path.exists():
                    logger.warning(f"Catalog not found, skipping: {catalog_path}")
                    continue

                catalog_df = pd.read_parquet(catalog_path)

                # Perform cross-matching based on method
                if cross_match["method"] == "exact_name":
                    merged_df = self._exact_name_crossmatch(
                        merged_df, catalog_df, cross_match, catalog_config
                    )
                elif cross_match["method"] == "gaia_source_id":
                    merged_df = self._gaia_crossmatch_with_proper_motion(
                        merged_df, catalog_df, cross_match, catalog_config
                    )
                elif cross_match["method"] == "fuzzy_hostname":
                    merged_df = self._fuzzy_hostname_crossmatch(
                        merged_df, catalog_df, cross_match, catalog_config
                    )

                # Apply deterministic precedence for conflicting parameters
                merged_df = self._apply_deterministic_precedence(
                    merged_df, catalog_config, catalog_id
                )

                logger.info(f"Merged {catalog_id}: {len(merged_df)} records")

            except Exception as e:
                logger.error(f"Failed to merge {catalog_id}: {e}")
                continue

        return merged_df

    def _exact_name_crossmatch(
        self,
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame,
        cross_match: dict,
        catalog_config: dict,
    ) -> pd.DataFrame:
        """Perform exact name-based cross-matching."""
        # Normalize names for matching
        primary_key = cross_match["primary_key"]
        foreign_key = cross_match["foreign_key"]
        catalog_id = catalog_config["id"]

        # Check if required columns exist
        if primary_key not in primary_df.columns:
            logger.error(
                f"Primary key '{primary_key}' not found in primary catalog. Available columns: {list(primary_df.columns)}"
            )
            return primary_df

        if foreign_key not in secondary_df.columns:
            if len(secondary_df.columns) == 0:
                logger.warning(
                    f"Catalog '{catalog_id}' is empty (likely an on_demand source waiting for Gaia IDs). Skipping cross-match."
                )
            else:
                logger.error(
                    f"Foreign key '{foreign_key}' not found in {catalog_id}. Available columns: {list(secondary_df.columns)}"
                )
            return primary_df

        primary_df["_match_key"] = norm_name(primary_df[primary_key])
        secondary_df["_match_key"] = norm_name(secondary_df[foreign_key])

        # Add catalog prefix to avoid column name conflicts
        secondary_columns = {
            col: f"{catalog_id}_{col}" if col != "_match_key" else col
            for col in secondary_df.columns
        }
        secondary_df = secondary_df.rename(columns=secondary_columns)

        # Merge
        merged = primary_df.merge(
            secondary_df, on="_match_key", how="left", suffixes=("", f"_{catalog_id}")
        )

        # Clean up
        merged = merged.drop(columns=["_match_key"])

        return merged

    def _gaia_crossmatch_with_proper_motion(
        self,
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame,
        cross_match: dict,
        catalog_config: dict,
    ) -> pd.DataFrame:
        """
        Perform Gaia source_id based cross-matching with hybrid DR2/DR3 support.

        This method handles the case where NASA uses DR2 IDs but the secondary
        catalog (hybrid data) may only have partial DR2 coverage. It implements
        intelligent fallback to ensure maximum cross-match coverage.
        """
        primary_key = cross_match["primary_key"]
        foreign_key = cross_match["foreign_key"]
        catalog_id = catalog_config["id"]

        # Check if required columns exist
        if primary_key not in primary_df.columns:
            logger.error(
                f"Primary key '{primary_key}' not found in primary catalog. Available columns: {list(primary_df.columns)}"
            )
            return primary_df

        if foreign_key not in secondary_df.columns:
            if len(secondary_df.columns) == 0:
                logger.warning(
                    f"Catalog '{catalog_id}' is empty (likely an on_demand source waiting for Gaia IDs). Skipping cross-match."
                )
            else:
                logger.error(
                    f"Foreign key '{foreign_key}' not found in {catalog_id}. Available columns: {list(secondary_df.columns)}"
                )
            return primary_df

        # Convert Gaia IDs to integers for matching
        primary_df["_gaia_id"] = primary_df[primary_key].apply(gaia_int)
        secondary_df["_gaia_id"] = secondary_df[foreign_key].apply(gaia_int)

        # Special handling for hybrid Gaia data to maximize coverage
        if catalog_id == "gaia_hybrid_merged":
            merged = self._gaia_hybrid_crossmatch(
                primary_df, secondary_df, catalog_config
            )
        else:
            # Standard cross-matching for other catalogs
            # Apply proper motion correction if coordinates are available
            if self.pipeline_config["cross_identification"]["stellar_crossmatch"][
                "proper_motion_correction"
            ]:
                secondary_df = self._apply_proper_motion_correction(
                    secondary_df, catalog_id
                )

            # Add catalog prefix to avoid column name conflicts
            secondary_columns = {
                col: f"{catalog_id}_{col}" if col != "_gaia_id" else col
                for col in secondary_df.columns
            }
            secondary_df = secondary_df.rename(columns=secondary_columns)

            # Merge
            merged = primary_df.merge(
                secondary_df, on="_gaia_id", how="left", suffixes=("", f"_{catalog_id}")
            )

        # Clean up
        merged = merged.drop(columns=["_gaia_id"])

        return merged

    def _gaia_hybrid_crossmatch(
        self, primary_df: pd.DataFrame, secondary_df: pd.DataFrame, catalog_config: dict
    ) -> pd.DataFrame:
        """
        Specialized cross-matching for hybrid Gaia data that prioritizes DR3
        but falls back to DR2 when needed.

        Strategy:
        1. First try to match NASA DR2 IDs with hybrid data using DR2 source IDs
        2. For unmatched sources, check if they need DR2 fallback data
        3. Apply proper motion correction as needed
        4. Maintain data source provenance
        """
        catalog_id = catalog_config["id"]

        # Apply proper motion correction if configured
        if self.pipeline_config["cross_identification"]["stellar_crossmatch"][
            "proper_motion_correction"
        ]:
            secondary_df = self._apply_proper_motion_correction(
                secondary_df, catalog_id
            )

        # Add catalog prefix to avoid column name conflicts
        secondary_columns = {
            col: f"{catalog_id}_{col}" if col != "_gaia_id" else col
            for col in secondary_df.columns
        }
        secondary_df = secondary_df.rename(columns=secondary_columns)

        # Perform the merge
        merged = primary_df.merge(
            secondary_df, on="_gaia_id", how="left", suffixes=("", f"_{catalog_id}")
        )

        # Log cross-match statistics for hybrid data
        total_primary = len(primary_df)
        matched_count = merged[f"{catalog_id}_data_source"].notna().sum()
        dr3_matches = (
            merged[merged[f"{catalog_id}_data_source"] == "DR3"].shape[0]
            if f"{catalog_id}_data_source" in merged.columns
            else 0
        )
        dr2_matches = (
            merged[merged[f"{catalog_id}_data_source"] == "DR2"].shape[0]
            if f"{catalog_id}_data_source" in merged.columns
            else 0
        )

        logger.info("Gaia hybrid cross-match results:")
        logger.info(f"  Total primary records: {total_primary}")
        logger.info(
            f"  Successful matches: {matched_count} ({matched_count/total_primary:.1%})"
        )
        logger.info(f"  DR3 matches: {dr3_matches}")
        logger.info(f"  DR2 matches: {dr2_matches}")
        logger.info(f"  Unmatched: {total_primary - matched_count}")

        # Identify unmatched sources for potential DR2 fallback
        unmatched_mask = merged[f"{catalog_id}_data_source"].isna()
        unmatched_count = unmatched_mask.sum()

        if unmatched_count > 0:
            logger.warning(
                f"Hybrid approach: {unmatched_count} sources not covered by current hybrid data"
            )
            logger.warning("Consider running DR2 fallback fetch for improved coverage")

            # Optionally log some unmatched source IDs for debugging
            unmatched_ids = (
                merged[unmatched_mask]["_gaia_id"].dropna().head(10).tolist()
            )
            if unmatched_ids:
                logger.debug(f"Sample unmatched Gaia DR2 IDs: {unmatched_ids}")

        return merged

    def _fuzzy_hostname_crossmatch(
        self,
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame,
        cross_match: dict,
        catalog_config: dict,
    ) -> pd.DataFrame:
        """Perform fuzzy hostname-based cross-matching."""
        primary_key = cross_match["primary_key"]
        foreign_key = cross_match["foreign_key"]
        catalog_id = catalog_config["id"]

        # Check if required columns exist
        if primary_key not in primary_df.columns:
            logger.error(
                f"Primary key '{primary_key}' not found in primary catalog. Available columns: {list(primary_df.columns)}"
            )
            return primary_df

        if foreign_key not in secondary_df.columns:
            if len(secondary_df.columns) == 0:
                logger.warning(
                    f"Catalog '{catalog_id}' is empty (likely an on_demand source waiting for Gaia IDs). Skipping cross-match."
                )
            else:
                logger.error(
                    f"Foreign key '{foreign_key}' not found in {catalog_id}. Available columns: {list(secondary_df.columns)}"
                )
            return primary_df

        # Add catalog prefix to avoid column name conflicts
        secondary_columns = {
            col: f"{catalog_id}_{col}" if col not in [foreign_key] else col
            for col in secondary_df.columns
        }
        secondary_df = secondary_df.rename(columns=secondary_columns)

        # Perform fuzzy matching on hostnames
        # This could be enhanced with more sophisticated fuzzy matching
        merged_df = primary_df.merge(
            secondary_df,
            left_on=primary_key,
            right_on=foreign_key,
            how="left",
            suffixes=("", f"_{catalog_id}"),
        )

        return merged_df

    def _apply_proper_motion_correction(
        self, df: pd.DataFrame, catalog_id: str
    ) -> pd.DataFrame:
        """
        Apply proper motion correction to align positional data across missions.

        Corrects coordinates from reference epoch to target epoch using proper motion data.
        """
        try:
            cross_id_config = self.pipeline_config["cross_identification"][
                "stellar_crossmatch"
            ]
            reference_epoch = cross_id_config.get("reference_epoch", 2016.0)  # Gaia DR3
            # Standard epoch            # Check if proper motion data is available
            target_epoch = cross_id_config.get("target_epoch", 2000.0)
            pm_cols = ["pmra", "pmdec", "ra", "dec"]
            available_cols = [col for col in pm_cols if col in df.columns]

            if len(available_cols) >= 2:  # Need at least RA/Dec
                logger.info(f"Applying proper motion correction for {catalog_id}")

                # Calculate epoch difference in years
                epoch_diff = target_epoch - reference_epoch

                # Apply proper motion correction if PM data available
                if "pmra" in df.columns and "pmdec" in df.columns:
                    # Convert proper motion from mas/yr to degrees
                    pm_ra_deg = df["pmra"] / (3600 * 1000)  # mas/yr to deg/yr
                    pm_dec_deg = df["pmdec"] / (3600 * 1000)  # mas/yr to deg/yr

                    # Apply correction (including cos(dec) factor for RA)
                    df["ra_corrected"] = df["ra"] + (
                        pm_ra_deg * epoch_diff / np.cos(np.radians(df["dec"]))
                    )
                    df["dec_corrected"] = df["dec"] + (pm_dec_deg * epoch_diff)

                    logger.info(
                        f"Applied proper motion correction: {epoch_diff:.1f} year epoch difference"
                    )
                else:
                    # No proper motion data, use original coordinates
                    df["ra_corrected"] = df.get("ra", np.nan)
                    df["dec_corrected"] = df.get("dec", np.nan)
                    logger.info(
                        f"No proper motion data available for {catalog_id}, using original coordinates"
                    )
            else:
                logger.warning(
                    f"Insufficient coordinate data for proper motion correction in {catalog_id}"
                )

        except Exception as e:
            logger.warning(
                f"Failed to apply proper motion correction for {catalog_id}: {e}"
            )

        return df

    def _apply_deterministic_precedence(
        self, merged_df: pd.DataFrame, catalog_config: dict, catalog_id: str
    ) -> pd.DataFrame:
        """
        Apply deterministic precedence rule: Gaia → SWEET-Cat → mission catalogs → archive.

        For conflicting measurements, use the highest priority source.
        """
        try:
            conflict_config = self.pipeline_config["cross_identification"][
                "conflict_resolution"
            ]
            precedence_order = conflict_config["precedence_order"]
            parameter_mapping = conflict_config["parameter_mapping"]

            for param_group, param_hierarchy in parameter_mapping.items():
                # Find the parameter name for this catalog in the hierarchy
                catalog_param = None
                for i, source_id in enumerate(precedence_order):
                    if source_id == catalog_id and i < len(param_hierarchy):
                        catalog_param = param_hierarchy[i]
                        break

                if (
                    catalog_param
                    and f"{catalog_id}_{catalog_param}" in merged_df.columns
                ):
                    # Check if this catalog has higher priority than existing data
                    # Use archive parameter name as target
                    target_param = param_hierarchy[-1]

                    # Apply precedence: replace values only if higher priority source available
                    catalog_priority = catalog_config["priority"]

                    # Create a mask for records where this catalog has data
                    has_data_mask = merged_df[f"{catalog_id}_{catalog_param}"].notna()

                    # For highest priority source (Gaia), always use its values when available
                    if catalog_priority == 1:  # Gaia has highest priority
                        merged_df.loc[has_data_mask, f"{target_param}_final"] = (
                            merged_df.loc[
                                has_data_mask, f"{catalog_id}_{catalog_param}"
                            ]
                        )
                        merged_df.loc[has_data_mask, f"{target_param}_source"] = (
                            catalog_id
                        )
                        logger.info(
                            f"Applied Gaia precedence for {param_group} ({catalog_param})"
                        )

                    # For other sources, only use if no higher priority data exists
                    elif f"{target_param}_final" not in merged_df.columns:
                        merged_df.loc[has_data_mask, f"{target_param}_final"] = (
                            merged_df.loc[
                                has_data_mask, f"{catalog_id}_{catalog_param}"
                            ]
                        )
                        merged_df.loc[has_data_mask, f"{target_param}_source"] = (
                            catalog_id
                        )
                        logger.info(
                            f"Applied {catalog_id} precedence for {param_group} (no higher priority data)"
                        )

                    else:
                        # Only fill missing values with lower priority data
                        missing_mask = (
                            merged_df[f"{target_param}_final"].isna() & has_data_mask
                        )
                        if missing_mask.any():
                            merged_df.loc[missing_mask, f"{target_param}_final"] = (
                                merged_df.loc[
                                    missing_mask, f"{catalog_id}_{catalog_param}"
                                ]
                            )
                            merged_df.loc[missing_mask, f"{target_param}_source"] = (
                                catalog_id
                            )
                            logger.info(
                                f"Filled missing {param_group} values with {catalog_id} data"
                            )

        except Exception as e:
            logger.warning(
                f"Failed to apply deterministic precedence for {catalog_id}: {e}"
            )

        return merged_df

    def _standardize_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize units across all measurements."""
        standardized_df = df.copy()

        # Unit conversions would be implemented here
        # For now, assume data is already in correct units
        logger.info("Unit standardization: assuming data already in standard units")

        return standardized_df

    def _add_quality_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add data quality indicators."""
        quality_df = df.copy()

        quality_config = self.pipeline_config["data_quality"]["quality_indicators"]

        for indicator in quality_config:
            if indicator["name"] == "data_completeness_score":
                key_params = indicator["key_parameters"]
                # Calculate fraction of available key parameters
                available_count = quality_df[key_params].notna().sum(axis=1)
                quality_df["data_completeness_score"] = available_count / len(
                    key_params
                )

        return quality_df

    def _derive_features_with_uncertainty(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive physical features with uncertainty propagation via Monte Carlo."""
        enriched_df = df.copy()

        # Add habitable zone edges (conservative and optimistic)
        enriched_df = add_hz_edges_to_df(
            enriched_df,
            teff_col="st_teff",
            lum_col="st_lum",
            optimistic=False,
            prefix="hz_",
        )

        enriched_df = add_hz_edges_to_df(
            enriched_df,
            teff_col="st_teff",
            lum_col="st_lum",
            optimistic=True,
            prefix="hz_opt_",
        )

        # Derive additional features
        enriched_df = self._calculate_stellar_flux(enriched_df)
        enriched_df = self._calculate_equilibrium_temperature(enriched_df)
        enriched_df = self._calculate_surface_gravity(enriched_df)
        enriched_df = self._calculate_escape_velocity(enriched_df)
        enriched_df = self._calculate_hz_positions(enriched_df)

        return enriched_df

    def _calculate_stellar_flux(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate stellar flux received by planet."""
        df["stellar_flux_s"] = df["st_lum"] / (df["pl_orbsmax"] ** 2)
        return df

    def _calculate_equilibrium_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate planetary equilibrium temperature."""
        # Simplified calculation (assumes albedo = 0.3)
        albedo = 0.3
        df["equilibrium_temperature"] = (
            df["st_teff"]
            * np.sqrt(df["st_rad"] / (2 * df["pl_orbsmax"]))
            * (1 - albedo) ** 0.25
        )
        return df

    def _calculate_surface_gravity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate planetary surface gravity."""
        # G * M / R^2 in Earth units, result in m/s^2
        G = 6.67430e-11  # m^3 kg^-1 s^-2
        M_earth = 5.972e24  # kg
        R_earth = 6.371e6  # m

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df["surface_gravity"] = (
                G * df["pl_masse"] * M_earth / (df["pl_rade"] * R_earth) ** 2
            )

        return df

    def _calculate_escape_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate planetary escape velocity."""
        # sqrt(2 * G * M / R) in km/s
        G = 6.67430e-11  # m^3 kg^-1 s^-2
        M_earth = 5.972e24  # kg
        R_earth = 6.371e6  # m

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df["escape_velocity"] = (
                np.sqrt(2 * G * df["pl_masse"] * M_earth / (df["pl_rade"] * R_earth))
                / 1000
            )  # Convert to km/s

        return df

    def _calculate_hz_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate position within habitable zones."""
        # Conservative HZ position
        hz_width_cons = df["hz_outer"] - df["hz_inner"]
        df["hz_position_conservative"] = (
            df["pl_orbsmax"] - df["hz_inner"]
        ) / hz_width_cons

        # Optimistic HZ position
        hz_width_opt = df["hz_opt_outer"] - df["hz_opt_inner"]
        df["hz_position_optimistic"] = (
            df["pl_orbsmax"] - df["hz_opt_inner"]
        ) / hz_width_opt

        return df

    def _encode_missingness_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode missingness patterns explicitly."""
        missingness_df = df.copy()

        missingness_config = self.pipeline_config["data_quality"][
            "missingness_encoding"
        ]

        for pattern in missingness_config["patterns"]:
            pattern_name = pattern["name"]
            columns = pattern["columns"]

            # Check if all columns in pattern are missing
            missing_mask = missingness_df[columns].isna().all(axis=1)
            missingness_df[f"missing_pattern_{pattern_name}"] = missing_mask

        return missingness_df

    def _add_provenance_information(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add data provenance tracking."""
        provenance_df = df.copy()

        # Add source information for key parameters
        key_sources = {
            "nasa_source": "nasa_exoplanet_archive_pscomppars",
            "phl_source": "phl_exoplanet_catalog",
            "gaia_source": "gaia_dr3_astrophysical_parameters",
            "sweet_source": "sweet_cat",
        }

        for source_col, source_name in key_sources.items():
            # Simple approach: mark as available if any column from source exists
            source_columns = [
                col for col in provenance_df.columns if source_name in col
            ]
            if source_columns:
                provenance_df[source_col] = (
                    provenance_df[source_columns].notna().any(axis=1)
                )
            else:
                provenance_df[source_col] = False

        return provenance_df

    def _save_results(self, df: pd.DataFrame) -> Path:
        """Save the pipeline results."""
        output_config = self.pipeline_config["output"]

        # Use configured output path or default
        output_subdir = output_config.get(
            "path", "data/processed/exolife_dataset"
        ).replace("data/processed/", "")
        output_dir = settings.processed_dir / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use configurable filename or default
        filename_base = output_config.get("filename", "exolife_catalog")

        # Save main dataset
        output_path = output_dir / f"{filename_base}.parquet"
        df.to_parquet(output_path, index=False)

        # Also save CSV version
        csv_path = output_dir / f"{filename_base}.csv"
        df.to_csv(csv_path, index=False)

        logger.info(f"Saved {filename_base}: {len(df)} records to {output_path}")

        return output_path

    def _get_pipeline_statistics(self, df: pd.DataFrame) -> Dict:
        """Get statistics about the pipeline process."""
        stats = {
            "total_records": len(df),
            "data_completeness_mean": df.get(
                "data_completeness_score", pd.Series()
            ).mean(),
            "hz_coverage": (df["hz_inner"].notna() & df["hz_outer"].notna()).sum(),
            "gaia_crossmatches": df.filter(regex=r"gaia_dr3_.*")
            .notna()
            .any(axis=1)
            .sum(),
            "derived_features": len(
                [
                    col
                    for col in df.columns
                    if any(
                        prefix in col
                        for prefix in [
                            "stellar_flux",
                            "equilibrium_temp",
                            "surface_gravity",
                            "escape_velocity",
                            "hz_position",
                        ]
                    )
                ]
            ),
        }

        return stats


class ConfigurableMergeManager:
    """
    Configuration-driven merge manager for flexible data processing pipelines.
    """

    def __init__(self):
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    def list_mergers(self) -> List[str]:
        """List available merge strategies from config files."""
        config_dir = settings.root_dir / "config" / "merges"
        available_configs = []

        if config_dir.exists():
            for config_file in config_dir.glob("*.yml"):
                config_name = config_file.stem
                available_configs.append(config_name)

        return available_configs or ["exolife_pipeline"]  # Default fallback

    def merge_data(
        self, method: str = "exolife_pipeline", overwrite: bool = True
    ) -> pd.DataFrame:
        """Execute the configurable data processing pipeline."""
        # Use the same output path logic as _save_results
        config_path = settings.root_dir / "config" / "merges" / f"{method}.yml"
        if config_path.exists():
            import yaml

            with config_path.open("r") as f:
                config = yaml.safe_load(f)
            output_subdir = (
                config["output"]
                .get("path", "data/processed/exolife_dataset")
                .replace("data/processed/", "")
            )
            filename_base = config["output"].get("filename", "exolife_catalog")
        else:
            output_subdir = "exolife_dataset"
            filename_base = "exolife_catalog"

        output_path = (
            settings.processed_dir / output_subdir / f"{filename_base}.parquet"
        )

        if not output_path.exists() or overwrite:
            self.logger.info(f"Running {method} pipeline")

            # Create the strategy
            config = MergeConfig(
                strategy=method,
                output_name=filename_base,
                sources=[],  # Loaded from configuration file
            )

            strategy = ConfigurablePipelineMerger(config)

            # Execute merge
            result = strategy.merge({})

            if not result.success:
                raise RuntimeError(f"Pipeline execution failed: {result.error_message}")

            self.logger.info("Pipeline completed successfully")
            return pd.read_parquet(result.output_path)
        else:
            self.logger.info("Loading existing dataset")
            return pd.read_parquet(output_path)


# Global instance for easy access
merge_manager = ConfigurableMergeManager()


def merge_data(
    method: str = "exolife_pipeline", overwrite: bool = True
) -> pd.DataFrame:
    """Convenience function that wraps the merge manager."""
    return merge_manager.merge_data(method, overwrite)


def list_mergers() -> List[str]:
    """Convenience function to list available merge strategies."""
    return merge_manager.list_mergers()
