"""
Pipeline merger that implements the full ExoLife data processing pipeline.

This merger maintains compatibility with the existing configuration-driven
pipeline while using the new modular architecture.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from exolife.data.merge.steps import (
    feature_engineering as fe,
    missingness as miss,
    normalization as norm_utils,
    quality as quality_utils,
    units as unit_utils,
)

from ..base_merger import BaseMerger
from .catalog_merger import CatalogMerger
from .coordinate_merger import CoordinateMerger
from .identifier_merger import IdentifierMerger


class PipelineMerger(BaseMerger):
    """
    Comprehensive pipeline merger that orchestrates the full data processing workflow.

    This merger implements the complete ExoLife pipeline including:
    - Multi-strategy cross-matching
    - Feature engineering with uncertainty propagation
    - Quality assessment and validation
    - Data normalization and output generation
    """

    def __init__(self, config):
        super().__init__(config)
        self.pipeline_config = self._load_pipeline_config()

        # Note: Child mergers will be created on-demand with appropriate configs
        # to avoid passing incompatible strategy names during initialization

    @property
    def merger_name(self) -> str:
        return "ExoLife Data Processing Pipeline"

    @property
    def supported_strategies(self) -> List[str]:
        return [
            "exolife_merge_v1",
            "exolife_pipeline",
            "comprehensive_merge",
            "multi_catalog_merge",
            # Backward compatibility aliases
            "unified_ingestion",
            "unified",
            "baseline",
            "gaia_enriched",
        ]

    def _load_pipeline_config(self) -> dict:
        """Load pipeline configuration from YAML file."""
        try:
            from exolife.settings import Settings

            settings = Settings()

            # Map strategy names to config file names
            strategy_to_config = {
                "unified_ingestion": "exolife_merge_v1",
                "unified": "exolife_merge_v1",
                "baseline": "exolife_merge_v1",
                "gaia_enriched": "exolife_merge_v1",
                "comprehensive": "exolife_merge_v1",
                "ingestion": "exolife_merge_v1",
            }

            config_name = strategy_to_config.get(
                self.config.strategy, self.config.strategy
            )
            config_path = (
                settings.root_dir / "config" / "mergers" / f"{config_name}.yml"
            )

            # Fallback to default config
            if not config_path.exists():
                config_path = (
                    settings.root_dir / "config" / "mergers" / "exolife_merge_v1.yml"
                )

            if not config_path.exists():
                self.logger.warning("No pipeline config found, using defaults")
                return self._get_default_config()

            with config_path.open("r") as f:
                return yaml.safe_load(f)

        except Exception as e:
            self.logger.warning(f"Failed to load pipeline config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """Provide default configuration when config file is not available."""
        return {
            "sources": {"primary": "nasa_exoplanet_archive_pscomppars", "catalogs": []},
            "cross_identification": {},
            "derived_features": {},
            "data_quality": {},
            "output": {},
        }

    def execute_merge(self, data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Execute the full ExoLife data processing pipeline."""
        self.logger.info("Starting ExoLife data processing pipeline")

        # Stage 1: Load primary catalog
        primary_df = self._load_primary_catalog(data_sources)
        self.logger.info(f"Primary catalog loaded: {len(primary_df)} records")

        # Stage 2: Cross-identification and merging
        merged_df = self._execute_cross_identification(primary_df, data_sources)
        self.logger.info(f"Cross-identification complete: {len(merged_df)} records")

        # Stage 3: Assign canonical identifiers
        merged_df = self._assign_canonical_identifiers(merged_df)

        # Stage 4: Unit standardization and validation
        merged_df = self._standardize_units_and_validate(merged_df)
        self.logger.info("Units standardized and validated")

        # Stage 5: Quality assessment
        merged_df = self._assess_data_quality(merged_df)
        self.logger.info("Data quality assessment complete")

        # Stage 6: Feature engineering
        merged_df = self._engineer_features(merged_df)
        self.logger.info("Feature engineering complete")

        # Stage 7: Uncertainty propagation
        merged_df = self._propagate_uncertainties(merged_df)
        self.logger.info("Uncertainty propagation complete")

        # Stage 8: Final processing
        merged_df = self._final_processing(merged_df)
        self.logger.info("Final processing complete")

        # Stage 9: Generate normalized outputs
        self._generate_normalized_outputs(merged_df)

        return merged_df

    def _load_primary_catalog(
        self, data_sources: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Load and prepare the primary catalog."""
        primary_id = self.pipeline_config.get("sources", {}).get(
            "primary", "nasa_exoplanet_archive_pscomppars"
        )

        if primary_id not in data_sources:
            # Try to load from disk
            try:
                from exolife.settings import Settings

                settings = Settings()
                source_dir = settings.raw_dir / primary_id
                parquet_path = source_dir / f"{primary_id}.parquet"
                csv_path = source_dir / f"{primary_id}.csv"

                if parquet_path.exists():
                    primary_df = pd.read_parquet(parquet_path)
                elif csv_path.exists():
                    primary_df = pd.read_csv(csv_path)
                else:
                    raise FileNotFoundError(f"Primary catalog not found: {primary_id}")
            except Exception as e:
                raise ValueError(f"Could not load primary catalog {primary_id}: {e}")
        else:
            primary_df = data_sources[primary_id].copy()

        # Add record identifiers
        if "record_id" not in primary_df.columns:
            if "pl_name" in primary_df.columns:
                primary_df["record_id"] = (
                    primary_df.index.astype(str)
                    + "_"
                    + primary_df["pl_name"].fillna("unknown")
                )
            else:
                primary_df["record_id"] = primary_df.index.astype(str)

        return primary_df

    def _execute_cross_identification(
        self, primary_df: pd.DataFrame, data_sources: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Execute cross-identification using appropriate strategies."""
        merged_df = primary_df.copy()

        # Get catalog configurations
        catalogs_config = self.pipeline_config.get("sources", {}).get("catalogs", [])

        # Sort catalogs by priority
        sorted_catalogs = sorted(catalogs_config, key=lambda x: x.get("priority", 999))

        for catalog_config in sorted_catalogs:
            catalog_id = catalog_config["id"]

            if catalog_id not in data_sources:
                self.logger.warning(f"Catalog not available: {catalog_id}")
                continue

            catalog_df = data_sources[catalog_id]
            cross_match_config = catalog_config.get("cross_match", {})
            method = cross_match_config.get("method", "exact_name")

            # Choose appropriate merger based on method
            merger_df = self._apply_cross_match_method(
                merged_df, catalog_df, catalog_id, method, cross_match_config
            )
            merged_df = merger_df

            self.logger.info(f"Cross-matched {catalog_id} using {method}")

        return merged_df

    def _apply_cross_match_method(
        self,
        primary_df: pd.DataFrame,
        catalog_df: pd.DataFrame,
        catalog_id: str,
        method: str,
        cross_match_config: dict,
    ) -> pd.DataFrame:
        """Apply specific cross-matching method."""
        # Create temporary config for the merger
        temp_config = self.config.model_copy()
        temp_config.sources = [catalog_id]

        if method == "gaia_source_id":
            temp_config.strategy = "gaia_source_id"
            temp_config.join_keys = {
                catalog_id: [cross_match_config.get("primary_key", "gaia_id")]
            }
            identifier_merger = IdentifierMerger(temp_config)
            return identifier_merger._merge_single_source(
                primary_df, catalog_df, catalog_id
            )

        elif method == "exact_name":
            temp_config.strategy = "exact_name"
            temp_config.join_keys = {
                catalog_id: [cross_match_config.get("primary_key", "pl_name")]
            }
            identifier_merger = IdentifierMerger(temp_config)
            return identifier_merger._merge_single_source(
                primary_df, catalog_df, catalog_id
            )

        elif method == "multiple_identifiers" or method == "coordinate_match":
            temp_config.strategy = "coordinate_match"
            temp_config.coordinate_tolerance = cross_match_config.get("tolerance", 3.0)
            temp_config.epoch_correction = cross_match_config.get(
                "epoch_correction", 0.0
            )

            coord_merger = CoordinateMerger(temp_config)
            return coord_merger._merge_by_coordinates(
                primary_df, catalog_df, catalog_id
            )

        elif method == "fuzzy_hostname":
            temp_config.strategy = "exact_name"  # Use exact name with hostname
            temp_config.join_keys = {
                catalog_id: [cross_match_config.get("primary_key", "hostname")]
            }
            identifier_merger = IdentifierMerger(temp_config)
            return identifier_merger._merge_single_source(
                primary_df, catalog_df, catalog_id
            )

        else:
            # Use catalog merger for unknown methods
            temp_config.strategy = "exoplanet_catalog"
            catalog_merger = CatalogMerger(temp_config)
            return catalog_merger._merge_catalog(primary_df, catalog_df, catalog_id)

    def _assign_canonical_identifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign canonical ExoLife identifiers."""
        return miss.assign_exolife_ids(df)

    def _standardize_units_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize units and apply physical validation."""
        # Standardize units
        df = self._standardize_units(df)

        # Apply physical range checks
        df = unit_utils.apply_physical_range_checks(df)

        return df

    def _standardize_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize measurement units across catalogs."""
        # Planetary radius conversion (Jupiter to Earth units)
        R_JUP_TO_REARTH = 11.209
        if "pl_rade" not in df.columns and "pl_radj" in df.columns:
            df["pl_rade"] = df["pl_radj"] * R_JUP_TO_REARTH
            df["pl_rade_units"] = "R_earth"

        # Planetary mass conversion
        M_JUP_TO_MEARTH = 317.83
        if "pl_masse" not in df.columns and "pl_massj" in df.columns:
            df["pl_masse"] = df["pl_massj"] * M_JUP_TO_MEARTH
            df["pl_masse_units"] = "M_earth"

        # Add unit labels
        unit_mapping = {
            "st_teff": "K",
            "st_lum": "L_sun",
            "st_mass": "M_sun",
            "st_rad": "R_sun",
            "pl_orbsmax": "AU",
            "pl_orbper": "days",
        }

        for param, unit in unit_mapping.items():
            if param in df.columns:
                df[f"{param}_units"] = unit

        return df

    def _assess_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assess data quality and add quality indicators."""
        df = quality_utils.add_quality_indicators(df, self.pipeline_config)
        df = quality_utils.assign_crossmatch_quality(df)
        df = quality_utils.assign_astrometry_quality_flag(df)
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer derived features with uncertainty handling."""
        return fe.derive_features_with_uncertainty(df)

    def _propagate_uncertainties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Propagate uncertainties using Monte Carlo methods."""
        return fe.monte_carlo_propagation(df, self.pipeline_config)

    def _final_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply final processing steps."""
        # Encode missingness patterns
        df = miss.encode_missingness_patterns(df, self.pipeline_config)

        # Add provenance information
        df = miss.add_provenance_information(df)

        # Compute distances
        df = fe.compute_distances(df, self.pipeline_config)

        return df

    def _generate_normalized_outputs(self, df: pd.DataFrame) -> None:
        """Generate normalized table outputs."""
        try:
            # Create normalized tables
            tables = norm_utils.normalize_output_tables(df)

            # Save normalized results
            output_paths = self._save_normalized_tables(tables)

            self.logger.info(
                f"Generated normalized tables: {list(output_paths.keys())}"
            )

        except Exception as e:
            self.logger.warning(f"Failed to generate normalized outputs: {e}")

    def _save_normalized_tables(
        self, tables: Dict[str, pd.DataFrame]
    ) -> Dict[str, Path]:
        """Save normalized tables to separate files in processed/normalized_tables."""
        try:
            from exolife.settings import Settings

            settings = Settings()

            # Always save normalized tables to the normalized_tables directory
            base_dir = settings.processed_dir / "normalized_tables"
            base_dir.mkdir(parents=True, exist_ok=True)

            output_paths = {}

            for table_name, table_df in tables.items():
                # Save as parquet
                parquet_path = (
                    base_dir / f"{self.config.output_name}_{table_name}.parquet"
                )
                table_df.to_parquet(parquet_path, index=False)
                output_paths[table_name] = parquet_path

            return output_paths

        except Exception as e:
            self.logger.error(f"Failed to save normalized tables: {e}")
            return {}

    def _load_data_sources(self) -> Dict[str, pd.DataFrame]:
        """Override to load all configured data sources."""
        data_sources = {}

        try:
            from exolife.settings import Settings

            settings = Settings()

            # Load primary catalog
            primary_id = self.pipeline_config.get("sources", {}).get("primary")
            if primary_id:
                primary_dir = settings.raw_dir / primary_id
                primary_path = primary_dir / f"{primary_id}.parquet"
                if not primary_path.exists():
                    primary_path = primary_dir / f"{primary_id}.csv"

                if primary_path.exists():
                    if primary_path.suffix == ".parquet":
                        data_sources[primary_id] = pd.read_parquet(primary_path)
                    else:
                        data_sources[primary_id] = pd.read_csv(primary_path)

            # Load secondary catalogs
            catalogs_config = self.pipeline_config.get("sources", {}).get(
                "catalogs", []
            )
            for catalog_config in catalogs_config:
                catalog_id = catalog_config["id"]
                catalog_dir = settings.raw_dir / catalog_id
                catalog_path = catalog_dir / f"{catalog_id}.parquet"

                if not catalog_path.exists():
                    catalog_path = catalog_dir / f"{catalog_id}.csv"

                if catalog_path.exists():
                    if catalog_path.suffix == ".parquet":
                        data_sources[catalog_id] = pd.read_parquet(catalog_path)
                    else:
                        data_sources[catalog_id] = pd.read_csv(catalog_path)
                else:
                    self.logger.warning(f"Catalog not found: {catalog_id}")

        except Exception as e:
            self.logger.error(f"Failed to load data sources: {e}")

        return data_sources
