"""
Specialized GAIA data fetcher with comprehensive DR2→DR3 cross-matching.

This fetcher implements sophisticated multi-tier cross-matching strategies
to achieve maximum coverage for GAIA astronomical data.
"""

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ....settings import settings
from ...utils import DataSource, fetch_adql, gaia_int
from ..fetcher_base import BaseFetcher, DataSourceConfig, FetchResult
from ..registry import register_fetcher

logger = logging.getLogger(__name__)


@register_fetcher("gaia")
class GaiaFetcher(BaseFetcher):
    """
    Specialized GAIA fetcher with comprehensive DR2→DR3 cross-matching.

    Implements a multi-tier approach to maximize coverage and data quality
    for GAIA astronomical observations with advanced cross-matching strategies.
    """

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.tap_endpoint = "https://gea.esac.esa.int/tap-server/tap/sync"

        # Cross-matching parameters
        self.angular_threshold = 200.0  # mas
        self.magnitude_threshold = 1.0  # mag
        self.parallax_sigma = 3.0
        self.epoch_dr2 = 2015.5
        self.epoch_dr3 = 2016.0

    @property
    def fetcher_type(self) -> str:
        return "gaia"

    def can_handle(self, config: DataSourceConfig) -> bool:
        """Check if this fetcher can handle GAIA-specific data sources."""
        return config.id in [
            "gaia_dr3_astrophysical_parameters",
            "gaia_dr2_astrophysical_parameters",
        ]

    def fetch(self, force: bool = False) -> FetchResult:
        """
        Fetch GAIA data with comprehensive cross-matching strategy.

        Args:
            force: Always fetch fresh data

        Returns:
            FetchResult with success status and coverage statistics
        """
        # Create source-specific directory
        source_dir = settings.raw_dir / self.config.id
        source_dir.mkdir(parents=True, exist_ok=True)
        output_path = source_dir / f"{self.config.id}.parquet"

        try:
            if self.config.id == "gaia_dr3_astrophysical_parameters":
                self._fetch_dr3_comprehensive(output_path)
            elif self.config.id == "gaia_dr2_astrophysical_parameters":
                self._fetch_dr2_fallback(output_path)
            else:
                raise ValueError(f"Unsupported GAIA source: {self.config.id}")

            # Get final statistics
            if output_path.exists():
                df = pd.read_parquet(output_path)
                return FetchResult(
                    source_id=self.config.id,
                    path=output_path,
                    success=True,
                    rows_fetched=len(df),
                    size_bytes=output_path.stat().st_size,
                )
            else:
                raise RuntimeError("Failed to create output file")

        except Exception as e:
            self.logger.error(f"GAIA fetch failed for {self.config.id}: {e}")
            return self._create_fallback_dataset(output_path, str(e))

    def _fetch_dr3_comprehensive(self, output_path: Path) -> Dict[str, Any]:
        """Fetch DR3 data with multi-tier cross-matching for maximum coverage."""

        # Get target GAIA IDs from NASA data
        nasa_gaia_ids = self._get_target_gaia_ids()
        if not nasa_gaia_ids:
            self._create_empty_dataset(output_path)
            return {"total_targets": 0, "total_coverage": 0}

        self.logger.info(
            f"Starting comprehensive GAIA DR3 fetch for {len(nasa_gaia_ids)} targets"
        )

        all_data = []
        unmatched_ids = set(nasa_gaia_ids)

        # Tier 1: Direct DR3 lookup
        tier1_data, tier1_matched = self._fetch_dr3_direct(list(unmatched_ids))
        if tier1_data is not None and not tier1_data.empty:
            all_data.append(tier1_data)
            unmatched_ids -= set(tier1_matched)

        # Tier 2: Position-based cross-matching
        if unmatched_ids:
            tier2_data, tier2_matched = self._fetch_dr3_crossmatch(list(unmatched_ids))
            if tier2_data is not None and not tier2_data.empty:
                all_data.append(tier2_data)
                unmatched_ids -= set(tier2_matched)

        # Tier 3: DR2 fallback
        if unmatched_ids:
            tier3_data, tier3_matched = self._fetch_dr2_direct(list(unmatched_ids))
            if tier3_data is not None and not tier3_data.empty:
                all_data.append(tier3_data)
                unmatched_ids -= set(tier3_matched)

        # Combine all tiers
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            final_df = final_df.drop_duplicates(subset=["gaia_dr2_source_id"])

            # Apply enhancements
            final_df = self._apply_enhancements(final_df)

            # Save results
            final_df.to_parquet(output_path, index=False)

            coverage = len(final_df)
            self.logger.info(
                f"GAIA comprehensive fetch completed: {coverage}/{len(nasa_gaia_ids)} "
                f"({coverage/len(nasa_gaia_ids)*100:.1f}%)"
            )
        else:
            self._create_empty_dataset(output_path)
            coverage = 0

        return {
            "total_targets": len(nasa_gaia_ids),
            "total_coverage": coverage,
        }

    def _get_target_gaia_ids(self) -> List[int]:
        """Extract target GAIA IDs from NASA exoplanet data."""
        nasa_dir = settings.raw_dir / "nasa_exoplanet_archive_pscomppars"
        nasa_path = nasa_dir / "nasa_exoplanet_archive_pscomppars.parquet"

        if not nasa_path.exists():
            self.logger.warning("NASA data not found for GAIA targeting")
            return []

        try:
            nasa_df = pd.read_parquet(nasa_path)
            nasa_gaia_ids = []

            for gaia_str in nasa_df["gaia_id"].dropna():
                gaia_id = gaia_int(gaia_str)
                if gaia_id is not None:
                    nasa_gaia_ids.append(gaia_id)

            return list(set(nasa_gaia_ids))  # Remove duplicates

        except Exception as e:
            self.logger.error(f"Failed to extract GAIA IDs from NASA data: {e}")
            return []

    def _fetch_dr3_direct(
        self, gaia_ids: List[int]
    ) -> Tuple[Optional[pd.DataFrame], List[int]]:
        """Tier 1: Direct DR3 source lookup."""
        try:
            matched_ids = []
            all_batches = []
            batch_size = 100

            for i in range(0, len(gaia_ids), batch_size):
                batch_ids = gaia_ids[i : i + batch_size]
                id_list = ",".join(map(str, batch_ids))

                query = f"""
                SELECT 
                    source_id as gaia_dr2_source_id,
                    source_id as gaia_dr3_source_id,
                    ra, dec, parallax, parallax_error,
                    pmra, pmra_error, pmdec, pmdec_error,
                    ruwe, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
                    bp_rp, radial_velocity, radial_velocity_error,
                    teff_gspphot, logg_gspphot, mh_gspphot,
                    distance_gspphot, ag_gspphot, ebpminrp_gspphot
                FROM gaiadr3.gaia_source
                WHERE source_id IN ({id_list})
                AND parallax IS NOT NULL
                AND phot_g_mean_mag IS NOT NULL
                """

                batch_df = self._execute_tap_query(
                    query, f"dr3_direct_batch_{i//batch_size}"
                )
                if batch_df is not None and not batch_df.empty:
                    # Add metadata
                    batch_df["data_source"] = "DR3_direct"
                    batch_df["crossmatch_method"] = "direct"
                    batch_df["crossmatch_quality"] = 1.0

                    all_batches.append(batch_df)
                    matched_ids.extend(batch_df["gaia_dr2_source_id"].tolist())

            if all_batches:
                result_df = pd.concat(all_batches, ignore_index=True)
                return result_df, matched_ids

            return None, []

        except Exception as e:
            self.logger.error(f"DR3 direct fetch failed: {e}")
            return None, []

    def _fetch_dr3_crossmatch(
        self, gaia_ids: List[int]
    ) -> Tuple[Optional[pd.DataFrame], List[int]]:
        """Tier 2: Position-based cross-matching using DR2 positions."""
        try:
            matched_ids = []
            all_matches = []
            batch_size = 50

            for i in range(0, len(gaia_ids), batch_size):
                batch_ids = gaia_ids[i : i + batch_size]

                # Get DR2 positions
                dr2_positions = self._get_dr2_positions(batch_ids)
                if dr2_positions.empty:
                    continue

                # Cross-match each position with DR3
                for _, dr2_row in dr2_positions.iterrows():
                    match = self._crossmatch_single_position(dr2_row)
                    if match is not None:
                        all_matches.append(match)
                        matched_ids.append(dr2_row["source_id"])

            if all_matches:
                result_df = pd.DataFrame(all_matches)
                return result_df, matched_ids

            return None, []

        except Exception as e:
            self.logger.error(f"DR3 crossmatch failed: {e}")
            return None, []

    def _get_dr2_positions(self, gaia_ids: List[int]) -> pd.DataFrame:
        """Get DR2 positions for cross-matching."""
        id_list = ",".join(map(str, gaia_ids))

        query = f"""
        SELECT source_id, ra, dec, phot_g_mean_mag, pmra, pmdec
        FROM gaiadr2.gaia_source
        WHERE source_id IN ({id_list})
        AND phot_g_mean_mag IS NOT NULL
        """

        result = self._execute_tap_query(query, "dr2_positions")
        return result if result is not None else pd.DataFrame()

    def _crossmatch_single_position(self, dr2_row: pd.Series) -> Optional[Dict]:
        """Cross-match a single DR2 position with DR3."""
        try:
            # Cone search in DR3
            cone_query = f"""
            SELECT 
                {dr2_row['source_id']} as gaia_dr2_source_id,
                source_id as gaia_dr3_source_id,
                ra, dec, parallax, parallax_error,
                pmra, pmra_error, pmdec, pmdec_error,
                ruwe, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
                bp_rp, radial_velocity, radial_velocity_error,
                teff_gspphot, logg_gspphot, mh_gspphot,
                distance_gspphot, ag_gspphot, ebpminrp_gspphot,
                DISTANCE(POINT('ICRS', {dr2_row['ra']}, {dr2_row['dec']}), 
                       POINT('ICRS', ra, dec)) * 3600000 as angular_separation
            FROM gaiadr3.gaia_source
            WHERE 1=CONTAINS(POINT('ICRS', ra, dec), 
                           CIRCLE('ICRS', {dr2_row['ra']}, {dr2_row['dec']}, 1.0/3600))
            AND ABS({dr2_row['phot_g_mean_mag']} - phot_g_mean_mag) < {self.magnitude_threshold}
            AND parallax IS NOT NULL
            ORDER BY DISTANCE(POINT('ICRS', {dr2_row['ra']}, {dr2_row['dec']}), 
                            POINT('ICRS', ra, dec))
            """

            matches_df = self._execute_tap_query(
                cone_query, f"cone_{dr2_row['source_id']}"
            )

            if matches_df is not None and not matches_df.empty:
                best_match = matches_df.iloc[0].to_dict()
                best_match["data_source"] = "DR3_crossmatch"
                best_match["crossmatch_method"] = "cone_search"
                best_match["crossmatch_quality"] = (
                    0.8 if best_match["angular_separation"] < 500 else 0.6
                )
                return best_match

            return None

        except Exception as e:
            self.logger.debug(f"Crossmatch failed for {dr2_row['source_id']}: {e}")
            return None

    def _fetch_dr2_direct(
        self, gaia_ids: List[int]
    ) -> Tuple[Optional[pd.DataFrame], List[int]]:
        """Tier 3: DR2 fallback for remaining sources."""
        try:
            matched_ids = []
            all_batches = []
            batch_size = 100

            for i in range(0, len(gaia_ids), batch_size):
                batch_ids = gaia_ids[i : i + batch_size]
                id_list = ",".join(map(str, batch_ids))

                query = f"""
                SELECT 
                    source_id as gaia_dr2_source_id,
                    NULL as gaia_dr3_source_id,
                    ra, dec, parallax, parallax_error,
                    pmra, pmra_error, pmdec, pmdec_error,
                    NULL as ruwe,
                    phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
                    bp_rp, radial_velocity, radial_velocity_error,
                    NULL as teff_gspphot, NULL as logg_gspphot, NULL as mh_gspphot,
                    NULL as distance_gspphot, NULL as ag_gspphot, NULL as ebpminrp_gspphot
                FROM gaiadr2.gaia_source
                WHERE source_id IN ({id_list})
                AND parallax > 0
                AND phot_g_mean_mag IS NOT NULL
                """

                batch_df = self._execute_tap_query(
                    query, f"dr2_fallback_batch_{i//batch_size}"
                )
                if batch_df is not None and not batch_df.empty:
                    # Add metadata
                    batch_df["data_source"] = "DR2_fallback"
                    batch_df["crossmatch_method"] = "direct"
                    batch_df["crossmatch_quality"] = 0.4

                    all_batches.append(batch_df)
                    matched_ids.extend(batch_df["gaia_dr2_source_id"].tolist())

            if all_batches:
                result_df = pd.concat(all_batches, ignore_index=True)
                return result_df, matched_ids

            return None, []

        except Exception as e:
            self.logger.error(f"DR2 fallback failed: {e}")
            return None, []

    def _fetch_dr2_fallback(self, output_path: Path) -> Dict[str, Any]:
        """Fetch DR2 fallback data for sources missing from DR3."""
        # Implementation for DR2-specific fetching
        # This would contain logic similar to the original _fetch_gaia_dr2_fallback
        self.logger.info("DR2 fallback fetch not yet implemented")
        self._create_empty_dataset(output_path)
        return {"total_targets": 0, "total_coverage": 0}

    def _execute_tap_query(self, query: str, query_id: str) -> Optional[pd.DataFrame]:
        """Execute a TAP query and return results as DataFrame."""
        try:
            ds = DataSource(
                id=query_id,
                name="GAIA TAP Query",
                description="GAIA query execution",
                download_url=self.tap_endpoint,
                adql=query,
                columns_to_keep=[],
            )

            raw_data = fetch_adql(ds)
            return pd.read_csv(io.BytesIO(raw_data))

        except Exception as e:
            self.logger.debug(f"TAP query failed for {query_id}: {e}")
            return None

    def _apply_enhancements(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality enhancements and corrections to the dataset."""
        if df.empty:
            return df

        df = df.copy()

        # Apply parallax correction (Lindegren et al. 2021)
        if "parallax" in df.columns:
            df["parallax_corrected"] = df["parallax"] - 0.017
            df["parallax_correction_applied"] = True

        # Compute quality scores
        df["overall_quality_score"] = self._compute_quality_score(df)

        return df

    def _compute_quality_score(self, df: pd.DataFrame) -> pd.Series:
        """Compute overall quality score for GAIA sources."""
        if df.empty:
            return pd.Series(dtype=float)

        scores = pd.Series(1.0, index=df.index)

        # Adjust for RUWE
        if "ruwe" in df.columns:
            ruwe_mask = df["ruwe"].notna()
            scores[ruwe_mask] *= 1.0 / (1.0 + (df.loc[ruwe_mask, "ruwe"] - 1.0) ** 2)

        # Adjust for parallax quality
        if "parallax" in df.columns and "parallax_error" in df.columns:
            parallax_mask = (
                df["parallax"].notna()
                & df["parallax_error"].notna()
                & (df["parallax_error"] > 0)
            )
            snr = (
                df.loc[parallax_mask, "parallax"]
                / df.loc[parallax_mask, "parallax_error"]
            )
            scores[parallax_mask] *= np.clip(snr / 10.0, 0, 1)

        return scores

    def _create_empty_dataset(self, output_path: Path) -> None:
        """Create empty dataset with proper schema."""
        columns = getattr(self.config, "columns_to_keep", [])
        if not columns:
            # Default GAIA columns if none specified
            columns = [
                "gaia_dr2_source_id",
                "gaia_dr3_source_id",
                "ra",
                "dec",
                "parallax",
                "parallax_error",
                "pmra",
                "pmra_error",
                "pmdec",
                "pmdec_error",
                "phot_g_mean_mag",
            ]

        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_parquet(output_path, index=False)

    def _create_fallback_dataset(
        self, output_path: Path, error_msg: str
    ) -> FetchResult:
        """Create fallback dataset when GAIA fetch fails."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._create_empty_dataset(output_path)

            return FetchResult(
                source_id=self.config.id,
                path=output_path,
                success=True,
                error_message=f"GAIA fetch failed, created empty dataset: {error_msg}",
                rows_fetched=0,
                size_bytes=output_path.stat().st_size,
            )

        except Exception as create_err:
            return FetchResult(
                source_id=self.config.id,
                path=None,
                success=False,
                error_message=f"Failed to create fallback: {create_err}",
            )
