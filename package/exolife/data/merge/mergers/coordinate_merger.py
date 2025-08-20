"""
Coordinate-based merger for astronomical position matching.

This merger handles cross-matching based on celestial coordinates with
proper motion correction, cone search, and uncertainty handling.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..base_merger import BaseMerger


class CoordinateMerger(BaseMerger):
    """
    Merger for coordinate-based astronomical cross-matching.

    Features:
    - Cone search with configurable tolerance
    - Proper motion correction between epochs
    - Multiple coordinate systems support
    - Quality metrics for matches
    """

    @property
    def merger_name(self) -> str:
        return "Coordinate-Based Merger"

    @property
    def supported_strategies(self) -> List[str]:
        return [
            "cone_search",
            "coordinate_match",
            "position_match",
            "astrometric_match",
        ]

    def execute_merge(self, data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Execute coordinate-based merging."""
        if not data_sources:
            raise ValueError("No data sources provided")

        # Start with the first dataset as primary
        primary_name = list(data_sources.keys())[0]
        merged_df = data_sources[primary_name].copy()

        self.logger.info(
            f"Starting coordinate merge with: {primary_name} ({len(merged_df)} rows)"
        )

        # Merge each additional dataset
        for source_name, source_df in list(data_sources.items())[1:]:
            merged_df = self._merge_by_coordinates(merged_df, source_df, source_name)

        return merged_df

    def _merge_by_coordinates(
        self, primary_df: pd.DataFrame, source_df: pd.DataFrame, source_name: str
    ) -> pd.DataFrame:
        """Merge datasets using coordinate matching with cone search."""
        # Get coordinate columns
        primary_coords = self._find_coordinate_columns(primary_df)
        source_coords = self._find_coordinate_columns(source_df)

        if not primary_coords or not source_coords:
            self.logger.warning(f"Missing coordinate columns for {source_name}")
            return primary_df

        primary_ra_col, primary_dec_col = primary_coords
        source_ra_col, source_dec_col = source_coords

        # Get tolerance from config or use default
        tolerance_arcsec = getattr(self.config, "coordinate_tolerance", 3.0)
        tolerance_deg = tolerance_arcsec / 3600.0

        self.logger.info(
            f'Coordinate matching {source_name} with {tolerance_arcsec}" tolerance'
        )

        # Apply proper motion correction if available
        source_df_corrected = self._apply_proper_motion_correction(
            source_df.copy(), source_ra_col, source_dec_col
        )

        # Perform cone search matching
        matches = self._cone_search_match(
            primary_df,
            source_df_corrected,
            primary_ra_col,
            primary_dec_col,
            source_ra_col,
            source_dec_col,
            tolerance_deg,
        )

        # Apply matches to create merged dataset
        merged_df = self._apply_coordinate_matches(
            primary_df, source_df_corrected, matches, source_name
        )

        self.logger.info(f"Found {len(matches)} coordinate matches for {source_name}")
        return merged_df

    def _find_coordinate_columns(self, df: pd.DataFrame) -> Optional[tuple[str, str]]:
        """Find RA and Dec columns in a DataFrame."""
        ra_col = None
        dec_col = None

        # Look for RA column
        for col in df.columns:
            if col.lower() in ["ra", "ra_deg", "right_ascension", "alpha"]:
                ra_col = col
                break

        # Look for Dec column
        for col in df.columns:
            if col.lower() in ["dec", "dec_deg", "declination", "delta"]:
                dec_col = col
                break

        return (ra_col, dec_col) if ra_col and dec_col else None

    def _apply_proper_motion_correction(
        self, df: pd.DataFrame, ra_col: str, dec_col: str
    ) -> pd.DataFrame:
        """Apply proper motion correction to coordinates."""
        # Look for proper motion columns
        pm_ra_cols = [
            col for col in df.columns if "pmra" in col.lower() or "pm_ra" in col.lower()
        ]
        pm_dec_cols = [
            col
            for col in df.columns
            if "pmdec" in col.lower() or "pm_dec" in col.lower()
        ]

        if not pm_ra_cols or not pm_dec_cols:
            self.logger.debug("No proper motion data available")
            return df

        pm_ra_col = pm_ra_cols[0]
        pm_dec_col = pm_dec_cols[0]

        # Default epoch correction (can be made configurable)
        epoch_diff_years = getattr(self.config, "epoch_correction", 0.0)

        if epoch_diff_years == 0.0:
            return df

        # Apply proper motion correction
        pm_ra_deg_per_yr = df[pm_ra_col] / (3600.0 * 1000.0)  # mas/yr to deg/yr
        pm_dec_deg_per_yr = df[pm_dec_col] / (3600.0 * 1000.0)  # mas/yr to deg/yr

        df[f"{ra_col}_corrected"] = df[ra_col] + (
            pm_ra_deg_per_yr * epoch_diff_years / np.cos(np.radians(df[dec_col]))
        )
        df[f"{dec_col}_corrected"] = df[dec_col] + (
            pm_dec_deg_per_yr * epoch_diff_years
        )

        self.logger.info(f"Applied {epoch_diff_years}yr proper motion correction")
        return df

    def _cone_search_match(
        self,
        primary_df: pd.DataFrame,
        source_df: pd.DataFrame,
        primary_ra_col: str,
        primary_dec_col: str,
        source_ra_col: str,
        source_dec_col: str,
        tolerance_deg: float,
    ) -> List[Dict]:
        """Perform cone search matching between two catalogs."""
        matches = []

        # Use corrected coordinates if available
        source_ra_col_actual = (
            f"{source_ra_col}_corrected"
            if f"{source_ra_col}_corrected" in source_df.columns
            else source_ra_col
        )
        source_dec_col_actual = (
            f"{source_dec_col}_corrected"
            if f"{source_dec_col}_corrected" in source_df.columns
            else source_dec_col
        )

        # Clean coordinate data
        primary_coords = primary_df[[primary_ra_col, primary_dec_col]].dropna()
        source_coords = source_df[
            [source_ra_col_actual, source_dec_col_actual]
        ].dropna()

        if len(primary_coords) == 0 or len(source_coords) == 0:
            return matches

        # For each primary source, find closest secondary source within tolerance
        for primary_idx, (primary_ra, primary_dec) in primary_coords.iterrows():
            # Calculate angular separations
            separations = self._calculate_angular_separation(
                primary_ra,
                primary_dec,
                source_coords[source_ra_col_actual].values,
                source_coords[source_dec_col_actual].values,
            )

            # Find matches within tolerance
            within_tolerance = separations <= tolerance_deg
            if np.any(within_tolerance):
                # Get closest match
                min_idx = np.argmin(separations)
                min_separation = separations[min_idx]

                if min_separation <= tolerance_deg:
                    source_idx = source_coords.index[min_idx]
                    matches.append(
                        {
                            "primary_idx": primary_idx,
                            "source_idx": source_idx,
                            "separation_arcsec": min_separation * 3600.0,
                            "separation_deg": min_separation,
                        }
                    )

        return matches

    def _calculate_angular_separation(
        self, ra1: float, dec1: float, ra2_array: np.ndarray, dec2_array: np.ndarray
    ) -> np.ndarray:
        """Calculate angular separation using haversine formula."""
        # Convert to radians
        ra1_rad = np.radians(ra1)
        dec1_rad = np.radians(dec1)
        ra2_rad = np.radians(ra2_array)
        dec2_rad = np.radians(dec2_array)

        # Haversine formula
        dra = ra2_rad - ra1_rad
        ddec = dec2_rad - dec1_rad

        a = (
            np.sin(ddec / 2) ** 2
            + np.cos(dec1_rad) * np.cos(dec2_rad) * np.sin(dra / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))

        return c  # Returns separation in radians, convert to degrees by caller

    def _apply_coordinate_matches(
        self,
        primary_df: pd.DataFrame,
        source_df: pd.DataFrame,
        matches: List[Dict],
        source_name: str,
    ) -> pd.DataFrame:
        """Apply coordinate matches to create merged dataset."""
        merged_df = primary_df.copy()

        if not matches:
            # Add empty columns for this source
            source_columns = [
                col for col in source_df.columns if not col.endswith("_corrected")
            ]
            for col in source_columns:
                merged_df[f"{source_name}_{col}"] = None
            return merged_df

        # Create mapping of enrichment data
        enrichment_data = {}
        for match in matches:
            primary_idx = match["primary_idx"]
            source_idx = match["source_idx"]

            # Add all source columns with prefix
            for col in source_df.columns:
                if not col.endswith("_corrected"):
                    enrichment_data.setdefault(f"{source_name}_{col}", {})[
                        primary_idx
                    ] = source_df.loc[source_idx, col]

            # Add match quality metrics
            enrichment_data.setdefault(f"{source_name}_match_separation_arcsec", {})[
                primary_idx
            ] = match["separation_arcsec"]
            enrichment_data.setdefault(f"{source_name}_match_quality", {})[
                primary_idx
            ] = "coordinate_match"

        # Apply enrichment data to merged DataFrame
        for col_name, data_dict in enrichment_data.items():
            merged_df[col_name] = merged_df.index.map(data_dict)

        return merged_df
