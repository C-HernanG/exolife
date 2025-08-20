"""
Unit standardisation and physical range checks for the ExoLife pipeline.

These functions encapsulate the logic for converting heterogeneous
astrophysical measurements into a common unit system and for enforcing
basic physical plausibility.  They operate on pandas DataFrames and
return transformed copies, leaving the original untouched.  This
module was extracted from the monolithic ``ConfigurablePipelineMerger``
to adhere to the Single Responsibility Principle and to make it
reusable in other contexts.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def standardize_units(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize units across all measurements.

    This implementation converts any planetary radii given in Jupiter
    units to Earth units, and planetary masses given in Jupiter units
    to Earth units.  New columns ``pl_rade`` and ``pl_masse`` are
    populated with the standardised values.  Where values are
    available in both unit systems, the Earth units take precedence.

    For each converted column a companion ``<col>_units`` column is
    created to record the original units (``R_earth``, ``R_jup``,
    ``M_earth``, ``M_jup``).  This aids reproducibility and enables
    CI tests to assert unit consistency.  Stellar quantities are
    assumed to already be in solar units as per the configuration.

    Args:
        df: Input DataFrame.

    Returns:
        A new DataFrame with unit columns added or updated.
    """
    standardized_df = df.copy()

    # ------------------------------------------------------------------
    # 1) Planetary radius conversion
    #    Accept values in either Earth radii (pl_rade) or Jupiter radii
    #    (pl_radj).  Convert Jupiter radii to Earth units and record
    #    provenance in a companion units column.  If both are present,
    #    prioritise Earth radius as canonical.
    R_JUP_TO_REARTH = 11.209
    if (
        "pl_rade" in standardized_df.columns
        and standardized_df["pl_rade"].notna().any()
    ):
        standardized_df["pl_rade_units"] = "R_earth"
    elif (
        "pl_radj" in standardized_df.columns
        and standardized_df["pl_radj"].notna().any()
    ):
        standardized_df["pl_rade"] = standardized_df["pl_radj"] * R_JUP_TO_REARTH
        standardized_df["pl_rade_units"] = "R_jup"
    # If neither column exists, leave radius undefined

    # 2) Planetary mass conversion
    #    Accept values in Earth masses (pl_masse) or Jupiter masses
    #    (pl_massj).  Convert Jupiter masses to Earth units and record
    #    provenance in a companion units column.  If both are present,
    #    prioritise Earth mass.
    M_JUP_TO_MEARTH = 317.83
    if (
        "pl_masse" in standardized_df.columns
        and standardized_df["pl_masse"].notna().any()
    ):
        standardized_df["pl_masse_units"] = "M_earth"
    elif (
        "pl_massj" in standardized_df.columns
        and standardized_df["pl_massj"].notna().any()
    ):
        standardized_df["pl_masse"] = standardized_df["pl_massj"] * M_JUP_TO_MEARTH
        standardized_df["pl_masse_units"] = "M_jup"

    # 3) Stellar units and other orbital parameters
    #    Assign explicit units for stellar parameters and orbital
    #    quantities if they exist.  Conversion is not performed here
    #    because the primary catalogs already provide values in
    #    canonical units, but unit labels are added for provenance.
    unit_map: Dict[str, str | None] = {
        "st_lum": "L_sun",
        "st_rad": "R_sun",
        "st_mass": "M_sun",
        "st_teff": "K",
        "pl_orbsmax": "AU",
        "pl_orbper": "days",
        "pl_orbeccen": None,  # dimensionless
    }
    for col, unit in unit_map.items():
        if col in standardized_df.columns:
            # Only set units if at least one non-null value exists
            if standardized_df[col].notna().any():
                standardized_df[f"{col}_units"] = unit

    # 4) Distances and parallaxes
    #    Parallaxes are assumed to be in milliarcseconds (mas).  If a
    #    parallax column exists, record units; distances will be
    #    computed later in the ``compute_distances`` function and assigned
    #    'pc' units.
    for par_col in [
        c
        for c in standardized_df.columns
        if c.lower().endswith("parallax") and not c.lower().endswith("error")
    ]:
        if standardized_df[par_col].notna().any():
            standardized_df[f"{par_col}_units"] = "mas"

    return standardized_df


def apply_physical_range_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce physical plausibility on key orbital and planetary parameters.

    For each parameter with known physical bounds, values falling
    outside the allowed domain are replaced with NaN and a boolean
    indicator column ``<param>_flag_invalid`` is set to ``True``.  The
    following checks are applied:

      - Orbital period (pl_orbper): must be positive
      - Semi-major axis (pl_orbsmax): must be positive
      - Eccentricity (pl_orbeccen): 0 ≤ e < 1
      - Planet radius (pl_rade): must be positive
      - Planet mass (pl_masse): must be positive

    This function operates on the standardised unit columns so that
    conversions have already been applied.  Flags for invalid values
    facilitate downstream quality assurance.

    Additionally, unrealistically long orbital periods (>1e5 days) are
    corrected using Kepler's third law if the necessary inputs are
    available.  A ``pl_orbper_corrected`` flag is added when such a
    correction is applied.

    Args:
        df: Input DataFrame after unit standardisation.

    Returns:
        A DataFrame with invalid values set to NaN and flag columns
        indicating which values were invalid or corrected.
    """
    result = df.copy()
    checks: Dict[str, callable] = {
        "pl_orbper": lambda x: x > 0,
        "pl_orbsmax": lambda x: x > 0,
        "pl_orbeccen": lambda x: (x >= 0) & (x < 1),
        "pl_rade": lambda x: x > 0,
        "pl_masse": lambda x: x > 0,
    }
    for col, cond in checks.items():
        if col in result.columns:
            mask = result[col].apply(lambda v: cond(v) if pd.notna(v) else True)
            # Flag invalid
            flag_col = f"{col}_flag_invalid"
            result[flag_col] = ~mask
            # Replace invalid values with NaN
            result.loc[~mask, col] = np.nan

    # Correct unrealistically long orbital periods using Kepler's law.
    # If pl_orbper is extremely large (e.g., >1e5 days), recompute
    # period from semi-major axis and stellar mass when available:
    # P (days) = sqrt(a^3 / M_star) * 365.25, where a (AU) and
    # M_star (M_sun).
    if all(c in result.columns for c in ["pl_orbper", "pl_orbsmax", "st_mass"]):
        mask_fix = (
            (result["pl_orbper"] > 1.0e5)
            & result["pl_orbsmax"].notna()
            & result["st_mass"].notna()
        )
        if mask_fix.any():
            a_vals = result.loc[mask_fix, "pl_orbsmax"].astype(float)
            mstar_vals = result.loc[mask_fix, "st_mass"].astype(float)
            # Kepler's third law: P(years) = sqrt(a^3 / M_star)
            # Convert to days
            period_days = np.sqrt(a_vals**3 / mstar_vals) * 365.25
            result.loc[mask_fix, "pl_orbper"] = period_days
            # Flag corrected records for provenance
            result.loc[mask_fix, "pl_orbper_corrected"] = True

    return result
