# File: src/exolife/data/utils/hz_utils.py
"""
Habitable Zone Utilities

Compute conservative and optimistic habitable-zone (HZ) limits using polynomial
prescriptions from Kopparapu et al. (2014).

Functions:
- hz_flux(teff, limit): Stellar flux (S_eff) at specified HZ boundary.
- hz_distance(seff, luminosity): Orbital distance (AU) given stellar flux & luminosity.
- hz_edges(teff, luminosity, optimistic): Inner & outer HZ distances.
- add_hz_edges(df, teff_col, lum_col, optimistic, prefix): Vectorized addition
  of HZ edge columns to a pandas DataFrame.
"""

from __future__ import annotations

import json
import logging
from typing import Literal, Tuple

import numpy as np
import pandas as pd

from exolife.config import EXTERNAL_DIR

# Initialize module logger
logger = logging.getLogger("exolife.data.utils.hz_utils")

# ------------------------------------------------------------------
# Coefficients for Kopparapu et al. (2014), Table 1
# ------------------------------------------------------------------
_KOPPARAPU_COEFFS = EXTERNAL_DIR / "kopparapu_coeffs.json"

try:
    with open(_KOPPARAPU_COEFFS) as f:
        raw = json.load(f)
    # Convert to a mapping of coefficients
    _KOPPARAPU_COEFFS: dict[str, Tuple[float, float, float, float, float]] = {
        k: tuple(v) for k, v in raw.items()
    }
    VALID_LIMITS = tuple(_KOPPARAPU_COEFFS.keys())
    logger.info("Loaded HZ coefficients for %d limits.", len(_KOPPARAPU_COEFFS))
except Exception as e:
    logger.error("Failed to load Kopparapu coefficients: %s", e)
    _KOPPARAPU_COEFFS = {}


def hz_flux(
    teff: float,
    limit: Literal[
        "RecentVenus", "RunawayGreenhouse", "MaximumGreenhouse", "EarlyMars"
    ] = "RunawayGreenhouse",
) -> float:
    """
    Compute stellar flux (S_eff) at a given HZ boundary for a star
    of temperature teff (K).

    Args:
        teff: Stellar effective temperature in Kelvin.
        limit: HZ boundary name; one of VALID_LIMITS.

    Returns:
        S_eff in Solar flux units.

    Raises:
        KeyError: If `limit` is not in VALID_LIMITS.
    """
    try:
        seff_sun, a, b, c, d = _KOPPARAPU_COEFFS[limit]
    except KeyError:
        logger.error("Invalid HZ limit '%s'; valid options: %s", limit, VALID_LIMITS)
        raise

    dt = teff - 5780.0
    # Polynomial expansion
    return seff_sun + a * dt + b * dt**2 + c * dt**3 + d * dt**4


def hz_distance(seff: float, luminosity: float) -> float:
    """
    Convert stellar flux (S_eff) and luminosity (L/L_sun) to orbital distance in AU.

    Args:
        seff: Stellar flux in Solar units.
        luminosity: Stellar luminosity in Solar units.

    Returns:
        Orbital distance in astronomical units (AU).
    """
    if seff <= 0 or luminosity < 0:
        logger.warning(
            "Non-positive flux or luminosity: seff=%s, luminosity=%s", seff, luminosity
        )
        return np.nan
    return np.sqrt(luminosity / seff)


def hz_edges(
    teff: float, luminosity: float, optimistic: bool = False
) -> Tuple[float, float]:
    """
    Compute inner and outer habitable zone (HZ) distances for a star.

    Args:
        teff: Stellar effective temperature (K).
        luminosity: Stellar luminosity (L/L_sun).
        optimistic: If True, use optimistic bounds (RecentVenus, EarlyMars).
                    Otherwise, use conservative bounds (RunawayGreenhouse,
                    MaximumGreenhouse).

    Returns:
        (inner_AU, outer_AU)
    """
    inner_limit = "RecentVenus" if optimistic else "RunawayGreenhouse"
    outer_limit = "EarlyMars" if optimistic else "MaximumGreenhouse"
    seff_inner = hz_flux(teff, inner_limit)
    seff_outer = hz_flux(teff, outer_limit)
    return hz_distance(seff_inner, luminosity), hz_distance(seff_outer, luminosity)


def add_hz_edges_to_df(
    df: pd.DataFrame,
    teff_col: str = "st_teff",
    lum_col: str = "st_lum",
    optimistic: bool = False,
    prefix: str = "hz_",
) -> pd.DataFrame:
    """
    Vectorized addition of HZ inner and outer edge columns to a DataFrame.

    Args:
        df: Input DataFrame containing stellar temperature and luminosity columns.
        teff_col: Column name for effective temperature.
        lum_col: Column name for luminosity.
        optimistic: Choose optimistic or conservative HZ boundaries.
        prefix: Prefix for new columns: '{prefix}inner', '{prefix}outer'.

    Returns:
        DataFrame with added HZ edge columns (modifies in-place).
    """
    # Extract arrays and mask invalid values
    teff = df[teff_col].to_numpy(dtype=float)
    lum = df[lum_col].to_numpy(dtype=float)

    # Compute all limits
    inner_seff = np.where(
        np.isnan(teff),
        np.nan,
        hz_flux(teff, "RecentVenus" if optimistic else "RunawayGreenhouse"),
    )
    outer_seff = np.where(
        np.isnan(teff),
        np.nan,
        hz_flux(teff, "EarlyMars" if optimistic else "MaximumGreenhouse"),
    )
    # Compute distances only where flux is positive, leave others as NaN
    inner = np.full_like(lum, np.nan, dtype=float)
    outer = np.full_like(lum, np.nan, dtype=float)

    valid_inner = inner_seff > 0
    valid_outer = outer_seff > 0
    inner[valid_inner] = np.sqrt(lum[valid_inner] / inner_seff[valid_inner])
    outer[valid_outer] = np.sqrt(lum[valid_outer] / outer_seff[valid_outer])

    # Assign to DataFrame
    df[f"{prefix}inner"] = inner
    df[f"{prefix}outer"] = outer
    logger.info(
        "Added HZ edges to DataFrame: columns '%sinner', '%souter'", prefix, prefix
    )
    return df
