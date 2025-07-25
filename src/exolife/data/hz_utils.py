"""
hz_utils – compute conservative / optimistic habitable‑zone limits.

Implements the polynomial prescription from Kopparapu et al. 2014 (ApJ 787, L29).
S_eff(T_eff) = S_eff,⊙ + a·T' + b·T'^2 + c·T'^3 + d·T'^4  ,
where T' = T_eff – 5780 K.

References
----------
- Kopparapu, R. K. et al. 2013, ApJ, 765, 131  (original)
- Kopparapu, R. K. et al. 2014, ApJ Letters, 787, L29  (updated coefficients)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Hard‑coded coefficients  (Table 1 in Kopparapu 2014)
# ------------------------------------------------------------------
_KOPPARAPU_COEFFS = {
    #  limit_name : (S_eff⊙,        a,          b,          c,           d)
    "RecentVenus": (1.776, 1.433e-4, 3.395e-9, 7.636e-12, 1.195e-15),
    "RunawayGreenhouse": (1.107, 1.332e-4, 3.097e-9, 7.256e-12, 1.195e-15),
    "MaximumGreenhouse": (0.356, 6.171e-5, 1.698e-9, 3.198e-12, 5.575e-16),
    "EarlyMars": (0.320, 5.547e-5, 1.526e-9, 2.874e-12, 5.011e-16),
}

__all__ = [
    "hz_flux",
    "hz_distance",
    "hz_edges",
    "add_hz_edges_to_df",
]


# ------------------------------------------------------------------
# Core helpers
# ------------------------------------------------------------------
def _seff(teff: float, coeffs: tuple[float, float, float, float, float]) -> float:
    """Stellar flux at HZ limit for a single T_eff (Kelvin)."""
    seff_sun, a, b, c, d = coeffs
    dt = teff - 5780.0
    return seff_sun + a * dt + b * dt**2 + c * dt**3 + d * dt**4


def hz_flux(teff: float, limit: str = "RunawayGreenhouse") -> float:
    """
    Returns S_eff (stellar flux in Solar units) at the chosen HZ limit.

    Parameters
    ----------
    teff : float
        Stellar effective temperature in Kelvin.
    limit : str
        One of 'RecentVenus', 'RunawayGreenhouse', 'MaximumGreenhouse', 'EarlyMars'.

    Notes
    -----
    - **Conservative HZ**: RunawayGreenhouse (inner) … MaximumGreenhouse (outer)
    - **Optimistic HZ**  : RecentVenus (inner) … EarlyMars (outer)
    """
    if limit not in _KOPPARAPU_COEFFS:
        raise KeyError(f"Unknown limit '{limit}'. Valid: {list(_KOPPARAPU_COEFFS)}")
    return _seff(teff, _KOPPARAPU_COEFFS[limit])


def hz_distance(seff: float, lum: float) -> float:
    """
    Convert stellar flux S_eff and luminosity L★ (in L_☉) to orbital distance (AU).

    Formula
    -------
        a_HZ = sqrt(L★ / S_eff)
    """
    return np.sqrt(lum / seff)


def hz_edges(teff: float, lum: float, optimistic: bool = False) -> tuple[float, float]:
    """
    Compute inner & outer HZ edges (AU).

    optimistic == False → conservative HZ (Runaway GH … Max GH)
    optimistic == True  → optimistic HZ  (Recent Venus … Early Mars)
    """
    if optimistic:
        inner = hz_distance(hz_flux(teff, "RecentVenus"), lum)
        outer = hz_distance(hz_flux(teff, "EarlyMars"), lum)
    else:
        inner = hz_distance(hz_flux(teff, "RunawayGreenhouse"), lum)
        outer = hz_distance(hz_flux(teff, "MaximumGreenhouse"), lum)
    return inner, outer


# ------------------------------------------------------------------
# Vectorised helper for data‑frames
# ------------------------------------------------------------------
def add_hz_edges_to_df(
    df: pd.DataFrame,
    teff_col: str = "st_teff",
    lum_col: str = "st_lum",
    optimistic: bool = False,
    prefix: str = "hz_",
) -> pd.DataFrame:
    """
    Append columns for inner & outer HZ limits (AU) to a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns with stellar T_eff [K] and luminosity [L_☉].
    teff_col : str
        Column name for T_eff.
    lum_col : str
        Column name for stellar luminosity.
    optimistic : bool
        If True use optimistic Recent Venus/Early Mars bounds, else conservative.
    prefix : str
        Added column names become '{prefix}inner' and '{prefix}outer'.

    Returns
    -------
    pandas.DataFrame  (same object, modified in‑place)
    """
    inner_vals, outer_vals = [], []
    for teff, lum in zip(df[teff_col].to_numpy(), df[lum_col].to_numpy()):
        if np.isnan(teff) or np.isnan(lum):
            inner_vals.append(np.nan)
            outer_vals.append(np.nan)
        else:
            inner, outer = hz_edges(float(teff), float(lum), optimistic=optimistic)
            inner_vals.append(inner)
            outer_vals.append(outer)

    df[f"{prefix}inner"] = inner_vals
    df[f"{prefix}outer"] = outer_vals
    return df
