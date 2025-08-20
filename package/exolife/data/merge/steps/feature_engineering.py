"""
Feature engineering utilities for the ExoLife pipeline.

This module encapsulates the derivation of physical metrics and
uncertainty propagation for astrophysical data.  Functions in this
module operate on pandas DataFrames and, where necessary, accept the
pipeline configuration to determine behaviour such as the number of
Monte Carlo samples or whether to apply zero‑point parallax corrections.

By extracting this logic from the monolithic merger class we enable
reuse of these feature engineering routines in isolation and make
testing and maintenance simpler.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Import from the top-level data utils rather than merge.utils.  The extra dot
# Import the shared helper function from the data utils module
from exolife.data.utils import add_hz_edges_to_df


def derive_features_with_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    """Derive physical features with uncertainty propagation via Monte Carlo.

    This function adds habitable zone edges (both conservative and
    optimistic) and computes a series of derived physical quantities
    including stellar flux, equilibrium temperature, surface gravity,
    escape velocity and the position of the planet within the habitable
    zone.  It delegates detailed calculations to specialised helper
    functions defined in this module.  The original DataFrame is not
    modified; a new enriched DataFrame is returned.

    Args:
        df: Input DataFrame containing at least the columns required
            for each derived metric (e.g., ``st_teff``, ``st_lum``,
            ``pl_orbsmax``).

    Returns:
        A new DataFrame with additional derived columns.
    """
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
    enriched_df = calculate_stellar_flux(enriched_df)
    enriched_df = calculate_equilibrium_temperature(enriched_df)
    enriched_df = calculate_surface_gravity(enriched_df)
    enriched_df = calculate_escape_velocity(enriched_df)
    enriched_df = calculate_hz_positions(enriched_df)
    return enriched_df


def calculate_stellar_flux(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate stellar flux received by each planet.

    Stellar flux is computed in units of the solar constant (``S_earth``)
    using the stellar luminosity in solar units and the orbital semi‑major
    axis.  If the luminosity is provided in logarithmic units (log10 of
    solar luminosity), it is converted to linear units prior to the
    calculation.  The result is stored in the ``stellar_flux_s`` column.

    Args:
        df: DataFrame containing ``st_lum`` and ``pl_orbsmax``.

    Returns:
        The input DataFrame with an additional ``stellar_flux_s`` column.
    """
    lum = df["st_lum"].copy()
    # Convert negative or zero luminosities (interpreted as log10)
    lum_linear = lum.where(lum > 0, 10**lum)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["stellar_flux_s"] = lum_linear / (df["pl_orbsmax"] ** 2)
    return df


def calculate_equilibrium_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the planetary equilibrium temperature.

    The equilibrium temperature is computed following a grey body
    approximation with explicit albedo (A) and heat redistribution
    factor (f).  By default, A=0.3 and f=1.0 if the columns are not
    present.  The formula used is:

        T_eq = T_eff * sqrt(R_star / (2 a)) * ( (1 - A)^(1/4) ) * f^(1/4)

    All quantities are assumed to already be in canonical units
    (Kelvin, Solar radii, AU).  Physically implausible temperatures
    (negative or exceeding ~20,000 K) are set to NaN.

    Args:
        df: DataFrame containing ``st_teff``, ``st_rad`` and ``pl_orbsmax``.

    Returns:
        The DataFrame with an additional ``equilibrium_temperature`` column.
    """
    # Ensure default albedo and redistribution columns exist
    if "albedo" not in df.columns:
        df["albedo"] = 0.3
    if "redistribution_factor" not in df.columns:
        df["redistribution_factor"] = 1.0
    # Compute equilibrium temperature
    with np.errstate(invalid="ignore"):
        term1 = df["st_teff"]
        term2 = np.sqrt(df["st_rad"] / (2.0 * df["pl_orbsmax"]))
        term3 = (1.0 - df["albedo"]) ** 0.25
        term4 = df["redistribution_factor"] ** 0.25
        df["equilibrium_temperature"] = term1 * term2 * term3 * term4
    # Flag physically implausible values
    mask_invalid = (df["equilibrium_temperature"] < 0) | (
        df["equilibrium_temperature"] > 2.0e4
    )
    df.loc[mask_invalid, "equilibrium_temperature"] = np.nan
    return df


def calculate_surface_gravity(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate planetary surface gravity.

    Surface gravity is computed as G * M / R^2, expressed in m/s^2.
    Mass and radius must be provided in Earth units.  Negative,
    non‑finite or exceedingly large gravities (>2000 m/s^2) are set
    to NaN.

    Args:
        df: DataFrame containing ``pl_masse`` and ``pl_rade``.

    Returns:
        DataFrame with ``surface_gravity`` column added.
    """
    G = 6.67430e-11  # m^3 kg^-1 s^-2
    M_earth = 5.972e24
    R_earth = 6.371e6
    with np.errstate(divide="ignore", invalid="ignore"):
        df["surface_gravity"] = (
            G * df["pl_masse"] * M_earth / (df["pl_rade"] * R_earth) ** 2
        )
    # Mask invalid or implausible values
    mask_invalid = (
        (df["surface_gravity"] < 0)
        | (~np.isfinite(df["surface_gravity"]))
        | (df["surface_gravity"] > 2.0e3)
    )
    df.loc[mask_invalid, "surface_gravity"] = np.nan
    return df


def calculate_escape_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate planetary escape velocity.

    Escape velocity is computed as sqrt(2 * G * M / R) and returned in
    km/s.  Mass and radius are assumed to be provided in Earth units.
    Negative, non‑finite or implausibly large values (>100 km/s) are
    set to NaN.

    Args:
        df: DataFrame containing ``pl_masse`` and ``pl_rade``.

    Returns:
        DataFrame with ``escape_velocity`` column added.
    """
    G = 6.67430e-11  # m^3 kg^-1 s^-2
    M_earth = 5.972e24
    R_earth = 6.371e6
    with np.errstate(divide="ignore", invalid="ignore"):
        df["escape_velocity"] = (
            np.sqrt(2 * G * df["pl_masse"] * M_earth / (df["pl_rade"] * R_earth))
            / 1000.0
        )
    mask_invalid = (
        (df["escape_velocity"] < 0)
        | (~np.isfinite(df["escape_velocity"]))
        | (df["escape_velocity"] > 1.0e2)
    )
    df.loc[mask_invalid, "escape_velocity"] = np.nan
    return df


def calculate_hz_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate position within habitable zones.

    Two dimensionless metrics are computed: ``hz_position_conservative`` and
    ``hz_position_optimistic``.  They measure the planet's orbital
    distance relative to the width of the conservative and optimistic
    habitable zones, respectively.  Specifically:

        hz_position_conservative = (a - hz_inner) / (hz_outer - hz_inner)
        hz_position_optimistic  = (a - hz_opt_inner) / (hz_opt_outer - hz_opt_inner)

    Args:
        df: DataFrame containing ``pl_orbsmax``, ``hz_inner``, ``hz_outer``,
            ``hz_opt_inner`` and ``hz_opt_outer``.

    Returns:
        DataFrame with additional columns describing relative HZ positions.
    """
    if "pl_orbsmax" in df.columns:
        hz_width_cons = df["hz_outer"] - df["hz_inner"]
        hz_width_opt = df["hz_opt_outer"] - df["hz_opt_inner"]
        df["hz_position_conservative"] = (
            df["pl_orbsmax"] - df["hz_inner"]
        ) / hz_width_cons
        df["hz_position_optimistic"] = (
            df["pl_orbsmax"] - df["hz_opt_inner"]
        ) / hz_width_opt
    return df


def compute_distances(
    df: pd.DataFrame, pipeline_config: Optional[dict] = None
) -> pd.DataFrame:
    """Derive stellar distances and provenance.

    Prefer dedicated DR3 geometric or photogeometric distances when
    available (e.g., columns ending with ``distance_gspphot``).  If
    unavailable, compute distance from parallax using the inverse
    relationship (distance_pc = 1000 / parallax_mas) with zero‑point
    correction if configured.  A ``distance_pc`` column and
    ``distance_provenance`` column are added.  A constant ``distance_units``
    column with value ``pc`` is also added.

    Args:
        df: Input DataFrame.
        pipeline_config: The pipeline configuration dict.  Only the
            ``cross_identification.stellar_crossmatch.parallax_zero_point_correction``
            flag is consulted.  This argument may be ``None``.

    Returns:
        DataFrame with additional distance columns.
    """
    result = df.copy()
    distance_col = None
    # Attempt to find geometric distance columns (e.g., from Gaia DR3 GSPPhot)
    for col in result.columns:
        if col.lower().endswith("distance_gspphot") and result[col].notna().any():
            distance_col = col
            break
    # Determine parallax column (prefer DR3 parallax over DR2)
    parallax_col = None
    for col in result.columns:
        if col.lower().endswith("parallax") and not col.lower().endswith("error"):
            parallax_col = col
            break
    distances = []
    provenance = []
    apply_zp_correction = False
    if pipeline_config:
        try:
            apply_zp_correction = bool(
                pipeline_config.get("cross_identification", {})
                .get("stellar_crossmatch", {})
                .get("parallax_zero_point_correction", False)
            )
        except Exception:
            apply_zp_correction = False
    zero_point_offset = 0.017
    for _, row in result.iterrows():
        dist = np.nan
        prov = None
        if distance_col and pd.notna(row[distance_col]):
            dist = row[distance_col]
            prov = "geometric"
        elif parallax_col and pd.notna(row[parallax_col]):
            parallax_mas = row[parallax_col]
            parallax_corr = (
                parallax_mas - zero_point_offset
                if apply_zp_correction
                else parallax_mas
            )
            if parallax_corr > 0:
                dist = 1000.0 / parallax_corr
                prov = "inv_parallax"
        distances.append(dist)
        provenance.append(prov)
    result["distance_pc"] = distances
    result["distance_provenance"] = provenance
    result["distance_units"] = "pc"
    return result


def monte_carlo_propagation(
    df: pd.DataFrame,
    pipeline_config: Optional[dict] = None,
    num_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Perform Monte Carlo sampling to propagate uncertainties in derived metrics.

    This method samples input parameters (``st_teff``, ``st_lum``, ``pl_rade``,
    ``pl_masse`` and ``pl_orbsmax``) according to their measurement errors,
    computes derived metrics (stellar flux, equilibrium temperature,
    surface gravity, escape velocity) for each draw, and stores summary
    statistics (mean, standard deviation, median, 16th and 84th
    percentiles) for each metric.  Derived metrics are suffixed with
    ``_mean``, ``_std``, ``_median``, ``_q16`` and ``_q84``.

    Args:
        df: Input DataFrame.
        pipeline_config: Pipeline configuration dict.  If provided and
            contains ``uncertainty_propagation.num_samples``, this value
            is used as the number of Monte Carlo samples.
        num_samples: Override the number of samples.  If ``None``, the
            pipeline configuration or a default of 1000 is used.

    Returns:
        DataFrame with additional summary statistics columns for each
        derived metric.
    """
    # Clone the input DataFrame to avoid side effects
    result = df.copy()
    # Determine number of samples
    if num_samples is None:
        if pipeline_config is not None:
            num_samples = pipeline_config.get("uncertainty_propagation", {}).get(
                "num_samples", 1000
            )
        else:
            num_samples = 1000

    # Predefine constants
    G = 6.67430e-11  # m^3 kg^-1 s^-2
    M_earth = 5.972e24  # kg
    R_earth = 6.371e6  # m

    # Helper to compute symmetric sigma from err1/err2
    def _get_sigma(row: pd.Series, param_name: str) -> float:
        err1_col = f"{param_name}err1"
        err2_col = f"{param_name}err2"
        sigma = 0.0
        if err1_col in row and pd.notna(row[err1_col]):
            sigma = max(sigma, abs(row[err1_col]))
        if err2_col in row and pd.notna(row[err2_col]):
            sigma = (
                max(sigma, abs(row[err2_col]))
                if sigma == 0
                else (abs(row[err1_col]) + abs(row[err2_col])) / 2.0
            )
        return sigma

    # Initialize containers for summary statistics
    flux_stats = {"mean": [], "std": [], "median": [], "q16": [], "q84": []}
    teq_stats = {"mean": [], "std": [], "median": [], "q16": [], "q84": []}
    g_stats = {"mean": [], "std": [], "median": [], "q16": [], "q84": []}
    vesc_stats = {"mean": [], "std": [], "median": [], "q16": [], "q84": []}

    # Iterate over rows to perform sampling
    for _, row in result.iterrows():
        # Retrieve means
        teff_mu = row.get("st_teff")
        lum_mu = row.get("st_lum")
        rad_mu = row.get("st_rad")
        a_mu = row.get("pl_orbsmax")
        rp_mu = row.get("pl_rade")
        mp_mu = row.get("pl_masse")
        albedo = row.get("albedo", 0.3)
        f_factor = row.get("redistribution_factor", 1.0)
        # Compute sigmas
        teff_sigma = _get_sigma(row, "st_teff")
        lum_sigma = _get_sigma(row, "st_lum")
        rad_sigma = _get_sigma(row, "st_rad")
        a_sigma = _get_sigma(row, "pl_orbsmax")
        rp_sigma = _get_sigma(row, "pl_rade")
        mp_sigma = _get_sigma(row, "pl_masse")
        # Draw samples; handle NaNs by using zero sigma (no variation)
        rng = np.random.default_rng(42)
        teff_samples = (
            rng.normal(teff_mu, teff_sigma, num_samples)
            if pd.notna(teff_mu)
            else np.full(num_samples, np.nan)
        )
        lum_samples = (
            rng.normal(lum_mu, lum_sigma, num_samples)
            if pd.notna(lum_mu)
            else np.full(num_samples, np.nan)
        )
        rad_samples = (
            rng.normal(rad_mu, rad_sigma, num_samples)
            if pd.notna(rad_mu)
            else np.full(num_samples, np.nan)
        )
        a_samples = (
            rng.normal(a_mu, a_sigma, num_samples)
            if pd.notna(a_mu)
            else np.full(num_samples, np.nan)
        )
        rp_samples = (
            rng.normal(rp_mu, rp_sigma, num_samples)
            if pd.notna(rp_mu)
            else np.full(num_samples, np.nan)
        )
        mp_samples = (
            rng.normal(mp_mu, mp_sigma, num_samples)
            if pd.notna(mp_mu)
            else np.full(num_samples, np.nan)
        )
        # Enforce positivity on semi-major axis, radius and mass
        a_samples = np.where(a_samples > 0, a_samples, np.nan)
        rp_samples = np.where(rp_samples > 0, rp_samples, np.nan)
        mp_samples = np.where(mp_samples > 0, mp_samples, np.nan)
        # Convert negative luminosity samples to linear units
        lum_linear = np.where(lum_samples > 0, lum_samples, 10**lum_samples)
        # Derived metrics: flux, equilibrium temperature, surface gravity, escape velocity
        with np.errstate(invalid="ignore", divide="ignore"):
            flux_samples = lum_linear / (a_samples**2)
            temp_samples = (
                teff_samples
                * np.sqrt(rad_samples / (2.0 * a_samples))
                * ((1.0 - albedo) ** 0.25)
                * (f_factor**0.25)
            )
            g_samples = G * mp_samples * M_earth / ((rp_samples * R_earth) ** 2)
            vesc_samples = (
                np.sqrt(2 * G * mp_samples * M_earth / (rp_samples * R_earth)) / 1000.0
            )
        # Helper to compute statistics ignoring NaNs

        def _stats(arr: np.ndarray) -> tuple[float, float, float, float, float]:
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                return np.nan, np.nan, np.nan, np.nan, np.nan
            return (
                float(np.mean(arr)),
                float(np.std(arr)),
                float(np.median(arr)),
                float(np.quantile(arr, 0.16)),
                float(np.quantile(arr, 0.84)),
            )

        flux_mean, flux_std, flux_med, flux_q16, flux_q84 = _stats(flux_samples)
        teq_mean, teq_std, teq_med, teq_q16, teq_q84 = _stats(temp_samples)
        g_mean, g_std, g_med, g_q16, g_q84 = _stats(g_samples)
        v_mean, v_std, v_med, v_q16, v_q84 = _stats(vesc_samples)
        # Append statistics
        for stat, value in zip(
            ["mean", "std", "median", "q16", "q84"],
            [flux_mean, flux_std, flux_med, flux_q16, flux_q84],
        ):
            flux_stats[stat].append(value)
        for stat, value in zip(
            ["mean", "std", "median", "q16", "q84"],
            [teq_mean, teq_std, teq_med, teq_q16, teq_q84],
        ):
            teq_stats[stat].append(value)
        for stat, value in zip(
            ["mean", "std", "median", "q16", "q84"],
            [g_mean, g_std, g_med, g_q16, g_q84],
        ):
            g_stats[stat].append(value)
        for stat, value in zip(
            ["mean", "std", "median", "q16", "q84"],
            [v_mean, v_std, v_med, v_q16, v_q84],
        ):
            vesc_stats[stat].append(value)

    # Assign aggregated columns to DataFrame
    for stat in ["mean", "std", "median", "q16", "q84"]:
        result[f"stellar_flux_s_{stat}"] = flux_stats[stat]
        result[f"equilibrium_temperature_{stat}"] = teq_stats[stat]
        result[f"surface_gravity_{stat}"] = g_stats[stat]
        result[f"escape_velocity_{stat}"] = vesc_stats[stat]
    return result
