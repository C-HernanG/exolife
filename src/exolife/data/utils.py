"""
Utility functions for data merging and preprocessing.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from ..settings import settings

config_path = settings.external_dir / "data_sources.json"

# Initialize module logger
logger = logging.getLogger("exolife.data.utils")


# ------------------------------------------------------------------
# Hz utils for calculating habitable zone distances and fluxes
# Compute conservative and optimistic habitable-zone (HZ) limits using polynomial
# prescriptions from Kopparapu et al. (2014).

# Functions:
# - hz_flux(teff, limit): Stellar flux (S_eff) at specified HZ boundary.
# - hz_distance(seff, luminosity): Orbital distance (AU) given stellar flux &
#   luminosity.
# - hz_edges(teff, luminosity, optimistic): Inner & outer HZ distances.
# - add_hz_edges(df, teff_col, lum_col, optimistic, prefix): Vectorized addition
#   of HZ edge columns to a pandas DataFrame.
# ------------------------------------------------------------------


# Coefficients for Kopparapu et al. (2014), Table 1
try:
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    _KOPPARAPU_COEFFS: dict[str, Tuple[float, float, float, float, float]] = {
        k: tuple(v) for k, v in cfg.get("kopparapu_hz_coefficients", []).items()
    }
    VALID_LIMITS = tuple(_KOPPARAPU_COEFFS.keys())
    logger.info("Loaded HZ coefficients for %d limits.", len(_KOPPARAPU_COEFFS))
except Exception as e:
    logger.error("Failed to load Kopparapu coefficients: %s", e)
    _KOPPARAPU_COEFFS = {}


def hz_flux(
    teff: float | np.ndarray,
    limit: Literal[
        "RecentVenus", "RunawayGreenhouse", "MaximumGreenhouse", "EarlyMars"
    ] = "RunawayGreenhouse",
) -> float | np.ndarray:
    """
    Compute stellar flux (S_eff) at a given HZ boundary for a star
    of temperature teff (K).

    Args:
        teff: Stellar effective temperature in Kelvin (scalar or array).
        limit: HZ boundary name; one of VALID_LIMITS.

    Returns:
        S_eff in Solar flux units (scalar or array).

    Raises:
        KeyError: If `limit` is not in VALID_LIMITS.
    """
    try:
        seff_sun, a, b, c, d = _KOPPARAPU_COEFFS[limit]
    except KeyError:
        logger.error("Invalid HZ limit '%s'; valid options: %s", limit, VALID_LIMITS)
        raise

    # Handle both scalar and array inputs
    teff = np.asarray(teff)
    dt = teff - 5780.0

    # Polynomial expansion with proper handling of invalid values
    with np.errstate(invalid="ignore"):
        result = seff_sun + a * dt + b * dt**2 + c * dt**3 + d * dt**4

    # Return scalar if input was scalar
    return result.item() if result.ndim == 0 else result


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

    # Compute all limits using vectorized hz_flux
    inner_limit = "RecentVenus" if optimistic else "RunawayGreenhouse"
    outer_limit = "EarlyMars" if optimistic else "MaximumGreenhouse"

    inner_seff = hz_flux(teff, inner_limit)
    outer_seff = hz_flux(teff, outer_limit)
    # Compute distances only where flux is positive and luminosity is valid
    inner = np.full_like(lum, np.nan, dtype=float)
    outer = np.full_like(lum, np.nan, dtype=float)

    # More robust validation: check for positive values and finite numbers
    valid_inner = (
        (inner_seff > 0) & np.isfinite(inner_seff) & (lum > 0) & np.isfinite(lum)
    )
    valid_outer = (
        (outer_seff > 0) & np.isfinite(outer_seff) & (lum > 0) & np.isfinite(lum)
    )

    # Suppress warnings for invalid value operations
    with np.errstate(invalid="ignore", divide="ignore"):
        inner[valid_inner] = np.sqrt(lum[valid_inner] / inner_seff[valid_inner])
        outer[valid_outer] = np.sqrt(lum[valid_outer] / outer_seff[valid_outer])

    # Assign to DataFrame
    df[f"{prefix}inner"] = inner
    df[f"{prefix}outer"] = outer
    logger.info(
        "Added HZ edges to DataFrame: columns '%sinner', '%souter'", prefix, prefix
    )
    return df


# ------------------------------------------------------------------
# Data fetching utilities
# ------------------------------------------------------------------


@dataclass(slots=True)
class DataSource:
    """
    Data source configuration for fetching data.
    """

    id: str
    name: str
    description: str
    download_url: Optional[str] = None
    adql: Optional[str] = None
    columns_to_keep: List[str] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    join_keys: dict[str, List[str]] = field(default_factory=dict)
    refresh: str = "static"
    format: Optional[str] = None


def load_config(path: Path = settings.external_dir / "data_sources.json") -> dict:
    """
    Load data source configuration from a JSON file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_sources(cfg: dict) -> dict[str, DataSource]:
    """
    Parse data sources from the configuration dictionary.
    """
    valid = set(DataSource.__dataclass_fields__)
    sources: dict[str, DataSource] = {}
    for entry in cfg.get("data_sources", []):
        data = {k: v for k, v in entry.items() if k in valid}
        sources[entry["id"]] = DataSource(**data)
    return sources


def _get_sources() -> dict[str, DataSource]:
    """Internal function to load and parse data sources."""
    return parse_sources(load_config())


def list_data_sources() -> List[str]:
    """
    List all available data source IDs from the config.
    """
    sources = _get_sources()
    return sorted(sources.keys())


def get_data_source(source_id: str) -> DataSource:
    """
    Get a data source configuration by ID.

    Args:
        source_id: The ID of the data source to retrieve.

    Returns:
        DataSource configuration object.

    Raises:
        KeyError: If the source_id is not found.
    """
    sources = _get_sources()
    if source_id not in sources:
        available = list(sources.keys())
        raise KeyError(f"Data source '{source_id}' not found. Available: {available}")
    return sources[source_id]


def fetch_adql(ds: DataSource, gaia_ids: List[int] | None = None) -> bytes:
    """
    Fetch data using ADQL query if available.

    Args:
        ds: DataSource configuration with ADQL query.
        gaia_ids: Optional list of Gaia IDs to substitute in query.

    Returns:
        Raw bytes from the ADQL query response.

    Raises:
        ValueError: If gaia_ids is required but not provided.
        RuntimeError: If query returns XML instead of CSV.
    """
    return _fetch_adql(ds, gaia_ids)


def timestamp() -> str:
    """
    Get the current UTC timestamp in the format YYYYMMDDTHHMMSSZ.
    """
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def stream_download(url: str) -> bytes:
    """
    Stream download content from a URL.
    """
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        return r.content


def write_csv_trimmed(raw: bytes, keep: List[str], out: Path) -> pd.DataFrame:
    """
    Write a trimmed CSV file from raw bytes, keeping only specified columns.
    """
    df = pd.read_csv(io.BytesIO(raw), usecols=lambda c: c in keep if keep else True)
    df.to_parquet(out, index=False)
    return df


def write_generic(raw: bytes, url: str, keep: List[str], out: Path) -> pd.DataFrame:
    """
    Write a generic file from raw bytes, keeping only specified columns.
    """
    if url.endswith(".csv") or ",format=csv" in url:
        return write_csv_trimmed(raw, keep, out)
    if url.endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(raw), columns=keep or None)
        df.to_parquet(out, index=False)
        return df
    # fallback to CSV
    return write_csv_trimmed(raw, keep, out)


def _fetch_adql(ds: "DataSource", gaia_ids: List[int] | None = None) -> bytes:
    """
    Fetch data using ADQL query if available.
    """
    query = ds.adql or ""
    if "<GAIA_ID_LIST>" in query:
        if not gaia_ids:
            raise ValueError("gaia_ids list must be provided to fill <GAIA_ID_LIST>")
        id_list = ",".join(str(i) for i in gaia_ids)
        query = query.replace("<GAIA_ID_LIST>", id_list)
    logger.info("Running ADQL query for %s", ds.id)
    r = requests.post(
        ds.download_url,
        data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query},
        timeout=300,
    )
    r.raise_for_status()
    if "xml" in r.headers.get("Content-Type", "").lower():
        raise RuntimeError("ADQL query returned XML instead of CSV")
    return r.content


# ----------------------------------------------------------------------
# Merging utilities
# ----------------------------------------------------------------------


def read_interim(src: str, cols: List[str] | None = None) -> pd.DataFrame:
    """
    Read an interim data source from the configured directory.
    """
    path = settings.interim_dir / f"{src}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path, columns=cols)


def norm_name(s: pd.Series) -> pd.Series:
    """
    Normalize a pandas Series by stripping whitespace and converting to lowercase.
    """
    return s.astype(str).str.strip().str.lower()


def find_source_id(prefix: str) -> str | None:
    """
    Find the first data source ID that starts with the given prefix.
    """
    available_sources = list_data_sources()
    return next((k for k in available_sources if k.startswith(prefix)), None)


def gaia_int(x):
    """
    Convert a value to a Gaia integer ID, extracting the first 10+ digit number.
    """
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return int(x)
    import re

    m = re.search(r"\d{10,}", str(x))
    return int(m.group(0)) if m else None


# ---------------------------------------------------------------------
# Data preprocessing utilities
# ---------------------------------------------------------------------


def load_default_drop_columns() -> list[str]:
    """
    Return the default list of columns to drop, read from *data_sources.json*.

    The JSON file is expected to live one directory above this module and to
    contain a top‑level key "columns_to_drop".  If the file or the key is
    missing, an empty list is returned so that the pruner becomes a no‑op.
    """

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        return list(cfg.get("columns_to_drop", []))
    except Exception:
        return []
