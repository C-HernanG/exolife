"""
Missingness encoding, provenance tracking and identifier assignment for the ExoLife pipeline.

These functions handle the encoding of missing data patterns, addition of
provenance indicators and assignment of canonical ExoLife identifiers
for stars and planets.  Extracting this logic allows reuse across
different processing pipelines and simplifies testing.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

# ``norm_name`` lives in the top-level ``exolife/data/utils.py``; three dots climb
# Import the normalization utility from the data utils module
from exolife.data.utils import norm_name


def encode_missingness_patterns(
    df: pd.DataFrame, pipeline_config: Dict
) -> pd.DataFrame:
    """Encode missingness patterns explicitly based on pipeline configuration.

    The configuration under ``data_quality.missingness_encoding.patterns``
    defines named groups of columns.  For each pattern a boolean
    indicator column ``missing_pattern_<name>`` is created that is True
    when all columns in that pattern are missing.  Additionally,
    individual ``<column>_missing`` indicators are added for each
    referenced column.

    Args:
        df: Input DataFrame.
        pipeline_config: Pipeline configuration dict containing the
            missingness encoding specification.

    Returns:
        DataFrame with additional missingness indicator columns.
    """
    missingness_df = df.copy()
    try:
        missingness_config = pipeline_config["data_quality"]["missingness_encoding"]
    except Exception:
        missingness_config = {"patterns": []}
    for pattern in missingness_config.get("patterns", []):
        pattern_name = pattern.get("name")
        columns = pattern.get("columns", [])
        # Only consider existing columns
        existing_cols = [c for c in columns if c in missingness_df.columns]
        if existing_cols:
            missing_mask = missingness_df[existing_cols].isna().all(axis=1)
        else:
            missing_mask = pd.Series(
                [True] * len(missingness_df), index=missingness_df.index
            )
        missingness_df[f"missing_pattern_{pattern_name}"] = missing_mask
    # Individual missing indicators
    created_indicators: set[str] = set()
    for pattern in missingness_config.get("patterns", []):
        for col in pattern.get("columns", []):
            if col in missingness_df.columns and col not in created_indicators:
                missingness_df[f"{col}_missing"] = missingness_df[col].isna()
                created_indicators.add(col)
    return missingness_df


def add_provenance_information(df: pd.DataFrame) -> pd.DataFrame:
    """Add data provenance tracking columns.

    Simple provenance flags are added for a handful of key sources.  A
    provenance flag is True if any column associated with that source is
    present and non-null in the row.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with additional provenance columns.
    """
    provenance_df = df.copy()
    key_sources = {
        "nasa_source": "nasa_exoplanet_archive_pscomppars",
        "phl_source": "phl_exoplanet_catalog",
        "gaia_source": "gaia_dr3_astrophysical_parameters",
        "sweet_source": "sweet_cat",
    }
    for source_col, source_name in key_sources.items():
        source_columns = [col for col in provenance_df.columns if source_name in col]
        if source_columns:
            provenance_df[source_col] = (
                provenance_df[source_columns].notna().any(axis=1)
            )
        else:
            provenance_df[source_col] = False
    return provenance_df


def assign_exolife_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Assign canonical ExoLife star and planet identifiers.

    The star identifier is derived by preferring the Gaia DR3 source
    identifier when present, falling back to the Gaia DR2 identifier
    otherwise.  The planet identifier is formed by normalising the
    planet name (lowercase, trimmed) using the ``norm_name`` utility.
    Duplicate star/planet pairs are removed.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with ``exolife_star_id`` and ``exolife_planet_id`` columns.
    """
    result = df.copy()

    def find_column(cols: List[str], suffixes: List[str]) -> str | None:
        for suffix in suffixes:
            matches = [c for c in cols if c.lower().endswith(suffix.lower())]
            if matches:
                return matches[0]
        return None

    cols = list(result.columns)
    dr3_suffixes = ["gaia_dr3_source_id", "source_id_dr3", "dr3_source_id"]
    dr2_suffixes = ["gaia_dr2_source_id", "source_id_dr2", "dr2_source_id"]
    pl_name_suffixes = ["pl_name", "planet_name", "pl_hostname"]
    dr3_col = find_column(cols, dr3_suffixes)
    dr2_col = find_column(cols, dr2_suffixes)
    name_col = find_column(cols, pl_name_suffixes)
    if dr3_col or dr2_col:
        dr3_series = result[dr3_col] if dr3_col else pd.Series([pd.NA] * len(result))
        dr2_series = result[dr2_col] if dr2_col else pd.Series([pd.NA] * len(result))
        star_id = dr3_series.combine_first(dr2_series)
    else:
        star_id = result.index.astype(str)
    result["exolife_star_id"] = star_id
    if name_col:
        result["exolife_planet_id"] = norm_name(result[name_col])
    else:
        result["exolife_planet_id"] = result.index.astype(str)
    result = result.drop_duplicates(
        subset=["exolife_star_id", "exolife_planet_id"], keep="first"
    ).reset_index(drop=True)
    return result
