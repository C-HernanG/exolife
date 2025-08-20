"""
Data quality indicators and flags for the ExoLife pipeline.

This module groups together functions that compute simple quality
metrics and flags based on catalogue contents.  Separating these
utilities from the merger logic allows them to be reused elsewhere and
tested independently.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def add_quality_indicators(df: pd.DataFrame, pipeline_config: Dict) -> pd.DataFrame:
    """Add data quality indicators defined in the pipeline configuration.

    The pipeline configuration can specify one or more quality indicators
    under ``data_quality.quality_indicators``.  Currently implemented
    indicators include a completeness score which measures the fraction
    of key parameters that are not missing.

    Args:
        df: Input DataFrame.
        pipeline_config: Pipeline configuration dict.  The field
            ``data_quality.quality_indicators`` is consulted.

    Returns:
        DataFrame with additional quality indicator columns.
    """
    quality_df = df.copy()
    try:
        quality_config = pipeline_config["data_quality"]["quality_indicators"]
    except Exception:
        quality_config = []
    for indicator in quality_config:
        if indicator.get("name") == "data_completeness_score":
            key_params = indicator.get("key_parameters", [])
            available_count = quality_df[key_params].notna().sum(axis=1)
            quality_df["data_completeness_score"] = available_count / len(key_params)
    return quality_df


def assign_astrometry_quality_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Compute an astrometry quality flag based on the RUWE value.

    According to Gaia astrometric practice, ``RUWE ≤ 1.4`` is considered
    good, ``RUWE > 1.4 and ≤ 2.0`` is suspect, and ``RUWE > 2.0`` is poor.
    A new column ``astrometry_quality_flag`` is created with values
    ``good``, ``suspect`` or ``poor``.  If no RUWE information is
    available the flag is set to ``None``.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with an additional ``astrometry_quality_flag`` column.
    """
    result = df.copy()
    ruwe_col = None
    for col in result.columns:
        if col.lower().endswith("ruwe"):
            ruwe_col = col
            break
    if ruwe_col:

        def classify_ruwe(val):
            try:
                v = float(val)
            except Exception:
                return None
            if pd.isna(v):
                return None
            if v <= 1.4:
                return "good"
            elif v <= 2.0:
                return "suspect"
            else:
                return "poor"

        result["astrometry_quality_flag"] = result[ruwe_col].apply(classify_ruwe)
    else:
        result["astrometry_quality_flag"] = None
    return result


def assign_crossmatch_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a categorical cross‑match quality flag.

    The flag categories follow the specification:

      - ``best_neighbour``: exactly one neighbour with high confidence
      - ``neighbourhood_tie``: multiple neighbours within tolerance
      - ``pm_cone``: match obtained via proper motion cone search
      - ``unmatched_dr2_only``: no DR3 match (DR2 only)

    Classification is heuristic based on available Gaia cross‑match
    metadata columns.  If no recognised metadata is present, the flag
    defaults to ``unmatched_dr2_only``.  A ``crossmatch_score`` is
    computed heuristically and normalised to the range [0, 1].

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with additional ``crossmatch_quality`` and
        ``crossmatch_score`` columns.
    """
    result = df.copy()
    gaia_prefix = None
    for col in result.columns:
        if col.endswith("_ambiguity_flag"):
            gaia_prefix = col.rsplit("_", 1)[0]
            break
    if not gaia_prefix:
        for col in result.columns:
            if "gaia_dr3" in col and "ambiguity" in col:
                gaia_prefix = col.rsplit("_", 1)[0]
                break
    quality_flags: List[str] = []
    crossmatch_scores: List[float] = []
    if gaia_prefix:
        amb_col = f"{gaia_prefix}_ambiguity_flag"
        method_col_pref = f"{gaia_prefix}_crossmatch_method"
        dr3_id_col_pref = f"{gaia_prefix}_gaia_dr3_source_id"
        dr2_id_col_pref = f"{gaia_prefix}_gaia_dr2_source_id"
        for _, row in result.iterrows():
            try:
                dr3_id = None
                for candidate in [
                    dr3_id_col_pref,
                    "gaia_dr3_source_id",
                    "gaia_id",
                    "source_id_dr3",
                ]:
                    if candidate in row and pd.notna(row[candidate]):
                        dr3_id = row[candidate]
                        break
                for candidate in [
                    dr2_id_col_pref,
                    "gaia_dr2_source_id",
                    "source_id_dr2",
                ]:
                    if candidate in row and pd.notna(row[candidate]):
                        row[candidate]
                        break
                ambiguity = row.get(amb_col)
                method_val = None
                for candidate in [
                    method_col_pref,
                    "crossmatch_method",
                    f"{gaia_prefix}_crossmatch_method",
                ]:
                    if candidate in row and pd.notna(row[candidate]):
                        method_val = row[candidate]
                        break
                method = (str(method_val) if method_val is not None else "").lower()
                sep = row.get("angular_separation", np.nan)
                parallax_dr2 = row.get("gaia_dr2_parallax", np.nan)
                parallax_dr3 = row.get("gaia_dr3_parallax", np.nan)
                ruwe_val = row.get("ruwe", np.nan) or row.get(
                    f"{gaia_prefix}_ruwe", np.nan
                )
                score = 1.0
                # penalise separation
                if pd.notna(sep):
                    sigma_sep = 1.0
                    score -= 0.3 * min(1.0, (sep / sigma_sep) ** 2)
                # penalise parallax difference
                if pd.notna(parallax_dr2) and pd.notna(parallax_dr3):
                    diff = abs(parallax_dr3 - parallax_dr2)
                    sigma_plx = max(1e-3, (abs(parallax_dr2) + abs(parallax_dr3)) / 2.0)
                    score -= 0.3 * min(1.0, (diff / sigma_plx) ** 2)
                # penalise RUWE > 1.4
                try:
                    ruwe = float(ruwe_val) if pd.notna(ruwe_val) else np.nan
                except Exception:
                    ruwe = np.nan
                if pd.notna(ruwe) and ruwe > 1.4:
                    penalty = 0.2 * min(1.0, (ruwe - 1.4) / 1.0)
                    score -= penalty
                # clamp score
                if score < 0.0:
                    score = 0.0
                if score > 1.0:
                    score = 1.0
                if pd.notna(dr3_id):
                    if ambiguity is True:
                        qual = "neighbourhood_tie"
                    else:
                        if "position" in method or "cone" in method:
                            qual = "pm_cone"
                        else:
                            qual = "best_neighbour"
                else:
                    qual = "unmatched_dr2_only"
                quality_flags.append(qual)
                crossmatch_scores.append(score)
            except Exception:
                quality_flags.append("unmatched_dr2_only")
                crossmatch_scores.append(0.0)
        result["crossmatch_quality"] = quality_flags
        result["crossmatch_score"] = crossmatch_scores
    else:
        result["crossmatch_quality"] = "unmatched_dr2_only"
        result["crossmatch_score"] = 0.0
    return result
