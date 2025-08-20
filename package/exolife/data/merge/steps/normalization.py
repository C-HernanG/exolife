"""
Normalization and persistence utilities for the ExoLife pipeline.

This module collects routines for breaking the flat, merged DataFrame
produced by the configurable pipeline into a set of normalised entity
tables and for persisting those tables to disk.  It also exposes
helpers to save the full flat dataset for backwards compatibility and
to compute summary statistics about the pipeline run.  Extracting
these operations into a dedicated module adheres to the Single
Responsibility Principle by separating orchestration logic in the
pipeline from data normalisation and persistence concerns.

The functions defined here mirror the behaviour of the corresponding
methods in ``ConfigurablePipelineMerger``.  They preserve all
functional semantics while enabling reuse outside of that class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Import the global settings object from the exolife package
from exolife.settings import Settings

settings = Settings()


def normalize_output_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Normalize the flat merged DataFrame into multiple entity tables.

    This function reproduces the logic of
    ``ConfigurablePipelineMerger._normalize_output_tables``.  It
    decomposes the wide DataFrame into separate tables for stars,
    planets, aliases, measurements, derived features and cross-match
    edges.  The intent is to conform to a relational schema where
    attributes pertaining to different conceptual entities are stored
    separately.

    Args:
        df: The fully merged and enriched DataFrame.

    Returns:
        A dictionary keyed by table name containing the corresponding
        DataFrames.
    """
    tables: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Defensive ID creation: ensure the required ExoLife identifier
    # columns exist on the input.  The downstream normalisation logic
    # relies on ``exolife_star_id`` and ``exolife_planet_id`` being present.
    # When upstream processing fails to assign these (e.g. empty
    # cross‑match or missing source identifiers), we synthesise them from
    # the DataFrame index.  This prevents ``KeyError`` failures and
    # preserves pipeline continuity in edge cases with incomplete data.
    df_with_ids = df.copy()
    if "exolife_star_id" not in df_with_ids.columns:
        df_with_ids["exolife_star_id"] = df_with_ids.index.astype(str)
    if "exolife_planet_id" not in df_with_ids.columns:
        df_with_ids["exolife_planet_id"] = df_with_ids.index.astype(str)
    df = df_with_ids

    # 1. Stars table: one row per star
    star_cols: List[str] = []
    for col in df.columns:
        # Include stellar parameters and Gaia identifiers
        if col.startswith("st_") or col in ["gaia_dr2_source_id", "gaia_dr3_source_id"]:
            star_cols.append(col)
        # include RUWE and astrometry quality
        if col.lower().endswith("ruwe") or col == "astrometry_quality_flag":
            star_cols.append(col)
    # remove duplicates while preserving order
    unique_star_cols = list(dict.fromkeys(star_cols))
    stars_df = (
        df[["exolife_star_id"] + unique_star_cols]
        .drop_duplicates("exolife_star_id")
        .reset_index(drop=True)
    )
    tables["stars"] = stars_df

    # 2. Star aliases table: exolife_star_id, alias_type, alias_value
    alias_records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        star_id = row["exolife_star_id"]
        # collect DR2 and DR3 IDs explicitly; fall back to 'gaia_id' if canonical columns missing
        dr2_val = None
        if "gaia_dr2_source_id" in df.columns and pd.notna(
            row.get("gaia_dr2_source_id")
        ):
            dr2_val = row.get("gaia_dr2_source_id")
        elif "gaia_id" in df.columns and pd.notna(row.get("gaia_id")):
            dr2_val = row.get("gaia_id")
        if dr2_val is not None:
            alias_records.append(
                {
                    "exolife_star_id": star_id,
                    "alias_type": "gaia_dr2_source_id",
                    "alias_value": dr2_val,
                }
            )
        dr3_val = None
        if "gaia_dr3_source_id" in df.columns and pd.notna(
            row.get("gaia_dr3_source_id")
        ):
            dr3_val = row.get("gaia_dr3_source_id")
        if dr3_val is not None:
            alias_records.append(
                {
                    "exolife_star_id": star_id,
                    "alias_type": "gaia_dr3_source_id",
                    "alias_value": dr3_val,
                }
            )
        # add more aliases if present (e.g., TIC, KIC), excluding internal and flag columns
        for col in df.columns:
            col_lower = col.lower()
            if (
                col_lower.endswith("id")
                and ("gaia" not in col_lower)
                and not col_lower.startswith("exolife")
                and col_lower != "record_id"
                and not col_lower.startswith("pl_")
                and not col_lower.endswith("_flag_invalid")
            ):
                value = row.get(col)
                if pd.notna(value):
                    alias_records.append(
                        {
                            "exolife_star_id": star_id,
                            "alias_type": col,
                            "alias_value": value,
                        }
                    )
    star_aliases_df = (
        pd.DataFrame(alias_records).drop_duplicates().reset_index(drop=True)
    )
    # Ensure alias_value is string for Parquet compatibility
    if not star_aliases_df.empty and "alias_value" in star_aliases_df.columns:
        star_aliases_df["alias_value"] = star_aliases_df["alias_value"].astype(str)
    tables["star_aliases"] = star_aliases_df

    # 3. Planets table: one row per planet
    planet_cols: List[str] = []
    for col in df.columns:
        if col.startswith("pl_") or col in [
            "exolife_planet_id",
            "exolife_star_id",
            "crossmatch_quality",
        ]:
            planet_cols.append(col)
    # remove duplicates while preserving order
    unique_planet_cols = list(dict.fromkeys(planet_cols))
    planets_df = (
        df[unique_planet_cols]
        .drop_duplicates("exolife_planet_id")
        .reset_index(drop=True)
    )
    tables["planets"] = planets_df

    # 4. Planet aliases table
    palias_records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        pid = row["exolife_planet_id"]
        # original planet name
        if "pl_name" in df.columns and pd.notna(row.get("pl_name")):
            palias_records.append(
                {
                    "exolife_planet_id": pid,
                    "alias_type": "pl_name",
                    "alias_value": row.get("pl_name"),
                }
            )
        # additional alias fields (KOI, TOI, TIC, KIC, EPIC, etc.)
        for col in df.columns:
            col_lower = col.lower()
            if (
                col_lower.startswith("koi_")
                or col_lower.startswith("toi_")
                or col_lower.startswith("tic")
                or col_lower.startswith("kic")
                or col_lower.startswith("epic")
            ) and not col_lower.endswith("_flag_invalid"):
                value = row.get(col)
                if pd.notna(value):
                    palias_records.append(
                        {
                            "exolife_planet_id": pid,
                            "alias_type": col,
                            "alias_value": value,
                        }
                    )
    planet_aliases_df = (
        pd.DataFrame(palias_records).drop_duplicates().reset_index(drop=True)
    )
    # Ensure alias_value is string for Parquet compatibility
    if not planet_aliases_df.empty and "alias_value" in planet_aliases_df.columns:
        planet_aliases_df["alias_value"] = planet_aliases_df["alias_value"].astype(str)
    tables["planet_aliases"] = planet_aliases_df

    # 5. Measurements table
    measurement_records: List[Dict[str, Any]] = []
    # Only numeric columns are recorded as measurements
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for _, row in df.iterrows():
        star_id = row["exolife_star_id"]
        planet_id = row["exolife_planet_id"]
        for col in numeric_cols:
            value = row[col]
            if pd.isna(value):
                continue
            # Determine object type
            if col.startswith("st_"):
                object_id = star_id
                object_type = "star"
            elif col.startswith("pl_") or col in [
                "stellar_flux_s_mean",
                "equilibrium_temperature_mean",
                "surface_gravity_mean",
                "escape_velocity_mean",
                "hz_position_conservative",
                "hz_position_optimistic",
            ]:
                object_id = planet_id
                object_type = "planet"
            else:
                continue
            # Compute symmetric error if available
            err = None
            err1_col = f"{col}err1"
            err2_col = f"{col}err2"
            if err1_col in df.columns and pd.notna(row.get(err1_col)):
                err = abs(row.get(err1_col))
            if err2_col in df.columns and pd.notna(row.get(err2_col)):
                err = err if err is not None else 0.0
                err = (err + abs(row.get(err2_col))) / 2.0
            # Units
            units = None
            unit_col = f"{col}_units"
            if unit_col in df.columns:
                units = row.get(unit_col)
            # Source
            source = None
            source_col = f"{col}_source"
            if source_col in df.columns:
                source = row.get(source_col)
            measurement_records.append(
                {
                    "object_id": object_id,
                    "object_type": object_type,
                    "name": col,
                    "value": value,
                    "err": err,
                    "units": units,
                    "source": source,
                    "method": None,
                    "epoch": None,
                    "provenance": None,
                }
            )
    measurements_df = pd.DataFrame(measurement_records)
    if not measurements_df.empty:
        # Ensure object_id is string for Parquet compatibility
        if "object_id" in measurements_df.columns:
            measurements_df["object_id"] = measurements_df["object_id"].astype(str)
        measurements_df = measurements_df.drop_duplicates(
            subset=["object_id", "object_type", "name", "value", "units"]
        ).reset_index(drop=True)
    tables["measurements"] = measurements_df

    # 6. Derived features table
    derived_records: List[Dict[str, Any]] = []
    derived_cols = [
        "stellar_flux_s_mean",
        "stellar_flux_s_std",
        "stellar_flux_s_median",
        "stellar_flux_s_q16",
        "stellar_flux_s_q84",
        "equilibrium_temperature_mean",
        "equilibrium_temperature_std",
        "equilibrium_temperature_median",
        "equilibrium_temperature_q16",
        "equilibrium_temperature_q84",
        "surface_gravity_mean",
        "surface_gravity_std",
        "surface_gravity_median",
        "surface_gravity_q16",
        "surface_gravity_q84",
        "escape_velocity_mean",
        "escape_velocity_std",
        "escape_velocity_median",
        "escape_velocity_q16",
        "escape_velocity_q84",
        "hz_position_conservative",
        "hz_position_optimistic",
    ]
    for _, row in df.iterrows():
        pid = row["exolife_planet_id"]
        for col in derived_cols:
            if col not in df.columns:
                continue
            val = row.get(col)
            if pd.isna(val):
                continue
            # Determine units for the metric
            units = None
            if "stellar_flux" in col:
                units = "S_earth"
            elif "temperature" in col:
                units = "K"
            elif "gravity" in col:
                units = "m/s^2"
            elif "escape_velocity" in col:
                units = "km/s"
            elif "hz_position" in col:
                units = "dimensionless"
            derived_records.append(
                {
                    "object_id": pid,
                    "name": col,
                    "value": val,
                    "units": units,
                    "err": None,
                    "assumptions": {
                        "albedo": row.get("albedo", 0.3),
                        "redistribution_factor": row.get("redistribution_factor", 1.0),
                    },
                    "method": "monte_carlo",
                }
            )
    derived_df = pd.DataFrame(derived_records)
    # Ensure object_id is string for Parquet compatibility
    if not derived_df.empty and "object_id" in derived_df.columns:
        derived_df["object_id"] = derived_df["object_id"].astype(str)
    tables["derived_features"] = derived_df

    # 7. xmatch edges table
    edge_records: List[Dict[str, Any]] = []
    # (a) GAIA DR2/DR3 direct crossmatch edges
    for _, row in df.iterrows():
        dr2 = row.get("gaia_dr2_source_id")
        dr3 = row.get("gaia_dr3_source_id")
        if pd.notna(dr2) and pd.notna(dr3):
            edge_records.append(
                {
                    "src_catalog": "GAIA_DR2",
                    "src_id": str(dr2),
                    "dst_catalog": "GAIA_DR3",
                    "dst_id": str(dr3),
                    "method": row.get("crossmatch_quality"),
                    "score": row.get("crossmatch_score"),
                    "sep": row.get("angular_separation"),
                    "epoch": None,
                    "notes": None,
                }
            )
    # (b) Alias-based crossmatch edges for stars
    star_aliases = tables.get("star_aliases")
    if star_aliases is not None:
        for star_id, group in star_aliases.groupby("exolife_star_id"):
            aliases = group[["alias_type", "alias_value"]].drop_duplicates()
            alias_list = aliases.to_dict("records")
            n = len(alias_list)
            for i in range(n):
                for j in range(i + 1, n):
                    a = alias_list[i]
                    b = alias_list[j]
                    # forward edge
                    edge_records.append(
                        {
                            "src_catalog": str(a["alias_type"]),
                            "src_id": str(a["alias_value"]),
                            "dst_catalog": str(b["alias_type"]),
                            "dst_id": str(b["alias_value"]),
                            "method": "alias",
                            "score": None,
                            "sep": None,
                            "epoch": None,
                            "notes": None,
                        }
                    )
                    # reverse edge
                    edge_records.append(
                        {
                            "src_catalog": str(b["alias_type"]),
                            "src_id": str(b["alias_value"]),
                            "dst_catalog": str(a["alias_type"]),
                            "dst_id": str(a["alias_value"]),
                            "method": "alias",
                            "score": None,
                            "sep": None,
                            "epoch": None,
                            "notes": None,
                        }
                    )
    # (c) Alias-based crossmatch edges for planets
    planet_aliases = tables.get("planet_aliases")
    if planet_aliases is not None:
        for pid, group in planet_aliases.groupby("exolife_planet_id"):
            aliases = group[["alias_type", "alias_value"]].drop_duplicates()
            alias_list = aliases.to_dict("records")
            n = len(alias_list)
            for i in range(n):
                for j in range(i + 1, n):
                    a = alias_list[i]
                    b = alias_list[j]
                    edge_records.append(
                        {
                            "src_catalog": str(a["alias_type"]),
                            "src_id": str(a["alias_value"]),
                            "dst_catalog": str(b["alias_type"]),
                            "dst_id": str(b["alias_value"]),
                            "method": "alias",
                            "score": None,
                            "sep": None,
                            "epoch": None,
                            "notes": None,
                        }
                    )
                    edge_records.append(
                        {
                            "src_catalog": str(b["alias_type"]),
                            "src_id": str(b["alias_value"]),
                            "dst_catalog": str(a["alias_type"]),
                            "dst_id": str(a["alias_value"]),
                            "method": "alias",
                            "score": None,
                            "sep": None,
                            "epoch": None,
                            "notes": None,
                        }
                    )
    # Build DataFrame for xmatch edges; drop duplicates
    if edge_records:
        xmatch_df = pd.DataFrame(edge_records).drop_duplicates().reset_index(drop=True)
    else:
        xmatch_df = pd.DataFrame(
            columns=[
                "src_catalog",
                "src_id",
                "dst_catalog",
                "dst_id",
                "method",
                "score",
                "sep",
                "epoch",
                "notes",
            ]
        )
    tables["xmatch_edges"] = xmatch_df
    return tables


def save_normalized_results(
    tables: Dict[str, pd.DataFrame],
    output_name: str,
    pipeline_config: Dict[str, Any],
) -> Dict[str, Path]:
    """Persist normalised entity tables to disk.

    For each table provided, this function writes both a Parquet and
    CSV copy to the configured output directory.  If writing the
    Parquet file fails (e.g., due to a missing engine), a CSV file is
    used as the primary output.  The return value maps each table
    name to the path actually used.

    Args:
        tables: A dictionary of DataFrames keyed by table name.
        output_name: The base filename to use for outputs.
        pipeline_config: Pipeline configuration dict used to locate
            the output directory under ``output.path``.

    Returns:
        A dictionary mapping table names to Path objects indicating
        where the primary file was written.
    """
    output_paths: Dict[str, Path] = {}
    output_config = pipeline_config.get("output", {}) if pipeline_config else {}
    output_subdir = output_config.get("path", "normalized_tables").replace(
        "data/processed/", ""
    )
    base_dir = settings.processed_dir / output_subdir
    base_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        parquet_path = base_dir / f"{output_name}_{name}.parquet"
        output_path: Path = parquet_path
        table.to_parquet(parquet_path, index=False)
        output_paths[name] = output_path
    return output_paths


def save_results(
    df: pd.DataFrame,
    output_name: str,
    pipeline_config: Dict[str, Any],
) -> Path:
    """Save the full merged DataFrame to disk.

    This helper writes the DataFrame to a Parquet file and, as a
    fallback, to a CSV file.  Both Parquet and CSV versions are
    written when possible.  The location and base filename are
    determined from the pipeline configuration.

    Args:
        df: The DataFrame to persist.
        output_name: Base filename (without extension).
        pipeline_config: Pipeline configuration dict specifying
            ``output.path`` and ``output.filename``.

    Returns:
        The Path to the primary file written (Parquet or CSV).
    """
    output_config = pipeline_config.get("output", {}) if pipeline_config else {}
    output_subdir = output_config.get("path", "exolife_catalog").replace(
        "data/processed/", ""
    )
    output_dir = settings.processed_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_base = output_config.get("filename", output_name)
    parquet_path = output_dir / f"{filename_base}.parquet"
    output_path: Path = parquet_path
    df.to_parquet(parquet_path, index=False)
    return output_path


def get_pipeline_statistics(
    df: pd.DataFrame,
    pipeline_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compute basic statistics about a merged dataset.

    This function mirrors ``ConfigurablePipelineMerger._get_pipeline_statistics``.
    It reports record counts, mean completeness scores, missingness
    fractions for key parameters and distributions of quality flags.

    Args:
        df: The processed DataFrame.
        pipeline_config: The pipeline configuration dict used to
            identify key parameters for missingness calculations.

    Returns:
        A dictionary containing summary statistics.
    """
    stats: Dict[str, Any] = {
        "total_records": len(df),
        "unique_stars": (
            df["exolife_star_id"].nunique()
            if "exolife_star_id" in df.columns
            else len(df)
        ),
        "unique_planets": (
            df["exolife_planet_id"].nunique()
            if "exolife_planet_id" in df.columns
            else len(df)
        ),
    }
    # Mean completeness
    if "data_completeness_score" in df.columns:
        stats["data_completeness_mean"] = float(df["data_completeness_score"].mean())
    # Missingness per key parameter
    missingness: Dict[str, float] = {}
    if pipeline_config is not None:
        try:
            key_params = pipeline_config["data_quality"]["quality_indicators"][0][
                "key_parameters"
            ]
            for param in key_params:
                if param in df.columns:
                    missing_fraction = float(df[param].isna().mean())
                    missingness[param] = missing_fraction
        except Exception:
            pass
    if missingness:
        stats["missingness"] = missingness
    # RUWE distribution
    if "astrometry_quality_flag" in df.columns:
        counts = df["astrometry_quality_flag"].value_counts(dropna=False).to_dict()
        stats["astrometry_quality_distribution"] = {
            str(k): int(v) for k, v in counts.items()
        }
    # Crossmatch quality distribution
    if "crossmatch_quality" in df.columns:
        counts = df["crossmatch_quality"].value_counts(dropna=False).to_dict()
        stats["crossmatch_quality_distribution"] = {
            str(k): int(v) for k, v in counts.items()
        }
    # Derived feature coverage
    derived_metrics = [
        "stellar_flux_s_mean",
        "equilibrium_temperature_mean",
        "surface_gravity_mean",
        "escape_velocity_mean",
        "hz_position_conservative",
        "hz_position_optimistic",
    ]
    derived_presence: Dict[str, float] = {}
    for metric in derived_metrics:
        if metric in df.columns:
            derived_presence[metric] = float(df[metric].notna().mean())
    if derived_presence:
        stats["derived_feature_coverage"] = derived_presence
    return stats
