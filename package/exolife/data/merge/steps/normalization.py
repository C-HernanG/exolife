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


def _get_units_for_measurement(col_name: str) -> str:
    """Get canonical units for a measurement column."""
    units_mapping = {
        # Stellar parameters
        "st_teff": "K",
        "st_lum": "L_sun",
        "st_mass": "M_sun",
        "st_rad": "R_sun",
        "st_logg": "log(cgs)",
        "st_met": "dex",
        "st_age": "Gyr",
        "st_dens": "g/cm^3",
        "sy_dist": "pc",
        "sy_gmag": "mag",
        "sy_kmag": "mag",
        "sy_tmag": "mag",
        # Planetary parameters
        "pl_orbper": "days",
        "pl_orbsmax": "AU",
        "pl_orbeccen": "dimensionless",
        "pl_orbincl": "deg",
        "pl_rade": "R_earth",
        "pl_radj": "R_jup",
        "pl_masse": "M_earth",
        "pl_massj": "M_jup",
        "pl_dens": "g/cm^3",
        "pl_eqt": "K",
        "pl_insol": "S_earth",
        # Derived features
        "stellar_flux_s": "S_earth",
        "equilibrium_temperature": "K",
        "surface_gravity": "m/s^2",
        "escape_velocity": "km/s",
        "hz_position_conservative": "dimensionless",
        "hz_position_optimistic": "dimensionless",
        "distance_pc": "pc",
    }

    # Handle columns with suffixes like _mean, _std, etc.
    base_col = col_name
    for suffix in ["_mean", "_std", "_median", "_q16", "_q84", "err1", "err2"]:
        if col_name.endswith(suffix):
            base_col = col_name.replace(suffix, "")
            break

    return units_mapping.get(base_col, "")


def _get_source_for_measurement(col_name: str, row: pd.Series) -> str:
    """Get the source catalog for a measurement."""
    # Check for explicit source columns first
    source_col = f"{col_name}_source"
    if source_col in row.index and pd.notna(row.get(source_col)):
        return str(row.get(source_col))

    # Determine source based on column prefix
    if col_name.startswith("gaia_dr3_"):
        return "GAIA_DR3"
    elif col_name.startswith("sweet_cat_"):
        return "SWEET_CAT"
    elif col_name.startswith("phl_"):
        return "PHL_EXOPLANET_CATALOG"
    elif col_name.startswith("toi_"):
        return "TOI_CANDIDATES"
    elif col_name.startswith("koi_"):
        return "KOI_CANDIDATES"
    elif col_name.startswith(
        (
            "stellar_flux",
            "equilibrium_temp",
            "surface_gravity",
            "escape_velocity",
            "hz_position",
        )
    ):
        return "EXOLIFE_DERIVED"
    else:
        return "NASA_EXOPLANET_ARCHIVE"


def _get_method_for_measurement(col_name: str, row: pd.Series) -> str:
    """Get the method used for a measurement."""
    # Check for explicit method columns first
    method_col = f"{col_name}_method"
    if method_col in row.index and pd.notna(row.get(method_col)):
        return str(row.get(method_col))

    # Determine method based on column characteristics
    if col_name.startswith(
        (
            "stellar_flux",
            "equilibrium_temp",
            "surface_gravity",
            "escape_velocity",
            "hz_position",
        )
    ):
        return "monte_carlo_propagation"
    elif "gaia" in col_name.lower():
        return (
            "astrometry"
            if any(x in col_name for x in ["parallax", "pmra", "pmdec"])
            else "photometry"
        )
    elif col_name.startswith("st_"):
        return "spectroscopy"
    elif col_name.startswith("pl_"):
        return "transit" if row.get("tran_flag", 0) == 1 else "radial_velocity"
    else:
        return "unknown"


def _get_feature_inputs(feature_name: str) -> str:
    """Get the input fields used to derive a feature."""
    input_mapping = {
        "stellar_flux_s": "st_lum,pl_orbsmax",
        "equilibrium_temperature": "st_teff,st_rad,pl_orbsmax,albedo,redistribution_factor",
        "surface_gravity": "pl_masse,pl_rade",
        "escape_velocity": "pl_masse,pl_rade",
        "hz_position_conservative": "st_teff,st_lum,pl_orbsmax",
        "hz_position_optimistic": "st_teff,st_lum,pl_orbsmax",
    }
    # Handle suffixed versions (_mean, _std, etc.)
    (
        feature_name.split("_")[0]
        + "_"
        + feature_name.split("_")[1]
        + "_"
        + feature_name.split("_")[2]
        if len(feature_name.split("_")) > 2
        else feature_name
    )
    for key in input_mapping:
        if key in feature_name:
            return input_mapping[key]
    return ""


def _canonicalize_alias_type(alias_type: str) -> str:
    """Canonicalize alias type to standard enumeration."""
    alias_type_lower = alias_type.lower().strip()

    # Canonical mappings
    canonical_types = {
        "gaia_dr2_source_id": "GAIA_DR2",
        "gaia_dr3_source_id": "GAIA_DR3",
        "gaia_id": "GAIA_DR3",
        "systemid": "HOSTNAME",
        "system_id": "HOSTNAME",
        "koi_candidates_kepid": "KOI",
        "kepid": "KOI",
        "koi": "KOI",
        "toi_candidates_tid": "TOI",
        "tid": "TOI",
        "toi": "TOI",
        "tic": "TIC",
        "tic_id": "TIC",
        "kic": "KIC",
        "kic_id": "KIC",
        "epic": "EPIC",
        "epic_id": "EPIC",
        "pl_name": "PLANET_NAME",
        "planet_name": "PLANET_NAME",
        "hostname": "HOST_STAR",
        "host_name": "HOST_STAR",
    }

    return canonical_types.get(alias_type_lower, alias_type.upper())


def _normalize_alias_value(alias_value: str) -> str:
    """Normalize alias value by trimming and lowercasing."""
    if pd.isna(alias_value):
        return ""
    return str(alias_value).strip().lower()


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
                    "alias_type": _canonicalize_alias_type("gaia_dr2_source_id"),
                    "alias_value": _normalize_alias_value(str(dr2_val)),
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
                    "alias_type": _canonicalize_alias_type("gaia_dr3_source_id"),
                    "alias_value": _normalize_alias_value(str(dr3_val)),
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
                            "alias_type": _canonicalize_alias_type(col),
                            "alias_value": _normalize_alias_value(str(value)),
                        }
                    )
    star_aliases_df = (
        pd.DataFrame(alias_records).drop_duplicates().reset_index(drop=True)
    )
    # Ensure alias_value is string for Parquet compatibility
    if not star_aliases_df.empty and "alias_value" in star_aliases_df.columns:
        star_aliases_df["alias_value"] = star_aliases_df["alias_value"].astype(str)
    tables["star_aliases"] = star_aliases_df

    # 3. Planets table: one row per planet (exclude candidate-specific columns)
    planet_cols: List[str] = []
    candidate_cols: List[str] = []

    for col in df.columns:
        if col.startswith("pl_") or col in [
            "exolife_planet_id",
            "exolife_star_id",
            "crossmatch_quality",
        ]:
            planet_cols.append(col)
        elif col.startswith(("toi_candidates_", "koi_candidates_")):
            candidate_cols.append(col)

    # remove duplicates while preserving order
    unique_planet_cols = list(dict.fromkeys(planet_cols))
    planets_df = (
        df[unique_planet_cols]
        .drop_duplicates("exolife_planet_id")
        .reset_index(drop=True)
    )
    tables["planets"] = planets_df

    # 3b. Candidate metadata tables
    # TOI candidates table
    toi_cols = ["exolife_planet_id"] + [
        col for col in candidate_cols if col.startswith("toi_")
    ]
    if toi_cols and len(toi_cols) > 1:  # more than just the ID column
        toi_df = df[toi_cols].dropna(
            subset=[col for col in toi_cols[1:3] if col in df.columns], how="all"
        )
        if not toi_df.empty:
            toi_df = toi_df.drop_duplicates("exolife_planet_id").reset_index(drop=True)
            tables["toi_candidates"] = toi_df

    # KOI candidates table
    koi_cols = ["exolife_planet_id"] + [
        col for col in candidate_cols if col.startswith("koi_")
    ]
    if koi_cols and len(koi_cols) > 1:  # more than just the ID column
        koi_df = df[koi_cols].dropna(
            subset=[col for col in koi_cols[1:3] if col in df.columns], how="all"
        )
        if not koi_df.empty:
            koi_df = koi_df.drop_duplicates("exolife_planet_id").reset_index(drop=True)
            tables["koi_candidates"] = koi_df

    # 4. Planet aliases table
    palias_records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        pid = row["exolife_planet_id"]
        # original planet name
        if "pl_name" in df.columns and pd.notna(row.get("pl_name")):
            palias_records.append(
                {
                    "exolife_planet_id": pid,
                    "alias_type": _canonicalize_alias_type("pl_name"),
                    "alias_value": _normalize_alias_value(str(row.get("pl_name"))),
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
                            "alias_type": _canonicalize_alias_type(col),
                            "alias_value": _normalize_alias_value(str(value)),
                        }
                    )
    # Remove duplicates based on all three keys
    planet_aliases_df = pd.DataFrame(palias_records)
    if not planet_aliases_df.empty:
        planet_aliases_df = planet_aliases_df.drop_duplicates(
            subset=["exolife_planet_id", "alias_type", "alias_value"]
        ).reset_index(drop=True)
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
            # Units - use mapping first, then explicit column
            units = _get_units_for_measurement(col)
            unit_col = f"{col}_units"
            if unit_col in df.columns and pd.notna(row.get(unit_col)):
                units = row.get(unit_col)

            # Source
            source = _get_source_for_measurement(col, row)

            # Method
            method = _get_method_for_measurement(col, row)

            # Epoch - derive from context or leave None
            epoch = None
            epoch_col = f"{col}_epoch"
            if epoch_col in df.columns and pd.notna(row.get(epoch_col)):
                epoch = str(row.get(epoch_col))

            # Provenance
            provenance = f"{source}:{method}"

            measurement_records.append(
                {
                    "object_id": object_id,
                    "object_type": object_type,
                    "name": col,
                    "value": value,
                    "err": err,
                    "units": units,
                    "source": source,
                    "method": method,
                    "epoch": epoch,
                    "provenance": provenance,
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
            # Flatten assumptions dict to separate columns for Parquet compatibility
            albedo_val = row.get("albedo", 0.3)
            redistribution_val = row.get("redistribution_factor", 1.0)

            derived_records.append(
                {
                    "object_id": pid,
                    "name": col,
                    "value": val,
                    "units": units,
                    "err": None,
                    "albedo_assumption": float(albedo_val),
                    "redistribution_factor_assumption": float(redistribution_val),
                    "method": "monte_carlo",
                    "provenance": "monte_carlo_propagation@v1.0",
                    "inputs": _get_feature_inputs(col),
                }
            )
    derived_df = pd.DataFrame(derived_records)
    # Ensure object_id is string for Parquet compatibility
    if not derived_df.empty and "object_id" in derived_df.columns:
        derived_df["object_id"] = derived_df["object_id"].astype(str)
    tables["derived_features"] = derived_df

    # 7. xmatch edges table - slim version with only essential columns
    edge_records: List[Dict[str, Any]] = []
    # Only include alias-based edges since geometric xmatch data is not populated
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
                    # Only forward edge to reduce redundancy
                    edge_records.append(
                        {
                            "src_catalog": str(a["alias_type"]),
                            "src_id": str(a["alias_value"]),
                            "dst_catalog": str(b["alias_type"]),
                            "dst_id": str(b["alias_value"]),
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
                    # Only forward edge to reduce redundancy
                    edge_records.append(
                        {
                            "src_catalog": str(a["alias_type"]),
                            "src_id": str(a["alias_value"]),
                            "dst_catalog": str(b["alias_type"]),
                            "dst_id": str(b["alias_value"]),
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

        # Optimize storage based on table type
        compression = "zstd"
        row_group_size = 200000

        if name == "xmatch_edges":
            # Optimize for largest table - use dictionary encoding for strings
            compression = "zstd"
            row_group_size = 128000

        elif name in ["planet_aliases", "star_aliases"]:
            # Medium tables - use higher compression
            compression = "zstd"
            row_group_size = 150000

        table.to_parquet(
            parquet_path,
            index=False,
            compression=compression,
            row_group_size=row_group_size,
        )
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

    # Use optimized storage settings
    df.to_parquet(parquet_path, index=False, compression="zstd", row_group_size=200000)
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
