"""
Integration test for derived features serialization fix.

This test ensures that the derived features table can be properly
serialized and deserialized without dict column issues.
"""

import pandas as pd

from exolife.data.merge.steps.normalization import normalize_output_tables
from exolife.settings import Settings


def test_derived_features_serialization():
    """Test that derived features table serializes properly without dict columns."""
    # Create sample data with derived features
    sample_data = {
        "exolife_planet_id": ["planet_1", "planet_2"],
        "exolife_star_id": [12345, 67890],
        "stellar_flux_s_mean": [1.2, 0.8],
        "equilibrium_temperature_mean": [288.0, 250.0],
        "surface_gravity_mean": [9.8, 12.1],
        "escape_velocity_mean": [11.2, 8.9],
        "hz_position_conservative": [0.95, 1.2],
        "albedo": [0.3, 0.3],
        "redistribution_factor": [1.0, 1.0],
        "gaia_dr3_source_id": [123456789, 987654321],
        "tic_id": [111111, 222222],
    }

    df = pd.DataFrame(sample_data)

    # Generate normalized tables
    tables = normalize_output_tables(df)

    # Check that derived_features table exists
    assert "derived_features" in tables
    derived_df = tables["derived_features"]

    # Check that there are no dict columns
    for col in derived_df.columns:
        sample_val = (
            derived_df[col].dropna().iloc[0]
            if len(derived_df[col].dropna()) > 0
            else None
        )
        assert not isinstance(sample_val, dict), f"Column {col} contains dict values"

    # Check that assumptions are flattened to separate columns
    assert "albedo_assumption" in derived_df.columns
    assert "redistribution_factor_assumption" in derived_df.columns

    # Test serialization to parquet (this should not raise an error)
    settings = Settings()
    test_path = settings.processed_dir / "test_derived_features.parquet"
    test_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        derived_df.to_parquet(test_path, index=False)

        # Test loading back
        loaded_df = pd.read_parquet(test_path)
        assert len(loaded_df) > 0
        assert "albedo_assumption" in loaded_df.columns
        assert "redistribution_factor_assumption" in loaded_df.columns

    finally:
        # Clean up
        if test_path.exists():
            test_path.unlink()


def test_measurements_table_populated():
    """Test that measurements table has proper units, source, method, provenance."""
    sample_data = {
        "exolife_planet_id": ["planet_1"],
        "exolife_star_id": [12345],
        "st_teff": [5778.0],
        "st_mass": [1.0],
        "pl_rade": [1.0],
        "pl_masse": [1.0],
        "tran_flag": [1],
        "gaia_dr3_source_id": [123456789],
        "tic_id": [111111],
    }

    df = pd.DataFrame(sample_data)
    tables = normalize_output_tables(df)

    assert "measurements" in tables
    measurements_df = tables["measurements"]

    # Check that required columns are present and populated
    assert "units" in measurements_df.columns
    assert "source" in measurements_df.columns
    assert "method" in measurements_df.columns
    assert "provenance" in measurements_df.columns

    # Check that measurements have non-null values for key fields
    units_coverage = measurements_df["units"].notna().mean()
    source_coverage = measurements_df["source"].notna().mean()
    method_coverage = measurements_df["method"].notna().mean()
    provenance_coverage = measurements_df["provenance"].notna().mean()

    assert units_coverage >= 0.95, f"Units coverage: {units_coverage}"
    assert source_coverage >= 0.95, f"Source coverage: {source_coverage}"
    assert method_coverage >= 0.95, f"Method coverage: {method_coverage}"
    assert provenance_coverage >= 0.95, f"Provenance coverage: {provenance_coverage}"


def test_xmatch_edges_optimized():
    """Test that xmatch edges table is optimized without empty columns."""
    sample_data = {
        "exolife_planet_id": ["planet_1", "planet_2"],
        "exolife_star_id": [12345, 67890],
        "gaia_dr3_source_id": [123456789, 987654321],
        "tic_id": [111111, 222222],
    }

    df = pd.DataFrame(sample_data)
    tables = normalize_output_tables(df)

    assert "xmatch_edges" in tables
    edges_df = tables["xmatch_edges"]

    # Check that only essential columns are present
    expected_cols = {"src_catalog", "src_id", "dst_catalog", "dst_id"}
    actual_cols = set(edges_df.columns)

    assert actual_cols == expected_cols, f"Expected {expected_cols}, got {actual_cols}"

    # Check that there are no null-only columns
    for col in edges_df.columns:
        null_rate = edges_df[col].isna().mean()
        assert null_rate < 1.0, f"Column {col} is 100% null"


def test_alias_canonicalization():
    """Test that alias types are properly canonicalized."""
    sample_data = {
        "exolife_planet_id": ["planet_1"],
        "exolife_star_id": [12345],
        "pl_name": ["Test Planet b"],
        "gaia_dr3_source_id": [123456789],
        "systemid": [42],
    }

    df = pd.DataFrame(sample_data)
    tables = normalize_output_tables(df)

    # Check planet aliases
    assert "planet_aliases" in tables
    planet_aliases = tables["planet_aliases"]

    # Check that alias types are canonicalized
    alias_types = set(planet_aliases["alias_type"].unique())
    expected_types = {"PLANET_NAME"}  # pl_name should be canonicalized
    assert expected_types.issubset(alias_types)

    # Check star aliases
    assert "star_aliases" in tables
    star_aliases = tables["star_aliases"]

    star_alias_types = set(star_aliases["alias_type"].unique())
    expected_star_types = {"GAIA_DR3", "HOSTNAME"}  # canonicalized forms
    assert expected_star_types.issubset(star_alias_types)

    # Check that alias values are normalized (lowercased, trimmed)
    assert all(isinstance(val, str) for val in planet_aliases["alias_value"])
    assert all(isinstance(val, str) for val in star_aliases["alias_value"])
