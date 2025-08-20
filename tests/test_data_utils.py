"""
Tests for the ExoLife data utilities module.
"""

import numpy as np
import pandas as pd
import pytest
import yaml
from io import BytesIO
from pathlib import Path
from unittest.mock import patch, mock_open

from exolife.data.utils import (
    hz_flux, hz_distance, hz_edges, add_hz_edges_to_df,
    DataSource, load_data_source_config, load_config, parse_sources,
    list_data_sources, get_data_source, timestamp, norm_name,
    gaia_int, find_source_id, load_default_drop_columns,
    read_interim, write_interim, write_csv_trimmed, write_generic
)


class TestHZUtils:
    """Test cases for habitable zone utility functions."""

    def test_hz_flux_scalar(self):
        """Test hz_flux with scalar input."""
        # Test with solar temperature (should return baseline values)
        solar_teff = 5780.0

        # For solar temperature, dt = 0, so result should be seff_sun
        runaway = hz_flux(solar_teff, "RunawayGreenhouse")
        assert abs(runaway - 1.107) < 1e-3  # From coefficients

        recent_venus = hz_flux(solar_teff, "RecentVenus")
        assert abs(recent_venus - 1.766) < 1e-3

    def test_hz_flux_array(self):
        """Test hz_flux with array input."""
        teff_array = np.array([5780.0, 4000.0, 6500.0])
        result = hz_flux(teff_array, "RunawayGreenhouse")

        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert abs(result[0] - 1.107) < 1e-3  # Solar case

    def test_hz_flux_invalid_limit(self):
        """Test hz_flux with invalid limit."""
        with pytest.raises(KeyError):
            hz_flux(5780.0, "InvalidLimit")

    def test_hz_distance(self):
        """Test hz_distance calculation."""
        # For solar values (seff=1, luminosity=1), distance should be 1 AU
        distance = hz_distance(1.0, 1.0)
        assert abs(distance - 1.0) < 1e-6

        # For seff=4, luminosity=1, distance should be 0.5 AU
        distance = hz_distance(4.0, 1.0)
        assert abs(distance - 0.5) < 1e-6

    def test_hz_distance_edge_cases(self):
        """Test hz_distance with edge cases."""
        # Zero or negative seff should return NaN
        assert np.isnan(hz_distance(0.0, 1.0))
        assert np.isnan(hz_distance(-1.0, 1.0))

        # Negative luminosity should return NaN
        assert np.isnan(hz_distance(1.0, -1.0))

    def test_hz_edges(self):
        """Test hz_edges calculation."""
        # Solar case
        inner, outer = hz_edges(5780.0, 1.0, optimistic=False)

        # Should return reasonable values for the solar system
        assert 0.5 < inner < 1.5
        assert 0.5 < outer < 2.0
        assert inner < outer

    def test_hz_edges_optimistic_vs_conservative(self):
        """Test difference between optimistic and conservative HZ boundaries."""
        teff, luminosity = 5780.0, 1.0

        inner_opt, outer_opt = hz_edges(teff, luminosity, optimistic=True)
        inner_cons, outer_cons = hz_edges(teff, luminosity, optimistic=False)

        # Optimistic boundaries should be wider
        assert inner_opt < inner_cons
        assert outer_opt > outer_cons

    def test_add_hz_edges_to_df(self, sample_dataframe):
        """Test adding HZ edges to DataFrame."""
        df = sample_dataframe.copy()
        result_df = add_hz_edges_to_df(
            df, teff_col="st_teff", lum_col="st_lum")

        # Check that new columns were added
        assert "hz_inner" in result_df.columns
        assert "hz_outer" in result_df.columns

        # Check that values are reasonable
        assert all(result_df["hz_inner"] < result_df["hz_outer"])
        assert all(result_df["hz_inner"] > 0)
        assert all(result_df["hz_outer"] > 0)

    def test_add_hz_edges_custom_prefix(self, sample_dataframe):
        """Test adding HZ edges with custom prefix."""
        df = sample_dataframe.copy()
        result_df = add_hz_edges_to_df(df, prefix="habitable_")

        assert "habitable_inner" in result_df.columns
        assert "habitable_outer" in result_df.columns

    def test_hz_with_log_luminosity(self, sample_dataframe):
        """Test HZ calculation with logarithmic luminosity values."""
        df = sample_dataframe.copy()

        # The sample data has log luminosity values (negative)
        # The function should handle this automatically
        result_df = add_hz_edges_to_df(df)

        # Should have valid results
        assert not result_df["hz_inner"].isna().all()
        assert not result_df["hz_outer"].isna().all()

    def test_hz_with_missing_hz_config(self, monkeypatch):
        """Test HZ functions when configuration file is missing."""
        # Mock empty coefficients dictionary
        import exolife.data.utils
        monkeypatch.setattr(exolife.data.utils, '_KOPPARAPU_COEFFS', {})

        with pytest.raises(KeyError):
            hz_flux(5780.0, "RunawayGreenhouse")


class TestDataSource:
    """Test cases for DataSource class."""

    def test_data_source_creation(self):
        """Test DataSource creation."""
        source = DataSource(
            id="test_source",
            name="Test Source",
            description="A test data source",
            download_url="https://example.com/data.csv",
            columns_to_keep=["col1", "col2"],
            primary_keys=["col1"],
            join_keys={"other": ["col2"]}
        )

        assert source.id == "test_source"
        assert source.name == "Test Source"
        assert source.download_url == "https://example.com/data.csv"
        assert source.columns_to_keep == ["col1", "col2"]

    def test_data_source_defaults(self):
        """Test DataSource with default values."""
        source = DataSource(
            id="minimal",
            name="Minimal Source",
            description="Minimal test source"
        )

        assert source.download_url is None
        assert source.adql is None
        assert source.columns_to_keep == []
        assert source.primary_keys == []
        assert source.join_keys == {}
        assert source.format is None


class TestDataSourceConfig:
    """Test cases for data source configuration functions."""

    def test_load_data_source_config(self, mock_source_configs, test_settings):
        """Test loading data source configuration."""
        with patch('exolife.data.utils.settings', test_settings):
            config = load_data_source_config("nasa_exoplanet_archive")

            assert config["id"] == "nasa_exoplanet_archive"
            assert config["name"] == "NASA Exoplanet Archive"
            assert "columns_to_keep" in config

    def test_load_nonexistent_config(self, test_settings):
        """Test loading non-existent configuration."""
        with patch('exolife.data.utils.settings', test_settings):
            with pytest.raises(FileNotFoundError):
                load_data_source_config("nonexistent_source")

    def test_load_config(self, mock_source_configs, test_settings):
        """Test loading all configurations."""
        with patch('exolife.data.utils.settings', test_settings):
            config = load_config()

            assert "data_sources" in config
            assert len(config["data_sources"]) >= 1  # At least one source

    def test_parse_sources(self):
        """Test parsing sources from configuration."""
        config = {
            "data_sources": [
                {
                    "id": "source1",
                    "name": "Source 1",
                    "description": "First source",
                    "download_url": "https://example.com/1.csv",
                    "columns_to_keep": ["col1", "col2"],
                    "extra_field": "ignored"  # Should be filtered out
                },
                {
                    "id": "source2",
                    "name": "Source 2",
                    "description": "Second source"
                }
            ]
        }

        sources = parse_sources(config)

        assert len(sources) == 2
        assert "source1" in sources
        assert "source2" in sources

        source1 = sources["source1"]
        assert source1.id == "source1"
        assert source1.download_url == "https://example.com/1.csv"
        assert source1.columns_to_keep == ["col1", "col2"]
        assert not hasattr(source1, "extra_field")

    def test_list_data_sources(self, mock_source_configs, test_settings):
        """Test listing available data sources."""
        with patch('exolife.data.utils.settings', test_settings):
            sources = list_data_sources()

            assert isinstance(sources, list)
            assert "nasa_exoplanet_archive" in sources
            assert "gaia_dr3_astrophysical_parameters" in sources

    def test_get_data_source(self, mock_source_configs, test_settings):
        """Test getting specific data source."""
        with patch('exolife.data.utils.settings', test_settings):
            source = get_data_source("nasa_exoplanet_archive")

            assert isinstance(source, DataSource)
            assert source.id == "nasa_exoplanet_archive"
            assert source.name == "NASA Exoplanet Archive"

    def test_get_nonexistent_data_source(self, mock_source_configs, test_settings):
        """Test getting non-existent data source."""
        with patch('exolife.data.utils.settings', test_settings):
            with pytest.raises(KeyError, match="Data source 'nonexistent' not found"):
                get_data_source("nonexistent")


class TestUtilityFunctions:
    """Test cases for various utility functions."""

    def test_timestamp(self):
        """Test timestamp generation."""
        ts = timestamp()

        # Should be a string in the format YYYYMMDDTHHMMSSZ
        assert isinstance(ts, str)
        assert len(ts) == 16
        assert ts.endswith("Z")
        assert "T" in ts

    def test_norm_name(self):
        """Test name normalization."""
        names = pd.Series(["  Kepler-442 b  ", "K2-18 B", "TRAPPIST-1 E"])
        normalized = norm_name(names)

        expected = pd.Series(["kepler-442 b", "k2-18 b", "trappist-1 e"])
        pd.testing.assert_series_equal(normalized, expected)

    def test_gaia_int(self):
        """Test Gaia ID integer conversion."""
        # Valid Gaia ID
        assert gaia_int("1234567890123456") == 1234567890123456
        assert gaia_int(1234567890123456) == 1234567890123456
        assert gaia_int(1234567890123456.0) == 1234567890123456

        # String with Gaia ID embedded
        assert gaia_int("Gaia DR3 1234567890123456") == 1234567890123456

        # Invalid cases
        assert gaia_int(None) is None
        assert gaia_int(pd.NA) is None
        assert gaia_int("no digits here") is None
        assert gaia_int("123456789") is None  # Too short

    def test_find_source_id(self, mock_source_configs, test_settings):
        """Test finding source ID by prefix."""
        with patch('exolife.data.utils.settings', test_settings):
            # Should find the NASA source
            found = find_source_id("nasa")
            assert found == "nasa_exoplanet_archive"

            # Should find the Gaia source
            found = find_source_id("gaia")
            assert found == "gaia_dr3_astrophysical_parameters"

            # Non-existent prefix
            found = find_source_id("nonexistent")
            assert found is None

    def test_load_default_drop_columns(self, test_settings):
        """Test loading default drop columns configuration."""
        # Create mock drop columns config
        drop_config = {"columns_to_drop": ["col1", "col2", "col3"]}

        config_dir = test_settings.root_dir / "config" / "constants"
        config_dir.mkdir(parents=True, exist_ok=True)

        drop_file = config_dir / "drop_columns.yml"
        with open(drop_file, 'w') as f:
            yaml.dump(drop_config, f)

        with patch('exolife.data.utils.settings', test_settings):
            columns = load_default_drop_columns()
            assert columns == ["col1", "col2", "col3"]

    def test_load_default_drop_columns_missing_file(self, test_settings):
        """Test loading drop columns when file is missing."""
        with patch('exolife.data.utils.settings', test_settings):
            columns = load_default_drop_columns()
            assert columns == []  # Should return empty list


class TestDataFileOperations:
    """Test cases for data file operations."""

    def test_write_csv_trimmed(self, temp_dir, sample_csv_content):
        """Test writing trimmed CSV data."""
        raw_data = sample_csv_content.encode('utf-8')
        output_path = temp_dir / "output.parquet"
        keep_cols = ["pl_name", "st_teff"]

        df = write_csv_trimmed(raw_data, keep_cols, output_path)

        # Check output file exists
        assert output_path.exists()

        # Check DataFrame content
        assert len(df.columns) == 2
        assert "pl_name" in df.columns
        assert "st_teff" in df.columns
        assert "pl_rade" not in df.columns  # Should be filtered out

    def test_write_csv_trimmed_no_filter(self, temp_dir, sample_csv_content):
        """Test writing CSV data without filtering."""
        raw_data = sample_csv_content.encode('utf-8')
        output_path = temp_dir / "output.parquet"

        df = write_csv_trimmed(raw_data, [], output_path)

        # Should keep all columns
        assert len(df.columns) == 4

    def test_write_generic_csv(self, temp_dir, sample_csv_content):
        """Test write_generic with CSV URL."""
        raw_data = sample_csv_content.encode('utf-8')
        output_path = temp_dir / "output.parquet"
        url = "https://example.com/data.csv"

        df = write_generic(raw_data, url, ["pl_name"], output_path)

        assert output_path.exists()
        assert len(df.columns) == 1
        assert "pl_name" in df.columns

    def test_write_generic_format_csv(self, temp_dir, sample_csv_content):
        """Test write_generic with format=csv in URL."""
        raw_data = sample_csv_content.encode('utf-8')
        output_path = temp_dir / "output.parquet"
        url = "https://example.com/data?format=csv"

        df = write_generic(raw_data, url, [], output_path)

        assert output_path.exists()

    def test_read_write_interim(self, temp_dir, sample_dataframe, test_settings):
        """Test reading and writing interim data."""
        with patch('exolife.data.utils.settings', test_settings):
            # Write data
            write_interim(sample_dataframe, "test_data", "test_stage")

            # Read it back
            df = read_interim("test_data", "test_stage")

            pd.testing.assert_frame_equal(df, sample_dataframe)

    def test_read_interim_with_columns(self, temp_dir, sample_dataframe, test_settings):
        """Test reading interim data with specific columns."""
        with patch('exolife.data.utils.settings', test_settings):
            # Write data
            write_interim(sample_dataframe, "test_data", "test_stage")

            # Read specific columns
            df = read_interim("test_data", "test_stage",
                              cols=["pl_name", "st_teff"])

            assert len(df.columns) == 2
            assert "pl_name" in df.columns
            assert "st_teff" in df.columns

    def test_read_interim_missing_file(self, test_settings):
        """Test reading non-existent interim data."""
        with patch('exolife.data.utils.settings', test_settings):
            with pytest.raises(FileNotFoundError):
                read_interim("nonexistent", "test_stage")


class TestDataIntegration:
    """Integration tests for data utilities."""

    def test_complete_hz_workflow(self, sample_dataframe):
        """Test complete HZ calculation workflow."""
        df = sample_dataframe.copy()

        # Add HZ edges
        df = add_hz_edges_to_df(df, optimistic=False)

        # Check that all systems have HZ boundaries
        assert not df["hz_inner"].isna().all()
        assert not df["hz_outer"].isna().all()

        # Calculate planetary positions relative to HZ
        df["pl_sma"] = 1.0  # Assume 1 AU for all planets
        df["in_hz"] = (df["pl_sma"] >= df["hz_inner"]) & (
            df["pl_sma"] <= df["hz_outer"])

        # Should have boolean column
        assert df["in_hz"].dtype == bool

    def test_data_source_workflow(self, mock_source_configs, test_settings):
        """Test complete data source configuration workflow."""
        with patch('exolife.data.utils.settings', test_settings):
            # List available sources
            sources = list_data_sources()
            assert len(sources) > 0

            # Get a specific source
            source = get_data_source(sources[0])
            assert isinstance(source, DataSource)

            # Verify source has required fields
            assert source.id
            assert source.name
            assert source.description
