"""
Tests for the ExoLife data fetcher modules.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import pandas as pd
import requests

from exolife.data.fetch.fetcher_base import (
    BaseFetcher, DataSourceConfig, FetchResult
)
from exolife.data.fetch.fetchers.http_fetcher import HttpFetcher
from exolife.data.fetch.manager import FetchManager
from exolife.data.fetch.registry import FetcherRegistry, register_fetcher


class TestDataSourceConfig:
    """Test cases for DataSourceConfig."""

    def test_data_source_config_creation(self):
        """Test basic DataSourceConfig creation."""
        config = DataSourceConfig(
            id="test_source",
            name="Test Source",
            description="A test data source",
            download_url="https://example.com/data.csv",
            columns_to_keep=["col1", "col2"],
            primary_keys=["id"],
            join_keys={"other": ["key"]},
            format="csv",
            timeout=120
        )

        assert config.id == "test_source"
        assert config.name == "Test Source"
        assert config.download_url == "https://example.com/data.csv"
        assert config.columns_to_keep == ["col1", "col2"]
        assert config.timeout == 120

    def test_data_source_config_defaults(self):
        """Test DataSourceConfig with default values."""
        config = DataSourceConfig(
            id="minimal",
            name="Minimal",
            description="Minimal config"
        )

        assert config.columns_to_keep == []
        assert config.primary_keys == []
        assert config.join_keys == {}
        assert config.timeout == 300  # Default timeout

    def test_data_source_config_extra_fields(self):
        """Test that extra fields are allowed."""
        config = DataSourceConfig(
            id="test",
            name="Test",
            description="Test",
            custom_field="custom_value"
        )

        # Should be accessible as attribute
        assert config.custom_field == "custom_value"


class TestFetchResult:
    """Test cases for FetchResult."""

    def test_fetch_result_success(self, temp_dir):
        """Test successful FetchResult creation."""
        output_path = temp_dir / "data.parquet"
        output_path.touch()

        result = FetchResult(
            source_id="test_source",
            path=output_path,
            success=True,
            rows_fetched=100,
            size_bytes=1024
        )

        assert result.source_id == "test_source"
        assert result.path == output_path
        assert result.success is True
        assert result.rows_fetched == 100
        assert result.size_bytes == 1024
        assert result.error_message is None

    def test_fetch_result_failure(self):
        """Test failed FetchResult creation."""
        result = FetchResult(
            source_id="failed_source",
            path=None,
            success=False,
            error_message="Download failed"
        )

        assert result.success is False
        assert result.error_message == "Download failed"
        assert result.rows_fetched is None


class MockFetcher(BaseFetcher):
    """Mock fetcher for testing."""

    def can_handle(self, config):
        return config.id.startswith("mock_")

    def fetch(self, force=False):
        return FetchResult(
            source_id=self.config.id,
            path=Path("mock_path.parquet"),
            success=True,
            rows_fetched=10
        )

    @property
    def fetcher_type(self):
        return "mock"


class TestBaseFetcher:
    """Test cases for BaseFetcher abstract base class."""

    def test_base_fetcher_instantiation(self, mock_data_source_config):
        """Test BaseFetcher instantiation with mock implementation."""
        fetcher = MockFetcher(mock_data_source_config)

        assert fetcher.config == mock_data_source_config
        assert fetcher.fetcher_type == "mock"

    def test_validate_config(self, mock_data_source_config):
        """Test config validation."""
        fetcher = MockFetcher(mock_data_source_config)
        assert fetcher.validate_config() is True

    def test_get_cache_path(self, mock_data_source_config, temp_dir):
        """Test cache path generation."""
        fetcher = MockFetcher(mock_data_source_config)
        cache_path = fetcher.get_cache_path(temp_dir)

        expected = temp_dir / f"{mock_data_source_config.id}.parquet"
        assert cache_path == expected

    def test_is_cached(self, mock_data_source_config, temp_dir):
        """Test cache checking."""
        fetcher = MockFetcher(mock_data_source_config)

        # Initially not cached
        assert fetcher.is_cached(temp_dir) is False

        # Create cache file
        cache_path = fetcher.get_cache_path(temp_dir)
        cache_path.touch()

        # Now should be cached
        assert fetcher.is_cached(temp_dir) is True

    def test_save_and_load_dataframe(self, mock_data_source_config, temp_dir, sample_dataframe):
        """Test DataFrame save and load operations."""
        fetcher = MockFetcher(mock_data_source_config)
        output_path = temp_dir / "test_data.parquet"

        # Save DataFrame
        fetcher.save_dataframe(sample_dataframe, output_path)
        assert output_path.exists()

        # Load DataFrame
        loaded_df = fetcher.load_dataframe(output_path)
        pd.testing.assert_frame_equal(loaded_df, sample_dataframe)

    def test_load_dataframe_specific_columns(self, mock_data_source_config, temp_dir, sample_dataframe):
        """Test loading specific columns from DataFrame."""
        fetcher = MockFetcher(mock_data_source_config)
        output_path = temp_dir / "test_data.parquet"

        # Save DataFrame
        fetcher.save_dataframe(sample_dataframe, output_path)

        # Load specific columns
        columns = ["pl_name", "st_teff"]
        loaded_df = fetcher.load_dataframe(output_path, columns=columns)

        assert list(loaded_df.columns) == columns
        assert len(loaded_df) == len(sample_dataframe)

    def test_load_dataframe_nonexistent_file(self, mock_data_source_config, temp_dir):
        """Test loading DataFrame from non-existent file."""
        fetcher = MockFetcher(mock_data_source_config)
        nonexistent_path = temp_dir / "nonexistent.parquet"

        with pytest.raises(FileNotFoundError):
            fetcher.load_dataframe(nonexistent_path)

    def test_fetcher_string_representations(self, mock_data_source_config):
        """Test string representations of fetcher."""
        fetcher = MockFetcher(mock_data_source_config)

        str_repr = str(fetcher)
        assert "MockFetcher" in str_repr
        assert mock_data_source_config.id in str_repr

        repr_str = repr(fetcher)
        assert "MockFetcher" in repr_str
        assert "config=" in repr_str


class TestHttpFetcher:
    """Test cases for HttpFetcher."""

    def test_can_handle_http_urls(self, mock_data_source_config):
        """Test that HttpFetcher can handle HTTP URLs."""
        # HTTP URL
        config = mock_data_source_config.model_copy()
        config.download_url = "http://example.com/data.csv"

        fetcher = HttpFetcher(config)
        assert fetcher.can_handle(config) is True

        # HTTPS URL
        config.download_url = "https://example.com/data.csv"
        assert fetcher.can_handle(config) is True

    def test_cannot_handle_non_http_urls(self, mock_data_source_config):
        """Test that HttpFetcher cannot handle non-HTTP URLs."""
        config = mock_data_source_config.model_copy()

        # FTP URL
        config.download_url = "ftp://example.com/data.csv"
        fetcher = HttpFetcher(config)
        assert fetcher.can_handle(config) is False

        # No URL
        config.download_url = None
        assert fetcher.can_handle(config) is False

    def test_fetcher_type(self, mock_data_source_config):
        """Test fetcher type property."""
        fetcher = HttpFetcher(mock_data_source_config)
        assert fetcher.fetcher_type == "http"

    @patch('exolife.data.fetch.fetchers.http_fetcher.stream_download')
    @patch('exolife.data.fetch.fetchers.http_fetcher.write_generic')
    def test_successful_fetch(self, mock_write, mock_download, mock_data_source_config,
                              temp_dir, sample_csv_content, test_settings):
        """Test successful HTTP fetch."""
        # Setup mocks
        mock_download.return_value = sample_csv_content.encode('utf-8')

        sample_df = pd.DataFrame({
            'pl_name': ['Test Planet'],
            'st_teff': [5000.0]
        })
        mock_write.return_value = sample_df

        # Configure fetcher
        config = mock_data_source_config.model_copy()
        config.download_url = "https://example.com/data.csv"

        with patch('exolife.data.fetch.fetchers.http_fetcher.settings', test_settings):
            fetcher = HttpFetcher(config)
            result = fetcher.fetch()

        # Verify result
        assert result.success is True
        assert result.source_id == config.id
        assert result.rows_fetched == 1

        # Verify mocks were called
        mock_download.assert_called_once_with(config.download_url)
        mock_write.assert_called_once()

    @patch('exolife.data.fetch.fetchers.http_fetcher.stream_download')
    def test_fetch_with_download_failure(self, mock_download, mock_data_source_config,
                                         temp_dir, test_settings):
        """Test fetch when download fails."""
        # Mock download failure
        mock_download.side_effect = requests.RequestException("Network error")

        config = mock_data_source_config.model_copy()
        config.download_url = "https://example.com/data.csv"

        with patch('exolife.data.fetch.fetchers.http_fetcher.settings', test_settings):
            fetcher = HttpFetcher(config)
            result = fetcher.fetch()

        # Should create fallback dataset
        assert result.success is True  # Fallback is considered successful
        assert "Download failed" in result.error_message

    def test_convert_to_source(self, mock_data_source_config):
        """Test conversion from DataSourceConfig to DataSource."""
        fetcher = HttpFetcher(mock_data_source_config)
        source = fetcher._convert_to_source(mock_data_source_config)

        assert source.id == mock_data_source_config.id
        assert source.name == mock_data_source_config.name
        assert source.download_url == mock_data_source_config.download_url

    def test_create_fallback_dataset(self, mock_data_source_config, temp_dir, test_settings):
        """Test fallback dataset creation."""
        config = mock_data_source_config.model_copy()
        config.columns_to_keep = ["col1", "col2"]
        config.primary_keys = ["col1"]

        with patch('exolife.data.fetch.fetchers.http_fetcher.settings', test_settings):
            fetcher = HttpFetcher(config)
            output_path = temp_dir / "fallback.parquet"

            result = fetcher._create_fallback_dataset(
                output_path, "Test error")

        assert result.success is True
        assert "Test error" in result.error_message
        assert output_path.exists()

        # Check fallback data
        df = pd.read_parquet(output_path)
        assert len(df) == 1
        assert "col1" in df.columns
        assert "col2" in df.columns


class TestFetcherRegistry:
    """Test cases for FetcherRegistry."""

    def test_registry_singleton(self):
        """Test that registry is a singleton."""
        registry1 = FetcherRegistry()
        registry2 = FetcherRegistry()
        assert registry1 is registry2

    def test_register_fetcher_decorator(self):
        """Test the register_fetcher decorator."""
        registry = FetcherRegistry()

        @register_fetcher("test_fetcher")
        class TestFetcher(BaseFetcher):
            def can_handle(self, config):
                return True

            def fetch(self, force=False):
                return FetchResult("test", Path("test"), True)

            @property
            def fetcher_type(self):
                return "test"

        # Check that fetcher was registered
        assert "test_fetcher" in registry._fetchers
        assert registry._fetchers["test_fetcher"] == TestFetcher

    def test_get_fetcher_for_config(self, mock_data_source_config):
        """Test getting appropriate fetcher for config."""
        registry = FetcherRegistry()

        # Register mock fetcher
        @register_fetcher("mock_test")
        class MockTestFetcher(BaseFetcher):
            def can_handle(self, config):
                return config.id == "test_source"

            def fetch(self, force=False):
                return FetchResult("test", Path("test"), True)

            @property
            def fetcher_type(self):
                return "mock_test"

        fetcher_class = registry.get_fetcher_for_config(
            mock_data_source_config)
        assert fetcher_class == MockTestFetcher

    def test_get_fetcher_no_match(self):
        """Test getting fetcher when no fetcher can handle config."""
        registry = FetcherRegistry()

        config = DataSourceConfig(
            id="unhandleable",
            name="Unhandleable",
            description="Cannot be handled"
        )

        with pytest.raises(ValueError, match="No fetcher found"):
            registry.get_fetcher_for_config(config)

    def test_list_fetchers(self):
        """Test listing available fetchers."""
        registry = FetcherRegistry()
        fetchers = registry.list_fetchers()

        assert isinstance(fetchers, list)
        # Should include at least the HTTP fetcher
        assert any("http" in name.lower() for name in fetchers)

    def test_create_fetcher(self, mock_data_source_config):
        """Test creating fetcher instance."""
        registry = FetcherRegistry()

        # HTTP fetcher should be available by default
        config = mock_data_source_config.model_copy()
        config.download_url = "https://example.com/data.csv"

        fetcher = registry.create_fetcher(config)
        assert isinstance(fetcher, HttpFetcher)
        assert fetcher.config == config


class TestFetchManager:
    """Test cases for FetchManager."""

    def test_fetch_manager_creation(self, test_settings):
        """Test FetchManager creation."""
        with patch('exolife.data.fetch.manager.settings', test_settings):
            manager = FetchManager()

            assert manager.settings == test_settings
            assert hasattr(manager, 'registry')

    def test_fetch_single_source(self, mock_data_source_config, test_settings, sample_csv_content):
        """Test fetching a single data source."""
        config = mock_data_source_config.model_copy()
        config.download_url = "https://example.com/data.csv"

        with patch('exolife.data.fetch.manager.settings', test_settings):
            with patch('exolife.data.fetch.fetchers.http_fetcher.stream_download') as mock_download:
                with patch('exolife.data.fetch.fetchers.http_fetcher.write_generic') as mock_write:
                    # Setup mocks
                    mock_download.return_value = sample_csv_content.encode(
                        'utf-8')
                    sample_df = pd.DataFrame({'col': [1, 2, 3]})
                    mock_write.return_value = sample_df

                    manager = FetchManager()
                    result = manager.fetch_source(config)

                    assert result.success is True
                    assert result.source_id == config.id

    def test_fetch_multiple_sources(self, test_settings):
        """Test fetching multiple data sources."""
        configs = [
            DataSourceConfig(
                id="source1",
                name="Source 1",
                description="First source",
                download_url="https://example.com/data1.csv"
            ),
            DataSourceConfig(
                id="source2",
                name="Source 2",
                description="Second source",
                download_url="https://example.com/data2.csv"
            )
        ]

        with patch('exolife.data.fetch.manager.settings', test_settings):
            with patch.object(FetchManager, 'fetch_source') as mock_fetch:
                mock_fetch.return_value = FetchResult(
                    "test", Path("test"), True)

                manager = FetchManager()
                results = manager.fetch_sources(configs)

                assert len(results) == 2
                assert all(r.success for r in results)

    def test_fetch_with_cache_check(self, mock_data_source_config, test_settings, temp_dir):
        """Test fetching with cache checking."""
        config = mock_data_source_config.model_copy()
        config.download_url = "https://example.com/data.csv"

        # Create cached file
        cache_dir = test_settings.raw_dir / config.id
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{config.id}.parquet"

        sample_df = pd.DataFrame({'col': [1, 2, 3]})
        sample_df.to_parquet(cache_file, index=False)

        with patch('exolife.data.fetch.manager.settings', test_settings):
            manager = FetchManager()

            # Should use cache
            result = manager.fetch_source(config, force=False)
            assert result.success is True

            # Should bypass cache
            with patch('exolife.data.fetch.fetchers.http_fetcher.stream_download') as mock_download:
                with patch('exolife.data.fetch.fetchers.http_fetcher.write_generic') as mock_write:
                    mock_download.return_value = b"new,data\n1,2"
                    mock_write.return_value = sample_df

                    result = manager.fetch_source(config, force=True)
                    assert result.success is True
                    mock_download.assert_called_once()
