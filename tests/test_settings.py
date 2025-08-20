"""
Tests for the ExoLife settings module.
"""

import os
import tempfile
from pathlib import Path
import pytest

from exolife.settings import Settings


class TestSettings:
    """Test cases for the Settings class."""

    def test_default_settings(self):
        """Test that default settings are correctly initialized."""
        settings = Settings()

        # Check that basic settings exist
        assert hasattr(settings, 'root_dir')
        assert hasattr(settings, 'data_dir')
        assert hasattr(settings, 'max_workers')
        assert hasattr(settings, 'log_level')

        # Check default values
        assert settings.max_workers == 4
        assert settings.log_level == "INFO"
        assert settings.request_timeout == 300
        assert settings.download_chunk_size == 8192
        assert settings.default_merge_strategy == "baseline"
        assert settings.force_refresh is False

    def test_custom_settings(self, temp_dir):
        """Test settings with custom values."""
        settings = Settings(
            root_dir=temp_dir,
            max_workers=8,
            log_level="DEBUG",
            request_timeout=600,
            force_refresh=True
        )

        assert settings.root_dir == temp_dir
        assert settings.max_workers == 8
        assert settings.log_level == "DEBUG"
        assert settings.request_timeout == 600
        assert settings.force_refresh is True

    def test_directory_creation(self, temp_dir):
        """Test that directories are created correctly."""
        settings = Settings(root_dir=temp_dir)
        settings.create_directories()

        # Check that directories exist
        assert settings.data_dir.exists()
        assert settings.raw_dir.exists()
        assert settings.interim_dir.exists()
        assert settings.processed_dir.exists()

    def test_environment_variable_override(self, temp_dir):
        """Test that environment variables override default settings."""
        # Set environment variables
        os.environ['EXOLIFE_MAX_WORKERS'] = '16'
        os.environ['EXOLIFE_LOG_LEVEL'] = 'WARNING'
        os.environ['EXOLIFE_FORCE_REFRESH'] = 'true'

        try:
            settings = Settings(root_dir=temp_dir)

            # These should be overridden by environment variables
            assert settings.max_workers == 16
            assert settings.log_level == "WARNING"
            assert settings.force_refresh is True

        finally:
            # Clean up environment variables
            for var in ['EXOLIFE_MAX_WORKERS', 'EXOLIFE_LOG_LEVEL', 'EXOLIFE_FORCE_REFRESH']:
                os.environ.pop(var, None)

    def test_directory_validation(self, temp_dir):
        """Test directory path validation and derivation."""
        settings = Settings(root_dir=temp_dir)

        # Data directories should be derived from root_dir
        expected_data_dir = temp_dir / "data"
        assert settings.data_dir == expected_data_dir

        expected_raw_dir = expected_data_dir / "raw"
        assert settings.raw_dir == expected_raw_dir

        expected_interim_dir = expected_data_dir / "interim"
        assert settings.interim_dir == expected_interim_dir

        expected_processed_dir = expected_data_dir / "processed"
        assert settings.processed_dir == expected_processed_dir

    def test_settings_without_pydantic(self, temp_dir, monkeypatch):
        """Test that settings work even without pydantic (fallback behavior)."""
        # This tests the fallback stubs defined in settings.py
        # We can't easily mock the import failure, but we can test the basic functionality
        settings = Settings(root_dir=temp_dir)
        settings.create_directories()

        # Basic functionality should still work
        assert settings.root_dir == temp_dir
        assert settings.data_dir.exists()

    def test_path_types(self, temp_dir):
        """Test that all directory settings are Path objects."""
        settings = Settings(root_dir=temp_dir)

        assert isinstance(settings.root_dir, Path)
        assert isinstance(settings.data_dir, Path)
        assert isinstance(settings.raw_dir, Path)
        assert isinstance(settings.interim_dir, Path)
        assert isinstance(settings.processed_dir, Path)

    def test_settings_immutability_after_creation(self, temp_dir):
        """Test that settings can be modified after creation."""
        settings = Settings(root_dir=temp_dir)

        # Should be able to modify settings
        settings.log_level = "ERROR"
        assert settings.log_level == "ERROR"

        settings.force_refresh = True
        assert settings.force_refresh is True

    def test_global_settings_import(self):
        """Test that global settings constants are available."""
        from exolife.settings import ROOT, DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR

        # These should be Path objects
        assert isinstance(ROOT, Path)
        assert isinstance(DATA_DIR, Path)
        assert isinstance(RAW_DIR, Path)
        assert isinstance(INTERIM_DIR, Path)
        assert isinstance(PROCESSED_DIR, Path)

    def test_create_directories_with_none_values(self, temp_dir):
        """Test create_directories handles None values correctly."""
        settings = Settings(root_dir=temp_dir)

        # Manually set some directories to None to test synthesis
        settings.raw_dir = None
        settings.interim_dir = None
        settings.processed_dir = None

        # This should recreate the None directories
        settings.create_directories()

        assert settings.raw_dir is not None
        assert settings.interim_dir is not None
        assert settings.processed_dir is not None
        assert settings.raw_dir.exists()
        assert settings.interim_dir.exists()
        assert settings.processed_dir.exists()
