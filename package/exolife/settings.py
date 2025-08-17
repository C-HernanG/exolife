"""
Configuration module for ExoLife project paths and environment overrides.
"""

from pathlib import Path
from typing import Optional

try:
    from pydantic import Field, field_validator
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings, Field, validator as field_validator


class Settings(BaseSettings):
    """
    Application settings using Pydantic for validation
    and environment variable support.
    """

    # Project root directory
    root_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )

    # Data directories
    data_dir: Optional[Path] = None
    raw_dir: Optional[Path] = None
    interim_dir: Optional[Path] = None
    processed_dir: Optional[Path] = None

    # Fetcher settings
    max_workers: int = Field(
        default=4, description="Maximum number of parallel workers for fetching"
    )

    request_timeout: int = Field(default=300, description="Request timeout in seconds")

    download_chunk_size: int = Field(
        default=8192, description="Download chunk size in bytes"
    )

    # Merger settings
    default_merge_strategy: str = Field(
        default="baseline", description="Default merge strategy"
    )

    # General settings
    log_level: str = Field(default="INFO", description="Logging level")

    force_refresh: bool = Field(
        default=False, description="Force refresh of cached data"
    )

    model_config = {
        "env_prefix": "EXOLIFE_",
        "case_sensitive": False,
    }

    @field_validator("data_dir")
    @classmethod
    def set_data_dir(cls, v, info):
        values = info.data if hasattr(info, "data") else getattr(info, "context", {})
        return (
            v
            or values.get("root_dir", Path(__file__).resolve().parent.parent.parent)
            / "data"
        )

    @field_validator("raw_dir")
    @classmethod
    def set_raw_dir(cls, v, info):
        values = info.data if hasattr(info, "data") else getattr(info, "context", {})
        data_dir = (
            values.get("data_dir")
            or values.get("root_dir", Path(__file__).resolve().parent.parent.parent)
            / "data"
        )
        return v or data_dir / "raw"

    @field_validator("interim_dir")
    @classmethod
    def set_interim_dir(cls, v, info):
        values = info.data if hasattr(info, "data") else getattr(info, "context", {})
        data_dir = (
            values.get("data_dir")
            or values.get("root_dir", Path(__file__).resolve().parent.parent.parent)
            / "data"
        )
        return v or data_dir / "interim"

    @field_validator("processed_dir")
    @classmethod
    def set_processed_dir(cls, v, info):
        values = info.data if hasattr(info, "data") else getattr(info, "context", {})
        data_dir = (
            values.get("data_dir")
            or values.get("root_dir", Path(__file__).resolve().parent.parent.parent)
            / "data"
        )
        return v or data_dir / "processed"

    def create_directories(self) -> None:
        """Create all data directories if they don't exist."""
        directories = [
            self.data_dir,
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
        ]
        for directory in directories:
            if directory:
                directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.create_directories()

ROOT = settings.root_dir
DATA_DIR = settings.data_dir
RAW_DIR = settings.raw_dir
INTERIM_DIR = settings.interim_dir
PROCESSED_DIR = settings.processed_dir
