"""
Configuration module for ExoLife project paths and environment overrides.
"""

from pathlib import Path
from typing import Optional

# Resolve pydantic imports in a way that is compatible with both
# pydantic v1 and v2.  In pydantic v2 the BaseSettings class was
# moved to the `pydantic-settings` package and importing it from
# `pydantic` raises a `PydanticImportError`.  We attempt to import
# BaseSettings from pydantic-settings first.  If that fails, we fall
# back to pydantic v1 style imports.  As a last resort we define
# minimal stubs to allow the package to load even without pydantic.
try:
    # Prefer pydantic v2 style separation
    from pydantic import Field, field_validator  # type: ignore
    from pydantic_settings import BaseSettings  # type: ignore
except Exception:
    try:
        # pydantic v1 (BaseSettings still in pydantic)
        from pydantic import (  # type: ignore
            BaseSettings,
            Field,
            validator as field_validator,  # type: ignore
        )
    except Exception:
        # Final fallback: define lightweight stubs so that the module can
        # still be imported even if pydantic is unavailable.  These
        # stubs do not provide full validation but satisfy attribute
        # access in downstream code.
        class BaseSettings:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        def Field(default=None, default_factory=None, **kwargs):  # type: ignore
            # Return value produced by default_factory if provided, else the default
            if default_factory is not None:
                try:
                    return default_factory()
                except Exception:
                    return default
            return default

        def field_validator(*args, **kwargs):  # type: ignore
            def decorator(fn):
                return fn

            return decorator


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
        """
        Create all data directories if they don't exist.  This method also
        synthesises missing directory fields when pydantic validation has
        not run (e.g., when BaseSettings is a stub).  If any of the
        directory attributes are ``None``, they will be constructed
        relative to ``root_dir``.
        """
        # Synthesize base data directory if absent
        if not getattr(self, "data_dir", None):
            self.data_dir = self.root_dir / "data"
        # Derive raw, interim and processed directories if absent
        if not getattr(self, "raw_dir", None):
            self.raw_dir = self.data_dir / "raw"
        if not getattr(self, "interim_dir", None):
            self.interim_dir = self.data_dir / "interim"
        if not getattr(self, "processed_dir", None):
            self.processed_dir = self.data_dir / "processed"

        directories = [
            self.data_dir,
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
        ]
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.create_directories()

ROOT = settings.root_dir
DATA_DIR = settings.data_dir
RAW_DIR = settings.raw_dir
INTERIM_DIR = settings.interim_dir
PROCESSED_DIR = settings.processed_dir
