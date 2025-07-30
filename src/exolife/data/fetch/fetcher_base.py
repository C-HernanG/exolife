"""
Abstract base class for data fetchers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DataSourceConfig(BaseModel):
    """
    Configuration for a data source.
    """

    id: str = Field(..., description="Unique identifier for the data source")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Description of the data source")
    download_url: Optional[str] = Field(None, description="URL to download the data")
    adql: Optional[str] = Field(None, description="ADQL query for astronomical data")
    columns_to_keep: List[str] = Field(
        default_factory=list, description="Columns to retain"
    )
    primary_keys: List[str] = Field(
        default_factory=list, description="Primary key columns"
    )
    join_keys: Dict[str, List[str]] = Field(
        default_factory=dict, description="Join keys for merging"
    )
    refresh: str = Field(default="static", description="Refresh strategy")
    format: Optional[str] = Field(None, description="Data format")
    timeout: int = Field(default=300, description="Request timeout in seconds")

    model_config = {"extra": "allow"}


class FetchResult(BaseModel):
    """
    Result of a fetch operation.
    """

    source_id: str
    path: Path
    success: bool
    error_message: Optional[str] = None
    rows_fetched: Optional[int] = None
    size_bytes: Optional[int] = None

    model_config = {"arbitrary_types_allowed": True}


class BaseFetcher(ABC):
    """
    Abstract base class for data fetchers implementing a pluggable architecture.
    """

    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    @abstractmethod
    def can_handle(self, config: DataSourceConfig) -> bool:
        """
        Check if this fetcher can handle the given data source configuration.
        """
        pass

    @abstractmethod
    def fetch(self, force: bool = False) -> FetchResult:
        """
        Fetch data from the source and return the result.
        """
        pass

    @property
    @abstractmethod
    def fetcher_type(self) -> str:
        """
        Return the type identifier for this fetcher.
        """
        pass

    def validate_config(self) -> bool:
        """
        Validate the configuration for this fetcher.
        """
        return True

    def get_cache_path(self, directory: Path) -> Path:
        """
        Get the cache path for this data source.
        """
        return directory / f"{self.config.id}.parquet"

    def is_cached(self, directory: Path) -> bool:
        """
        Check if data is already cached.
        """
        return self.get_cache_path(directory).exists()

    def save_dataframe(self, df: pd.DataFrame, path: Path) -> None:
        """
        Save DataFrame to the specified path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        self.logger.info(f"Saved {len(df)} rows to {path}")

    def load_dataframe(
        self, path: Path, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load DataFrame from the specified path.
        """
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        return pd.read_parquet(path, columns=columns)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(source_id={self.config.id})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
