"""
Abstract base class for data preprocessors.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PreprocessorConfig(BaseModel):
    """
    Configuration for preprocessor operations.
    """

    name: str = Field(..., description="Name of the preprocessor")
    enabled: bool = Field(
        default=True, description="Whether the preprocessor is enabled"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Preprocessor parameters"
    )

    model_config = {"extra": "allow"}


class PreprocessorResult(BaseModel):
    """
    Result of a preprocessing operation.
    """

    preprocessor_name: str
    success: bool
    error_message: Optional[str] = None
    rows_processed: Optional[int] = None
    columns_added: List[str] = Field(default_factory=list)
    columns_removed: List[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class BasePreprocessor(ABC):
    """
    Abstract base class for data preprocessors.
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.config = config or PreprocessorConfig(name=self.__class__.__name__)
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the DataFrame and return the transformed version.
        """
        pass

    @property
    @abstractmethod
    def preprocessor_name(self) -> str:
        """
        Return the name identifier for this preprocessor.
        """
        pass

    def validate_config(self) -> bool:
        """
        Validate the configuration for this preprocessor.
        """
        return True

    def can_process(self, df: pd.DataFrame) -> bool:
        """
        Check if this preprocessor can process the given DataFrame.
        """
        return True

    def get_required_columns(self) -> List[str]:
        """
        Return a list of columns required by this preprocessor.
        """
        return []

    def check_requirements(self, df: pd.DataFrame) -> bool:
        """
        Check if the DataFrame meets the requirements for processing.
        """
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            self.logger.warning(f"Missing required columns: {missing_cols}")
            return False

        return True

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
