"""
Registry for dynamically loading preprocessor classes.
"""

import logging
from typing import Dict, List, Type

from .base_preprocessor import BasePreprocessor

logger = logging.getLogger(__name__)


class PreprocessorRegistry:
    """
    Registry for dynamically loading preprocessor classes.
    """

    _registry: Dict[str, Type[BasePreprocessor]] = {}

    @classmethod
    def register(cls, name: str, preprocessor_class: Type[BasePreprocessor]) -> None:
        """
        Register a preprocessor class.
        """
        cls._registry[name] = preprocessor_class
        logger.debug(f"Registered preprocessor: {name}")

    @classmethod
    def get(cls, name: str) -> Type[BasePreprocessor]:
        """
        Get a preprocessor class by name.
        """
        if name not in cls._registry:
            # Try to import from default location
            try:
                from .preprocessors import (
                    ColumnsPruner,
                    DataTypeOptimizer,
                    HZEdgesOptimisticPreprocessor,
                    HZEdgesPreprocessor,
                    MissingValueHandler,
                )

                # Auto-register built-in preprocessors
                cls._registry.update(
                    {
                        "HZEdgesPreprocessor": HZEdgesPreprocessor,
                        "HZEdgesOptimisticPreprocessor": HZEdgesOptimisticPreprocessor,
                        "ColumnsPruner": ColumnsPruner,
                        "DataTypeOptimizer": DataTypeOptimizer,
                        "MissingValueHandler": MissingValueHandler,
                    }
                )
            except ImportError as e:
                logger.warning(f"Failed to auto-register preprocessors: {e}")

        if name not in cls._registry:
            raise ValueError(f"Unknown preprocessor: {name}")

        return cls._registry[name]

    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all available preprocessor names.
        """
        # Ensure built-ins are registered
        try:
            cls.get("ColumnsPruner")  # Trigger auto-registration
        except ValueError:
            pass
        return list(cls._registry.keys())

    @classmethod
    def get_preprocessor_info(cls) -> Dict[str, str]:
        """
        Get information about available preprocessors.
        """
        # Ensure built-ins are registered
        try:
            cls.get("ColumnsPruner")  # Trigger auto-registration
        except ValueError:
            pass

        info = {}
        for name, preprocessor_class in cls._registry.items():
            info[name] = (
                f"{preprocessor_class.__name__} - "
                f"{preprocessor_class.__doc__ or 'No description'}"
            )
        return info
