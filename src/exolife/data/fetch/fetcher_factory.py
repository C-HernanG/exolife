"""
Scalable fetcher factory with automatic registration and extensibility.
"""

import logging
from typing import Dict, List, Optional, Type

from .fetcher_base import BaseFetcher, DataSourceConfig

logger = logging.getLogger(__name__)


class FetcherRegistry:
    """
    Registry for fetcher classes with automatic registration support.
    Designed for maximum scalability and easy extension.
    """

    _fetchers: Dict[str, Type[BaseFetcher]] = {}
    _default_fetcher: Optional[Type[BaseFetcher]] = None

    @classmethod
    def register(
        cls,
        fetcher_type: str,
        fetcher_class: Type[BaseFetcher],
        is_default: bool = False,
    ) -> None:
        """
        Register a fetcher class for a specific type.

        Args:
            fetcher_type: Unique identifier for the fetcher
            fetcher_class: Fetcher class that inherits from BaseFetcher
            is_default: Whether this should be the default fetcher
        """
        if not issubclass(fetcher_class, BaseFetcher):
            raise ValueError(
                f"Fetcher class must inherit from BaseFetcher: {fetcher_class}"
            )

        cls._fetchers[fetcher_type] = fetcher_class

        if is_default:
            cls._default_fetcher = fetcher_class

        logger.debug(f"Registered fetcher: {fetcher_type} -> {fetcher_class.__name__}")

    @classmethod
    def unregister(cls, fetcher_type: str) -> None:
        """
        Remove a fetcher from the registry.
        """
        if fetcher_type in cls._fetchers:
            del cls._fetchers[fetcher_type]
            logger.debug(f"Unregistered fetcher: {fetcher_type}")

    @classmethod
    def get_available_types(cls) -> List[str]:
        """
        Get list of all registered fetcher types.
        """
        return list(cls._fetchers.keys())

    @classmethod
    def get_fetcher_class(cls, fetcher_type: str) -> Type[BaseFetcher]:
        """
        Get a specific fetcher class by type.
        """
        if fetcher_type not in cls._fetchers:
            raise ValueError(f"Unknown fetcher type: {fetcher_type}")
        return cls._fetchers[fetcher_type]

    @classmethod
    def create_fetcher(
        cls, config: DataSourceConfig, fetcher_type: Optional[str] = None
    ) -> BaseFetcher:
        """
        Create the best fetcher for the given configuration.

        Args:
            config: Data source configuration
            fetcher_type: Specific fetcher type to use, or None for auto-detection

        Returns:
            Configured fetcher instance
        """
        # If specific type requested, use it
        if fetcher_type:
            fetcher_class = cls.get_fetcher_class(fetcher_type)
            return fetcher_class(config)

        # Try to find the best fetcher by asking each one if it can handle the config
        compatible_fetchers = []
        for ftype, fetcher_class in cls._fetchers.items():
            temp_fetcher = fetcher_class(config)
            if temp_fetcher.can_handle(config):
                compatible_fetchers.append((ftype, fetcher_class))

        if not compatible_fetchers:
            # Fall back to default fetcher if available
            if cls._default_fetcher:
                logger.debug(f"Using default fetcher for source {config.id}")
                return cls._default_fetcher(config)
            raise ValueError(f"No compatible fetcher found for source: {config.id}")

        # Use the first compatible fetcher (could add priority logic here)
        fetcher_type, fetcher_class = compatible_fetchers[0]
        logger.debug(f"Using {fetcher_class.__name__} for source {config.id}")
        return fetcher_class(config)

    @classmethod
    def get_info(cls) -> Dict[str, str]:
        """
        Get information about all registered fetchers.
        """
        info = {}
        for fetcher_type, fetcher_class in cls._fetchers.items():
            default_marker = (
                " (default)" if fetcher_class == cls._default_fetcher else ""
            )
            info[fetcher_type] = (
                f"{fetcher_class.__name__}{default_marker} - "
                f"{fetcher_class.__doc__ or 'No description'}"
            )
        return info


def register_fetcher(
    fetcher_type: str,
    fetcher_class: Optional[Type[BaseFetcher]] = None,
    is_default: bool = False,
):
    """
    Decorator and function for registering fetchers.

    Can be used as:
    1. Function: register_fetcher("my_type", MyFetcher)
    2. Decorator: @register_fetcher("my_type")
    3. Decorator with default: @register_fetcher("my_type", is_default=True)
    """

    def decorator(cls: Type[BaseFetcher]) -> Type[BaseFetcher]:
        FetcherRegistry.register(fetcher_type, cls, is_default)
        return cls

    if fetcher_class is not None:
        # Used as function
        FetcherRegistry.register(fetcher_type, fetcher_class, is_default)
        return fetcher_class
    else:
        # Used as decorator
        return decorator


def get_fetcher(
    config: DataSourceConfig, fetcher_type: Optional[str] = None
) -> BaseFetcher:
    """
    Create the best fetcher for the given configuration.

    This is the main entry point for creating fetchers.
    """
    return FetcherRegistry.create_fetcher(config, fetcher_type)


def list_fetcher_types() -> List[str]:
    """
    List all available fetcher types.
    """
    return FetcherRegistry.get_available_types()


def get_fetcher_info() -> Dict[str, str]:
    """
    Get information about all available fetchers.
    """
    return FetcherRegistry.get_info()
