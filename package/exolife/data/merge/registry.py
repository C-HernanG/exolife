"""
Simplified merger factory and registry for discovering and creating mergers.

This module provides a clean, discoverable way to register and create
data mergers without complex abstractions.
"""

import logging
from typing import Dict, List, Optional, Type

from .base_merger import BaseMerger, MergeConfig
from .mergers import CatalogMerger, CoordinateMerger, IdentifierMerger, PipelineMerger

logger = logging.getLogger(__name__)


class MergerRegistry:
    """
    Registry for discovering and creating data mergers.

    This class maintains a registry of available mergers and provides
    methods to create instances based on strategy names or requirements.
    """

    def __init__(self):
        self._mergers: Dict[str, Type[BaseMerger]] = {}
        self._register_default_mergers()

    def _register_default_mergers(self) -> None:
        """Register the default set of mergers."""
        # Register individual mergers
        self.register(IdentifierMerger)
        self.register(CoordinateMerger)
        self.register(CatalogMerger)
        self.register(PipelineMerger)

        logger.info(f"Registered {len(self._mergers)} merger types")

    def register(self, merger_class: Type[BaseMerger]) -> None:
        """
        Register a merger class.

        Args:
            merger_class: Class that inherits from BaseMerger
        """
        if not issubclass(merger_class, BaseMerger):
            raise ValueError(f"Merger must inherit from BaseMerger: {merger_class}")

        # Create temporary instance to get supported strategies
        temp_config = MergeConfig(strategy="temp", output_name="temp")
        temp_instance = merger_class(temp_config)

        # Register each supported strategy
        for strategy in temp_instance.supported_strategies:
            self._mergers[strategy.lower()] = merger_class
            logger.debug(f"Registered strategy '{strategy}' -> {merger_class.__name__}")

    def unregister(self, strategy: str) -> None:
        """Unregister a strategy."""
        strategy_lower = strategy.lower()
        if strategy_lower in self._mergers:
            del self._mergers[strategy_lower]
            logger.debug(f"Unregistered strategy: {strategy}")

    def get_available_strategies(self) -> List[str]:
        """Get list of all available strategy names."""
        return list(self._mergers.keys())

    def get_merger_for_strategy(self, strategy: str) -> Optional[Type[BaseMerger]]:
        """Get the merger class that handles a specific strategy."""
        return self._mergers.get(strategy.lower())

    def create_merger(self, config: MergeConfig) -> BaseMerger:
        """
        Create a merger instance for the given configuration.

        Args:
            config: Merge configuration specifying strategy and parameters

        Returns:
            Configured merger instance

        Raises:
            ValueError: If strategy is not supported
        """
        strategy_lower = config.strategy.lower()

        if strategy_lower not in self._mergers:
            available = ", ".join(self.get_available_strategies())
            raise ValueError(
                f"Unknown merge strategy: '{config.strategy}'. "
                f"Available strategies: {available}"
            )

        merger_class = self._mergers[strategy_lower]
        merger = merger_class(config)

        # Validate that the merger can handle this configuration
        if not merger.can_handle(config.strategy):
            raise ValueError(
                f"Merger {merger_class.__name__} cannot handle strategy '{config.strategy}'"
            )

        logger.debug(
            f"Created {merger_class.__name__} for strategy '{config.strategy}'"
        )
        return merger

    def get_merger_info(self) -> Dict[str, str]:
        """Get information about available mergers and their strategies."""
        info = {}

        # Group strategies by merger class
        merger_to_strategies = {}
        for strategy, merger_class in self._mergers.items():
            if merger_class not in merger_to_strategies:
                merger_to_strategies[merger_class] = []
            merger_to_strategies[merger_class].append(strategy)

        # Create info string for each merger
        for merger_class, strategies in merger_to_strategies.items():
            temp_config = MergeConfig(strategy="temp", output_name="temp")
            temp_instance = merger_class(temp_config)

            info[merger_class.__name__] = {
                "name": temp_instance.merger_name,
                "strategies": strategies,
                "description": (
                    merger_class.__doc__.strip()
                    if merger_class.__doc__
                    else "No description"
                ),
            }

        return info


# Global registry instance
merger_registry = MergerRegistry()


def create_merger(config: MergeConfig) -> BaseMerger:
    """
    Convenience function to create a merger using the global registry.

    Args:
        config: Merge configuration

    Returns:
        Configured merger instance
    """
    return merger_registry.create_merger(config)


def register_merger(merger_class: Type[BaseMerger]) -> None:
    """
    Convenience function to register a custom merger.

    Args:
        merger_class: Merger class to register
    """
    merger_registry.register(merger_class)


def get_available_strategies() -> List[str]:
    """Convenience function to get available strategies."""
    return merger_registry.get_available_strategies()


def get_merger_info() -> Dict[str, str]:
    """Convenience function to get merger information."""
    return merger_registry.get_merger_info()


def merge_data(strategy: str, output_name: str, **kwargs) -> BaseMerger:
    """
    Convenience function to create and execute a merge operation.

    Args:
        strategy: Merge strategy to use
        output_name: Base name for output files
        **kwargs: Additional configuration parameters

    Returns:
        Merger instance (can be used to access results)
    """
    config = MergeConfig(strategy=strategy, output_name=output_name, **kwargs)

    merger = create_merger(config)
    result = merger.merge()

    if not result.success:
        raise RuntimeError(f"Merge failed: {result.error_message}")

    logger.info(f"Merge completed: {result.output_path}")
    return merger
