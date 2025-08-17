"""
Factory for creating merge strategies based on configuration.
"""

import logging
from typing import Dict, List, Type

from .merge_base import BaseMergeStrategy, MergeConfig
from .pipeline_merger import ConfigurablePipelineMerger

logger = logging.getLogger(__name__)


class MergeStrategyFactory:
    """
    Factory for creating appropriate merge strategies based on configuration.
    """

    def __init__(self):
        self._strategies: Dict[str, Type[BaseMergeStrategy]] = {}
        self._register_default_strategies()

    def _register_default_strategies(self) -> None:
        """
        Register the default merge strategies.
        """
        # Register the configurable pipeline merger for all strategy names
        self.register_strategy("exolife_pipeline", ConfigurablePipelineMerger)

        # Legacy compatibility aliases
        self.register_strategy("unified_ingestion", ConfigurablePipelineMerger)
        self.register_strategy("unified", ConfigurablePipelineMerger)
        self.register_strategy("ingestion", ConfigurablePipelineMerger)
        self.register_strategy("baseline", ConfigurablePipelineMerger)
        self.register_strategy("gaia_enriched", ConfigurablePipelineMerger)

        # Gaia-specific hybrid merger
        self.register_strategy("gaia_hybrid", GaiaHybridMergeStrategy)

    def register_strategy(
        self, strategy_name: str, strategy_class: Type[BaseMergeStrategy]
    ) -> None:
        """
        Register a merge strategy for a specific name.
        """
        if not issubclass(strategy_class, BaseMergeStrategy):
            raise ValueError(
                f"Strategy class must inherit from BaseMergeStrategy: {strategy_class}"
            )

        self._strategies[strategy_name] = strategy_class
        logger.debug(
            f"Registered merge strategy: {strategy_name} -> {strategy_class.__name__}"
        )

    def unregister_strategy(self, strategy_name: str) -> None:
        """
        Unregister a merge strategy.
        """
        if strategy_name in self._strategies:
            del self._strategies[strategy_name]
            logger.debug(f"Unregistered merge strategy: {strategy_name}")

    def get_available_strategies(self) -> List[str]:
        """
        Get list of available merge strategy names.
        """
        return list(self._strategies.keys())

    def create_strategy(self, config: MergeConfig) -> BaseMergeStrategy:
        """
        Create appropriate merge strategy based on configuration.
        """
        strategy_name = config.strategy.lower()

        if strategy_name not in self._strategies:
            raise ValueError(
                f"Unknown merge strategy: {config.strategy}. "
                f"Available strategies: {self.get_available_strategies()}"
            )

        strategy_class = self._strategies[strategy_name]
        strategy = strategy_class(config)

        # Validate that the strategy can handle this configuration
        if not strategy.can_handle(config):
            raise ValueError(
                f"Strategy {strategy_name} cannot handle the provided configuration"
            )

        logger.debug(f"Created {strategy_class.__name__} for merge operation")
        return strategy

    def get_strategy_info(self) -> Dict[str, str]:
        """
        Get information about available merge strategies.
        """
        info = {}
        for strategy_name, strategy_class in self._strategies.items():
            info[strategy_name] = (
                f"{strategy_class.__name__} - "
                f"{strategy_class.__doc__ or 'No description'}"
            )
        return info


class GaiaHybridMergeStrategy(BaseMergeStrategy):
    """
    Specialized strategy for merging Gaia DR2+DR3 hybrid datasets.
    """

    def can_handle(self, config: MergeConfig) -> bool:
        """Check if this strategy can handle the configuration."""
        return config.strategy.lower() == "gaia_hybrid"

    def merge(self) -> None:
        """Execute the Gaia hybrid merge."""
        from .gaia_hybrid_merger import merge_gaia_hybrid

        # Use the hybrid merger function
        output_path = merge_gaia_hybrid(self.config.dict())
        logger.info(f"Gaia hybrid merge completed: {output_path}")


# Global factory instance
merge_factory = MergeStrategyFactory()


def get_merge_strategy(config: MergeConfig) -> BaseMergeStrategy:
    """
    Convenience function to get a merge strategy for the given configuration.
    """
    return merge_factory.create_strategy(config)


def register_custom_strategy(
    strategy_name: str, strategy_class: Type[BaseMergeStrategy]
) -> None:
    """
    Convenience function to register a custom merge strategy.
    """
    merge_factory.register_strategy(strategy_name, strategy_class)


def get_merge_strategy_info():
    """
    Convenience function to get information about available merge strategies.
    """
    return merge_factory.get_strategy_info()
