"""
Merge strategies using the proven logic.
"""

import io
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ...settings import settings
from ..utils import fetch_adql, gaia_int, get_data_source, norm_name
from .merge_base import BaseMergeStrategy, MergeConfig, MergeResult

logger = logging.getLogger(__name__)


class BaselineMerger(BaseMergeStrategy):
    """
    Merger that implements the baseline merge logic.
    """

    @property
    def strategy_name(self) -> str:
        return "baseline"

    def can_handle(self, config: MergeConfig) -> bool:
        """
        Check if this strategy can handle the configuration.
        """
        return config.strategy.lower() == "baseline"

    def merge(self, data_sources: Dict[str, pd.DataFrame]) -> MergeResult:
        """
        Execute the baseline merge using the old proven logic.
        """
        try:
            # Use the baseline logic directly
            merged_df = self._baseline_merge()

            # Save the result
            output_path = (
                settings.merged_dir / "parquet" / f"merged_{self.strategy_name}.parquet"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            merged_df.to_parquet(output_path, index=False)

            # Also save CSV version
            csv_path = settings.merged_dir / "csv" / f"merged_{self.strategy_name}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            merged_df.to_csv(csv_path, index=False)

            return MergeResult(
                output_name=self.config.output_name,
                output_path=output_path,
                success=True,
                input_sources=self.config.sources,
                rows_before={source: len(df) for source, df in data_sources.items()},
                rows_after=len(merged_df),
                merge_statistics=self.get_merge_statistics(data_sources, merged_df),
            )

        except Exception as e:
            logger.error(f"Baseline merge failed: {e}")
            return MergeResult(
                output_name=self.config.output_name,
                output_path=Path("/tmp/failed"),  # Provide a valid path for Pydantic
                success=False,
                error_message=str(e),
                input_sources=self.config.sources,
                rows_before={},
            )

    def _baseline_merge(self) -> pd.DataFrame:
        """
        Execute the baseline merge logic.
        """
        # Use raw_dir instead of interim for new system compatibility
        nasa_path = settings.raw_dir / "nasa_exoplanet_archive_pscomppars.parquet"
        phl_path = settings.raw_dir / "phl_exoplanet_catalog.parquet"

        if not nasa_path.exists() or not phl_path.exists():
            raise FileNotFoundError(
                f"Required data files not found."
                f"NASA: {nasa_path.exists()}, PHL: {phl_path.exists()}"
            )

        nasa = pd.read_parquet(nasa_path)
        phl = pd.read_parquet(phl_path)

        nasa["_key"] = norm_name(nasa["pl_name"])
        phl["_key"] = norm_name(phl["P_NAME"])
        df = nasa.merge(phl, on="_key", how="left").drop(
            columns=["_key", "P_NAME"], errors="ignore"
        )
        return df


class GaiaEnrichedMerger(BaseMergeStrategy):
    """
    Merger that implements the gaia_enriched merge logic.
    """

    @property
    def strategy_name(self) -> str:
        return "gaia_enriched"

    def can_handle(self, config: MergeConfig) -> bool:
        """
        Check if this strategy can handle the configuration.
        """
        return config.strategy.lower() == "gaia_enriched"

    def merge(self, data_sources: Dict[str, pd.DataFrame]) -> MergeResult:
        """
        Execute the gaia_enriched merge using the old proven logic.
        """
        try:
            # Use the gaia_enriched logic directly
            merged_df = self._gaia_enriched_merge()

            # Save the result
            output_path = (
                settings.merged_dir / "parquet" / f"merged_{self.strategy_name}.parquet"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            merged_df.to_parquet(output_path, index=False)

            # Also save CSV version
            csv_path = settings.merged_dir / "csv" / f"merged_{self.strategy_name}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            merged_df.to_csv(csv_path, index=False)

            return MergeResult(
                output_name=self.config.output_name,
                output_path=output_path,
                success=True,
                input_sources=self.config.sources,
                rows_before={source: len(df) for source, df in data_sources.items()},
                rows_after=len(merged_df),
                merge_statistics=self.get_merge_statistics(data_sources, merged_df),
            )

        except Exception as e:
            logger.error(f"Gaia enriched merge failed: {e}")
            return MergeResult(
                output_name=self.config.output_name,
                output_path=Path("/tmp/failed"),  # Provide a valid path for Pydantic
                success=False,
                error_message=str(e),
                input_sources=self.config.sources,
                rows_before={},
            )

    def _gaia_enriched_merge(self) -> pd.DataFrame:
        """
        Execute the gaia_enriched merge logic.
        """
        # Start with baseline
        base = self._baseline_merge()
        base["_gaia_src"] = base["gaia_id"].apply(gaia_int)

        # Handle Gaia data
        ds = get_data_source("gaia_dr3_astrophysical_parameters")
        interim_path = settings.raw_dir / f"{ds.id}.parquet"

        gaia_ids = base["_gaia_src"].dropna().astype(int).unique().tolist()
        if gaia_ids:
            raw = fetch_adql(ds, gaia_ids)
            params = pd.read_csv(io.BytesIO(raw))[ds.columns_to_keep]
            params.to_parquet(interim_path, index=False)
        else:
            pd.DataFrame(columns=ds.columns_to_keep).to_parquet(
                interim_path, index=False
            )

        params = pd.read_parquet(interim_path)
        if "source_id" in params.columns:
            params = params.rename(columns={"source_id": "_gaia_src"})
        base = base.merge(params, on="_gaia_src", how="left")

        # Try to add SWEET-Cat data
        try:
            sweet_path = settings.raw_dir / "sweet_cat.parquet"
            if sweet_path.exists():
                sweet = pd.read_parquet(sweet_path)
                base = base.merge(
                    sweet,
                    left_on="hostname",
                    right_on="Name",
                    how="left",
                    suffixes=("", "_sweet"),
                )
        except Exception as e:
            logger.info("SWEET-Cat data not found or failed to load; skipping: %s", e)

        return base

    def _baseline_merge(self) -> pd.DataFrame:
        """
        Execute the baseline merge as a helper.
        """
        # Use raw_dir instead of interim for new system compatibility
        nasa_path = settings.raw_dir / "nasa_exoplanet_archive_pscomppars.parquet"
        phl_path = settings.raw_dir / "phl_exoplanet_catalog.parquet"

        if not nasa_path.exists() or not phl_path.exists():
            raise FileNotFoundError(
                f"Required data files not found."
                f"NASA: {nasa_path.exists()}, PHL: {phl_path.exists()}"
            )

        nasa = pd.read_parquet(nasa_path)
        phl = pd.read_parquet(phl_path)

        nasa["_key"] = norm_name(nasa["pl_name"])
        phl["_key"] = norm_name(phl["P_NAME"])
        df = nasa.merge(phl, on="_key", how="left").drop(
            columns=["_key", "P_NAME"], errors="ignore"
        )
        return df


class MergeManager:
    """
    Manager that provides the merge_data functionality
    integrating with the architecture.
    """

    def __init__(self):
        self.strategies = {
            "baseline": BaselineMerger,
            "gaia_enriched": GaiaEnrichedMerger,
        }
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    def list_mergers(self) -> List[str]:
        """
        List available merge strategies.
        """
        return list(self.strategies.keys())

    def merge_data(self, method: str, overwrite: bool = True) -> pd.DataFrame:
        """
        Execute merge using the proven logic.
        """
        if method not in self.strategies:
            raise KeyError(f"Unknown merge method: {method}")

        output_path = settings.merged_dir / "parquet" / f"merged_{method}.parquet"

        if not output_path.exists() or overwrite:
            self.logger.info("Merging data using method: %s", method)

            # Create the strategy
            config = MergeConfig(
                strategy=method,
                output_name=f"merged_{method}",
                sources=[],  # Not used in legacy logic
            )

            strategy_class = self.strategies[method]
            strategy = strategy_class(config)

            # Execute merge (data_sources parameter not used in this implementation)
            result = strategy.merge({})

            if not result.success:
                raise RuntimeError(f"Merge failed: {result.error_message}")

            self.logger.info("Wrote merged data for %s", method)
            return pd.read_parquet(result.output_path)
        else:
            return pd.read_parquet(output_path)


# Global instance for easy access
merge_manager = MergeManager()


def merge_data(method: str, overwrite: bool = True) -> pd.DataFrame:
    """
    Convenience function that wraps the merge manager.
    """
    return merge_manager.merge_data(method, overwrite)


def list_mergers() -> List[str]:
    """
    Convenience function to list available merge strategies.
    """
    return merge_manager.list_mergers()
