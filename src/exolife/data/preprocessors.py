from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BasePreprocessor(ABC):
    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame: ...


class HZEdgesPreprocessor(BasePreprocessor):
    """Append conservative habitable-zone edges to DataFrame."""

    def __init__(
        self, teff_col: str = "st_teff", lum_col: str = "st_lum", prefix: str = "hz_"
    ):
        self.teff_col = teff_col
        self.lum_col = lum_col
        self.prefix = prefix

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        from .utils.hz_utils import add_hz_edges_to_df

        return add_hz_edges_to_df(
            df, self.teff_col, self.lum_col, optimistic=False, prefix=self.prefix
        )


class HZEdgesOptimisticPreprocessor(BasePreprocessor):
    """Append optimistic habitable-zone edges to DataFrame."""

    def __init__(
        self,
        teff_col: str = "st_teff",
        lum_col: str = "st_lum",
        prefix: str = "hz_opt_",
    ):
        self.teff_col = teff_col
        self.lum_col = lum_col
        self.prefix = prefix

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        from .utils.hz_utils import add_hz_edges_to_df

        return add_hz_edges_to_df(
            df, self.teff_col, self.lum_col, optimistic=True, prefix=self.prefix
        )


__all__ = ["BasePreprocessor", "HZEdgesPreprocessor", "HZEdgesOptimisticPreprocessor"]
