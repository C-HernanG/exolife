"""
Concrete implementations of data preprocessors using the new architecture.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import pandas as pd

from .base_preprocessor import BasePreprocessor, PreprocessorConfig


class HZEdgesPreprocessor(BasePreprocessor):
    """
    Append conservative (Runaway ⇄ Maximum-Greenhouse) HZ edges.
    """

    def __init__(
        self,
        config: PreprocessorConfig = None,
        teff_col: str = "st_teff",
        lum_col: str = "st_lum",
        prefix: str = "hz_",
    ) -> None:
        super().__init__(config or PreprocessorConfig(name="HZEdgesPreprocessor"))
        self.teff_col = teff_col
        self.lum_col = lum_col
        self.prefix = prefix

    @property
    def preprocessor_name(self) -> str:
        return "hz_edges"

    def get_required_columns(self) -> List[str]:
        return [self.teff_col, self.lum_col]

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.check_requirements(df):
            self.logger.warning("Requirements not met, skipping HZ edges calculation")
            return df

        from ..utils import add_hz_edges_to_df

        return add_hz_edges_to_df(
            df, self.teff_col, self.lum_col, optimistic=False, prefix=self.prefix
        )


class HZEdgesOptimisticPreprocessor(BasePreprocessor):
    """
    Append optimistic (Recent-Venus ⇄ Early-Mars) HZ edges.
    """

    def __init__(
        self,
        config: PreprocessorConfig = None,
        teff_col: str = "st_teff",
        lum_col: str = "st_lum",
        prefix: str = "hz_opt_",
    ) -> None:
        super().__init__(
            config or PreprocessorConfig(name="HZEdgesOptimisticPreprocessor")
        )
        self.teff_col = teff_col
        self.lum_col = lum_col
        self.prefix = prefix

    @property
    def preprocessor_name(self) -> str:
        return "hz_edges_optimistic"

    def get_required_columns(self) -> List[str]:
        return [self.teff_col, self.lum_col]

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.check_requirements(df):
            self.logger.warning(
                "Requirements not met, skipping optimistic HZ edges calculation"
            )
            return df

        from ..utils import add_hz_edges_to_df

        return add_hz_edges_to_df(
            df, self.teff_col, self.lum_col, optimistic=True, prefix=self.prefix
        )


class ColumnsPruner(BasePreprocessor):
    """
    Drop superfluous columns after the merge stage.
    """

    def __init__(
        self,
        config: PreprocessorConfig = None,
        columns: Iterable[str] = None,
    ) -> None:
        super().__init__(config or PreprocessorConfig(name="ColumnsPruner"))

        if columns is not None:
            self.columns: Sequence[str] = list(columns)
        else:
            from ..utils import load_default_drop_columns

            self.columns = load_default_drop_columns()

    @property
    def preprocessor_name(self) -> str:
        return "columns_pruner"

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.columns:
            self.logger.info("No columns to drop")
            return df

        existing = [c for c in self.columns if c in df.columns]
        if existing:
            df = df.drop(columns=existing, errors="ignore")
            self.logger.info(
                f"Dropped {len(existing)} columns: "
                f"{existing[:5]}{'...' if len(existing) > 5 else ''}"
            )
        else:
            self.logger.info("No specified columns found to drop")

        return df


class DataTypeOptimizer(BasePreprocessor):
    """
    Optimize DataFrame data types to reduce memory usage.
    """

    def __init__(self, config: PreprocessorConfig = None):
        super().__init__(config or PreprocessorConfig(name="DataTypeOptimizer"))

    @property
    def preprocessor_name(self) -> str:
        return "data_type_optimizer"

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_memory = df.memory_usage(deep=True).sum()

        # Optimize numeric columns
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")

        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")

        # Optimize object columns to category if beneficial
        for col in df.select_dtypes(include=["object"]).columns:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if (
                num_unique_values / num_total_values < 0.5
            ):  # Less than 50% unique values
                df[col] = df[col].astype("category")

        final_memory = df.memory_usage(deep=True).sum()
        memory_reduction = (initial_memory - final_memory) / initial_memory * 100

        self.logger.info(
            f"Memory usage reduced by {memory_reduction:.1f}% "
            f"({initial_memory / 1024**2:.1f}MB → {final_memory / 1024**2:.1f}MB)"
        )

        return df


class MissingValueHandler(BasePreprocessor):
    """
    Handle missing values using various strategies.
    """

    def __init__(
        self,
        config: PreprocessorConfig = None,
        strategy: str = "drop",
        threshold: float = 0.5,
    ):
        super().__init__(config or PreprocessorConfig(name="MissingValueHandler"))
        self.strategy = strategy  # 'drop', 'fill_mean', 'fill_median', 'fill_mode'
        self.threshold = threshold  # For 'drop' strategy

    @property
    def preprocessor_name(self) -> str:
        return "missing_value_handler"

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_rows = len(df)

        if self.strategy == "drop":
            # Drop columns with more than threshold missing values
            missing_ratio = df.isnull().sum() / len(df)
            cols_to_drop = missing_ratio[missing_ratio > self.threshold].index.tolist()

            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                self.logger.info(
                    f"Dropped {len(cols_to_drop)} "
                    f"columns with >{self.threshold*100}% missing values"
                )

            # Drop rows with any missing values in remaining columns
            df = df.dropna()

        elif self.strategy == "fill_mean":
            numeric_cols = df.select_dtypes(include=["number"]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        elif self.strategy == "fill_median":
            numeric_cols = df.select_dtypes(include=["number"]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        elif self.strategy == "fill_mode":
            for col in df.columns:
                if df[col].isnull().any():
                    mode_value = (
                        df[col].mode().iloc[0] if not df[col].mode().empty else 0
                    )
                    df[col] = df[col].fillna(mode_value)

        final_rows = len(df)
        if initial_rows != final_rows:
            self.logger.info(
                f"Rows: {initial_rows} → {final_rows} "
                f"({initial_rows - final_rows} removed)"
            )

        return df


__all__ = [
    "HZEdgesPreprocessor",
    "HZEdgesOptimisticPreprocessor",
    "ColumnsPruner",
    "DataTypeOptimizer",
    "MissingValueHandler",
]
