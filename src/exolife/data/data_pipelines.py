from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from .fetchers import fetch_all_sources
from .mergers import list_mergers, merge_data
from .preprocessors import (
    BasePreprocessor,
    HZEdgesOptimisticPreprocessor,
    HZEdgesPreprocessor,
)

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Runs full data workflow: fetch → merge → preprocess
    """

    def __init__(
        self,
        merge_method: str,
        preprocessors: Optional[List[BasePreprocessor]] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        force: bool = False,
    ):
        if merge_method not in list_mergers():
            raise KeyError(f"Unknown merge method: {merge_method}")
        self.merge_method = merge_method
        self.preprocessors = preprocessors or []
        self.parallel = parallel
        self.max_workers = max_workers
        self.force = force

    def run(self) -> pd.DataFrame:
        # Fetch all sources
        logger.info("Starting fetch for all sources...")
        fetch_all_sources(
            parallel=self.parallel, max_workers=self.max_workers, force=self.force
        )

        # Merge
        logger.info("Merging using '%s'...", self.merge_method)
        df = merge_data(self.merge_method)

        # Preprocess
        for proc in self.preprocessors:
            logger.info("Applying preprocessor: %s", proc.__class__.__name__)
            df = proc.process(df)
        return df


def available_data_pipelines() -> List[str]:
    """List preset data pipeline names."""
    return ["baseline", "gaia_enriched"]


def get_data_pipeline(name: str, optimistic: bool = False) -> DataPipeline:
    """Return a configured DataPipeline by name."""
    if name == "baseline":
        preprocessors: List[BasePreprocessor] = []
    elif name == "gaia_enriched":
        preprocessors = [
            HZEdgesPreprocessor(),
            HZEdgesOptimisticPreprocessor(),
        ]
    else:
        raise KeyError(name)
    return DataPipeline(name, preprocessors)


__all__ = ["DataPipeline", "available_data_pipelines", "get_data_pipeline"]
