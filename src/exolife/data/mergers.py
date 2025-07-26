from __future__ import annotations

import io
import logging
from typing import Callable, List

import pandas as pd

from exolife.config import INTERIM_DIR, MERGED_DIR

from .fetchers import _SOURCES, _fetch_adql

logger = logging.getLogger(__name__)
MergerFunc = Callable[[], pd.DataFrame]
_MERGERS: dict[str, MergerFunc] = {}


def merger(name: str):
    def wrap(fn: MergerFunc):
        _MERGERS[name] = fn
        return fn

    return wrap


def list_mergers() -> List[str]:
    """List registered merging strategies."""
    return sorted(_MERGERS)


def merge_data(method: str, overwrite: bool = True) -> pd.DataFrame:
    """Run the named merger, cache to disk, and return the merged DataFrame."""
    PARQUET = MERGED_DIR / "parquet"
    CSV = MERGED_DIR / "csv"
    PARQUET.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)

    if method not in _MERGERS:
        raise KeyError(method)
    out_pq = PARQUET / f"merged_{method}.parquet"
    out_csv = CSV / f"merged_{method}.csv"

    if not out_pq.exists() or overwrite:
        logger.info("Merging data using method: %s", method)
        df = _MERGERS[method]()
        df.to_parquet(out_pq, index=False)
        df.to_csv(out_csv, index=False)
        logger.info("Wrote merged data for %s", method)
    return pd.read_parquet(out_pq)


# Internal readers & utils


def _read_interim(src: str, cols: List[str] | None = None) -> pd.DataFrame:
    path = INTERIM_DIR / f"{src}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path, columns=cols)


def _norm_name(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def _find_source_id(prefix: str) -> str | None:
    return next((k for k in _SOURCES if k.startswith(prefix)), None)


def _gaia_int(x):
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return int(x)
    import re

    m = re.search(r"\d{10,}", str(x))
    return int(m.group(0)) if m else None


@merger("baseline")
def _baseline() -> pd.DataFrame:
    nasa = _read_interim(_find_source_id("nasa_exoplanet_archive_pscomppars"))
    phl = _read_interim("phl_exoplanet_catalog")
    nasa["_key"] = _norm_name(nasa["pl_name"])
    phl["_key"] = _norm_name(phl["P_NAME"])
    df = nasa.merge(phl, on="_key", how="left").drop(
        columns=["_key", "P_NAME"], errors="ignore"
    )
    return df


@merger("gaia_enriched")
def _gaia_enriched() -> pd.DataFrame:
    base = _baseline()
    base["_gaia_src"] = base["gaia_id"].apply(_gaia_int)
    ds = _SOURCES["gaia_dr3_astrophysical_parameters"]
    interim_path = INTERIM_DIR / f"{ds.id}.parquet"
    gaia_ids = base["_gaia_src"].dropna().astype(int).unique().tolist()
    if gaia_ids:
        raw = _fetch_adql(ds, gaia_ids)
        params = pd.read_csv(io.BytesIO(raw))[ds.columns_to_keep]
        params.to_parquet(interim_path, index=False)
    else:
        pd.DataFrame(columns=ds.columns_to_keep).to_parquet(interim_path, index=False)

    params = pd.read_parquet(interim_path)
    if "source_id" in params.columns:
        params = params.rename(columns={"source_id": "_gaia_src"})
    base = base.merge(params, on="_gaia_src", how="left")
    try:
        sweet = _read_interim("sweet_cat")
        base = base.merge(
            sweet,
            left_on="hostname",
            right_on="Name",
            how="left",
            suffixes=("", "_sweet"),
        )
    except FileNotFoundError:
        logger.info("SWEET-Cat interim data not found; skipping")

    from .utils.hz_utils import add_hz_edges_to_df

    if {"st_teff", "st_lum"}.issubset(base.columns):
        add_hz_edges_to_df(base, "st_teff", "st_lum", optimistic=False, prefix="hz_")
        add_hz_edges_to_df(base, "st_teff", "st_lum", optimistic=True, prefix="hz_opt_")
    return base


__all__ = ["list_mergers", "merge_data"]
