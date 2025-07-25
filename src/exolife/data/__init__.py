from __future__ import annotations

import concurrent.futures as _futures
import io
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .hz_utils import add_hz_edges_to_df

# Configure module-level logger
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR = Path(os.environ.get("EXOLIFE_DATA_DIR", DEFAULT_DATA_DIR)).resolve()
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
MERGED_DIR = DATA_DIR / "merged"
PARQUET_DIR = MERGED_DIR / "parquet"
CSV_DIR = MERGED_DIR / "csv"
CONFIG_PATH = Path(
    os.environ.get(
        "EXOLIFE_CONFIG", PROJECT_ROOT / "data" / "external" / "data_sources.json"
    )
)

for d in (RAW_DIR, INTERIM_DIR, PARQUET_DIR, CSV_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# DataSource definition
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DataSource:
    id: str
    name: str
    description: str
    download_url: str | None = None
    adql: str | None = None
    columns_to_keep: list[str] = field(default_factory=list)
    primary_keys: list[str] = field(default_factory=list)
    join_keys: dict[str, list[str]] = field(default_factory=dict)
    refresh: str = "static"
    format: str | None = None


# ---------------------------------------------------------------------------
# Load and parse data_sources config
# ---------------------------------------------------------------------------


def _load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_sources(cfg: dict) -> dict[str, DataSource]:
    valid = set(DataSource.__dataclass_fields__)
    sources: dict[str, DataSource] = {}
    for entry in cfg.get("data_sources", []):
        data = {k: v for k, v in entry.items() if k in valid}
        sources[entry["id"]] = DataSource(**data)
    return sources


_SOURCES = _parse_sources(_load_config())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _stream_download(url: str) -> bytes:
    import requests

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        return r.content


def _write_csv_trimmed(raw: bytes, keep: list[str], out: Path) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), usecols=lambda c: c in keep if keep else True)
    df.to_parquet(out, index=False)
    return df


def _write_generic(raw: bytes, url: str, keep: list[str], out: Path) -> pd.DataFrame:
    if url.endswith(".csv") or ",format=csv" in url:
        return _write_csv_trimmed(raw, keep, out)
    if url.endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(raw), columns=keep or None)
        df.to_parquet(out, index=False)
        return df
    return _write_csv_trimmed(raw, keep, out)


def _gaia_int(x):
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return int(x)
    m = re.search(r"\d{10,}", str(x))
    return int(m.group(0)) if m else None


# ---------------------------------------------------------------------------
# Fetch API
# ---------------------------------------------------------------------------


def list_data_sources() -> list[str]:
    return sorted(_SOURCES)


def _fetch_adql(ds: DataSource, gaia_ids: list[int] | None = None) -> bytes:
    query = ds.adql or ""
    if "<GAIA_ID_LIST>" in query:
        if not gaia_ids:
            raise ValueError("gaia_ids list must be provided to fill <GAIA_ID_LIST>")
        id_list = ",".join(str(i) for i in gaia_ids)
        query = query.replace("<GAIA_ID_LIST>", id_list)
    import requests

    logger.info("Running ADQL query for %s", ds.id)
    r = requests.post(
        ds.download_url,
        data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query},
        timeout=300,
    )
    r.raise_for_status()
    if "xml" in r.headers.get("Content-Type", "").lower():
        raise RuntimeError("ADQL query returned XML instead of CSV")
    return r.content


def fetch_source(src_id: str, force: bool = False) -> Path:
    if src_id not in _SOURCES:
        raise KeyError(src_id)
    ds = _SOURCES[src_id]
    interim = INTERIM_DIR / f"{ds.id}.parquet"
    # Skip ADQL on-demand unless forced
    if ds.adql and ds.refresh == "on_demand" and not force:
        if interim.exists():
            logger.info("Using cached on_demand source %s", ds.id)
            return interim
        # create empty stub
        pd.DataFrame(columns=ds.columns_to_keep).to_parquet(interim, index=False)
        return interim

    if interim.exists() and not force:
        logger.info("Using cached source %s", ds.id)
        return interim

    # Download and trim
    raw = _fetch_adql(ds) if ds.adql else _stream_download(ds.download_url)
    raw_path = RAW_DIR / ds.id / f"{ds.id}_{_timestamp()}.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw)

    _write_generic(raw, ds.download_url or "", ds.columns_to_keep, interim)
    logger.info("Fetched and stored interim data for %s", ds.id)
    return interim


def fetch_all_sources(
    parallel: bool = True, max_workers: int | None = None, force: bool = False
) -> dict[str, Path]:
    ids = list_data_sources()
    results: dict[str, Path] = {}
    logger.info("Fetching all sources (parallel=%s)...", parallel)
    if parallel:
        with _futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(fetch_source, i, force): i for i in ids}
            for f in _futures.as_completed(futures):
                src = futures[f]
                results[src] = f.result()
    else:
        for i in ids:
            results[i] = fetch_source(i, force)
    logger.info("Completed fetching %d sources", len(ids))
    return results


# ---------------------------------------------------------------------------
# Merge registry
# ---------------------------------------------------------------------------
MergerFunc = Callable[[], pd.DataFrame]
_MERGERS: dict[str, MergerFunc] = {}


def merger(name: str):
    def wrap(fn: MergerFunc):
        _MERGERS[name] = fn
        return fn

    return wrap


def list_mergers() -> list[str]:
    return sorted(_MERGERS)


def merge_data(method: str, overwrite: bool = True) -> Path:
    if method not in _MERGERS:
        raise KeyError(method)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    out_parquet = PARQUET_DIR / f"merged_{method}.parquet"
    out_csv = CSV_DIR / f"merged_{method}.csv"
    if not out_parquet.exists() or overwrite:
        logger.info("Merging data using method: %s", method)
        df = _MERGERS[method]()
        df.to_parquet(out_parquet, index=False)
        df.to_csv(out_csv, index=False)
        logger.info(
            "Wrote merged data: %s (parquet) and %s (csv)", out_parquet, out_csv
        )
    return out_parquet


# ---------------------------------------------------------------------------
# Internal readers
# ---------------------------------------------------------------------------


def _read_interim(src: str, cols: Iterable[str] | None = None) -> pd.DataFrame:
    path = INTERIM_DIR / f"{src}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path, columns=list(cols) if cols else None)


def _norm_name(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def _find_source_id(prefix: str) -> str | None:
    return next((k for k in _SOURCES if k.startswith(prefix)), None)


# ---------------------------------------------------------------------------
# Mergers
# ---------------------------------------------------------------------------


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

    # Fetch Gaia astrophysical parameters
    ds = _SOURCES["gaia_dr3_astrophysical_parameters"]
    interim_path = INTERIM_DIR / f"{ds.id}.parquet"
    gaia_ids = base["_gaia_src"].dropna().astype(int).unique().tolist()
    if gaia_ids:
        raw = _fetch_adql(ds, gaia_ids)
        df_params = pd.read_csv(io.BytesIO(raw))[ds.columns_to_keep]
        df_params.to_parquet(interim_path, index=False)
    else:
        pd.DataFrame(columns=ds.columns_to_keep).to_parquet(interim_path, index=False)

    # Merge Gaia and SWEET-Cat
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

    # Add habitable-zone edges
    if {"st_teff", "st_lum"}.issubset(base.columns):
        base = add_hz_edges_to_df(
            base, "st_teff", "st_lum", optimistic=False, prefix="hz_"
        )
        base = add_hz_edges_to_df(
            base, "st_teff", "st_lum", optimistic=True, prefix="hz_opt_"
        )

    return base


__all__ = [
    "list_data_sources",
    "fetch_source",
    "fetch_all_sources",
    "list_mergers",
    "merge_data",
    "RAW_DIR",
    "INTERIM_DIR",
    "PARQUET_DIR",
    "CSV_DIR",
]
