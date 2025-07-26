from __future__ import annotations

import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

from exolife.config import EXTERNAL_DIR, INTERIM_DIR, RAW_DIR

# Configure logger
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DataSource:
    id: str
    name: str
    description: str
    download_url: Optional[str] = None
    adql: Optional[str] = None
    columns_to_keep: List[str] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    join_keys: dict[str, List[str]] = field(default_factory=dict)
    refresh: str = "static"
    format: Optional[str] = None


def _load_config(path: Path = EXTERNAL_DIR / "data_sources.json") -> dict:
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


def list_data_sources() -> List[str]:
    """List all available data source IDs from the config."""
    return sorted(_SOURCES)


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _stream_download(url: str) -> bytes:
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        return r.content


def _write_csv_trimmed(raw: bytes, keep: List[str], out: Path) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), usecols=lambda c: c in keep if keep else True)
    df.to_parquet(out, index=False)
    return df


def _write_generic(raw: bytes, url: str, keep: List[str], out: Path) -> pd.DataFrame:
    if url.endswith(".csv") or ",format=csv" in url:
        return _write_csv_trimmed(raw, keep, out)
    if url.endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(raw), columns=keep or None)
        df.to_parquet(out, index=False)
        return df
    # fallback to CSV
    return _write_csv_trimmed(raw, keep, out)


def _fetch_adql(ds: DataSource, gaia_ids: List[int] | None = None) -> bytes:
    query = ds.adql or ""
    if "<GAIA_ID_LIST>" in query:
        if not gaia_ids:
            raise ValueError("gaia_ids list must be provided to fill <GAIA_ID_LIST>")
        id_list = ",".join(str(i) for i in gaia_ids)
        query = query.replace("<GAIA_ID_LIST>", id_list)
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
    """Fetch or load a single data source by its ID into INTERIM_DIR."""
    if src_id not in _SOURCES:
        raise KeyError(src_id)
    ds = _SOURCES[src_id]
    interim = INTERIM_DIR / f"{ds.id}.parquet"

    # handle on_demand ADQL
    if ds.adql and ds.refresh == "on_demand" and not force:
        if interim.exists():
            logger.info("Using cached on_demand source %s", ds.id)
            return interim
        pd.DataFrame(columns=ds.columns_to_keep).to_parquet(interim, index=False)
        return interim

    # cached
    if interim.exists() and not force:
        logger.info("Using cached source %s", ds.id)
        return interim

    # download
    raw = _fetch_adql(ds) if ds.adql else _stream_download(ds.download_url or "")
    raw_path = RAW_DIR / ds.id / f"{ds.id}_{_timestamp()}.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw)

    # write trimmed
    _write_generic(raw, ds.download_url or "", ds.columns_to_keep, interim)
    logger.info("Fetched and stored interim data for %s", ds.id)
    return interim


def fetch_all_sources(
    parallel: bool = True, max_workers: int | None = None, force: bool = False
) -> dict[str, Path]:
    """Fetch or load all configured sources, optionally in parallel."""
    ids = list_data_sources()
    results: dict[str, Path] = {}
    logger.info("Fetching all sources (parallel=%s)...", parallel)
    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(fetch_source, i, force): i for i in ids}
            for f in as_completed(futures):
                results[futures[f]] = f.result()
    else:
        for i in ids:
            results[i] = fetch_source(i, force)
    logger.info("Completed fetching %d sources", len(ids))
    return results


__all__ = [
    "DataSource",
    "list_data_sources",
    "fetch_source",
    "fetch_all_sources",
    "RAW_DIR",
    "INTERIM_DIR",
]
