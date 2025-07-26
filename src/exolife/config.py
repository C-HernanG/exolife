"""
Configuration module for ExoLife project paths and environment overrides.
This ensures a consistent, centralized definition of data directories and
auto-creates them on import.
"""

import os
from pathlib import Path

# Project root directory
ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MERGED_DIR = DATA_DIR / "merged"

# Overriding via environment variables
RAW_DIR = Path(os.getenv("EXOLIFE_RAW_DIR", RAW_DIR))
EXTERNAL_DIR = Path(os.getenv("EXOLIFE_EXTERNAL_DIR", EXTERNAL_DIR))
INTERIM_DIR = Path(os.getenv("EXOLIFE_INTERIM_DIR", INTERIM_DIR))
PROCESSED_DIR = Path(os.getenv("EXOLIFE_PROCESSED_DIR", PROCESSED_DIR))
MERGED_DIR = Path(os.getenv("EXOLIFE_MERGED_DIR", MERGED_DIR))

# Ensure every directory exists
for d in (EXTERNAL_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MERGED_DIR):
    d.mkdir(parents=True, exist_ok=True)
