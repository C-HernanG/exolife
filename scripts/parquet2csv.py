#!/usr/bin/env python3
"""
Script: convert_parquet.py
Convert a Parquet file to CSV.

Usage:
    python convert_parquet.py input.parquet output.csv

    or

    ./parquet2csv.py input.parquet output.csv
"""
import logging
import os
import sys

import pandas as pd


def setup_logger():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    return logging.getLogger(__name__)


logger = setup_logger()


def convert_parquet_to_csv(parquet_path: str, csv_path: str):
    """Convert a Parquet file to CSV."""
    if not os.path.exists(parquet_path):
        logger.error(f"Input file not found: {parquet_path}")
        sys.exit(1)
    try:
        df = pd.read_parquet(parquet_path)
        logger.info(f"Read Parquet file: {parquet_path} (rows={len(df)})")
    except Exception as e:
        logger.error(f"Failed to read Parquet file: {e}")
        sys.exit(1)
    try:
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV file: {csv_path} (rows={len(df)})")
    except Exception as e:
        logger.error(f"Failed to write CSV file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_parquet.py <input.parquet> <output.csv>")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    convert_parquet_to_csv(input_path, output_path)
