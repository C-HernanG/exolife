#!/usr/bin/env python3
"""
parquet2csv.py

A simple CLI tool to convert a Parquet file into a CSV file.

Usage:
    python parquet2csv.py input.parquet output.csv [--compression gzip]
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


def configure_logger() -> logging.Logger:
    """
    Configure and return a logger with INFO level and a standardized format.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = configure_logger()


def convert_parquet_to_csv(
    parquet_path: Path, csv_path: Path, compression: str | None = None
) -> None:
    """
    Read a Parquet file and write its contents to a CSV file.

    Args:
        parquet_path (Path): Path to the input Parquet file.
        csv_path (Path): Path where the output CSV will be saved.
        compression (str | None): Optional compression format for the CSV.
            Supported values: 'gzip', 'bz2', 'zip', 'xz'.

    Raises:
        FileNotFoundError: If the input Parquet file does not exist.
        pd.errors.EmptyDataError: If the Parquet file is empty.
        Exception: For any other read/write errors.
    """
    if not parquet_path.is_file():
        logger.error("Input file not found: %s", parquet_path)
        raise FileNotFoundError(f"Input file not found: {parquet_path}")

    try:
        df = pd.read_parquet(parquet_path)
        logger.info(
            "Loaded Parquet file '%s' with %d rows and %d columns",
            parquet_path,
            df.shape[0],
            df.shape[1],
        )
    except Exception as e:
        logger.exception("Failed to read Parquet file: %s", e)
        raise

    try:
        df.to_csv(csv_path, index=False, compression=compression)
        logger.info(
            "Saved CSV file '%s' with %d rows and %d columns",
            csv_path,
            df.shape[0],
            df.shape[1],
        )
    except Exception as e:
        logger.exception("Failed to write CSV file: %s", e)
        raise


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """
    Parse and return command-line arguments.

    Args:
        args (list[str] | None): List of arguments to parse (defaults to sys.argv[1:]).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Convert a Parquet file to CSV.")
    parser.add_argument("input", type=Path, help="Path to the input Parquet file.")
    parser.add_argument("output", type=Path, help="Path for the output CSV file.")
    parser.add_argument(
        "--compression",
        choices=["gzip", "bz2", "zip", "xz"],
        help="Compress the output CSV with the given format.",
    )
    return parser.parse_args(args)


def main() -> None:
    """
    Main entry point for the script.

    Parses arguments and executes the conversion, handling errors gracefully.
    """
    args = parse_args()

    try:
        convert_parquet_to_csv(
            parquet_path=args.input, csv_path=args.output, compression=args.compression
        )
    except FileNotFoundError:
        sys.exit(1)
    except Exception:
        logger.error("Conversion failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
