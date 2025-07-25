"""Lightweight command‑line interface for common Exolife tasks.

$ exolife fetch               # download raw sources
$ exolife merge               # clean + merge into processed parquet
$ exolife info                # version & data‑path diagnostics
"""

import logging
from importlib.metadata import PackageNotFoundError, version

import click

from exolife.data import (
    fetch_all_sources,
    fetch_source,
    list_data_sources,
    list_mergers,
    merge_data,
)

# Configure logger for the CLI
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)


@click.group()
def main():
    """ExoLife CLI"""
    pass


@main.command()
@click.argument("source", required=True)
def fetch(source):
    """Fetch data for given DATA_SOURCE or 'all'."""
    available = list_data_sources()
    if source != "all" and source not in available:
        logger.error(
            f"Unknown data source '{source}'. Available: {', '.join(available)}"
        )
        return
    if source == "all":
        fetch_all_sources()
    else:
        fetch_source(source)


@main.command()
@click.argument("method", required=True)
def merge(method):
    """Merge data using METHOD."""
    available = list_mergers()
    if method not in available:
        logger.error(
            f"Unknown merge method '{method}'. Available: {', '.join(available)}"
        )
        return
    merge_data(method)


@main.command()
def info():
    """Display ExoLife package information and available commands."""
    # Package version
    try:
        pkg_version = version(__name__.split(".")[0])
    except PackageNotFoundError:
        pkg_version = "0.0.0-dev"
    click.echo(f"ExoLife version: {pkg_version}")

    # Data sources
    sources = list_data_sources()
    click.echo("\nAvailable data sources:")
    for src in sources:
        click.echo(f"  - {src}")

    # Merge methods
    methods = list_mergers()
    click.echo("\nAvailable merge methods:")
    for m in methods:
        click.echo(f"  - {m}")


if __name__ == "__main__":
    main()
