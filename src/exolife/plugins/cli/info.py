"""
CLI command: info

Displays ExoLife package version, available data sources, merge methods,
and data pipelines.
"""

import logging
from importlib.metadata import PackageNotFoundError, version

import click

from exolife.data import available_data_pipelines, list_data_sources, list_mergers

# Configure module-level logger
logger = logging.getLogger("exolife.cli.info")


@click.command("info")
def cli() -> None:
    """
    Show package metadata and registered pipeline components.
    """
    # Retrieve package version, fallback if not installed
    try:
        pkg_version = version("exolife")
        logger.debug("Retrieved package version: %s", pkg_version)
    except PackageNotFoundError:
        pkg_version = "0.0.0-dev"
        logger.warning(
            "Package 'exolife' not found; using development version placeholder."
        )

    click.echo(f"ExoLife version: {pkg_version}")

    # List data sources
    click.echo("\nAvailable data sources:")
    for src in list_data_sources():
        click.echo(f"  - {src}")

    # List merge methods
    click.echo("\nAvailable merge methods:")
    for method in list_mergers():
        click.echo(f"  - {method}")

    # List data pipelines
    click.echo("\nAvailable data pipelines:")
    for pipeline in available_data_pipelines():
        click.echo(f"  - {pipeline}")
