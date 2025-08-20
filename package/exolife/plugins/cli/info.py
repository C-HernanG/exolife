"""
CLI command: info

Displays ExoLife package version, available data sources, merge methods,
and data pipelines.
"""

import logging
from importlib.metadata import PackageNotFoundError, version

import click

from exolife.data.fetch import get_fetcher_info
from exolife.data.merge import get_merger_info
from exolife.data.preprocess import get_preprocessor_info

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

    # Show available data fetchers
    click.echo("\nAvailable data fetchers:")
    try:
        fetcher_info = get_fetcher_info()
        for fetcher_type, description in fetcher_info.items():
            click.echo(f"  - {fetcher_type}: {description}")
    except Exception as e:
        click.echo(f"  Error loading fetcher information: {e}")

    # List available merge strategies
    click.echo("\nAvailable merge strategies:")
    try:
        merger_info = get_merger_info()
        for merger_name, details in merger_info.items():
            click.echo(f"  {details['name']}:")
            for strategy in details["strategies"]:
                click.echo(f"    - {strategy}")
    except Exception as e:
        click.echo(f"  Error loading merge strategies: {e}")

    # Show available preprocessors
    click.echo("\nAvailable preprocessors:")
    try:
        preprocessor_info = get_preprocessor_info()
        for preprocessor_name, description in preprocessor_info.items():
            click.echo(f"  - {preprocessor_name}: {description}")
    except Exception as e:
        click.echo(f"  Error loading preprocessor information: {e}")

    # Show current configuration
    click.echo("\nCurrent configuration:")
    from exolife.settings import Settings

    settings = Settings()
    click.echo(f"  Data directory: {settings.data_dir}")
    click.echo(f"  Log level: {settings.log_level}")
    click.echo(f"  Max workers: {settings.max_workers}")
