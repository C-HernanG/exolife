"""
CLI command: fetch
Fetches and preprocesses data for one or all sources.
"""

import logging

import click

from exolife.data import fetch_all_sources, fetch_source, list_data_sources

# Configure module-level logger
logger = logging.getLogger("exolife.cli.fetch")


@click.command("fetch")
@click.argument("source", type=click.STRING, required=True)
def cli(source: str) -> None:
    """
    Fetch data for the specified DATA_SOURCE or for all sources.

    Args:
        source: Identifier of the data source, or 'all' to fetch every source.
    """
    # Retrieve available sources from configuration
    available = list_data_sources()

    # Validate input
    if source.lower() != "all" and source not in available:
        logger.error(
            "Unknown data source '%s'; available options: %s",
            source,
            ", ".join(available),
        )
        click.echo(
            f"Error: Unknown data source '{source}'.\nAvailable: {', '.join(available)}"
        )
        raise click.Abort()

    # Execute fetch
    try:
        if source.lower() == "all":
            logger.info("Fetching all data sources...")
            fetch_all_sources()
        else:
            logger.info("Fetching data source '%s'...", source)
            fetch_source(source)
    except Exception as exc:
        logger.exception("Failed during fetch operation: %s", exc)
        click.echo("An error occurred during fetching; see logs for details.")
        raise click.Abort()
