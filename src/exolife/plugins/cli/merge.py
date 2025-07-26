"""
CLI command: merge
Merges interim datasets into a processed output using registered merge strategies.
"""

import logging

import click

from exolife.data import list_mergers, merge_data

# Configure module-level logger
logger = logging.getLogger("exolife.cli.merge")


@click.command("merge")
@click.argument("method", type=click.STRING, required=True)
def cli(method: str) -> None:
    """
    Merge data using the specified METHOD.

    Args:
        method: Name of the registered merge strategy (e.g., 'concat').
    """
    # List available merge methods
    available = list_mergers()
    if method not in available:
        logger.error(
            "Unknown merge method '%s'; available options: %s",
            method,
            ", ".join(available),
        )
        click.echo(
            f"Error: Unknown merge method '{method}'.\n"
            f"Available: {', '.join(available)}"
        )
        raise click.Abort()

    # Perform merge
    try:
        logger.info("Merging data using method '%s'...", method)
        merge_data(method)
        logger.info("Merge completed successfully.")
    except Exception as exc:
        logger.exception("Merge operation failed: %s", exc)
        click.echo("An error occurred during merge; see logs for details.")
        raise click.Abort()
