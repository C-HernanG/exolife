"""
CLI command: merge

Merges interim datasets into a processed output using registered merge strategies.
"""

import logging

import click

from exolife.data.merge.merge_factory import merge_factory
from exolife.data.merge.mergers import merge_manager

# Configure module-level logger
logger = logging.getLogger("exolife.cli.merge")


@click.command("merge")
@click.argument("method", type=click.STRING, required=False)
@click.option("--force", is_flag=True, help="Force refresh of merged data")
def cli(method: str, force: bool) -> None:
    """
    Merge data using the specified METHOD.

    Args:
        method: Name of the registered merge strategy.
        Run without method to see available options.

    Note: For complex workflows, consider using the DAG system:
        exolife dag run src/exolife/pipeline/dagspec.yaml
    """
    # Get available strategies dynamically from merge factory
    available = merge_factory.get_available_strategies()

    # If no method specified, show available options
    if not method:
        click.echo("Available merge methods:")
        for strategy in sorted(available):
            click.echo(f"  - {strategy}")
        click.echo()
        click.echo(
            "💡 Tip: For complex workflows with dependencies, use the DAG system:"
        )
        click.echo("   exolife dag run src/exolife/pipeline/dagspec.yaml")
        return

    # Validate method
    if method not in available:
        logger.error(
            "Unknown merge method '%s'; available options: %s",
            method,
            ", ".join(sorted(available)),
        )
        click.echo(
            f"Error: Unknown merge method '{method}'.\n"
            f"Available: {', '.join(sorted(available))}"
        )
        raise click.Abort()

    # Perform merge operation
    try:
        logger.info("Merging data using method '%s'...", method)
        click.echo(f"Merging data using '{method}' strategy...")

        # Use merge manager which handles the proven strategies
        result = merge_manager.merge_data(method, overwrite=force)

        if hasattr(result, "__len__"):
            click.echo(f"✓ Merged {len(result)} rows from multiple sources")
        else:
            click.echo("✓ Merge completed successfully")

        logger.info("Merge completed successfully.")

    except Exception as exc:
        logger.exception("Merge operation failed: %s", exc)
        click.echo(f"✗ Merge failed: {exc}")
        raise click.Abort()
