"""
CLI command: merge

Merges interim datasets into a processed output using registered merge strategies.
"""

import logging

import click

from exolife.data.merge import get_available_strategies, get_merger_info, merge_data

# Configure module-level logger
logger = logging.getLogger("exolife.cli.merge")


@click.command("merge")
@click.argument("method", type=click.STRING, required=False)
@click.option("--force", is_flag=True, help="Force refresh of merged data")
@click.option(
    "--list-strategies", is_flag=True, help="List all available merge strategies"
)
@click.option("--info", is_flag=True, help="Show detailed information about mergers")
def cli(method: str, force: bool, list_strategies: bool, info: bool) -> None:
    """
    Execute data merging using the modular merge system.

    Args:
        method: Merge strategy to use (e.g., 'exolife_merge_v1', 'gaia_source_id')
        Run without method to see available options.

    The ExoLife merge system provides multiple strategies:
    - Individual mergers: gaia_source_id, exact_name, coordinate_match
    - Catalog-specific: exoplanet_catalog, candidate_validation
    - Pipeline mergers: exolife_merge_v1 (full pipeline)

    For complex workflows, use the DAG system:
        exolife dag run config/dags/dagspec.yaml
    """

    # Handle info display options
    if list_strategies:
        available = get_available_strategies()
        click.echo("Available merge strategies:")
        click.echo("=" * 40)
        for strategy in sorted(available):
            click.echo(f"  • {strategy}")
        return

    if info:
        merger_info = get_merger_info()
        click.echo("ExoLife Merge System Information")
        click.echo("=" * 40)
        for merger_name, details in merger_info.items():
            click.echo(f"\n{details['name']} ({merger_name}):")
            click.echo(f"  Strategies: {', '.join(details['strategies'])}")
            click.echo(f"  Description: {details['description'][:100]}...")
        return

    # Get available strategies dynamically
    available = get_available_strategies()

    # If no method specified, show available options
    if not method:
        click.echo("ExoLife Merge System")
        click.echo("=" * 30)
        click.echo()
        click.echo("Available strategies:")

        # Group strategies by merger type
        merger_info = get_merger_info()
        for merger_name, details in merger_info.items():
            click.echo(f"\n{details['name']}:")
            for strategy in details["strategies"]:
                if strategy == "exolife_merge_v1":
                    click.echo(f"  ✓ {strategy} (recommended full pipeline)")
                else:
                    click.echo(f"  • {strategy}")

        click.echo()
        click.echo("Example usage:")
        click.echo("  exolife merge exolife_merge_v1        # Full pipeline")
        click.echo("  exolife merge gaia_source_id          # Gaia ID matching only")
        click.echo("  exolife merge --list-strategies       # List all strategies")
        click.echo("  exolife merge --info                  # Detailed merger info")
        return

    # Validate method
    if method not in available:
        logger.error(
            "Unknown merge strategy '%s'; available options: %s",
            method,
            ", ".join(sorted(available)),
        )
        click.echo(
            f"Error: Unknown strategy '{method}'.\n"
            f"Use --list-strategies to see available options."
        )
        raise click.Abort()

    # Perform merge operation
    try:
        logger.info("Running merge with strategy '%s'...", method)
        click.echo(f"Running merge with strategy: {method}")

        # Use the new merge system
        merger = merge_data(
            strategy=method,
            output_name="exolife_catalog",
            sources=[],  # Will be loaded from config for pipeline strategies
            drop_duplicates=True,
        )

        # Get the result
        result = merger.merge()

        if result.success:
            click.echo(f"✓ Merge completed: {result.rows_processed} records processed")
            click.echo(f"  Output: {result.output_path}")

            if result.execution_time_seconds:
                click.echo(f"  Time: {result.execution_time_seconds:.2f}s")

            if result.statistics:
                stats = result.statistics
                if "merge_efficiency" in stats:
                    click.echo(f"  Efficiency: {stats['merge_efficiency']:.2%}")

                if "input_sources" in stats:
                    sources_count = len(stats["input_sources"])
                    click.echo(f"  Sources merged: {sources_count}")
        else:
            click.echo(f"✗ Merge failed: {result.error_message}")
            raise click.Abort()

        logger.info("Merge completed successfully.")

    except Exception as exc:
        logger.exception("Merge operation failed: %s", exc)
        click.echo(f"✗ Merge failed: {exc}")
        raise click.Abort()
