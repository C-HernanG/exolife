"""
CLI command: fetch

Fetches and preprocesses data for one or all sources.
"""

import logging

import click

from exolife.data.fetch.fetchers import fetch_manager
from exolife.data.utils import list_data_sources

# Configure module-level logger
logger = logging.getLogger("exolife.cli.fetch")


@click.command("fetch")
@click.argument("source", type=click.STRING, required=False)
@click.option("--force", is_flag=True, help="Force refresh of cached data")
@click.option("--parallel/--sequential", default=True, help="Use parallel fetching")
def cli(source: str, force: bool, parallel: bool) -> None:
    """
    Fetch data for the specified DATA_SOURCE or for all sources.

    Args:
        source: Identifier of the data source, or 'all' to fetch every source.
                If not specified, lists available sources.

    Note: For complex workflows, consider using the DAG system:
        exolife dag run src/exolife/pipeline/dagspec.yaml
    """
    # Show deprecation notice for complex workflows
    if source == "all":
        click.echo(
            "💡 Tip: For complex workflows with dependencies, use the DAG system:"
        )
        click.echo("   exolife dag run src/exolife/pipeline/dagspec.yaml")
        click.echo()

    # Get available sources from data layer
    available_sources = list_data_sources()

    # If no source specified, list available sources
    if not source:
        click.echo("Available data sources:")
        for source_id in available_sources:
            click.echo(f"  - {source_id}")
        click.echo("\nUsage:")
        click.echo("  exolife fetch <source_id>    # Fetch specific source")
        click.echo("  exolife fetch all            # Fetch all sources")
        return

    # Validate input
    if source.lower() != "all" and source not in available_sources:
        available = available_sources + ["all"]
        logger.error(
            "Unknown data source '%s'; available options: %s",
            source,
            ", ".join(available),
        )
        click.echo(
            f"Error: Unknown data source '{source}'.\nAvailable: {', '.join(available)}"
        )
        raise click.Abort()

    # Execute fetch using data layer
    try:
        if source.lower() == "all":
            logger.info("Fetching all data sources...")
            click.echo(f"Fetching all {len(available_sources)} data sources...")

            results = fetch_manager.fetch_all_sources(parallel=parallel, force=force)

            successful = 0
            for source_id, result_path in results.items():
                if result_path and result_path.exists():
                    try:
                        import pandas as pd

                        df = pd.read_parquet(result_path)
                        click.echo(f"✓ {source_id}: {len(df)} rows")
                        successful += 1
                    except Exception:
                        click.echo(f"✗ {source_id}: file exists but corrupted")
                else:
                    click.echo(f"✗ {source_id}: failed to fetch")

            click.echo(
                f"\nCompleted: {successful}/{len(results)} sources fetched successfully"
            )

        else:
            logger.info("Fetching data source '%s'...", source)
            click.echo(f"Fetching data source: {source}")

            try:
                result_path = fetch_manager._fetch_single_source(source, force=force)

                if result_path and result_path.exists():
                    import pandas as pd

                    df = pd.read_parquet(result_path)
                    click.echo(f"✓ {source}: {len(df)} rows")
                else:
                    click.echo(f"✗ {source}: failed to fetch")
                    raise click.Abort()

            except Exception as e:
                click.echo(f"✗ {source}: {e}")
                raise click.Abort()

    except Exception as exc:
        logger.exception("Failed during fetch operation: %s", exc)
        click.echo("An error occurred during fetching; see logs for details.")
        raise click.Abort()
