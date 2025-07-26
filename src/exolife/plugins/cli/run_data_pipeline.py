"""
CLI command: data-pipeline

Lists or executes full data pipelines by name, with optional
force and parallel flags.
"""

import logging

import click

from exolife.data import available_data_pipelines, fetch_all_sources, get_data_pipeline

# Configure module-level logger
logger = logging.getLogger("exolife.cli.data_pipeline")


@click.command("data-pipeline")
@click.argument("name", required=False, type=click.STRING)
@click.option(
    "--force", is_flag=True, help="Refetch all sources before running pipeline"
)
@click.option(
    "--parallel/--no-parallel",
    default=True,
    help="Enable or disable parallel fetching of sources",
)
def cli(name: str, force: bool, parallel: bool) -> None:
    """
    List or run a full data pipeline by NAME.

    Without NAME, lists all available pipelines.
    With NAME, executes the pipeline,
        optionally refetching data and toggling parallelism.
    """
    # Retrieve pipeline names
    names = available_data_pipelines()

    # If no pipeline specified, list and exit
    if not name:
        click.echo("Available data pipelines:")
        for p in names:
            click.echo(f"  - {p}")
        logger.info("Displayed %d available pipelines.", len(names))
        return

    # Validate pipeline name
    if name not in names:
        logger.error(
            "Unknown data pipeline '%s'; available: %s", name, ", ".join(names)
        )
        click.echo(
            f"Error: Unknown data pipeline '{name}'.\nAvailable: {', '.join(names)}"
        )
        raise click.Abort()

    # Optionally refetch raw data
    if force:
        logger.info("Forcing refetch of all data sources")
        try:
            fetch_all_sources()
        except Exception as exc:
            logger.exception("Failed to refetch sources: %s", exc)
            click.echo("Error during data refetch; see logs for details.")
            raise click.Abort()

    # Retrieve and configure pipeline
    pipeline = get_data_pipeline(name)
    pipeline.parallel = parallel  # assume pipeline object supports this flag
    logger.info("Running data pipeline '%s' (parallel=%s)", name, parallel)

    # Execute pipeline
    try:
        df = pipeline.run()
        row_count = len(df) if hasattr(df, "__len__") else "unknown"
        click.echo(f"'{name}' completed successfully: {row_count} rows processed.")
        logger.info("Data pipeline '%s' finished: %s rows.", name, row_count)
    except Exception as exc:
        logger.exception("Data pipeline '%s' execution failed: %s", name, exc)
        click.echo("Data pipeline execution error; check logs for details.")
        raise click.Abort()
