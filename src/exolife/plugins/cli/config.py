"""
CLI command: config

Configuration management commands.
"""

import logging
from pathlib import Path

import click

from exolife.settings import Settings

# Configure module-level logger
logger = logging.getLogger("exolife.cli.config")


@click.group("config")
def cli():
    """
    Configuration management commands.
    """
    pass


@cli.command("show")
def show_config():
    """
    Show current configuration.
    """
    settings = Settings()

    click.echo("ExoLife Configuration")
    click.echo("=" * 30)
    click.echo(f"Root Directory: {settings.root_dir}")
    click.echo(f"Data Directory: {settings.data_dir}")
    click.echo(f"Raw Directory: {settings.raw_dir}")
    click.echo(f"Interim Directory: {settings.interim_dir}")
    click.echo(f"Processed Directory: {settings.processed_dir}")
    click.echo(f"Merged Directory: {settings.merged_dir}")
    click.echo(f"Log Level: {settings.log_level}")
    click.echo(f"Max Workers: {settings.max_workers}")
    click.echo(f"Force Refresh: {settings.force_refresh}")


@cli.command("dags")
def list_dags():
    """
    List available DAG workflow specifications.
    """
    try:
        # Look for DAG YAML files in the pipeline directory
        pipeline_dir = Path("src/exolife/pipeline")
        dag_files = list(pipeline_dir.glob("*.yaml")) + list(pipeline_dir.glob("*.yml"))

        if dag_files:
            click.echo("Available DAG Workflows:")
            click.echo("=" * 30)
            for dag_file in dag_files:
                click.echo(f"  - {dag_file.name}")
            click.echo("\nUse 'exolife dag run <file>' to execute a workflow")
        else:
            click.echo("No DAG workflow files found in src/exolife/pipeline/")

    except Exception as e:
        click.echo(f"Error: Unable to list DAG files: {e}")
        logger.error("Failed to list available DAG workflows.")
