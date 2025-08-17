"""
CLI command: data

Data management commands including status and data source information.
"""

import logging

import click
import pandas as pd
import yaml

from exolife.data.utils import list_data_sources, load_data_source_config
from exolife.settings import Settings

# Configure module-level logger
logger = logging.getLogger("exolife.cli.data")


@click.group("data")
def cli():
    """
    Data management commands.
    """
    pass


@cli.command("status")
def data_status():
    """
    Show status of data sources and processed datasets.
    """
    settings = Settings()

    click.echo("Data Status Report")
    click.echo("=" * 50)

    # Check raw data
    click.echo("\nRaw Data:")
    if settings.raw_dir.exists():
        for raw_file in settings.raw_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(raw_file)
                source_id = raw_file.stem
                click.echo(f"  ✓ {source_id}: {len(df)} rows")
            except Exception:
                click.echo(f"  ⚠ {raw_file.name}: file exists but corrupted")
    else:
        click.echo("  No raw data directory found")

    # Check merged data
    merged_dir = settings.merged_dir / "parquet"
    click.echo("\nMerged Data:")
    if merged_dir.exists():
        for merged_file in merged_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(merged_file)
                strategy_name = merged_file.stem
                click.echo(f"  ✓ {strategy_name}: {len(df)} rows")
            except Exception:
                click.echo(f"  ⚠ {merged_file.name}: corrupted")
    else:
        click.echo("  No merged data directory found")

    # Check processed data
    processed_dir = settings.processed_dir / "parquet"
    click.echo("\nProcessed Data:")
    if processed_dir.exists():
        for processed_file in processed_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(processed_file)
                pipeline_name = processed_file.stem
                click.echo(f"  ✓ {pipeline_name}: {len(df)} rows")
            except Exception:
                click.echo(f"  ⚠ {processed_file.name}: corrupted")
    else:
        click.echo("  No processed data directory found")


@cli.command("sources")
@click.option(
    "--detailed", "-d", is_flag=True, help="Show detailed information about each source"
)
def list_sources(detailed):
    """
    List available data sources from configuration.
    """
    available_sources = list_data_sources()

    if not detailed:
        click.echo("Available Data Sources:")
        click.echo("=" * 30)
        for source_id in available_sources:
            click.echo(f"  - {source_id}")
        click.echo("\nUse --detailed (-d) flag for more information.")
        return

    # For detailed info, read the configuration files
    click.echo("Available Data Sources (Detailed)")
    click.echo("=" * 50)

    try:
        for i, source_id in enumerate(available_sources, 1):
            source_config = load_data_source_config(source_id)

            name = source_config.get("name", "Unknown")
            description = source_config.get("description", "No description")

            click.echo(f"{i}. {source_id}")
            click.echo(f"   Name: {name}")
            click.echo(f"   Description: {description}")

            # Show URL or ADQL query
            if "download_url" in source_config:
                url = source_config["download_url"]
                if len(url) > 80:
                    url = url[:77] + "..."
                click.echo(f"   URL: {url}")
            elif "adql" in source_config:
                click.echo("   Query: ADQL query (Gaia TAP)")

            # Show primary keys
            if "primary_keys" in source_config:
                keys = ", ".join(source_config["primary_keys"])
                click.echo(f"   Primary Keys: {keys}")

            # Show column count
            if "columns_to_keep" in source_config:
                col_count = len(source_config["columns_to_keep"])
                click.echo(f"   Columns: {col_count} selected")

            # Show join relationships
            if "join_keys" in source_config and source_config["join_keys"]:
                click.echo(
                    f"   Joins with: {', '.join(source_config['join_keys'].keys())}"
                )

            click.echo()

    except Exception as e:
        click.echo(f"✗ Error reading data sources configuration: {e}")


@cli.command("config")
def show_data_config():
    """
    Show data configuration details.
    """
    settings = Settings()

    try:
        available_sources = list_data_sources()

        click.echo("Data Configuration")
        click.echo("=" * 50)

        click.echo(f"Configuration Location: {settings.root_dir / 'config'}")
        click.echo(f"Data Sources: {len(available_sources)}")

        for source_id in available_sources:
            try:
                click.echo(f"  - {source_id}")
            except Exception:
                click.echo(f"  - {source_id} (config error)")

        # Show constants configuration
        constants_dir = settings.root_dir / "config" / "constants"
        if constants_dir.exists():
            click.echo("\nConfiguration Files:")
            for config_file in constants_dir.glob("*.yml"):
                click.echo(f"  - {config_file.name}")

        # Show HZ coefficients if available
        hz_config_path = settings.root_dir / "config" / "constants" / "hz.yml"
        if hz_config_path.exists():
            try:
                with hz_config_path.open("r") as f:
                    hz_config = yaml.safe_load(f)
                click.echo(f"\nHabitable Zone Models: {len(hz_config)}")
                for model in hz_config.keys():
                    click.echo(f"  - {model}")
            except Exception:
                click.echo("\nHabitable Zone Models: (error loading)")

    except Exception as e:
        click.echo(f"✗ Error reading data configuration: {e}")


@cli.command("validate")
def validate_sources():
    """
    Validate data sources configuration and check data availability.
    """
    settings = Settings()

    # Use data layer to get available sources
    try:
        available_sources = list_data_sources()
    except Exception as e:
        click.echo(f"✗ Error loading data sources configuration: {e}")
        return

    click.echo("Validating Data Sources")
    click.echo("=" * 50)

    valid_count = 0
    total_count = len(available_sources)

    for source_id in available_sources:
        click.echo(f"\nValidating: {source_id}")

        raw_file = settings.raw_dir / f"{source_id}.parquet"
        if raw_file.exists():
            try:
                import pandas as pd

                df = pd.read_parquet(raw_file)
                click.echo(
                    f"  ✓ Raw data exists: {len(df)} rows, {len(df.columns)} columns"
                )
                valid_count += 1
            except Exception as e:
                click.echo(f"  ⚠ Raw data exists but corrupted: {e}")
                continue
        else:
            click.echo(f"  ⚠ Raw data not found: {raw_file}")

        click.echo("  ✓ Configuration valid")

    click.echo(
        f"\nValidation Summary: {valid_count}/{total_count} sources have valid data"
    )
