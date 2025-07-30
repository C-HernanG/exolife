"""
CLI command: data

Data management commands including status and data source information.
"""

import json
import logging

import click
import pandas as pd

from exolife.data.utils import list_data_sources
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

    # For detailed info, read the configuration file
    settings = Settings()
    data_sources_file = settings.external_dir / "data_sources.json"
    if not data_sources_file.exists():
        click.echo(f"✗ Data sources configuration not found at: {data_sources_file}")
        return

    try:
        with open(data_sources_file, "r") as f:
            config = json.load(f)

        data_sources = config.get("data_sources", [])

        click.echo("Available Data Sources (Detailed)")
        click.echo("=" * 50)
        click.echo(f"Project: {config.get('project', 'Unknown')}")
        click.echo(f"Version: {config.get('version', 'Unknown')}")
        click.echo(f"Last Updated: {config.get('last_updated', 'Unknown')}")
        click.echo()

        for i, source in enumerate(data_sources, 1):
            source_id = source.get("id", "unknown")
            name = source.get("name", "Unknown")
            description = source.get("description", "No description")
            refresh = source.get("refresh", "unknown")

            click.echo(f"{i}. {source_id}")
            click.echo(f"   Name: {name}")
            click.echo(f"   Description: {description}")
            click.echo(f"   Refresh: {refresh}")

            # Show URL or ADQL query
            if "download_url" in source:
                url = source["download_url"]
                if len(url) > 80:
                    url = url[:77] + "..."
                click.echo(f"   URL: {url}")
            elif "adql" in source:
                click.echo("   Query: ADQL query (Gaia TAP)")

            # Show primary keys
            if "primary_keys" in source:
                keys = ", ".join(source["primary_keys"])
                click.echo(f"   Primary Keys: {keys}")

            # Show column count
            if "columns_to_keep" in source:
                col_count = len(source["columns_to_keep"])
                click.echo(f"   Columns: {col_count} selected")

            # Show join relationships
            if "join_keys" in source and source["join_keys"]:
                click.echo(f"   Joins with: {', '.join(source['join_keys'].keys())}")

            click.echo()

    except Exception as e:
        click.echo(f"✗ Error reading data sources configuration: {e}")


@cli.command("config")
def show_data_config():
    """
    Show data configuration details.
    """
    settings = Settings()
    data_sources_file = settings.external_dir / "data_sources.json"
    if not data_sources_file.exists():
        click.echo(f"✗ Data sources configuration not found at: {data_sources_file}")
        return

    try:
        with open(data_sources_file, "r") as f:
            config = json.load(f)

        click.echo("Data Configuration")
        click.echo("=" * 50)

        # Project info
        click.echo(f"Project: {config.get('project', 'Unknown')}")
        click.echo(f"Version: {config.get('version', 'Unknown')}")
        click.echo(f"Last Updated: {config.get('last_updated', 'Unknown')}")

        if "notes" in config:
            click.echo("\nNotes:")
            click.echo(f"  {config['notes']}")

        # Data sources summary
        data_sources = config.get("data_sources", [])
        click.echo(f"\nData Sources: {len(data_sources)}")
        for source in data_sources:
            refresh = source.get("refresh", "unknown")
            click.echo(f"  - {source.get('id', 'unknown')} (refresh: {refresh})")

        # Feature engineering
        if "feature_engineering" in config:
            fe = config["feature_engineering"]
            click.echo("\nFeature Engineering:")
            if "derived_planet_metrics" in fe:
                click.echo(f"  Planet metrics: {len(fe['derived_planet_metrics'])}")
            if "derived_stellar_metrics" in fe:
                click.echo(f"  Stellar metrics: {len(fe['derived_stellar_metrics'])}")

        # Quality filters
        if "quality_filters" in config:
            qf = config["quality_filters"]
            click.echo("\nQuality Filters:")
            if "min_required_columns" in qf:
                click.echo(f"  Required columns: {len(qf['min_required_columns'])}")
            if "max_fractional_uncertainty" in qf:
                click.echo(
                    f"  Uncertainty limits: {len(qf['max_fractional_uncertainty'])}"
                )

        # Habitable zone coefficients
        if "kopparapu_hz_coefficients" in config:
            hz = config["kopparapu_hz_coefficients"]
            click.echo(f"\nHabitable Zone Models: {len(hz)}")
            for model in hz.keys():
                click.echo(f"  - {model}")

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
