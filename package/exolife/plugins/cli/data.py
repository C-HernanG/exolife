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
        for source_dir in settings.raw_dir.iterdir():
            if source_dir.is_dir():
                parquet_file = source_dir / f"{source_dir.name}.parquet"
                csv_file = source_dir / f"{source_dir.name}.csv"

                if parquet_file.exists():
                    try:
                        df = pd.read_parquet(parquet_file)
                        click.echo(f"  ✓ {source_dir.name}: {len(df)} rows (parquet)")
                        if csv_file.exists():
                            click.echo("    + CSV version available")
                    except Exception:
                        click.echo(f"  ⚠ {source_dir.name}: parquet corrupted")
                elif csv_file.exists():
                    try:
                        df = pd.read_csv(csv_file)
                        click.echo(f"  ✓ {source_dir.name}: {len(df)} rows (csv only)")
                    except Exception:
                        click.echo(f"  ⚠ {source_dir.name}: csv corrupted")
                else:
                    click.echo(f"  ⚠ {source_dir.name}: no data files found")
    else:
        click.echo("  No raw data directory found")

    # Check interim data (staged processing)
    click.echo("\nInterim Data (Processing Stages):")
    if settings.interim_dir.exists():
        for stage_dir in settings.interim_dir.iterdir():
            if stage_dir.is_dir():
                click.echo(f"\n  Stage: {stage_dir.name}")

                parquet_files = list(stage_dir.glob("*.parquet"))
                csv_files = list(stage_dir.glob("*.csv"))

                if parquet_files or csv_files:
                    for parquet_file in parquet_files:
                        try:
                            df = pd.read_parquet(parquet_file)
                            click.echo(
                                f"    ✓ {parquet_file.stem}: {len(df)} rows (parquet)"
                            )
                        except Exception:
                            click.echo(f"    ⚠ {parquet_file.name}: parquet corrupted")

                    csv_only = [
                        f
                        for f in csv_files
                        if not (stage_dir / f"{f.stem}.parquet").exists()
                    ]
                    for csv_file in csv_only:
                        try:
                            df = pd.read_csv(csv_file)
                            click.echo(
                                f"    ✓ {csv_file.stem}: {len(df)} rows (csv only)"
                            )
                        except Exception:
                            click.echo(f"    ⚠ {csv_file.name}: csv corrupted")
                else:
                    click.echo("    (empty)")
    else:
        click.echo("  No interim data directory found")

    # Check processed data
    click.echo("\nProcessed Data:")
    if settings.processed_dir.exists():
        for dataset_dir in settings.processed_dir.iterdir():
            if dataset_dir.is_dir():
                click.echo(f"\n  Dataset: {dataset_dir.name}")

                parquet_files = list(dataset_dir.glob("*.parquet"))
                csv_files = list(dataset_dir.glob("*.csv"))

                if parquet_files:
                    for parquet_file in parquet_files:
                        try:
                            df = pd.read_parquet(parquet_file)
                            click.echo(
                                f"    ✓ {parquet_file.stem}: {len(df)} rows (parquet)"
                            )
                        except Exception:
                            click.echo(f"    ⚠ {parquet_file.name}: parquet corrupted")

                if csv_files:
                    csv_only = [
                        f
                        for f in csv_files
                        if not (dataset_dir / f"{f.stem}.parquet").exists()
                    ]
                    for csv_file in csv_only:
                        try:
                            df = pd.read_csv(csv_file)
                            click.echo(
                                f"    ✓ {csv_file.stem}: {len(df)} rows (csv only)"
                            )
                        except Exception:
                            click.echo(f"    ⚠ {csv_file.name}: csv corrupted")

                if not parquet_files and not csv_files:
                    click.echo(f"    ⚠ {dataset_dir.name}: no data files found")
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

        source_dir = settings.raw_dir / source_id
        raw_file = source_dir / f"{source_id}.parquet"
        csv_file = source_dir / f"{source_id}.csv"

        if raw_file.exists():
            try:
                import pandas as pd

                df = pd.read_parquet(raw_file)
                click.echo(
                    f"  ✓ Raw data exists: {len(df)} rows, {len(df.columns)} columns (parquet)"
                )
                if csv_file.exists():
                    click.echo("    + CSV version available")
                valid_count += 1
            except Exception as e:
                click.echo(f"  ⚠ Raw parquet data corrupted: {e}")
                if csv_file.exists():
                    try:
                        df = pd.read_csv(csv_file)
                        click.echo(
                            f"  ✓ Raw CSV data exists: {len(df)} rows, {len(df.columns)} columns"
                        )
                        valid_count += 1
                    except Exception as csv_e:
                        click.echo(f"  ⚠ Raw CSV data also corrupted: {csv_e}")
                        continue
                else:
                    continue
        elif csv_file.exists():
            try:
                import pandas as pd

                df = pd.read_csv(csv_file)
                click.echo(
                    f"  ✓ Raw CSV data exists: {len(df)} rows, {len(df.columns)} columns"
                )
                valid_count += 1
            except Exception as e:
                click.echo(f"  ⚠ Raw CSV data corrupted: {e}")
                continue
        else:
            click.echo(f"  ⚠ Raw data not found in: {source_dir}")

        click.echo("  ✓ Configuration valid")

    click.echo(
        f"\nValidation Summary: {valid_count}/{total_count} sources have valid data"
    )
