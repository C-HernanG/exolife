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
    Execute data ingestion using the unified pipeline.

    Args:
        method: Ingestion method (all methods now use unified_ingestion).
        Run without method to see available options.

    Note: ExoLife now uses a single unified ingestion pipeline that:
    - Cross-identifies sources via Gaia source_id and (host, letter)
    - Propagates uncertainties via Monte Carlo sampling
    - Derives features with uncertainty quantification
    - Maintains data provenance and quality indicators

    For complex workflows, use the DAG system:
        exolife dag run config/dags/dagspec.yaml
    """
    # Get available strategies dynamically from merge factory
    available = merge_factory.get_available_strategies()

    # If no method specified, show available options
    if not method:
        click.echo("ExoLife Unified Ingestion Pipeline")
        click.echo("=" * 40)
        click.echo()
        click.echo("Available methods (all use unified_ingestion):")
        for strategy in sorted(available):
            if strategy == "unified_ingestion":
                click.echo(f"  ✓ {strategy} (recommended)")
            else:
                click.echo(f"  - {strategy} (legacy alias)")
        click.echo()
        click.echo("The unified pipeline harmonizes multi-mission catalogs with:")
        click.echo("• Cross-identification via Gaia source_id and (host, letter)")
        click.echo("• Unit standardization across all sources")
        click.echo("• Monte Carlo uncertainty propagation (N=1000)")
        click.echo(
            "• Derived features: stellar flux, Teq, surface gravity, HZ distances"
        )
        click.echo("• Explicit missingness encoding")
        click.echo("• Data provenance tracking")
        click.echo()
        click.echo(
            "💡 Tip: For complete workflows with validation, use the DAG system:"
        )
        click.echo("   exolife dag run config/dags/dagspec.yaml")
        return

    # Validate method (all methods now map to unified_ingestion)
    if method not in available:
        logger.error(
            "Unknown ingestion method '%s'; available options: %s",
            method,
            ", ".join(sorted(available)),
        )
        click.echo(
            f"Error: Unknown method '{method}'.\n"
            f"Available: {', '.join(sorted(available))}"
        )
        raise click.Abort()

    # Perform ingestion operation
    try:
        logger.info("Running unified ingestion with method '%s'...", method)
        click.echo("Running ExoLife unified ingestion pipeline...")

        if method != "unified_ingestion":
            click.echo(f"Note: '{method}' now maps to the unified ingestion pipeline")

        # Use merge manager which handles the unified strategy
        result = merge_manager.merge_data(method, overwrite=force)

        if hasattr(result, "__len__"):
            click.echo(f"✓ Ingestion completed: {len(result)} records processed")

            # Show some statistics if available
            if "data_completeness_score" in result.columns:
                avg_completeness = result["data_completeness_score"].mean()
                click.echo(f"  • Average data completeness: {avg_completeness:.2%}")

            if any("hz_" in col for col in result.columns):
                hz_coverage = result.filter(regex=r"hz_.*").notna().any(axis=1).sum()
                click.echo(f"  • Records with HZ data: {hz_coverage}")

            derived_features = len(
                [
                    col
                    for col in result.columns
                    if any(
                        prefix in col
                        for prefix in [
                            "stellar_flux",
                            "equilibrium_temp",
                            "surface_gravity",
                            "escape_velocity",
                        ]
                    )
                ]
            )
            if derived_features > 0:
                click.echo(f"  • Derived features added: {derived_features}")
        else:
            click.echo("✓ Ingestion completed successfully")

        logger.info("Unified ingestion completed successfully.")

    except Exception as exc:
        logger.exception("Ingestion operation failed: %s", exc)
        click.echo(f"✗ Ingestion failed: {exc}")
        raise click.Abort()
