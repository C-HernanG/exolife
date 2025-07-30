"""
DAG CLI Commands for workflow orchestration.
"""

import logging
import pathlib
from typing import Optional

import click

from exolife.pipeline import (
    DAGExecutor,
    DataPipelineTaskExecutor,
    TaskStatus,
    load_dag_from_yaml,
)
from exolife.settings import settings

logger = logging.getLogger(__name__)


def get_dags_directory() -> pathlib.Path:
    """
    Get the DAGs directory, creating it if it doesn't exist.
    """
    dags_dir = settings.root_dir / "dags"
    dags_dir.mkdir(exist_ok=True)
    return dags_dir


def list_available_dags() -> list[tuple[str, pathlib.Path]]:
    """
    List all available DAG files in the dags directory.
    """
    dags_dir = get_dags_directory()
    dag_files = []

    for pattern in ["*.yaml", "*.yml"]:
        for dag_file in dags_dir.glob(pattern):
            dag_name = dag_file.stem
            dag_files.append((dag_name, dag_file))

    return sorted(dag_files)


def resolve_dag_file(dag_input: str) -> pathlib.Path:
    """
    Resolve a DAG input to a file path.

    Args:
        dag_input: Either a DAG name or a file path

    Returns:
        Path to the DAG file

    Raises:
        FileNotFoundError: If the DAG cannot be found
    """
    # If it's already a path that exists, use it
    dag_path = pathlib.Path(dag_input)
    if dag_path.exists():
        return dag_path

    # Try to find it in the dags directory
    dags_dir = get_dags_directory()

    # Try with .yaml extension
    yaml_path = dags_dir / f"{dag_input}.yaml"
    if yaml_path.exists():
        return yaml_path

    # Try with .yml extension
    yml_path = dags_dir / f"{dag_input}.yml"
    if yml_path.exists():
        return yml_path

    # Try exact match in dags directory
    exact_path = dags_dir / dag_input
    if exact_path.exists():
        return exact_path

    # List available DAGs for error message
    available_dags = list_available_dags()
    dag_names = [name for name, _ in available_dags]

    raise FileNotFoundError(
        f"DAG '{dag_input}' not found. Available DAGs: "
        f"{', '.join(dag_names) or 'none'}\nDAGs directory: {dags_dir}"
    )


@click.group("dag")
def cli():
    """
    DAG workflow orchestration commands.
    """
    pass


@cli.command()
@click.argument("dag_file", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option(
    "--mode",
    type=click.Choice(["sequential", "parallel"]),
    default="sequential",
    help="Execution mode for the DAG",
)
@click.option("--dry-run", is_flag=True, help="Validate DAG without executing tasks")
@click.option("--task", help="Execute only a specific task (and its dependencies)")
@click.pass_context
def run(ctx, dag_file: pathlib.Path, mode: str, dry_run: bool, task: Optional[str]):
    """
    Execute a DAG workflow from a YAML specification file.

    Examples:
        exolife dag run dagspec.yaml
        exolife dag run dagspec.yaml --mode parallel
        exolife dag run dagspec.yaml --dry-run
        exolife dag run dagspec.yaml --task merge_datasets
    """
    try:
        # Load DAG from YAML
        dag = load_dag_from_yaml(dag_file)
        click.echo(f"✓ Loaded DAG with {len(dag.tasks)} tasks from {dag_file}")

        # Validate DAG
        validation_errors = dag.validate()
        if validation_errors:
            click.echo("✗ DAG validation failed:")
            for error in validation_errors:
                click.echo(f"  - {error}")
            return

        click.echo("✓ DAG validation passed")

        if dry_run:
            click.echo("✓ Dry run completed successfully")
            return

        # Create executor
        task_executor = DataPipelineTaskExecutor()
        executor = DAGExecutor(task_executor)

        # Execute DAG
        click.echo(f"🚀 Starting DAG execution in {mode} mode...")

        if task:
            # Execute specific task and dependencies
            result = executor.execute_task(dag, task)
            if result.status == TaskStatus.SUCCESS:
                click.echo(f"✓ Task '{task}' completed successfully")
            else:
                click.echo(f"✗ Task '{task}' failed: {result.error_message}")
                return
        else:
            # Execute entire DAG
            results = executor.execute_dag(dag, mode=mode)

            # Report results
            succeeded = sum(
                1 for r in results.values() if r.status == TaskStatus.SUCCESS
            )
            failed = sum(1 for r in results.values() if r.status == TaskStatus.FAILED)

            click.echo("\n📊 Execution Summary:")
            click.echo(f"  ✓ Succeeded: {succeeded}")
            click.echo(f"  ✗ Failed: {failed}")
            click.echo(f"  📋 Total: {len(results)}")

            if failed > 0:
                click.echo("\n❌ Failed tasks:")
                for task_id, result in results.items():
                    if result.status == TaskStatus.FAILED:
                        click.echo(f"  - {task_id}: {result.error_message}")
                return

            click.echo("\n🎉 All tasks completed successfully!")

    except Exception as e:
        logger.error(f"DAG execution failed: {e}")
        click.echo(f"✗ DAG execution failed: {e}")


@cli.command()
@click.argument("dag_file", type=click.Path(exists=True, path_type=pathlib.Path))
def validate(dag_file: pathlib.Path):
    """
    Validate a DAG specification file.

    Examples:
        exolife dag validate dagspec.yaml
    """
    try:
        # Load DAG from YAML
        dag = load_dag_from_yaml(dag_file)
        click.echo(f"✓ Loaded DAG with {len(dag.tasks)} tasks from {dag_file}")

        # Validate DAG
        validation_errors = dag.validate()
        if validation_errors:
            click.echo("✗ DAG validation failed:")
            for error in validation_errors:
                click.echo(f"  - {error}")
            return

        click.echo("✓ DAG validation passed")

        # Show DAG structure
        click.echo("\n📋 DAG Structure:")
        execution_order = dag.topological_sort()
        for i, task_id in enumerate(execution_order, 1):
            task = dag.tasks[task_id]
            deps = ", ".join(task.dependencies) if task.dependencies else "None"
            click.echo(f"  {i}. {task_id} (deps: {deps})")

    except Exception as e:
        logger.error(f"DAG validation failed: {e}")
        click.echo(f"✗ DAG validation failed: {e}")


@cli.command()
@click.argument("dag_file", type=click.Path(exists=True, path_type=pathlib.Path))
def info(dag_file: pathlib.Path):
    """
    Show information about a DAG specification file.

    Examples:
        exolife dag info dagspec.yaml
    """
    try:
        # Load DAG from YAML
        dag = load_dag_from_yaml(dag_file)

        click.echo(f"📋 DAG Information: {dag_file}")
        click.echo(f"  Name: {dag.name}")
        click.echo(f"  Description: {dag.description}")
        click.echo(f"  Tasks: {len(dag.tasks)}")

        # Show task details
        click.echo("\n🔧 Tasks:")
        for task_id, task in dag.tasks.items():
            click.echo(f"  - {task_id}")
            click.echo(f"    Type: {task.task_type}")
            click.echo(
                f"    Dependencies: "
                f"{', '.join(task.dependencies) if task.dependencies else 'None'}"
            )
            if task.config:
                config_summary = {
                    k: str(v)[:50] + "..." if len(str(v)) > 50 else v
                    for k, v in task.config.items()
                }
                click.echo(f"    Config: {config_summary}")

        # Show execution order
        click.echo("\n🔄 Execution Order:")
        execution_order = dag.topological_sort()
        for i, task_id in enumerate(execution_order, 1):
            click.echo(f"  {i}. {task_id}")

    except Exception as e:
        logger.error(f"DAG info failed: {e}")
        click.echo(f"✗ DAG info failed: {e}")
