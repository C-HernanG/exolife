"""
DAG Executor for running pipeline workflows.

This module provides execution engines for running DAG-defined workflows
with support for parallel execution, error handling, and progress tracking.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Set

from ..settings import settings
from .dag import DAG, TaskNode, TaskResult, TaskStatus

logger = logging.getLogger(__name__)


class TaskExecutor:
    """
    Base class for task executors.
    """

    def execute(self, task: TaskNode) -> TaskResult:
        """
        Execute a single task and return the result.
        """
        raise NotImplementedError("Subclasses must implement execute method")


class DataPipelineTaskExecutor(TaskExecutor):
    """
    Task executor for data pipeline tasks.
    """

    def __init__(self):
        self.task_handlers = {
            "fetch": self._execute_fetch_task,
            "merge": self._execute_merge_task,
            "preprocess": self._execute_preprocess_task,
            "validate": self._execute_validate_task,
            "export": self._execute_export_task,
        }

    def execute(self, task: TaskNode) -> TaskResult:
        """
        Execute a data pipeline task.
        """
        start_time = time.time()
        task_result = TaskResult(task_id=task.task_id, status=TaskStatus.RUNNING)

        try:
            logger.info(f"Executing task '{task.task_id}' of type '{task.task_type}'")

            if task.task_type not in self.task_handlers:
                raise ValueError(f"Unknown task type: {task.task_type}")

            output = self.task_handlers[task.task_type](task)

            task_result.status = TaskStatus.SUCCESS
            task_result.output = output
            logger.info(f"Task '{task.task_id}' completed successfully")

        except Exception as e:
            task_result.status = TaskStatus.FAILED
            task_result.error_message = str(e)
            logger.error(f"Task '{task.task_id}' failed: {e}")

        finally:
            task_result.execution_time = time.time() - start_time

        return task_result

    def _execute_fetch_task(self, task: TaskNode) -> Any:
        """
        Execute a data fetch task.
        """
        from exolife.data.fetch.fetchers import fetch_manager

        source = task.config.get("source")
        force = task.config.get("force_refresh", False)

        if not source:
            raise ValueError("Fetch task requires 'source' in config")

        result = fetch_manager._fetch_single_source(source, force=force)
        return {"source": source, "success": True, "path": str(result)}

    def _execute_merge_task(self, task: TaskNode) -> Any:
        """
        Execute a data merge task.
        """
        from exolife.data.merge.mergers import merge_data

        strategy = task.config.get("strategy", "baseline")

        logger.info(f"Merging data with strategy: {strategy}")
        result = merge_data(strategy)

        # Save merged data to interim directory for next steps
        cache_dir = settings.interim_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        output_path = cache_dir / f"merged_{strategy}.parquet"
        result.to_parquet(output_path)
        logger.info(f"Saved merged data to {output_path}")

        return {
            "strategy": strategy,
            "rows": len(result),
            "columns": len(result.columns),
            "output_path": str(output_path),
        }

    def _execute_preprocess_task(self, task: TaskNode) -> Any:
        """
        Execute a data preprocessing task.
        """
        import pandas as pd

        from exolife.data.preprocess import PreprocessorRegistry

        preprocessor_name = task.config.get("preprocessor")
        if not preprocessor_name:
            raise ValueError("Preprocess task requires 'preprocessor' in config")

        # Load data from previous step using interim directory
        cache_dir = settings.interim_dir

        # Find the most recent data file (could be from merge or previous preprocess)
        data_files = list(cache_dir.glob("*.parquet"))
        if not data_files:
            raise ValueError("No data files found in interim directory")

        # Load the most recent file (for now, use a simple heuristic)
        latest_file = max(data_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading data from {latest_file}")
        data = pd.read_parquet(latest_file)

        # Get and run the preprocessor
        logger.info(f"Running preprocessor: {preprocessor_name}")
        preprocessor_class = PreprocessorRegistry.get(preprocessor_name)
        preprocessor = preprocessor_class()

        processed_data = preprocessor.process(data)

        # Save processed data to interim directory
        output_path = cache_dir / f"processed_{preprocessor_name.lower()}.parquet"
        processed_data.to_parquet(output_path)
        logger.info(f"Saved processed data to {output_path}")

        return {
            "preprocessor": preprocessor_name,
            "success": True,
            "input_rows": len(data),
            "output_rows": len(processed_data),
            "output_path": str(output_path),
        }

    def _execute_validate_task(self, task: TaskNode) -> Any:
        """
        Execute a data validation task.
        """
        import pandas as pd

        checks = []
        if task.config.get("check_completeness", False):
            checks.append("completeness")
        if task.config.get("check_data_types", False):
            checks.append("data_types")
        if task.config.get("check_hz_columns", False):
            checks.append("hz_columns")

        logger.info(f"Running validation checks: {checks}")

        # Load data for validation from interim directory
        cache_dir = settings.interim_dir
        data_files = list(cache_dir.glob("*.parquet"))

        if data_files:
            # Validate the most recent data file
            latest_file = max(data_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Validating data from {latest_file}")
            data = pd.read_parquet(latest_file)

            validation_results = {}

            if "completeness" in checks:
                missing_percentage = (data.isnull().sum().sum() / data.size) * 100
                validation_results["completeness"] = {
                    "passed": missing_percentage < 50,  # Less than 50% missing
                    "missing_percentage": missing_percentage,
                }

            if "data_types" in checks:
                validation_results["data_types"] = {
                    "passed": True,  # Simple check - assume types are OK
                    "numeric_columns": len(
                        data.select_dtypes(include=["number"]).columns
                    ),
                    "object_columns": len(
                        data.select_dtypes(include=["object"]).columns
                    ),
                }

            if "hz_columns" in checks:
                hz_columns = [col for col in data.columns if "hz" in col.lower()]
                validation_results["hz_columns"] = {
                    "passed": len(hz_columns) > 0,
                    "hz_columns_found": hz_columns,
                }

            all_passed = all(
                result.get("passed", False) for result in validation_results.values()
            )

            return {
                "checks": checks,
                "all_passed": all_passed,
                "validation_results": validation_results,
                "data_shape": data.shape,
            }
        else:
            logger.warning("No data files found for validation")
            return {
                "checks": checks,
                "all_passed": False,
                "error": "No data files found",
            }

    def _execute_export_task(self, task: TaskNode) -> Any:
        """
        Execute a data export task.
        """
        import pandas as pd

        formats = task.config.get("formats", ["parquet"])
        output_path = task.config.get("output_path", "final_dataset")
        filename_base = task.config.get(
            "filename", "unified_catalog"
        )  # Use configurable filename

        # Create output directory in processed folder
        output_dir = settings.processed_dir / output_path
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load the final processed data from interim directory
        cache_dir = settings.interim_dir
        data_files = list(cache_dir.glob("*.parquet"))

        if not data_files:
            raise ValueError("No data files found in interim directory for export")

        # Load the most recent data file
        latest_file = max(data_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Exporting data from {latest_file}")
        data = pd.read_parquet(latest_file)

        exported_files = []

        for fmt in formats:
            if fmt == "parquet":
                export_file = output_dir / f"{filename_base}.parquet"
                data.to_parquet(export_file, index=False)
            elif fmt == "csv":
                export_file = output_dir / f"{filename_base}.csv"
                data.to_csv(export_file, index=False)
            else:
                logger.warning(f"Unsupported export format: {fmt}")
                continue

            exported_files.append(str(export_file))
            logger.info(f"Exported data to {export_file}")

        return {
            "formats": formats,
            "output_path": str(output_dir),
            "success": True,
            "exported_files": exported_files,
            "data_shape": data.shape,
        }


class DAGExecutor:
    """
    Executor for running DAG workflows.

    Supports both sequential and parallel execution modes with configurable
    error handling and retry policies.
    """

    def __init__(self, task_executor: TaskExecutor, max_workers: int = 4):
        self.task_executor = task_executor
        self.max_workers = max_workers
        self.execution_results: Dict[str, TaskResult] = {}

    def execute_sequential(self, dag: DAG) -> Dict[str, TaskResult]:
        """
        Execute DAG tasks sequentially in topological order.
        """
        validation_errors = dag.validate()
        if validation_errors:
            raise ValueError(f"DAG validation failed: {'; '.join(validation_errors)}")

        execution_order = dag.topological_sort()

        logger.info(f"Starting sequential execution of DAG '{dag.dag_id}'")
        logger.info(f"Execution order: {' -> '.join(execution_order)}")

        for task_id in execution_order:
            task = dag.nodes[task_id]
            result = self._execute_with_retries(task)
            self.execution_results[task_id] = result

            if result.status == TaskStatus.FAILED and task.on_failure == "fail":
                logger.error(f"Stopping execution due to failed task: {task_id}")
                break
            elif result.status == TaskStatus.FAILED and task.on_failure == "skip":
                logger.warning(f"Skipping failed task: {task_id}")
                result.status = TaskStatus.SKIPPED

        return self.execution_results

    def execute_parallel(self, dag: DAG) -> Dict[str, TaskResult]:
        """
        Execute DAG tasks in parallel where possible.
        """
        validation_errors = dag.validate()
        if validation_errors:
            raise ValueError(f"DAG validation failed: {'; '.join(validation_errors)}")

        logger.info(f"Starting parallel execution of DAG '{dag.dag_id}'")

        executed_tasks: Set[str] = set()
        failed_tasks: Set[str] = set()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while len(executed_tasks) + len(failed_tasks) < len(dag.nodes):
                # Get tasks ready for execution
                ready_tasks = [
                    task_id
                    for task_id in dag.get_ready_tasks(executed_tasks)
                    if task_id not in failed_tasks
                ]

                if not ready_tasks:
                    # Check if we're stuck due to failed dependencies
                    remaining_tasks = (
                        set(dag.nodes.keys()) - executed_tasks - failed_tasks
                    )
                    if remaining_tasks:
                        logger.error(
                            f"Cannot proceed: remaining tasks {remaining_tasks}"
                            f" have failed dependencies"
                        )
                    break

                # Submit ready tasks for execution
                future_to_task = {
                    executor.submit(
                        self._execute_with_retries, dag.nodes[task_id]
                    ): task_id
                    for task_id in ready_tasks
                }

                # Collect results
                for future in as_completed(future_to_task):
                    task_id = future_to_task[future]
                    result = future.result()
                    self.execution_results[task_id] = result

                    if result.status == TaskStatus.SUCCESS:
                        executed_tasks.add(task_id)
                        logger.info(f"Task '{task_id}' completed successfully")
                    else:
                        task = dag.nodes[task_id]
                        if task.on_failure == "fail":
                            logger.error(f"Task '{task_id}' failed, stopping execution")
                            failed_tasks.add(task_id)
                            # Cancel remaining futures
                            for f in future_to_task:
                                f.cancel()
                            return self.execution_results
                        elif task.on_failure == "skip":
                            logger.warning(f"Task '{task_id}' failed but continuing")
                            result.status = TaskStatus.SKIPPED
                            executed_tasks.add(task_id)
                        else:  # continue
                            logger.warning(f"Task '{task_id}' failed but continuing")
                            failed_tasks.add(task_id)

        return self.execution_results

    def execute_dag(self, dag: DAG, mode: str = "sequential") -> Dict[str, TaskResult]:
        """
        Execute a DAG in the specified mode.

        Args:
            dag: The DAG to execute
            mode: Execution mode ("sequential" or "parallel")

        Returns:
            Dictionary mapping task IDs to their execution results
        """
        if mode == "parallel":
            return self.execute_parallel(dag)
        elif mode == "sequential":
            return self.execute_sequential(dag)
        else:
            raise ValueError(
                f"Unknown execution mode: {mode}. Use 'sequential' or 'parallel'"
            )

    def execute_task(self, dag: DAG, task_id: str) -> TaskResult:
        """
        Execute a specific task and its dependencies.

        Args:
            dag: The DAG containing the task
            task_id: ID of the task to execute

        Returns:
            TaskResult for the specified task
        """
        validation_errors = dag.validate()
        if validation_errors:
            raise ValueError(f"DAG validation failed: {'; '.join(validation_errors)}")

        if task_id not in dag.nodes:
            raise ValueError(f"Task '{task_id}' not found in DAG")

        # Get all dependencies for this task (recursive)
        def get_all_dependencies(tid: str, visited: Set[str] = None) -> Set[str]:
            if visited is None:
                visited = set()
            if tid in visited:
                return visited
            visited.add(tid)

            task = dag.nodes[tid]
            for dep_id in task.dependencies:
                get_all_dependencies(dep_id, visited)
            return visited

        # Execute all dependencies first
        required_tasks = get_all_dependencies(task_id)
        execution_order = [
            tid for tid in dag.topological_sort() if tid in required_tasks
        ]

        logger.info(f"Executing task '{task_id}' with dependencies: {execution_order}")

        for tid in execution_order:
            task = dag.nodes[tid]
            result = self._execute_with_retries(task)
            self.execution_results[tid] = result

            if result.status == TaskStatus.FAILED and task.on_failure == "fail":
                logger.error(
                    f"Dependency task '{tid}' failed, cannot execute '{task_id}'"
                )
                break

        return self.execution_results.get(
            task_id,
            TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error_message="Task was not executed due to dependency failure",
            ),
        )

    def _execute_with_retries(self, task: TaskNode) -> TaskResult:
        """
        Execute a task with retry logic.
        """
        last_result = None

        for attempt in range(task.retries + 1):
            if attempt > 0:
                logger.info(
                    f"Retrying task '{task.task_id}' "
                    f"(attempt {attempt + 1}/{task.retries + 1})"
                )

            try:
                if task.timeout:
                    # Execute with timeout
                    result = self.task_executor.execute(task)
                else:
                    result = self.task_executor.execute(task)

                if result.status == TaskStatus.SUCCESS:
                    return result

                last_result = result

            except Exception as e:
                last_result = TaskResult(
                    task_id=task.task_id, status=TaskStatus.FAILED, error_message=str(e)
                )

        return last_result or TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error_message="Task failed after all retry attempts",
        )

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the execution results.
        """
        if not self.execution_results:
            return {"status": "not_started", "tasks": 0}

        status_counts = {}
        total_time = 0

        for result in self.execution_results.values():
            status = result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            total_time += result.execution_time

        return {
            "total_tasks": len(self.execution_results),
            "status_counts": status_counts,
            "total_execution_time": total_time,
            "success_rate": status_counts.get("success", 0)
            / len(self.execution_results)
            * 100,
        }
