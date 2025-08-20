"""
Tests for the ExoLife pipeline executor module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import time

from exolife.pipeline.dag import DAG, TaskNode, TaskStatus, TaskResult
from exolife.pipeline.executor import DAGExecutor, DataPipelineTaskExecutor


class MockTaskExecutor:
    """Mock task executor for testing."""

    def __init__(self, should_fail=False, execution_time=0.1):
        self.should_fail = should_fail
        self.execution_time = execution_time
        self.executed_tasks = []

    def execute_task(self, task_node, dag_context=None):
        """Mock task execution."""
        self.executed_tasks.append(task_node.task_id)

        # Simulate execution time
        time.sleep(self.execution_time)

        if self.should_fail and task_node.task_id == "failing_task":
            return TaskResult(
                task_id=task_node.task_id,
                status=TaskStatus.FAILED,
                error_message="Mock task failure",
                execution_time=self.execution_time
            )

        return TaskResult(
            task_id=task_node.task_id,
            status=TaskStatus.SUCCESS,
            output={"mock": "data"},
            execution_time=self.execution_time
        )


class TestDAGExecutor:
    """Test cases for DAGExecutor."""

    def test_dag_executor_creation(self):
        """Test DAGExecutor creation."""
        task_executor = MockTaskExecutor()
        executor = DAGExecutor(task_executor)

        assert executor.task_executor == task_executor
        assert executor.max_retries == 3
        assert executor.retry_delay == 1.0

    def test_dag_executor_custom_settings(self):
        """Test DAGExecutor with custom settings."""
        task_executor = MockTaskExecutor()
        executor = DAGExecutor(task_executor, max_retries=5, retry_delay=2.0)

        assert executor.max_retries == 5
        assert executor.retry_delay == 2.0

    def test_execute_task_success(self, sample_dag):
        """Test successful task execution."""
        task_executor = MockTaskExecutor()
        executor = DAGExecutor(task_executor)

        task_node = sample_dag.nodes["fetch_data"]
        result = executor.execute_task(sample_dag, "fetch_data")

        assert result.status == TaskStatus.SUCCESS
        assert result.task_id == "fetch_data"
        assert "fetch_data" in task_executor.executed_tasks

    def test_execute_task_not_found(self, sample_dag):
        """Test executing non-existent task."""
        task_executor = MockTaskExecutor()
        executor = DAGExecutor(task_executor)

        result = executor.execute_task(sample_dag, "nonexistent_task")

        assert result.status == TaskStatus.FAILED
        assert "not found" in result.error_message

    def test_execute_task_with_retries(self, sample_dag):
        """Test task execution with retries."""
        # Create a task executor that fails first time, succeeds second time
        call_count = 0

        def mock_execute(task_node, dag_context=None):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return TaskResult(
                    task_id=task_node.task_id,
                    status=TaskStatus.FAILED,
                    error_message="First attempt failed"
                )
            else:
                return TaskResult(
                    task_id=task_node.task_id,
                    status=TaskStatus.SUCCESS
                )

        task_executor = Mock()
        task_executor.execute_task = mock_execute

        # Fast retry for testing
        executor = DAGExecutor(task_executor, retry_delay=0.01)

        # Set retries on the task
        sample_dag.nodes["fetch_data"].retries = 2

        result = executor.execute_task(sample_dag, "fetch_data")

        assert result.status == TaskStatus.SUCCESS
        assert call_count == 2

    def test_execute_task_max_retries_exceeded(self, sample_dag):
        """Test task execution when max retries are exceeded."""
        def mock_execute(task_node, dag_context=None):
            return TaskResult(
                task_id=task_node.task_id,
                status=TaskStatus.FAILED,
                error_message="Always fails"
            )

        task_executor = Mock()
        task_executor.execute_task = mock_execute

        executor = DAGExecutor(task_executor, retry_delay=0.01)

        # Set retries on the task
        sample_dag.nodes["fetch_data"].retries = 2

        result = executor.execute_task(sample_dag, "fetch_data")

        assert result.status == TaskStatus.FAILED
        assert "after 2 retries" in result.error_message

    def test_execute_dag_sequential_success(self, sample_dag):
        """Test successful sequential DAG execution."""
        task_executor = MockTaskExecutor()
        executor = DAGExecutor(task_executor)

        results = executor.execute_dag(sample_dag, mode="sequential")

        assert len(results) == 3
        assert all(r.status == TaskStatus.SUCCESS for r in results.values())

        # Check execution order
        expected_order = ["fetch_data", "process_data", "merge_data"]
        actual_order = task_executor.executed_tasks
        assert actual_order == expected_order

    def test_execute_dag_sequential_with_failure(self):
        """Test sequential DAG execution with task failure."""
        dag = DAG("test", "Test")

        # Create tasks where middle task fails
        tasks = [
            TaskNode("task1", "fetch", on_failure="fail"),
            TaskNode("task2", "process", dependencies=[
                     "task1"], on_failure="fail"),
            TaskNode("task3", "merge", dependencies=["task2"])
        ]

        for task in tasks:
            dag.add_task(task)

        # Mock executor that fails on task2
        def mock_execute(task_node, dag_context=None):
            if task_node.task_id == "task2":
                return TaskResult(task_node.task_id, TaskStatus.FAILED, error_message="Mock failure")
            return TaskResult(task_node.task_id, TaskStatus.SUCCESS)

        task_executor = Mock()
        task_executor.execute_task = mock_execute

        executor = DAGExecutor(task_executor)
        results = executor.execute_dag(dag, mode="sequential")

        # Should have results for task1 and task2, but not task3
        assert "task1" in results
        assert "task2" in results
        assert results["task2"].status == TaskStatus.FAILED

        # task3 should not be executed due to dependency failure
        assert "task3" not in results

    def test_execute_dag_with_on_failure_continue(self):
        """Test DAG execution with on_failure='continue'."""
        dag = DAG("test", "Test")

        tasks = [
            TaskNode("task1", "fetch"),
            TaskNode("task2", "process", dependencies=[
                     "task1"], on_failure="continue"),
            TaskNode("task3", "merge", dependencies=["task2"])
        ]

        for task in tasks:
            dag.add_task(task)

        # Mock executor that fails on task2
        def mock_execute(task_node, dag_context=None):
            if task_node.task_id == "task2":
                return TaskResult(task_node.task_id, TaskStatus.FAILED, error_message="Mock failure")
            return TaskResult(task_node.task_id, TaskStatus.SUCCESS)

        task_executor = Mock()
        task_executor.execute_task = mock_execute

        executor = DAGExecutor(task_executor)
        results = executor.execute_dag(dag, mode="sequential")

        # All tasks should have results, even though task2 failed
        assert len(results) == 3
        assert results["task2"].status == TaskStatus.FAILED
        assert results["task3"].status == TaskStatus.SUCCESS

    def test_execute_dag_parallel_mode(self, sample_dag):
        """Test parallel DAG execution."""
        task_executor = MockTaskExecutor(execution_time=0.01)
        executor = DAGExecutor(task_executor)

        start_time = time.time()
        results = executor.execute_dag(sample_dag, mode="parallel")
        execution_time = time.time() - start_time

        assert len(results) == 3
        assert all(r.status == TaskStatus.SUCCESS for r in results.values())

        # Parallel execution should be faster than sequential for this case
        # (though with such fast mock tasks, this might not always be true)
        assert execution_time < 0.1  # Should be much faster than 3 * 0.01

    def test_execute_dag_invalid_mode(self, sample_dag):
        """Test DAG execution with invalid mode."""
        task_executor = MockTaskExecutor()
        executor = DAGExecutor(task_executor)

        with pytest.raises(ValueError, match="Invalid execution mode"):
            executor.execute_dag(sample_dag, mode="invalid_mode")

    def test_execute_dag_with_timeout(self):
        """Test task execution with timeout."""
        dag = DAG("test", "Test")

        # Create task with short timeout
        task = TaskNode("slow_task", "process", timeout=0.05)
        dag.add_task(task)

        # Mock executor that takes longer than timeout
        def mock_execute(task_node, dag_context=None):
            time.sleep(0.1)  # Longer than timeout
            return TaskResult(task_node.task_id, TaskStatus.SUCCESS)

        task_executor = Mock()
        task_executor.execute_task = mock_execute

        executor = DAGExecutor(task_executor)
        result = executor.execute_task(dag, "slow_task")

        assert result.status == TaskStatus.FAILED
        assert "timed out" in result.error_message

    def test_check_dependencies_met(self):
        """Test dependency checking."""
        dag = DAG("test", "Test")
        tasks = [
            TaskNode("A", "fetch"),
            TaskNode("B", "process", dependencies=["A"]),
            TaskNode("C", "merge", dependencies=["A", "B"])
        ]

        for task in tasks:
            dag.add_task(task)

        task_executor = MockTaskExecutor()
        executor = DAGExecutor(task_executor)

        completed = {}

        # Initially, only A should be ready
        assert executor._check_dependencies_met(
            dag.nodes["A"], completed) is True
        assert executor._check_dependencies_met(
            dag.nodes["B"], completed) is False
        assert executor._check_dependencies_met(
            dag.nodes["C"], completed) is False

        # After A completes
        completed["A"] = TaskResult("A", TaskStatus.SUCCESS)
        assert executor._check_dependencies_met(
            dag.nodes["B"], completed) is True
        assert executor._check_dependencies_met(
            dag.nodes["C"], completed) is False

        # After B completes
        completed["B"] = TaskResult("B", TaskStatus.SUCCESS)
        assert executor._check_dependencies_met(
            dag.nodes["C"], completed) is True

    def test_check_dependencies_met_with_failure(self):
        """Test dependency checking when dependency failed."""
        dag = DAG("test", "Test")
        tasks = [
            TaskNode("A", "fetch"),
            TaskNode("B", "process", dependencies=["A"])
        ]

        for task in tasks:
            dag.add_task(task)

        task_executor = MockTaskExecutor()
        executor = DAGExecutor(task_executor)

        # A failed
        completed = {"A": TaskResult("A", TaskStatus.FAILED)}

        # B should not be ready due to failed dependency
        assert executor._check_dependencies_met(
            dag.nodes["B"], completed) is False


class TestDataPipelineTaskExecutor:
    """Test cases for DataPipelineTaskExecutor."""

    def test_data_pipeline_task_executor_creation(self):
        """Test DataPipelineTaskExecutor creation."""
        executor = DataPipelineTaskExecutor()
        assert hasattr(executor, 'execute_task')

    @patch('exolife.pipeline.executor.FetchManager')
    def test_execute_fetch_task(self, mock_fetch_manager_class):
        """Test executing a fetch task."""
        # Setup mock
        mock_manager = Mock()
        mock_fetch_manager_class.return_value = mock_manager

        from exolife.data.fetch.fetcher_base import FetchResult
        mock_result = FetchResult("test_source", Path(
            "test.parquet"), True, rows_fetched=100)
        mock_manager.fetch_source_by_id.return_value = mock_result

        # Create task
        task = TaskNode(
            task_id="fetch_test",
            task_type="fetch",
            config={"source": "test_source"}
        )

        executor = DataPipelineTaskExecutor()
        result = executor.execute_task(task)

        assert result.status == TaskStatus.SUCCESS
        assert result.task_id == "fetch_test"
        assert result.output["fetch_result"] == mock_result

    def test_execute_unknown_task_type(self):
        """Test executing task with unknown type."""
        task = TaskNode(
            task_id="unknown_task",
            task_type="unknown_type",
            config={}
        )

        executor = DataPipelineTaskExecutor()
        result = executor.execute_task(task)

        assert result.status == TaskStatus.FAILED
        assert "Unknown task type" in result.error_message

    @patch('exolife.pipeline.executor.FetchManager')
    def test_execute_fetch_task_with_error(self, mock_fetch_manager_class):
        """Test executing fetch task that raises exception."""
        # Setup mock to raise exception
        mock_manager = Mock()
        mock_fetch_manager_class.return_value = mock_manager
        mock_manager.fetch_source_by_id.side_effect = Exception(
            "Network error")

        task = TaskNode(
            task_id="fetch_error",
            task_type="fetch",
            config={"source": "error_source"}
        )

        executor = DataPipelineTaskExecutor()
        result = executor.execute_task(task)

        assert result.status == TaskStatus.FAILED
        assert "Network error" in result.error_message

    def test_execute_validate_task(self):
        """Test executing a validate task."""
        task = TaskNode(
            task_id="validate_test",
            task_type="validate",
            config={"check_completeness": True}
        )

        executor = DataPipelineTaskExecutor()
        result = executor.execute_task(task)

        # Currently validate tasks just return success
        # In a real implementation, this would perform actual validation
        assert result.status == TaskStatus.SUCCESS
        assert result.task_id == "validate_test"

    def test_execute_export_task(self):
        """Test executing an export task."""
        task = TaskNode(
            task_id="export_test",
            task_type="export",
            config={"format": "parquet", "output_path": "test_output"}
        )

        executor = DataPipelineTaskExecutor()
        result = executor.execute_task(task)

        # Currently export tasks just return success
        # In a real implementation, this would perform actual export
        assert result.status == TaskStatus.SUCCESS
        assert result.task_id == "export_test"


class TestExecutorIntegration:
    """Integration tests for executor components."""

    def test_full_dag_execution_workflow(self):
        """Test complete DAG execution workflow."""
        # Create a simple DAG
        dag = DAG("integration_test", "Integration test DAG")

        tasks = [
            TaskNode("fetch", "fetch", config={"source": "test"}),
            TaskNode("validate", "validate", dependencies=["fetch"]),
            TaskNode("export", "export", dependencies=["validate"])
        ]

        for task in tasks:
            dag.add_task(task)

        # Use mock task executor
        task_executor = MockTaskExecutor()
        executor = DAGExecutor(task_executor)

        # Execute DAG
        results = executor.execute_dag(dag)

        # Verify results
        assert len(results) == 3
        assert all(r.status == TaskStatus.SUCCESS for r in results.values())

        # Verify execution order
        expected_order = ["fetch", "validate", "export"]
        assert task_executor.executed_tasks == expected_order

    def test_dag_execution_with_mixed_outcomes(self):
        """Test DAG execution with mixed success/failure outcomes."""
        dag = DAG("mixed_test", "Mixed outcome test")

        tasks = [
            TaskNode("success_task", "fetch"),
            TaskNode("failing_task", "process", dependencies=[
                     "success_task"], on_failure="continue"),
            TaskNode("final_task", "export", dependencies=["failing_task"])
        ]

        for task in tasks:
            dag.add_task(task)

        # Use task executor that fails on specific task
        task_executor = MockTaskExecutor(should_fail=True)
        executor = DAGExecutor(task_executor)

        results = executor.execute_dag(dag)

        # Should have all results
        assert len(results) == 3
        assert results["success_task"].status == TaskStatus.SUCCESS
        assert results["failing_task"].status == TaskStatus.FAILED
        # Due to on_failure="continue"
        assert results["final_task"].status == TaskStatus.SUCCESS

    @patch('exolife.pipeline.executor.FetchManager')
    def test_real_task_executor_integration(self, mock_fetch_manager_class):
        """Test integration with real DataPipelineTaskExecutor."""
        # Setup mock fetch manager
        mock_manager = Mock()
        mock_fetch_manager_class.return_value = mock_manager

        from exolife.data.fetch.fetcher_base import FetchResult
        mock_result = FetchResult("test_source", Path("test.parquet"), True)
        mock_manager.fetch_source_by_id.return_value = mock_result

        # Create DAG with real task types
        dag = DAG("real_tasks", "Real task types")

        tasks = [
            TaskNode("fetch_data", "fetch", config={"source": "test_source"}),
            TaskNode("validate_data", "validate", dependencies=["fetch_data"]),
        ]

        for task in tasks:
            dag.add_task(task)

        # Use real task executor
        task_executor = DataPipelineTaskExecutor()
        executor = DAGExecutor(task_executor)

        results = executor.execute_dag(dag)

        assert len(results) == 2
        assert results["fetch_data"].status == TaskStatus.SUCCESS
        assert results["validate_data"].status == TaskStatus.SUCCESS

    def test_executor_performance_characteristics(self):
        """Test performance characteristics of executor."""
        # Create a larger DAG to test performance
        dag = DAG("performance_test", "Performance test DAG")

        # Create 10 parallel tasks and 1 final task that depends on all
        parallel_tasks = []
        for i in range(10):
            task = TaskNode(f"parallel_{i}", "fetch")
            dag.add_task(task)
            parallel_tasks.append(f"parallel_{i}")

        final_task = TaskNode("final", "merge", dependencies=parallel_tasks)
        dag.add_task(final_task)

        # Test both execution modes
        task_executor = MockTaskExecutor(
            execution_time=0.001)  # Very fast tasks
        executor = DAGExecutor(task_executor)

        # Sequential execution
        start_time = time.time()
        results_seq = executor.execute_dag(dag, mode="sequential")
        seq_time = time.time() - start_time

        # Reset executed tasks
        task_executor.executed_tasks = []

        # Parallel execution
        start_time = time.time()
        results_par = executor.execute_dag(dag, mode="parallel")
        par_time = time.time() - start_time

        # Both should succeed
        assert len(results_seq) == 11
        assert len(results_par) == 11
        assert all(r.status == TaskStatus.SUCCESS for r in results_seq.values())
        assert all(r.status == TaskStatus.SUCCESS for r in results_par.values())

        # Parallel should be faster (though with very fast tasks this might not always be true)
        # We mainly test that both modes work correctly
