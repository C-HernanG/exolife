"""
Tests for the ExoLife DAG (Directed Acyclic Graph) module.
"""

import pytest
import yaml
from pathlib import Path

from exolife.pipeline.dag import (
    DAG, TaskNode, TaskStatus, TaskResult, CycleDetectedError,
    load_dag_from_yaml, save_dag_to_yaml
)


class TestTaskNode:
    """Test cases for TaskNode class."""

    def test_task_node_creation(self):
        """Test basic TaskNode creation."""
        task = TaskNode(
            task_id="test_task",
            task_type="fetch",
            config={"source": "test"},
            dependencies=["dep1", "dep2"],
            retries=3,
            timeout=60.0,
            on_failure="skip"
        )

        assert task.task_id == "test_task"
        assert task.task_type == "fetch"
        assert task.config == {"source": "test"}
        assert task.dependencies == ["dep1", "dep2"]
        assert task.retries == 3
        assert task.timeout == 60.0
        assert task.on_failure == "skip"

    def test_task_node_defaults(self):
        """Test TaskNode with default values."""
        task = TaskNode(task_id="simple_task", task_type="process")

        assert task.task_id == "simple_task"
        assert task.task_type == "process"
        assert task.config == {}
        assert task.dependencies == []
        assert task.retries == 0
        assert task.timeout is None
        assert task.on_failure == "fail"

    def test_task_node_validation(self):
        """Test TaskNode validation."""
        # Empty task_id should raise ValueError
        with pytest.raises(ValueError, match="Task ID cannot be empty"):
            TaskNode(task_id="", task_type="fetch")

        # Empty task_type should raise ValueError
        with pytest.raises(ValueError, match="Task type cannot be empty"):
            TaskNode(task_id="test", task_type="")

        # Negative retries should raise ValueError
        with pytest.raises(ValueError, match="Retries must be non-negative"):
            TaskNode(task_id="test", task_type="fetch", retries=-1)


class TestTaskResult:
    """Test cases for TaskResult class."""

    def test_task_result_creation(self):
        """Test TaskResult creation."""
        result = TaskResult(
            task_id="test_task",
            status=TaskStatus.SUCCESS,
            output={"rows": 100},
            execution_time=5.5,
            metadata={"source": "test"}
        )

        assert result.task_id == "test_task"
        assert result.status == TaskStatus.SUCCESS
        assert result.output == {"rows": 100}
        assert result.execution_time == 5.5
        assert result.metadata == {"source": "test"}

    def test_task_result_defaults(self):
        """Test TaskResult with default values."""
        result = TaskResult(task_id="test", status=TaskStatus.PENDING)

        assert result.task_id == "test"
        assert result.status == TaskStatus.PENDING
        assert result.output is None
        assert result.error_message is None
        assert result.execution_time == 0.0
        assert result.metadata == {}


class TestDAG:
    """Test cases for DAG class."""

    def test_dag_creation(self):
        """Test basic DAG creation."""
        dag = DAG("test_dag", "Test DAG description")

        assert dag.dag_id == "test_dag"
        assert dag.description == "Test DAG description"
        assert len(dag.nodes) == 0
        assert dag.name == "test_dag"  # Alias for dag_id
        assert dag.tasks == dag.nodes  # Alias for nodes

    def test_add_task(self, sample_dag):
        """Test adding tasks to DAG."""
        assert len(sample_dag.nodes) == 3
        assert "fetch_data" in sample_dag.nodes
        assert "process_data" in sample_dag.nodes
        assert "merge_data" in sample_dag.nodes

    def test_add_duplicate_task(self, sample_dag):
        """Test that adding duplicate task raises error."""
        duplicate_task = TaskNode(task_id="fetch_data", task_type="fetch")

        with pytest.raises(ValueError, match="Task 'fetch_data' already exists"):
            sample_dag.add_task(duplicate_task)

    def test_add_dependency(self, sample_dag):
        """Test adding dependencies between tasks."""
        # Add a new task
        new_task = TaskNode(task_id="export_data", task_type="export")
        sample_dag.add_task(new_task)

        # Add dependency
        sample_dag.add_dependency("export_data", "merge_data")

        assert "merge_data" in sample_dag.nodes["export_data"].dependencies

    def test_add_dependency_errors(self, sample_dag):
        """Test error cases for adding dependencies."""
        # Non-existent task
        with pytest.raises(ValueError, match="Task 'nonexistent' not found"):
            sample_dag.add_dependency("nonexistent", "fetch_data")

        # Non-existent dependency
        with pytest.raises(ValueError, match="Dependency task 'nonexistent' not found"):
            sample_dag.add_dependency("fetch_data", "nonexistent")

    def test_validation_success(self, sample_dag):
        """Test successful DAG validation."""
        errors = sample_dag.validate()
        assert errors == []
        assert sample_dag._validated is True

    def test_validation_missing_dependency(self):
        """Test validation with missing dependency."""
        dag = DAG("test", "Test")
        task = TaskNode(
            task_id="task1",
            task_type="fetch",
            dependencies=["missing_task"]
        )
        dag.add_task(task)

        errors = dag.validate()
        assert len(errors) == 1
        assert "depends on missing task 'missing_task'" in errors[0]

    def test_cycle_detection(self):
        """Test cycle detection in DAG."""
        dag = DAG("test", "Test")

        # Create a cycle: A -> B -> C -> A
        task_a = TaskNode(task_id="A", task_type="fetch", dependencies=["C"])
        task_b = TaskNode(task_id="B", task_type="process", dependencies=["A"])
        task_c = TaskNode(task_id="C", task_type="merge", dependencies=["B"])

        dag.add_task(task_a)
        dag.add_task(task_b)
        dag.add_task(task_c)

        errors = dag.validate()
        assert len(errors) == 1
        assert "Cycle detected" in errors[0]

    def test_execution_order(self, sample_dag):
        """Test getting execution order."""
        order = sample_dag.get_execution_order()

        # fetch_data should come first (no dependencies)
        assert order[0] == "fetch_data"

        # process_data should come before merge_data
        process_idx = order.index("process_data")
        merge_idx = order.index("merge_data")
        assert process_idx < merge_idx

    def test_topological_sort_alias(self, sample_dag):
        """Test that topological_sort is an alias for get_execution_order."""
        order1 = sample_dag.get_execution_order()
        order2 = sample_dag.topological_sort()
        assert order1 == order2

    def test_get_ready_tasks(self, sample_dag):
        """Test getting tasks ready for execution."""
        # Initially, only fetch_data should be ready
        ready = sample_dag.get_ready_tasks(set())
        assert ready == ["fetch_data"]

        # After fetch_data is executed, process_data should be ready
        ready = sample_dag.get_ready_tasks({"fetch_data"})
        assert ready == ["process_data"]

        # After both are executed, merge_data should be ready
        ready = sample_dag.get_ready_tasks({"fetch_data", "process_data"})
        assert ready == ["merge_data"]

    def test_get_task_dependencies(self, sample_dag):
        """Test getting task dependencies."""
        deps = sample_dag.get_task_dependencies("merge_data")
        assert deps == ["process_data"]

        deps = sample_dag.get_task_dependencies("fetch_data")
        assert deps == []

        # Non-existent task
        with pytest.raises(ValueError, match="Task 'nonexistent' not found"):
            sample_dag.get_task_dependencies("nonexistent")

    def test_get_task_dependents(self, sample_dag):
        """Test getting task dependents."""
        dependents = sample_dag.get_task_dependents("fetch_data")
        assert dependents == ["process_data"]

        dependents = sample_dag.get_task_dependents("process_data")
        assert dependents == ["merge_data"]

        dependents = sample_dag.get_task_dependents("merge_data")
        assert dependents == []

        # Non-existent task
        with pytest.raises(ValueError, match="Task 'nonexistent' not found"):
            sample_dag.get_task_dependents("nonexistent")

    def test_get_statistics(self, sample_dag):
        """Test getting DAG statistics."""
        stats = sample_dag.get_statistics()

        assert stats["dag_id"] == "test_dag"
        assert stats["total_tasks"] == 3
        assert stats["root_tasks"] == 1  # Only fetch_data has no dependencies
        assert stats["leaf_tasks"] == 1  # Only merge_data has no dependents
        # fetch_data -> process_data -> merge_data
        assert stats["max_depth"] == 3
        assert stats["validated"] is True  # Should be validated by now

    def test_to_dict(self, sample_dag):
        """Test converting DAG to dictionary."""
        dag_dict = sample_dag.to_dict()

        assert dag_dict["dag_id"] == "test_dag"
        assert dag_dict["description"] == "Test DAG for unit testing"
        assert len(dag_dict["tasks"]) == 3

        # Check task structure
        fetch_task = dag_dict["tasks"]["fetch_data"]
        assert fetch_task["task_type"] == "fetch"
        assert fetch_task["config"] == {"source": "test_source"}
        assert fetch_task["dependencies"] == []

    def test_from_dict(self, dag_yaml_config):
        """Test creating DAG from dictionary."""
        dag = DAG.from_dict(dag_yaml_config)

        assert dag.dag_id == "test_pipeline"
        assert dag.description == "Test pipeline for unit testing"
        assert len(dag.nodes) == 3

        # Check that dependencies are properly set
        assert dag.nodes["validate_data"].dependencies == ["fetch_nasa"]
        assert dag.nodes["export_results"].dependencies == ["validate_data"]

    def test_empty_dag(self):
        """Test behavior with empty DAG."""
        dag = DAG("empty", "Empty DAG")

        assert dag.get_execution_order() == []
        assert dag.get_ready_tasks(set()) == []

        stats = dag.get_statistics()
        assert stats["total_tasks"] == 0
        assert stats["max_depth"] == 0


class TestDAGFileOperations:
    """Test cases for DAG file I/O operations."""

    def test_save_and_load_dag(self, sample_dag, temp_dir):
        """Test saving and loading DAG to/from YAML file."""
        yaml_file = temp_dir / "test_dag.yaml"

        # Save DAG
        save_dag_to_yaml(sample_dag, yaml_file)
        assert yaml_file.exists()

        # Load DAG
        loaded_dag = load_dag_from_yaml(yaml_file)

        # Compare
        assert loaded_dag.dag_id == sample_dag.dag_id
        assert loaded_dag.description == sample_dag.description
        assert len(loaded_dag.nodes) == len(sample_dag.nodes)

        # Check task details
        for task_id in sample_dag.nodes:
            original = sample_dag.nodes[task_id]
            loaded = loaded_dag.nodes[task_id]

            assert loaded.task_id == original.task_id
            assert loaded.task_type == original.task_type
            assert loaded.config == original.config
            assert loaded.dependencies == original.dependencies

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading DAG from non-existent file."""
        nonexistent_file = temp_dir / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            load_dag_from_yaml(nonexistent_file)

    def test_load_empty_yaml(self, temp_dir):
        """Test loading DAG from empty YAML file."""
        empty_file = temp_dir / "empty.yaml"
        empty_file.write_text("")

        with pytest.raises(ValueError, match="Empty or invalid YAML"):
            load_dag_from_yaml(empty_file)

    def test_load_invalid_yaml(self, temp_dir):
        """Test loading DAG from invalid YAML file."""
        invalid_file = temp_dir / "invalid.yaml"
        invalid_file.write_text("invalid: yaml: content: [")

        with pytest.raises(Exception):  # YAML parsing error
            load_dag_from_yaml(invalid_file)

    def test_roundtrip_conversion(self, dag_yaml_config, temp_dir):
        """Test that DAG survives roundtrip conversion (dict -> DAG -> dict)."""
        # Create DAG from dict
        dag = DAG.from_dict(dag_yaml_config)

        # Convert back to dict
        roundtrip_dict = dag.to_dict()

        # Should be equivalent (order might differ)
        assert roundtrip_dict["dag_id"] == dag_yaml_config["dag_id"]
        assert roundtrip_dict["description"] == dag_yaml_config["description"]
        assert len(roundtrip_dict["tasks"]) == len(dag_yaml_config["tasks"])

        for task_id in dag_yaml_config["tasks"]:
            original = dag_yaml_config["tasks"][task_id]
            roundtrip = roundtrip_dict["tasks"][task_id]

            assert roundtrip["task_type"] == original["task_type"]
            assert roundtrip["dependencies"] == original["dependencies"]


class TestDAGValidationEdgeCases:
    """Test edge cases for DAG validation."""

    def test_self_dependency(self):
        """Test task depending on itself."""
        dag = DAG("test", "Test")
        task = TaskNode(
            task_id="self_dep",
            task_type="fetch",
            dependencies=["self_dep"]
        )
        dag.add_task(task)

        errors = dag.validate()
        assert len(errors) == 1
        assert "Cycle detected" in errors[0]

    def test_complex_cycle(self):
        """Test more complex cycle detection."""
        dag = DAG("test", "Test")

        # Create: A -> B -> C -> D -> B (cycle involving B, C, D)
        tasks = [
            TaskNode(task_id="A", task_type="fetch", dependencies=[]),
            TaskNode(task_id="B", task_type="process",
                     dependencies=["A", "D"]),
            TaskNode(task_id="C", task_type="transform", dependencies=["B"]),
            TaskNode(task_id="D", task_type="merge", dependencies=["C"]),
        ]

        for task in tasks:
            dag.add_task(task)

        errors = dag.validate()
        assert len(errors) == 1
        assert "Cycle detected" in errors[0]

    def test_multiple_validation_errors(self):
        """Test DAG with multiple validation errors."""
        dag = DAG("test", "Test")

        # Task with missing dependency + cycle
        task1 = TaskNode(task_id="task1", task_type="fetch",
                         dependencies=["missing"])
        task2 = TaskNode(task_id="task2", task_type="process",
                         dependencies=["task3"])
        task3 = TaskNode(task_id="task3", task_type="merge",
                         dependencies=["task2"])

        dag.add_task(task1)
        dag.add_task(task2)
        dag.add_task(task3)

        errors = dag.validate()
        assert len(errors) == 2  # Missing dependency + cycle

        # Check both error types are present
        error_text = " ".join(errors)
        assert "missing task" in error_text
        assert "Cycle detected" in error_text
