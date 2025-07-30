"""
DAG (Directed Acyclic Graph) implementation for pipeline orchestration.

This module provides a flexible workflow system where tasks can be defined
with dependencies and executed in the correct order.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """
    Status of a task in the DAG.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskResult:
    """
    Result of a task execution.
    """

    task_id: str
    status: TaskStatus
    output: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskNode:
    """
    A node in the DAG representing a single task.
    """

    task_id: str
    task_type: str
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    retries: int = 0
    timeout: Optional[float] = None
    on_failure: str = "fail"  # Options: "fail", "skip", "continue"

    def __post_init__(self):
        """
        Validate the task node configuration.
        """
        if not self.task_id:
            raise ValueError("Task ID cannot be empty")
        if not self.task_type:
            raise ValueError("Task type cannot be empty")
        if self.retries < 0:
            raise ValueError("Retries must be non-negative")


class CycleDetectedError(Exception):
    """
    Raised when a cycle is detected in the DAG.
    """

    pass


class DAG:
    """
    Directed Acyclic Graph for workflow orchestration.

    Provides functionality to define tasks with dependencies and execute
    them in the correct order while handling failures and retries.
    """

    def __init__(self, dag_id: str, description: str = ""):
        self.dag_id = dag_id
        self.description = description
        self.nodes: Dict[str, TaskNode] = {}
        self.execution_results: Dict[str, TaskResult] = {}
        self._validated = False

    @property
    def tasks(self) -> Dict[str, TaskNode]:
        """
        Get all tasks in the DAG. Alias for nodes for backward compatibility.
        """
        return self.nodes

    @property
    def name(self) -> str:
        """
        Get DAG name. Alias for dag_id.
        """
        return self.dag_id

    def add_task(self, task: TaskNode) -> None:
        """
        Add a task to the DAG.
        """
        if task.task_id in self.nodes:
            raise ValueError(f"Task '{task.task_id}' already exists in DAG")

        self.nodes[task.task_id] = task
        self._validated = False
        logger.debug(f"Added task '{task.task_id}' to DAG '{self.dag_id}'")

    def add_dependency(self, task_id: str, dependency_id: str) -> None:
        """
        Add a dependency between two tasks.
        """
        if task_id not in self.nodes:
            raise ValueError(f"Task '{task_id}' not found in DAG")
        if dependency_id not in self.nodes:
            raise ValueError(f"Dependency task '{dependency_id}' not found in DAG")

        if dependency_id not in self.nodes[task_id].dependencies:
            self.nodes[task_id].dependencies.append(dependency_id)
            self._validated = False
            logger.debug(f"Added dependency: '{task_id}' depends on '{dependency_id}'")

    def validate(self) -> List[str]:
        """
        Validate the DAG structure.

        Checks for:
        1. Cycle detection
        2. Missing dependencies
        3. Orphaned tasks (optional warning)

        Returns:
            List of validation error messages. Empty list if valid.
        """
        errors = []

        # Check for missing dependencies
        for task_id, task in self.nodes.items():
            for dep_id in task.dependencies:
                if dep_id not in self.nodes:
                    errors.append(
                        f"Task '{task_id}' depends on missing task '{dep_id}'"
                    )

        # Detect cycles using DFS
        try:
            self._detect_cycles()
        except CycleDetectedError as e:
            errors.append(str(e))

        if not errors:
            self._validated = True
            logger.info(f"DAG '{self.dag_id}' validated successfully")

        return errors

    def _detect_cycles(self) -> None:
        """
        Detect cycles in the DAG using DFS.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {task_id: WHITE for task_id in self.nodes}

        def dfs(task_id: str, path: List[str]) -> None:
            if colors[task_id] == GRAY:
                cycle_start = path.index(task_id)
                cycle = " -> ".join(path[cycle_start:] + [task_id])
                raise CycleDetectedError(f"Cycle detected: {cycle}")

            if colors[task_id] == BLACK:
                return

            colors[task_id] = GRAY
            path.append(task_id)

            for dep_id in self.nodes[task_id].dependencies:
                dfs(dep_id, path)

            path.pop()
            colors[task_id] = BLACK

        for task_id in self.nodes:
            if colors[task_id] == WHITE:
                dfs(task_id, [])

    def get_execution_order(self) -> List[str]:
        """
        Get the topological order for task execution.

        Returns a list of task IDs in the order they should be executed.
        """
        validation_errors = self.validate()
        if validation_errors:
            raise ValueError(f"DAG validation failed: {'; '.join(validation_errors)}")

        # Calculate in-degrees
        in_degree = {task_id: 0 for task_id in self.nodes}
        for task_id, task in self.nodes.items():
            for dep_id in task.dependencies:
                in_degree[task_id] += 1

        # Topological sort using Kahn's algorithm
        queue = deque([task_id for task_id, degree in in_degree.items() if degree == 0])
        execution_order = []

        while queue:
            current = queue.popleft()
            execution_order.append(current)

            # Reduce in-degree for dependent tasks
            for task_id, task in self.nodes.items():
                if current in task.dependencies:
                    in_degree[task_id] -= 1
                    if in_degree[task_id] == 0:
                        queue.append(task_id)

        if len(execution_order) != len(self.nodes):
            raise CycleDetectedError(
                "Unable to determine execution order due to cycles"
            )

        return execution_order

    def topological_sort(self) -> List[str]:
        """
        Get the topological sort of tasks. Alias for get_execution_order.

        Returns a list of task IDs in dependency-resolved order.
        """
        return self.get_execution_order()

    def get_ready_tasks(self, executed_tasks: Set[str]) -> List[str]:
        """
        Get tasks that are ready to execute (all dependencies completed).
        """
        ready_tasks = []

        for task_id, task in self.nodes.items():
            if task_id not in executed_tasks and all(
                dep_id in executed_tasks for dep_id in task.dependencies
            ):
                ready_tasks.append(task_id)

        return ready_tasks

    def get_task_dependencies(self, task_id: str) -> List[str]:
        """
        Get the dependencies for a specific task.
        """
        if task_id not in self.nodes:
            raise ValueError(f"Task '{task_id}' not found in DAG")
        return self.nodes[task_id].dependencies.copy()

    def get_task_dependents(self, task_id: str) -> List[str]:
        """
        Get tasks that depend on the specified task.
        """
        if task_id not in self.nodes:
            raise ValueError(f"Task '{task_id}' not found in DAG")

        dependents = []
        for other_task_id, other_task in self.nodes.items():
            if task_id in other_task.dependencies:
                dependents.append(other_task_id)

        return dependents

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get DAG statistics.
        """
        return {
            "dag_id": self.dag_id,
            "total_tasks": len(self.nodes),
            "root_tasks": len([t for t in self.nodes.values() if not t.dependencies]),
            "leaf_tasks": len(
                [
                    t
                    for t_id, t in self.nodes.items()
                    if not self.get_task_dependents(t_id)
                ]
            ),
            "max_depth": self._calculate_max_depth(),
            "validated": self._validated,
        }

    def _calculate_max_depth(self) -> int:
        """
        Calculate the maximum depth of the DAG.
        """
        if not self.nodes:
            return 0

        depths = {}

        def calculate_depth(task_id: str) -> int:
            if task_id in depths:
                return depths[task_id]

            task = self.nodes[task_id]
            if not task.dependencies:
                depth = 1
            else:
                depth = 1 + max(calculate_depth(dep_id) for dep_id in task.dependencies)

            depths[task_id] = depth
            return depth

        return max(calculate_depth(task_id) for task_id in self.nodes)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert DAG to dictionary representation.
        """
        return {
            "dag_id": self.dag_id,
            "description": self.description,
            "tasks": {
                task_id: {
                    "task_type": task.task_type,
                    "config": task.config,
                    "dependencies": task.dependencies,
                    "retries": task.retries,
                    "timeout": task.timeout,
                    "on_failure": task.on_failure,
                }
                for task_id, task in self.nodes.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DAG":
        """
        Create DAG from dictionary representation.
        """
        dag = cls(data["dag_id"], data.get("description", ""))

        for task_id, task_data in data["tasks"].items():
            task = TaskNode(
                task_id=task_id,
                task_type=task_data["task_type"],
                config=task_data.get("config", {}),
                dependencies=task_data.get("dependencies", []),
                retries=task_data.get("retries", 0),
                timeout=task_data.get("timeout"),
                on_failure=task_data.get("on_failure", "fail"),
            )
            dag.add_task(task)

        return dag


def load_dag_from_yaml(file_path: Union[str, Path]) -> DAG:
    """
    Load a DAG from a YAML specification file.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"DAG specification file not found: {file_path}")

    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Empty or invalid YAML file: {file_path}")

    return DAG.from_dict(data)


def save_dag_to_yaml(dag: DAG, file_path: Union[str, Path]) -> None:
    """
    Save a DAG to a YAML specification file.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        yaml.dump(dag.to_dict(), f, default_flow_style=False, indent=2)

    logger.info(f"DAG '{dag.dag_id}' saved to {file_path}")
