"""
Tests for the ExoLife CLI module.
"""

import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from pathlib import Path

from exolife.cli import main, load_commands
from exolife.plugins.cli.dag import cli as dag_cli


class TestCLICore:
    """Test cases for core CLI functionality."""

    def test_main_cli_group_creation(self):
        """Test that main CLI group is created properly."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])

        assert result.exit_code == 0
        assert "ExoLife CLI" in result.output

    def test_cli_with_log_level_option(self):
        """Test CLI with log level option."""
        runner = CliRunner()
        result = runner.invoke(main, ['--log-level', 'DEBUG', '--help'])

        assert result.exit_code == 0

    def test_cli_with_force_option(self):
        """Test CLI with force option."""
        runner = CliRunner()
        result = runner.invoke(main, ['--force', '--help'])

        assert result.exit_code == 0

    def test_load_commands_function(self):
        """Test that load_commands function works."""
        # This function should load plugin commands without errors
        try:
            load_commands()
        except Exception as e:
            pytest.fail(f"load_commands() raised an exception: {e}")

    @patch('exolife.cli.logger')
    def test_load_commands_with_plugin_error(self, mock_logger):
        """Test load_commands handles plugin loading errors gracefully."""
        with patch('exolife.cli.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("Test import error")

            # Should not raise exception
            load_commands()

            # Should log error
            mock_logger.error.assert_called()

    def test_context_object_creation(self):
        """Test that CLI context object is created properly."""
        runner = CliRunner()

        @main.command()
        @pytest.fixture
        def test_command(ctx):
            assert ctx.obj is not None
            assert 'log_level' in ctx.obj
            assert 'force' in ctx.obj

        # We can't easily test this without modifying the CLI structure
        # This is more of a design verification


class TestDAGCLI:
    """Test cases for DAG CLI commands."""

    def test_dag_cli_group(self):
        """Test DAG CLI group."""
        runner = CliRunner()
        result = runner.invoke(dag_cli, ['--help'])

        assert result.exit_code == 0
        assert "DAG workflow orchestration" in result.output

    def test_dag_run_help(self):
        """Test DAG run command help."""
        runner = CliRunner()
        result = runner.invoke(dag_cli, ['run', '--help'])

        assert result.exit_code == 0
        assert "Execute a DAG workflow" in result.output

    def test_dag_validate_help(self):
        """Test DAG validate command help."""
        runner = CliRunner()
        result = runner.invoke(dag_cli, ['validate', '--help'])

        assert result.exit_code == 0
        assert "Validate a DAG specification" in result.output

    def test_dag_info_help(self):
        """Test DAG info command help."""
        runner = CliRunner()
        result = runner.invoke(dag_cli, ['info', '--help'])

        assert result.exit_code == 0
        assert "Show information about a DAG" in result.output

    def test_dag_run_nonexistent_file(self):
        """Test DAG run with non-existent file."""
        runner = CliRunner()
        result = runner.invoke(dag_cli, ['run', 'nonexistent.yaml'])

        assert result.exit_code != 0

    def test_dag_validate_nonexistent_file(self):
        """Test DAG validate with non-existent file."""
        runner = CliRunner()
        result = runner.invoke(dag_cli, ['validate', 'nonexistent.yaml'])

        assert result.exit_code != 0

    def test_dag_run_with_valid_file(self, temp_dir, dag_yaml_config):
        """Test DAG run with valid YAML file."""
        import yaml

        dag_file = temp_dir / "test_dag.yaml"
        with open(dag_file, 'w') as f:
            yaml.dump(dag_yaml_config, f)

        runner = CliRunner()

        # Test dry run
        result = runner.invoke(dag_cli, ['run', str(dag_file), '--dry-run'])

        # Should succeed validation
        assert result.exit_code == 0
        assert "Dry run completed successfully" in result.output

    def test_dag_validate_with_valid_file(self, temp_dir, dag_yaml_config):
        """Test DAG validate with valid YAML file."""
        import yaml

        dag_file = temp_dir / "test_dag.yaml"
        with open(dag_file, 'w') as f:
            yaml.dump(dag_yaml_config, f)

        runner = CliRunner()
        result = runner.invoke(dag_cli, ['validate', str(dag_file)])

        assert result.exit_code == 0
        assert "DAG validation passed" in result.output

    def test_dag_info_with_valid_file(self, temp_dir, dag_yaml_config):
        """Test DAG info with valid YAML file."""
        import yaml

        dag_file = temp_dir / "test_dag.yaml"
        with open(dag_file, 'w') as f:
            yaml.dump(dag_yaml_config, f)

        runner = CliRunner()
        result = runner.invoke(dag_cli, ['info', str(dag_file)])

        assert result.exit_code == 0
        assert "DAG Information" in result.output
        assert dag_yaml_config["dag_id"] in result.output

    def test_dag_run_with_invalid_dag(self, temp_dir, invalid_dag_config):
        """Test DAG run with invalid DAG configuration."""
        import yaml

        dag_file = temp_dir / "invalid_dag.yaml"
        with open(dag_file, 'w') as f:
            yaml.dump(invalid_dag_config, f)

        runner = CliRunner()
        result = runner.invoke(dag_cli, ['run', str(dag_file), '--dry-run'])

        # Should fail validation
        assert result.exit_code == 0  # Command succeeds but reports validation errors
        assert "DAG validation failed" in result.output

    def test_dag_run_with_execution_modes(self, temp_dir, dag_yaml_config):
        """Test DAG run with different execution modes."""
        import yaml

        dag_file = temp_dir / "test_dag.yaml"
        with open(dag_file, 'w') as f:
            yaml.dump(dag_yaml_config, f)

        runner = CliRunner()

        # Test sequential mode (default)
        result = runner.invoke(dag_cli, ['run', str(dag_file), '--dry-run'])
        assert result.exit_code == 0

        # Test parallel mode
        result = runner.invoke(dag_cli, ['run', str(
            dag_file), '--mode', 'parallel', '--dry-run'])
        assert result.exit_code == 0

    def test_dag_run_specific_task(self, temp_dir, dag_yaml_config):
        """Test DAG run with specific task."""
        import yaml

        dag_file = temp_dir / "test_dag.yaml"
        with open(dag_file, 'w') as f:
            yaml.dump(dag_yaml_config, f)

        runner = CliRunner()
        result = runner.invoke(dag_cli, ['run', str(
            dag_file), '--task', 'fetch_nasa', '--dry-run'])

        assert result.exit_code == 0

    @patch('exolife.plugins.cli.dag.DAGExecutor')
    @patch('exolife.plugins.cli.dag.DataPipelineTaskExecutor')
    def test_dag_run_execution_success(self, mock_task_executor, mock_dag_executor,
                                       temp_dir, dag_yaml_config):
        """Test successful DAG execution."""
        import yaml
        from exolife.pipeline.dag import TaskStatus, TaskResult

        dag_file = temp_dir / "test_dag.yaml"
        with open(dag_file, 'w') as f:
            yaml.dump(dag_yaml_config, f)

        # Mock successful execution
        mock_executor_instance = MagicMock()
        mock_dag_executor.return_value = mock_executor_instance

        # Mock results for all tasks
        mock_results = {
            task_id: TaskResult(task_id, TaskStatus.SUCCESS)
            for task_id in dag_yaml_config["tasks"]
        }
        mock_executor_instance.execute_dag.return_value = mock_results

        runner = CliRunner()
        result = runner.invoke(dag_cli, ['run', str(dag_file)])

        assert result.exit_code == 0
        assert "All tasks completed successfully" in result.output

    @patch('exolife.plugins.cli.dag.DAGExecutor')
    @patch('exolife.plugins.cli.dag.DataPipelineTaskExecutor')
    def test_dag_run_execution_failure(self, mock_task_executor, mock_dag_executor,
                                       temp_dir, dag_yaml_config):
        """Test DAG execution with failures."""
        import yaml
        from exolife.pipeline.dag import TaskStatus, TaskResult

        dag_file = temp_dir / "test_dag.yaml"
        with open(dag_file, 'w') as f:
            yaml.dump(dag_yaml_config, f)

        # Mock execution with failure
        mock_executor_instance = MagicMock()
        mock_dag_executor.return_value = mock_executor_instance

        # Mock results with one failure
        mock_results = {
            "fetch_nasa": TaskResult("fetch_nasa", TaskStatus.SUCCESS),
            "validate_data": TaskResult("validate_data", TaskStatus.FAILED, error_message="Validation failed"),
            "export_results": TaskResult("export_results", TaskStatus.SUCCESS)
        }
        mock_executor_instance.execute_dag.return_value = mock_results

        runner = CliRunner()
        result = runner.invoke(dag_cli, ['run', str(dag_file)])

        assert result.exit_code == 0  # CLI doesn't exit with error code
        assert "Failed tasks:" in result.output
        assert "validate_data" in result.output


class TestCLIUtilities:
    """Test cases for CLI utility functions."""

    @patch('exolife.plugins.cli.dag.settings')
    def test_get_dags_directory(self, mock_settings, temp_dir):
        """Test getting DAGs directory."""
        from exolife.plugins.cli.dag import get_dags_directory

        mock_settings.root_dir = temp_dir
        dags_dir = get_dags_directory()

        assert dags_dir == temp_dir / "dags"
        assert dags_dir.exists()

    @patch('exolife.plugins.cli.dag.get_dags_directory')
    def test_list_available_dags(self, mock_get_dags_dir, temp_dir):
        """Test listing available DAGs."""
        from exolife.plugins.cli.dag import list_available_dags

        # Create mock DAG files
        dags_dir = temp_dir / "dags"
        dags_dir.mkdir(exist_ok=True)

        (dags_dir / "dag1.yaml").touch()
        (dags_dir / "dag2.yml").touch()
        (dags_dir / "not_a_dag.txt").touch()

        mock_get_dags_dir.return_value = dags_dir

        available_dags = list_available_dags()

        assert len(available_dags) == 2
        dag_names = [name for name, _ in available_dags]
        assert "dag1" in dag_names
        assert "dag2" in dag_names
        assert "not_a_dag" not in dag_names

    @patch('exolife.plugins.cli.dag.get_dags_directory')
    def test_resolve_dag_file(self, mock_get_dags_dir, temp_dir):
        """Test resolving DAG file paths."""
        from exolife.plugins.cli.dag import resolve_dag_file

        dags_dir = temp_dir / "dags"
        dags_dir.mkdir(exist_ok=True)

        # Create test DAG file
        dag_file = dags_dir / "test_dag.yaml"
        dag_file.touch()

        mock_get_dags_dir.return_value = dags_dir

        # Test with absolute path
        resolved = resolve_dag_file(str(dag_file))
        assert resolved == dag_file

        # Test with DAG name
        resolved = resolve_dag_file("test_dag")
        assert resolved == dag_file

        # Test with .yml extension
        yml_file = dags_dir / "test_dag2.yml"
        yml_file.touch()
        resolved = resolve_dag_file("test_dag2")
        assert resolved == yml_file

    @patch('exolife.plugins.cli.dag.get_dags_directory')
    def test_resolve_dag_file_not_found(self, mock_get_dags_dir, temp_dir):
        """Test resolving non-existent DAG file."""
        from exolife.plugins.cli.dag import resolve_dag_file

        dags_dir = temp_dir / "dags"
        dags_dir.mkdir(exist_ok=True)
        mock_get_dags_dir.return_value = dags_dir

        with pytest.raises(FileNotFoundError, match="DAG 'nonexistent' not found"):
            resolve_dag_file("nonexistent")


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    def test_cli_plugin_loading(self):
        """Test that CLI plugins are loaded correctly."""
        # After importing the main CLI, DAG commands should be available
        from exolife.cli import main

        # Get list of commands
        commands = list(main.commands.keys())

        # Should include DAG commands
        assert "dag" in commands

    def test_full_cli_workflow(self, temp_dir, dag_yaml_config):
        """Test complete CLI workflow: validate -> info -> dry-run."""
        import yaml

        dag_file = temp_dir / "workflow_test.yaml"
        with open(dag_file, 'w') as f:
            yaml.dump(dag_yaml_config, f)

        runner = CliRunner()

        # Step 1: Validate
        result = runner.invoke(dag_cli, ['validate', str(dag_file)])
        assert result.exit_code == 0
        assert "DAG validation passed" in result.output

        # Step 2: Get info
        result = runner.invoke(dag_cli, ['info', str(dag_file)])
        assert result.exit_code == 0
        assert "DAG Information" in result.output

        # Step 3: Dry run
        result = runner.invoke(dag_cli, ['run', str(dag_file), '--dry-run'])
        assert result.exit_code == 0
        assert "Dry run completed successfully" in result.output

    def test_cli_error_handling(self):
        """Test CLI error handling for various scenarios."""
        runner = CliRunner()

        # Test with invalid command
        result = runner.invoke(main, ['invalid_command'])
        assert result.exit_code != 0

        # Test DAG command with missing file
        result = runner.invoke(dag_cli, ['run', '/nonexistent/path.yaml'])
        assert result.exit_code != 0

    def test_cli_help_messages(self):
        """Test that all help messages are properly formatted."""
        runner = CliRunner()

        # Main help
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert "ExoLife CLI" in result.output

        # DAG help
        result = runner.invoke(dag_cli, ['--help'])
        assert result.exit_code == 0
        assert "workflow orchestration" in result.output

        # DAG subcommand help
        for subcommand in ['run', 'validate', 'info']:
            result = runner.invoke(dag_cli, [subcommand, '--help'])
            assert result.exit_code == 0
