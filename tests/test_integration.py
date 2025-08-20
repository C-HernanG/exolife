"""
Integration tests for the ExoLife project.

These tests verify that different components work together correctly.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, Mock
import yaml
import tempfile

from exolife.settings import Settings
from exolife.data.utils import load_config, get_data_source
from exolife.data.fetch.manager import FetchManager
from exolife.pipeline.dag import load_dag_from_yaml
from exolife.pipeline.executor import DAGExecutor, DataPipelineTaskExecutor


class TestDataIntegration:
    """Integration tests for data components."""

    @pytest.mark.integration
    def test_complete_data_source_workflow(self, mock_source_configs, test_settings):
        """Test complete data source configuration and loading workflow."""
        with patch('exolife.data.utils.settings', test_settings):
            # Load configuration
            config = load_config()
            assert "data_sources" in config
            assert len(config["data_sources"]) >= 1

            # Get specific source
            source = get_data_source("nasa_exoplanet_archive")
            assert source.id == "nasa_exoplanet_archive"
            assert source.download_url is not None

    @pytest.mark.integration
    @patch('exolife.data.fetch.fetchers.http_fetcher.stream_download')
    @patch('exolife.data.fetch.fetchers.http_fetcher.write_generic')
    def test_fetch_manager_integration(self, mock_write, mock_download,
                                       mock_source_configs, test_settings, sample_csv_content):
        """Test FetchManager with real configuration."""
        # Setup mocks
        mock_download.return_value = sample_csv_content.encode('utf-8')
        sample_df = pd.DataFrame({'pl_name': ['Test'], 'st_teff': [5000]})
        mock_write.return_value = sample_df

        with patch('exolife.data.utils.settings', test_settings):
            with patch('exolife.data.fetch.manager.settings', test_settings):
                # Create manager and fetch source
                manager = FetchManager()
                result = manager.fetch_source_by_id("nasa_exoplanet_archive")

                assert result.success is True
                assert result.source_id == "nasa_exoplanet_archive"

    @pytest.mark.integration
    def test_hz_calculation_integration(self, sample_dataframe, mock_hz_config, test_settings):
        """Test HZ calculation with real stellar data."""
        from exolife.data.utils import add_hz_edges_to_df

        with patch('exolife.data.utils.settings', test_settings):
            df = sample_dataframe.copy()

            # Add HZ boundaries
            result_df = add_hz_edges_to_df(df)

            # Verify results
            assert "hz_inner" in result_df.columns
            assert "hz_outer" in result_df.columns
            assert not result_df["hz_inner"].isna().all()
            assert not result_df["hz_outer"].isna().all()
            assert all(result_df["hz_inner"] < result_df["hz_outer"])


class TestPipelineIntegration:
    """Integration tests for pipeline components."""

    @pytest.mark.integration
    def test_dag_yaml_loading_and_execution(self, temp_dir, dag_yaml_config):
        """Test loading DAG from YAML and executing it."""
        # Save DAG to file
        dag_file = temp_dir / "test_dag.yaml"
        with open(dag_file, 'w') as f:
            yaml.dump(dag_yaml_config, f)

        # Load DAG
        dag = load_dag_from_yaml(dag_file)
        assert dag.dag_id == dag_yaml_config["dag_id"]
        assert len(dag.nodes) == len(dag_yaml_config["tasks"])

        # Validate DAG
        errors = dag.validate()
        assert errors == []

        # Execute with mock task executor
        from tests.test_executor import MockTaskExecutor
        task_executor = MockTaskExecutor()
        executor = DAGExecutor(task_executor)

        results = executor.execute_dag(dag)
        assert len(results) == len(dag_yaml_config["tasks"])
        assert all(r.success for r in results.values()
                   if hasattr(r, 'success'))

    @pytest.mark.integration
    @patch('exolife.pipeline.executor.FetchManager')
    def test_real_pipeline_task_execution(self, mock_fetch_manager_class,
                                          mock_source_configs, test_settings):
        """Test executing real pipeline tasks."""
        # Setup mock fetch manager
        from exolife.data.fetch.fetcher_base import FetchResult

        mock_manager = Mock()
        mock_fetch_manager_class.return_value = mock_manager
        mock_result = FetchResult("test_source", Path(
            "test.parquet"), True, rows_fetched=100)
        mock_manager.fetch_source_by_id.return_value = mock_result

        # Create task executor
        from exolife.pipeline.dag import TaskNode
        executor = DataPipelineTaskExecutor()

        # Test fetch task
        fetch_task = TaskNode("fetch_test", "fetch", config={
                              "source": "test_source"})
        result = executor.execute_task(fetch_task)

        assert result.status.value == "success"
        assert result.output["fetch_result"] == mock_result

        # Test validate task
        validate_task = TaskNode("validate_test", "validate", config={
                                 "check_completeness": True})
        result = executor.execute_task(validate_task)

        assert result.status.value == "success"

    @pytest.mark.integration
    def test_full_pipeline_workflow(self, temp_dir, test_settings):
        """Test complete pipeline workflow from configuration to execution."""
        # Create minimal DAG configuration
        dag_config = {
            "dag_id": "integration_test",
            "description": "Integration test pipeline",
            "tasks": {
                "fetch_data": {
                    "task_type": "fetch",
                    "config": {"source": "test_source"},
                    "dependencies": []
                },
                "validate_data": {
                    "task_type": "validate",
                    "config": {"check_completeness": True},
                    "dependencies": ["fetch_data"]
                }
            }
        }

        # Save to file
        dag_file = temp_dir / "integration_dag.yaml"
        with open(dag_file, 'w') as f:
            yaml.dump(dag_config, f)

        # Load and execute
        dag = load_dag_from_yaml(dag_file)

        # Use mock task executor for integration test
        from tests.test_executor import MockTaskExecutor
        task_executor = MockTaskExecutor()
        executor = DAGExecutor(task_executor)

        # Execute pipeline
        results = executor.execute_dag(dag, mode="sequential")

        # Verify complete pipeline execution
        assert len(results) == 2
        assert "fetch_data" in results
        assert "validate_data" in results
        assert all(r.status.value == "success" for r in results.values())

        # Verify execution order
        assert task_executor.executed_tasks == ["fetch_data", "validate_data"]


class TestCLIIntegration:
    """Integration tests for CLI components."""

    @pytest.mark.integration
    def test_cli_dag_command_integration(self, temp_dir, dag_yaml_config):
        """Test CLI DAG commands with real configuration."""
        from click.testing import CliRunner
        from exolife.plugins.cli.dag import cli as dag_cli

        # Create DAG file
        dag_file = temp_dir / "cli_test.yaml"
        with open(dag_file, 'w') as f:
            yaml.dump(dag_yaml_config, f)

        runner = CliRunner()

        # Test validate command
        result = runner.invoke(dag_cli, ['validate', str(dag_file)])
        assert result.exit_code == 0
        assert "DAG validation passed" in result.output

        # Test info command
        result = runner.invoke(dag_cli, ['info', str(dag_file)])
        assert result.exit_code == 0
        assert dag_yaml_config["dag_id"] in result.output

        # Test run with dry-run
        result = runner.invoke(dag_cli, ['run', str(dag_file), '--dry-run'])
        assert result.exit_code == 0
        assert "Dry run completed successfully" in result.output

    @pytest.mark.integration
    def test_cli_main_integration(self):
        """Test main CLI integration with plugins."""
        from click.testing import CliRunner
        from exolife.cli import main

        runner = CliRunner()

        # Test main help
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert "ExoLife CLI" in result.output

        # Test that DAG commands are loaded
        result = runner.invoke(main, ['dag', '--help'])
        assert result.exit_code == 0
        assert "workflow orchestration" in result.output


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @pytest.mark.integration
    @patch('exolife.data.fetch.fetchers.http_fetcher.stream_download')
    @patch('exolife.data.fetch.fetchers.http_fetcher.write_generic')
    def test_complete_exolife_workflow(self, mock_write, mock_download,
                                       mock_source_configs, mock_hz_config,
                                       test_settings, sample_csv_content):
        """Test complete ExoLife workflow from data fetch to analysis."""
        # Setup mocks
        mock_download.return_value = sample_csv_content.encode('utf-8')
        stellar_df = pd.DataFrame({
            'pl_name': ['Kepler-442 b', 'K2-18 b'],
            'st_teff': [4402.0, 3457.0],
            'st_lum': [-0.377, -1.255],
            'pl_rade': [1.34, 2.3],
            'gaia_id': [1234567890123456, 2345678901234567]
        })
        mock_write.return_value = stellar_df

        with patch('exolife.data.utils.settings', test_settings):
            with patch('exolife.data.fetch.manager.settings', test_settings):
                # Step 1: Fetch data
                manager = FetchManager()
                result = manager.fetch_source_by_id("nasa_exoplanet_archive")
                assert result.success is True

                # Step 2: Process data (calculate HZ)
                from exolife.data.utils import add_hz_edges_to_df
                df_with_hz = add_hz_edges_to_df(stellar_df)

                # Verify HZ calculation
                assert "hz_inner" in df_with_hz.columns
                assert "hz_outer" in df_with_hz.columns

                # Step 3: Save processed data
                from exolife.data.utils import write_interim
                write_interim(df_with_hz, "processed_exoplanets", "final")

                # Step 4: Read processed data
                from exolife.data.utils import read_interim
                final_df = read_interim("processed_exoplanets", "final")

                # Verify complete workflow
                assert len(final_df) == 2
                assert all(col in final_df.columns for col in [
                           'pl_name', 'hz_inner', 'hz_outer'])
                assert not final_df["hz_inner"].isna().all()

    @pytest.mark.integration
    def test_settings_integration_across_modules(self, temp_dir):
        """Test that settings work consistently across all modules."""
        # Create settings
        settings = Settings(
            root_dir=temp_dir,
            max_workers=8,
            log_level="DEBUG"
        )
        settings.create_directories()

        # Test that directories exist
        assert settings.data_dir.exists()
        assert settings.raw_dir.exists()
        assert settings.interim_dir.exists()
        assert settings.processed_dir.exists()

        # Test settings are accessible from different modules
        with patch('exolife.data.utils.settings', settings):
            with patch('exolife.data.fetch.manager.settings', settings):
                # Should be able to use settings consistently
                from exolife.data.utils import load_config
                from exolife.data.fetch.manager import FetchManager

                # Both should work with the same settings
                config = load_config()  # Uses settings.root_dir for config path
                manager = FetchManager()  # Uses settings for directory configuration

                assert manager.settings == settings

    @pytest.mark.integration
    def test_error_handling_integration(self, temp_dir):
        """Test error handling across integrated components."""
        # Test with invalid DAG
        invalid_dag = {
            "dag_id": "invalid_test",
            "tasks": {
                "broken_task": {
                    "task_type": "fetch",
                    "dependencies": ["missing_dependency"]
                }
            }
        }

        dag_file = temp_dir / "invalid.yaml"
        with open(dag_file, 'w') as f:
            yaml.dump(invalid_dag, f)

        # Should handle gracefully
        dag = load_dag_from_yaml(dag_file)
        errors = dag.validate()

        assert len(errors) > 0
        assert "missing_dependency" in errors[0]

        # CLI should also handle this gracefully
        from click.testing import CliRunner
        from exolife.plugins.cli.dag import cli as dag_cli

        runner = CliRunner()
        result = runner.invoke(dag_cli, ['validate', str(dag_file)])

        assert result.exit_code == 0  # Command succeeds but reports validation error
        assert "DAG validation failed" in result.output
