"""
Fixtures and test configuration for the ExoLife test suite.
"""

import tempfile
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import pytest
import yaml

from exolife.settings import Settings
from exolife.data.utils import DataSource
from exolife.data.fetch.fetcher_base import DataSourceConfig
from exolife.pipeline.dag import DAG, TaskNode


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_settings(temp_dir):
    """Create test settings with temporary directories."""
    settings = Settings(
        root_dir=temp_dir,
        data_dir=temp_dir / "data",
        raw_dir=temp_dir / "data" / "raw",
        interim_dir=temp_dir / "data" / "interim",
        processed_dir=temp_dir / "data" / "processed",
        log_level="DEBUG",
        max_workers=2,
        request_timeout=30,
    )
    settings.create_directories()
    return settings


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'pl_name': ['Kepler-442 b', 'K2-18 b', 'TRAPPIST-1 e'],
        'st_teff': [4402.0, 3457.0, 2566.0],
        'st_lum': [-0.377, -1.255, -3.255],  # log values
        'pl_orbper': [112.3, 32.9, 6.1],
        'pl_rade': [1.34, 2.3, 0.92],
        'pl_masse': [2.3, 8.6, 0.77],
        'st_rad': [0.61, 0.41, 0.121],
        'st_mass': [0.68, 0.45, 0.089],
        'gaia_id': [1234567890123456, 2345678901234567, 3456789012345678]
    })


@pytest.fixture
def mock_data_source():
    """Create a mock DataSource configuration."""
    return DataSource(
        id="test_source",
        name="Test Data Source",
        description="A test data source for unit testing",
        download_url="https://example.com/test_data.csv",
        columns_to_keep=["pl_name", "st_teff", "pl_rade"],
        primary_keys=["pl_name"],
        join_keys={"gaia": ["gaia_id"]}
    )


@pytest.fixture
def mock_data_source_config():
    """Create a mock DataSourceConfig for fetchers."""
    return DataSourceConfig(
        id="test_source",
        name="Test Data Source",
        description="A test data source for unit testing",
        download_url="https://example.com/test_data.csv",
        columns_to_keep=["pl_name", "st_teff", "pl_rade"],
        primary_keys=["pl_name"],
        join_keys={"gaia": ["gaia_id"]},
        format="csv",
        timeout=30
    )


@pytest.fixture
def sample_dag():
    """Create a sample DAG for testing."""
    dag = DAG("test_dag", "Test DAG for unit testing")

    # Add tasks
    task1 = TaskNode(
        task_id="fetch_data",
        task_type="fetch",
        config={"source": "test_source"},
        dependencies=[]
    )

    task2 = TaskNode(
        task_id="process_data",
        task_type="preprocess",
        config={"operation": "clean"},
        dependencies=["fetch_data"]
    )

    task3 = TaskNode(
        task_id="merge_data",
        task_type="merge",
        config={"strategy": "baseline"},
        dependencies=["process_data"]
    )

    dag.add_task(task1)
    dag.add_task(task2)
    dag.add_task(task3)

    return dag


@pytest.fixture
def dag_yaml_config():
    """Sample DAG YAML configuration."""
    return {
        "dag_id": "test_pipeline",
        "description": "Test pipeline for unit testing",
        "tasks": {
            "fetch_nasa": {
                "task_type": "fetch",
                "config": {"source": "nasa_exoplanet_archive"},
                "dependencies": [],
                "retries": 2,
                "timeout": 300
            },
            "validate_data": {
                "task_type": "validate",
                "config": {"check_completeness": True},
                "dependencies": ["fetch_nasa"],
                "retries": 1,
                "timeout": 120
            },
            "export_results": {
                "task_type": "export",
                "config": {"format": "parquet"},
                "dependencies": ["validate_data"],
                "retries": 1,
                "timeout": 60
            }
        }
    }


@pytest.fixture
def mock_hz_config(temp_dir):
    """Create mock HZ configuration file."""
    hz_config = {
        "RecentVenus": [1.766, 2.136e-4, 2.533e-8, -1.332e-11, -3.097e-15],
        "RunawayGreenhouse": [1.107, 1.332e-4, 1.580e-8, -8.308e-12, -1.931e-15],
        "MaximumGreenhouse": [0.356, 6.171e-5, 1.698e-9, -2.442e-12, -8.023e-16],
        "EarlyMars": [0.320, 5.547e-5, 1.526e-9, -2.874e-12, -5.011e-16]
    }

    config_dir = temp_dir / "config" / "constants"
    config_dir.mkdir(parents=True, exist_ok=True)

    hz_file = config_dir / "hz.yml"
    with open(hz_file, 'w') as f:
        yaml.dump(hz_config, f)

    return hz_file


@pytest.fixture
def mock_source_configs(temp_dir):
    """Create mock source configuration files."""
    sources_dir = temp_dir / "config" / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    # NASA source config
    nasa_config = {
        "id": "nasa_exoplanet_archive",
        "name": "NASA Exoplanet Archive",
        "description": "Planetary systems composite parameters",
        "download_url": "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI",
        "columns_to_keep": ["pl_name", "st_teff", "pl_rade"],
        "primary_keys": ["pl_name"],
        "format": "csv"
    }

    with open(sources_dir / "nasa_exoplanet_archive.yml", 'w') as f:
        yaml.dump(nasa_config, f)

    # GAIA source config
    gaia_config = {
        "id": "gaia_dr3_astrophysical_parameters",
        "name": "Gaia DR3 Astrophysical Parameters",
        "description": "Stellar parameters from Gaia DR3",
        "download_url": "https://gea.esac.esa.int/tap-server/tap/sync",
        "adql": "SELECT source_id, teff_gspphot FROM gaiadr3.astrophysical_parameters WHERE source_id IN (<GAIA_ID_LIST>)",
        "columns_to_keep": ["source_id", "teff_gspphot"],
        "primary_keys": ["source_id"],
        "format": "csv"
    }

    with open(sources_dir / "gaia_dr3_astrophysical_parameters.yml", 'w') as f:
        yaml.dump(gaia_config, f)

    return sources_dir


@pytest.fixture
def sample_csv_content():
    """Sample CSV content for testing downloads."""
    return """pl_name,st_teff,pl_rade,pl_masse
Kepler-442 b,4402.0,1.34,2.3
K2-18 b,3457.0,2.3,8.6
TRAPPIST-1 e,2566.0,0.92,0.77
"""


@pytest.fixture
def invalid_dag_config():
    """DAG configuration with validation errors."""
    return {
        "dag_id": "invalid_dag",
        "tasks": {
            "task_with_missing_dep": {
                "task_type": "fetch",
                "dependencies": ["nonexistent_task"]
            },
            "task1": {
                "task_type": "process",
                "dependencies": ["task2"]
            },
            "task2": {
                "task_type": "merge",
                "dependencies": ["task1"]  # Creates cycle
            }
        }
    }
