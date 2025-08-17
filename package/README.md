# ExoLife Package

This directory contains the main source code for the ExoLife project - a machine learning system for predicting exoplanet habitability to support astrobiological discovery.

## Structure

```
package/
├── exolife/                    # Main package
│   ├── __init__.py
│   ├── cli.py                  # Command-line interface
│   ├── settings.py             # Configuration settings
│   ├── data/                   # Data processing modules
│   │   ├── fetch/              # Data fetching components
│   │   ├── merge/              # Data merging strategies
│   │   ├── preprocess/         # Data preprocessing pipelines
│   │   └── utils.py            # Data utilities
│   ├── evaluation/             # Model evaluation components
│   ├── pipeline/               # DAG workflow management
│   ├── plugins/                # CLI plugin system
│   │   └── cli/                # CLI command plugins
│   └── training/               # Model training components
└── README.md                   # This file
```

## Key Components

### Command Line Interface (CLI)
- **`cli.py`**: Main CLI entry point with dynamic plugin loading
- **`plugins/cli/`**: Modular CLI commands for different operations
  - `fetch.py`: Data fetching commands
  - `merge.py`: Data merging commands
  - `config.py`: Configuration management

### Data Processing
- **`data/fetch/`**: Automated data retrieval from various astronomical databases
- **`data/merge/`**: Intelligent data fusion strategies for combining multiple sources
- **`data/preprocess/`**: Data cleaning, feature engineering, and quality filtering

### Pipeline Management
- **`pipeline/`**: DAG-based workflow orchestration for complex data processing pipelines

## Usage

The package is designed to be used via the command-line interface:

```bash
# Install the package in development mode
pip install -e .

# Fetch data from all sources
exolife fetch all

# Merge data using a specific strategy
exolife merge baseline

# Run a complete pipeline
exolife dag run config/dags/dagspec.yaml

# Show configuration
exolife config show
```

## Development

This package follows modern Python packaging standards and includes:
- Modular, plugin-based architecture
- Comprehensive logging and error handling
- Type hints and documentation
- Configuration management via `settings.py`
- Extensible command system

For more information about the project, see the main README.md in the project root.
