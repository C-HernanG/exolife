# 🌍 ExoLife: Estimating Exoplanet Habitability

<div align="center">

*Supporting astrobiological discovery through machine learning*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

ExoLife is a comprehensive machine learning framework designed to estimate exoplanet habitability using astrophysical data from multiple astronomical surveys. The project integrates observations from different catalogs to create a robust, uncertainty-aware habitability estimator that supports astrobiological research and discovery.

The framework employs advanced data harmonization techniques, cross-matching algorithms, physics-based feature engineering, and explainable machine learning models to provide scientifically interpretable habitability assessments for known exoplanets.

## Key Features

### Data Integration & Processing
- **Multi-Mission Catalog Integration**: Harmonizes 6+ major astronomical catalogs (KOI, TOI, NASA Exoplanet Archive, PHL, SWEET-Cat, Gaia DR3)
- **Intelligent Cross-Matching**: Advanced algorithms for matching objects across different naming conventions and coordinate systems
- **Data Quality Assurance**: Comprehensive validation and quality control throughout the pipeline

### Scientific Computing
- **Physics-Based Feature Engineering**: Derives key astrophysical parameters including stellar flux, equilibrium temperature, habitable zone distances, and tidal forces
- **Uncertainty Propagation**: Monte Carlo sampling techniques to handle observational uncertainties
- **Unit Standardization**: Automatic detection and conversion to consistent physical units

### Machine Learning & Analysis
- **Explainable AI**: SHAP values and feature importance analysis for scientific interpretability
- **Model Calibration**: Uncertainty-aware machine learning with confidence intervals
- **Interactive Notebooks**: Jupyter-based workflow for data exploration and model interpretation

### Development & Deployment
- **Containerized Environment**: Complete Docker-based development and deployment setup
- **RESTful API**: FastAPI server for model inference and integration
- **Comprehensive Testing**: Unit, integration, and performance testing suite
- **CI/CD Ready**: Automated quality checks and deployment workflows

📖 **Read more in the [ExoLife Paper](docs/Exolife.pdf)**

## Table of Contents

- [Overview](#overview)
- [Key Features](#-key-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Development Environment](#-development-environment)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Data Pipeline](#-data-pipeline)
- [API Reference](#-api-reference)
- [Development](#-development)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)

## Prerequisites

Before getting started, ensure you have the following installed:

- **Docker** and **Docker Compose** v2.0+ ([Installation Guide](https://docs.docker.com/get-docker/))
- **Make** (included on macOS/Linux, [Windows Installation](https://gnuwin32.sourceforge.net/packages/make.htm))
- **Git** for version control ([Download](https://git-scm.com/downloads))

> **Note**: ExoLife uses a Docker-first approach for reproducibility and ease of deployment. All development and execution happens within containerized environments.

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/C-HernanG/exolife.git
cd exolife
```

### Step 2: Build Development Environment

```bash
# Build and start Docker images and install the package
make build-base
make up-base
```

The development environment will be available at: **http://localhost:8888**

### Step 3: Alternative Environments

Depending on your use case, you can start different environments:

```bash
# API server for model inference
make build-api
make up-api             # Available at http://localhost:8000

# GPU-enabled training environment with MLflow tracking
make build-gpu
make up-gpu             # MLflow UI at http://localhost:5000

# Start all services
make build-all
make up-all

# Interactive shell for development
make shell              # Direct container access
```

## Development Environment

ExoLife provides several pre-configured environments for different workflows:

| Environment | Command | Port | Purpose |
|-------------|---------|------|---------|
| **Development** | `make up-base` | 8888 | JupyterLab with full development tools |
| **API Server** | `make up-api` | 8000 | FastAPI server for model serving |
| **Training** | `make up-gpu` | 5000 | GPU-enabled training with MLflow |
| **Shell** | `make shell` | - | Interactive container shell |

### Environment Features

- **Persistent Data**: All environments mount the project directory for persistent storage
- **Package Auto-Install**: The ExoLife package is automatically installed in development mode
- **GPU Support**: Training environment includes CUDA support for accelerated computing
- **MLflow Integration**: Experiment tracking and model registry for ML workflows

## Project Structure

The ExoLife project follows a modular architecture organized around data processing, machine learning, and deployment:

```
exolife/
├── api/                                     # FastAPI server implementation
├── config/                                  # Configuration management
│   ├── constants/                           # Feature definitions and physical constants
│   │   ├── drop_columns.yml                 # Columns to exclude from processing
│   │   ├── feature_engineering.yml          # Feature derivation parameters
│   │   ├── hz.yml                           # Habitable zone definitions
│   │   └── quality_filters.yml              # Data quality thresholds
│   ├── dags/                                # Pipeline workflow definitions (legacy)
│   ├── mergers/                             # Data merging configurations
│   └── sources/                             # External data source specifications
├── data/                                    # Data storage hierarchy
│   ├── raw/                                 # Original source data (KOI, TOI, Gaia, etc.)
│   ├── interim/                             # Intermediate processing results
│   └── processed/                           # Final processed datasets
├── docker/                                  # Docker configurations
│   ├── Dockerfile.base                      # Base development image
│   ├── Dockerfile.gpu                       # GPU-enabled training image
│   ├── Dockerfile.api                       # Production API image
│   └── entrypoints/                         # Container entry scripts
├── docs/                                    # Base documentation
├── notebooks/                               # Jupyter analysis notebooks
│   ├── 01_ingestion-harmonization/          # Data ingestion and catalog merging
│   ├── 02_quality-eda/                      # Quality assessment and EDA
│   ├── 03_feature_engineering-uncertainty/  # Physics-based features
│   ├── 04_selection-function_labels/        # Target variable definition
│   ├── 05_modeling-calibration/             # ML model development
│   ├── 06_interpretability-diagnostics/     # Model interpretation
│   └── 07_deployment-monitoring/            # Production deployment
├── package/exolife/                         # Core Python package
│   ├── data/                                # Data processing modules
│   │   ├── fetch/                           # Data acquisition utilities
│   │   ├── merge/                           # Cross-catalog merging
│   │   └── preprocess/                      # Data cleaning and preparation
│   ├── pipeline/                            # Workflow execution engine (legacy)
│   ├── plugins/                             # Extensible plugin system
│   ├── training/                            # Model training utilities
│   ├── cli.py                               # Command-line interface
│   └── settings.py                          # Configuration management
├── scripts/                                 # Utility scripts
├── tests/                                   # Comprehensive test suite
│   ├── test_*.py                            # Unit and integration tests
│   └── conftest.py                          # Test configurations
├── compose.yaml
├── Makefile
└── pyproject.toml
```

## Usage

### ExoLife Command-Line Interface

The `exolife` CLI provides comprehensive access to all framework functionality. The CLI is automatically available in all Docker environments.

#### Information Commands
```bash
exolife info               # Display package information and available components
exolife data status        # Check data availability and processing status
exolife config show        # Display current configuration settings
exolife --help             # Show all available commands and options
```

#### Data Management
```bash
exolife fetch <source>     # Fetch specific data source (e.g., 'koi', 'toi', 'gaia')
exolife fetch all          # Fetch all configured data sources
```

#### Workflow Execution
```bash
exolife dag run <dag.yaml> # Execute a workflow DAG configuration
```

#### Configuration Management
```bash
exolife config edit        # Edit configuration files interactively
```

### Essential Make Commands

The Makefile provides high-level automation for common development tasks:

```bash
# Environment Management
make help                  # Show all available commands with descriptions
make up-base               # Start JupyterLab development environment
make up-api                # Start FastAPI server for model inference
make shell                 # Access interactive container shell

# Development Workflow
make test                  # Run complete test suite
make test-unit             # Run unit tests only
make test-integration      # Run integration tests only
make lint                  # Code quality checks and formatting
make format                # Auto-format code with black and ruff

# Data Analysis
make analyze-data          # Analyze processed datasets

# Cleanup and Maintenance
make clean                 # Clean up build artifacts and temporary files
make clean-docker          # Remove Docker containers and images
make status                # Check project and Docker status
```

## Data Pipeline

ExoLife implements a comprehensive data processing pipeline organized into sequential stages:

### Stage 1: Ingestion & Harmonization
- **Catalog Snapshots**: Load and validate raw astronomical catalogs
- **Cross-Matching**: Match objects across different catalogs using identifiers and coordinates
- **Unit Standardization**: Convert all measurements to consistent physical units

### Stage 2: Quality Assessment & EDA
- **Schema Validation**: Ensure data consistency using Great Expectations
- **Statistical Analysis**: Comprehensive exploratory data analysis
- **Quality Filters**: Apply astrophysical constraints and quality thresholds

### Stage 3: Feature Engineering & Uncertainty
- **Physics-Based Features**: Derive equilibrium temperature, stellar flux, habitable zone metrics
- **Uncertainty Propagation**: Monte Carlo sampling for observational uncertainties
- **Feature Selection**: Identify most informative features for habitability prediction

### Stage 4: Selection Function & Labels
- **Target Definition**: Define habitability labels based on astrophysical criteria
- **Selection Bias**: Account for observational selection effects
- **Data Splitting**: Create training/validation/test sets with proper stratification

### Stage 5: Modeling & Calibration
- **Model Training**: Train ensemble models with cross-validation
- **Hyperparameter Optimization**: Automated parameter tuning
- **Uncertainty Calibration**: Ensure model confidence aligns with prediction accuracy

### Stage 6: Interpretability & Diagnostics
- **Feature Importance**: SHAP values and permutation importance analysis
- **Model Diagnostics**: Residual analysis and performance metrics
- **Scientific Validation**: Compare predictions with known astrophysical constraints

### Stage 7: Deployment & Monitoring
- **Model Serving**: Deploy models via FastAPI for real-time inference
- **Performance Monitoring**: Track model performance over time
- **Data Drift Detection**: Monitor for changes in input data distribution

> **Note**: Each stage is implemented as Jupyter notebooks in the `notebooks/` directory, providing interactive exploration and documentation of the entire workflow.

## API Reference

ExoLife provides a RESTful API built with FastAPI for model inference and integration with external systems.

### Starting the API Server

```bash
make up-api    # Start API server at http://localhost:8000
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message and API information |
| `/health` | GET | Health check endpoint for monitoring |
| `/docs` | GET | Interactive API documentation (Swagger UI) |
| `/redoc` | GET | Alternative API documentation |

### Example Usage

```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs
```

> **Note**: The current API provides basic infrastructure. Model prediction endpoints will be added as the ML pipeline matures.

## Development

### Development Workflow

1. **Environment Setup**
   ```bash
   make build-base         # Build Docker images
   make up-base            # Start development environment
   ```

2. **Daily Development**
   ```bash
   make test-unit          # Run unit tests during development
   make lint               # Check code quality
   make format             # Auto-format code
   ```

3. **Data Analysis**
   ```bash
   make analyze-data       # Analyze processed datasets
   make jupyter            # Alternative way to start Jupyter
   ```

4. **Debugging and Monitoring**
   ```bash
   make logs               # View Docker container logs
   make status             # Check service status
   make shell              # Access container shell for debugging
   ```

### Code Quality Standards

ExoLife maintains high code quality through automated tools:

- **Formatting**: Black (line length: 88 characters)
- **Linting**: Ruff for fast Python linting
- **Type Checking**: Static type hints throughout the codebase
- **Testing**: Pytest with comprehensive test coverage
- **Pre-commit Hooks**: Automated quality checks on commit

```bash
# Install pre-commit hooks (optional, for local development)
make pre-commit-install

# Run all quality checks
make quality-check    # Combines lint + test
```

### Project Configuration

Key configuration files:

- **`pyproject.toml`**: Python package configuration, dependencies, and tool settings
- **`compose.yaml`**: Docker Compose service definitions
- **`config/`**: Application-specific configurations for data sources and processing
- **`Makefile`**: Development workflow automation

## Testing

ExoLife includes a comprehensive testing suite to ensure reliability and correctness.

### Test Categories

| Test Type | Command | Description |
|-----------|---------|-------------|
| **Unit Tests** | `make test-unit` | Fast tests for individual components |
| **Integration Tests** | `make test-integration` | End-to-end workflow testing |
| **All Tests** | `make test` | Complete test suite |
| **Coverage** | `make test-coverage` | Tests with coverage reporting |
| **Watch Mode** | `make test-watch` | Continuous testing during development |

### Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_cli.py              # Command-line interface tests
├── test_dag.py              # Workflow execution tests
├── test_data_*.py           # Data processing tests
├── test_integration.py      # End-to-end integration tests
├── test_executor.py         # Pipeline execution tests
└── test_settings.py         # Configuration tests
```

### Running Tests

```bash
# Quick test run (fastest tests only)
make quick-test

# Verbose test output
make test-verbose

# Test with coverage report
make test-coverage

# Watch mode for development
make test-watch
```

### Test Coverage

Test coverage reports are generated in `htmlcov/` and can be viewed in a browser after running `make test-coverage`.

## Contributing

We welcome contributions to ExoLife! Whether you're fixing bugs, adding features, improving documentation, or enhancing the scientific methodology, your contributions are valuable.

### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/exolife.git
   cd exolife
   ```
3. **Set up the development environment**:
   ```bash
   make build-base
   make up-base
   ```

### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and ensure they meet our standards:
   ```bash
   make test           # Run all tests
   make lint           # Check code quality
   make format         # Auto-format code
   ```

3. **Write or update tests** for your changes

4. **Commit your changes** with a descriptive message:
   ```bash
   git commit -m "feat: add new habitability metric calculation"
   ```

5. **Push to your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** on GitHub

### Contribution Guidelines

#### Prerequisites

To ensure consistency across developers, ensure you have the following installed:

- **pre-commit**

```bash
pip install pre-commit
pre-commit install
```

#### Code Quality
- Follow PEP 8 style guidelines (enforced by Black and Ruff)
- Add type hints for all new functions and methods
- Write docstrings for public APIs
- Ensure all tests pass before submitting

#### Commit Messages
Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for adding tests
- `refactor:` for code refactoring

#### Scientific Contributions
- Validate new features against established astrophysical principles
- Include references to relevant scientific literature
- Document assumptions and limitations
- Provide example usage in notebooks

### Types of Contributions

- **Bug Reports**: Use GitHub issues with detailed reproduction steps
- **Feature Requests**: Propose new functionality with scientific justification
- **Documentation**: Improve README, docstrings, or notebook explanations
- **Scientific Methods**: Enhance algorithms or add new astrophysical calculations
- **Testing**: Improve test coverage or add new test cases
- **Performance**: Optimize data processing or model training efficiency

### Development Resources

- **Architecture**: See `docs/` for detailed technical documentation
- **Scientific Background**: Review the [ExoLife Paper](docs/Exolife.pdf)
- **API Reference**: Use `make api` and visit http://localhost:8000/docs
- **Notebook Examples**: Explore `notebooks/` for usage patterns

## Citation

If you use ExoLife in your research, please cite:

```bibtex
@software{exolife,
  title={ExoLife: Estimating Exoplanet Habitability},
  author={Guirao, Carlos Hernán},
  year={2025},
  url={https://github.com/C-HernanG/exolife},
  version={0.1.0}
}
```

## Support

- **Documentation**: Check the [docs/](docs/) directory and notebook examples
- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/C-HernanG/exolife/issues)
- **Discussions**: Join conversations in [GitHub Discussions](https://github.com/C-HernanG/exolife/discussions)
- **Email**: Contact the maintainer at carlos.hernangui@gmail.com

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

The MIT License allows for both academic and commercial use, with attribution required. This aligns with our goal of supporting open science and astrobiological discovery.

---

<div align="center">

**Built for the astrobiology and exoplanet science community**

[⭐ Star us on GitHub](https://github.com/C-HernanG/exolife) | [🚀 Get Started](#-quick-start) | [📖 Read the Paper](docs/Exolife.pdf)

</div>