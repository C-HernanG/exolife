# ExoLife Makefile
# A comprehensive build and development automation tool for the ExoLife project

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON_VERSION := 3.11
CONDA_ENV := exolife
PACKAGE_DIR := package
TEST_DIR := tests
DOCS_DIR := docs
DATA_DIR := data

# Color codes for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Check if conda is available
CONDA_EXISTS := $(shell command -v conda 2> /dev/null)

# PHONY targets
.PHONY: help install install-env install-dev install-jupyter install-all
.PHONY: update update-env update-exolife update-all
.PHONY: test test-verbose test-coverage test-unit test-integration analyze-data
.PHONY: lint lint-check format format-check quality-check
.PHONY: clean clean-pyc clean-build clean-data clean-all
.PHONY: kernel jupyter notebook
.PHONY: build dist upload
.PHONY: docs docs-serve
.PHONY: env-info project-info check-env activate status

##@ Setup and Installation
install: install-env install-dev ## Install everything (environment + development)
	@printf "$(GREEN)✓ Complete installation finished$(NC)\n"

install-env: ## Install conda environment from environment.yml
ifndef CONDA_EXISTS
	@printf "$(RED)✗ Conda not found. Please install conda or miniconda first.$(NC)\n"
	@exit 1
endif
	@printf "$(BLUE)Installing conda environment...$(NC)\n"
	conda env create -f environment.yml --force
	@if [ -f .gitmessage ]; then \
		git config commit.template .gitmessage; \
		printf "$(GREEN)✓ Git commit template configured$(NC)\n"; \
	fi
	@printf "$(GREEN)✓ Conda environment '$(CONDA_ENV)' created$(NC)\n"

install-dev: ## Install package in development mode
	@printf "$(BLUE)Installing package in development mode...$(NC)\n"
	pip install -e $(PACKAGE_DIR)
	@printf "$(GREEN)✓ ExoLife package installed in development mode$(NC)\n"

install-jupyter: ## Install Jupyter extensions and kernel
	@printf "$(BLUE)Installing Jupyter kernel...$(NC)\n"
	python -m ipykernel install --user --name=$(CONDA_ENV) --display-name="Python ($(CONDA_ENV))"
	@printf "$(GREEN)✓ Jupyter kernel '$(CONDA_ENV)' installed$(NC)\n"

install-all: install install-jupyter ## Install everything including Jupyter

##@ Development Tools
update: update-env update-exolife ## Update environment and package

update-env: ## Update conda environment
	@printf "$(BLUE)Updating conda environment...$(NC)\n"
	conda env update -f environment.yml --prune
	@printf "$(GREEN)✓ Environment updated$(NC)\n"

update-exolife: ## Update ExoLife package
	@printf "$(BLUE)Updating ExoLife package...$(NC)\n"
	pip install -e . --upgrade
	@printf "$(GREEN)✓ Package updated$(NC)\n"

update-all: update ## Alias for update

##@ Testing
test: ## Run all tests
	@printf "$(BLUE)Running tests...$(NC)\n"
	python -m pytest $(TEST_DIR) -v

test-verbose: ## Run tests with verbose output
	@printf "$(BLUE)Running tests (verbose)...$(NC)\n"
	python -m pytest $(TEST_DIR) -v -s

test-coverage: ## Run tests with coverage report
	@printf "$(BLUE)Running tests with coverage...$(NC)\n"
	python -m pytest $(TEST_DIR) --cov=exolife --cov-report=html --cov-report=term --cov-report=xml

test-unit: ## Run unit tests only
	@printf "$(BLUE)Running unit tests...$(NC)\n"
	python -m pytest $(TEST_DIR) -v -k "not integration"

test-integration: ## Run integration tests only
	@printf "$(BLUE)Running integration tests...$(NC)\n"
	python -m pytest $(TEST_DIR) -v -k "integration"

test-quick: ## Run quick test suite (unit tests only, no coverage)
	@printf "$(BLUE)Running quick tests...$(NC)\n"
	python -m pytest $(TEST_DIR) -x -k "not integration"

test-parallel: ## Run tests in parallel (requires pytest-xdist)
	@printf "$(BLUE)Running tests in parallel...$(NC)\n"
	python -m pytest $(TEST_DIR) -n auto -v

test-runner: ## Use custom test runner script
	@printf "$(BLUE)Using custom test runner...$(NC)\n"
	python run_tests.py --verbose

test-runner-coverage: ## Use test runner with coverage
	@printf "$(BLUE)Using test runner with coverage...$(NC)\n"
	python run_tests.py --coverage --verbose

analyze-data: ## Analyze processed data
	@printf "$(BLUE)Analyzing processed data...$(NC)\n"
	python -m scripts.analyze_processed_data
	@printf "$(GREEN)✓ Data analysis completed$(NC)\n"

##@ Code Quality
lint: lint-check ## Alias for lint-check

lint-check: ## Check code style and quality
	@printf "$(BLUE)Checking code style with ruff...$(NC)\n"
	ruff check --diff $(PACKAGE_DIR)
	@printf "$(BLUE)Checking code formatting with black...$(NC)\n"
	black --check --diff $(PACKAGE_DIR)

format: ## Format code with black and ruff
	@printf "$(BLUE)Formatting code with ruff...$(NC)\n"
	ruff check --fix --unsafe-fixes $(PACKAGE_DIR)
	@printf "$(BLUE)Formatting code with black...$(NC)\n"
	black $(PACKAGE_DIR)
	@printf "$(GREEN)✓ Code formatted$(NC)\n"

format-check: lint-check ## Alias for lint-check

quality-check: lint-check test ## Run all quality checks (lint + tests)

##@ Jupyter and Notebooks
kernel: install-jupyter ## Alias for install-jupyter

jupyter: ## Start Jupyter Lab
	@printf "$(BLUE)Starting Jupyter Lab...$(NC)\n"
	jupyter lab

notebook: jupyter ## Alias for jupyter

##@ Cleanup
clean: clean-pyc ## Clean Python cache files
	@printf "$(GREEN)✓ Cleanup completed$(NC)\n"

clean-pyc: ## Remove Python cache files
	@printf "$(BLUE)Cleaning Python cache files...$(NC)\n"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.orig" -delete

clean-build: ## Remove build artifacts
	@printf "$(BLUE)Cleaning build artifacts...$(NC)\n"
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

clean-data: ## Remove data directories (CAUTION: This will delete all data!)
	@printf "$(RED)⚠ WARNING: This will delete all data files!$(NC)\n"
	@printf "$(YELLOW)Continue? [y/N]: $(NC)" && read ans && [ $${ans:-N} = y ]
	@printf "$(BLUE)Cleaning data directories...$(NC)\n"
	rm -rf $(DATA_DIR)/raw/*
	rm -rf $(DATA_DIR)/processed/*
	rm -rf $(DATA_DIR)/interim/*
	@if [ -d "$(DATA_DIR)/merged" ]; then rm -rf $(DATA_DIR)/merged/*; fi
	@printf "$(GREEN)✓ Data directories cleaned$(NC)\n"

clean-all: clean-pyc clean-build ## Clean everything except data

##@ Build and Distribution
build: clean-build ## Build package
	@printf "$(BLUE)Building package...$(NC)\n"
	cd $(PACKAGE_DIR) && python -m build
	@printf "$(GREEN)✓ Package built$(NC)\n"

dist: build ## Create distribution

##@ Information and Status
env-info: ## Show environment information
	@printf "$(BLUE)Environment Information:$(NC)\n"
	@printf "Conda environment: $(CONDA_ENV)\n"
	@printf "Python version: $(PYTHON_VERSION)\n"
	@printf "Package directory: $(PACKAGE_DIR)\n"
	@printf "Test directory: $(TEST_DIR)\n"
	@printf "Data directory: $(DATA_DIR)\n"

project-info: ## Show project information
	@printf "$(BLUE)Project Information:$(NC)\n"
	@printf "Name: ExoLife\n"
	@printf "Description: Predicting Exoplanet Habitability to Support Astrobiological Discovery\n"
	@printf "Version: 0.1.0\n"
	@printf "Python: >=3.11\n"

check-env: ## Check if conda environment is activated
	@if [ "$${CONDA_DEFAULT_ENV}" != "$(CONDA_ENV)" ]; then \
		printf "$(YELLOW)⚠ Conda environment '$(CONDA_ENV)' is not activated$(NC)\n"; \
		printf "$(BLUE)Run: conda activate $(CONDA_ENV)$(NC)\n"; \
	else \
		printf "$(GREEN)✓ Environment '$(CONDA_ENV)' is active$(NC)\n"; \
	fi

activate: ## Show activation command
	@printf "$(BLUE)To activate the environment, run:$(NC)\n"
	@printf "conda activate $(CONDA_ENV)\n"

status: check-env project-info ## Show project and environment status

##@ Help
help: ## Display this help message
	@printf "$(BLUE)ExoLife Makefile$(NC)\n"
	@printf "$(BLUE)=================$(NC)\n\n"
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(BLUE)<target>$(NC)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) }' $(MAKEFILE_LIST)