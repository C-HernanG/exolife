# ExoLife Makefile
# Docker-focused build and development automation tool for the ExoLife project

# Default target
.DEFAULT_GOAL := help

# Project configuration
PROJECT_NAME := exolife
PYTHON_VERSION := 3.11
PACKAGE_DIR := package
TEST_DIR := tests
DOCS_DIR := docs
DATA_DIR := data

# Docker configuration
DOCKER_REGISTRY := exolife
BASE_IMAGE := $(DOCKER_REGISTRY)/base:latest
GPU_IMAGE := $(DOCKER_REGISTRY)/gpu:latest
API_IMAGE := $(DOCKER_REGISTRY)/api:latest

# Color codes for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Docker check function
DOCKER_AVAILABLE := $(shell docker --version 2>/dev/null)
ifndef DOCKER_AVAILABLE
$(error Docker is not available. Please install Docker first.)
endif

# PHONY targets
.PHONY: help clean build-all test lint format
.PHONY: build-base build-gpu build-api up-dev down logs shell jupyter
.PHONY: up-gpu up-api up-all down-base down-gpu down-api down-all restart
.PHONY: test-verbose test-coverage test-unit test-integration test-watch
.PHONY: security-check quality-check ci-local analyze-data mlflow cli-help
.PHONY: quick-test quick-lint clean-pyc clean-build clean-data clean-docker clean-all
.PHONY: env-info docker-status dist benchmark memory-usage

# Helper functions
define docker_run_base
	docker run --rm -it \
		-v "$(shell pwd):/workspace" \
		-w /workspace \
		$(BASE_IMAGE) $(1)
endef

define check_image
	@if ! docker image inspect $(1) >/dev/null 2>&1; then \
		printf "$(RED)✗ Image $(1) not found. Run 'make $(2)' first.$(NC)\n"; \
		exit 1; \
	fi
endef

##@ Build
build-base: ## Build the base Docker image (runtime + dev dependencies)
	@printf "$(BLUE)Building base image...$(NC)\n"
	@docker build -f docker/Dockerfile.base -t $(BASE_IMAGE) . || \
		(printf "$(RED)✗ Failed to build base image$(NC)\n"; exit 1)
	@printf "$(GREEN)✓ Base image built: $(BASE_IMAGE)$(NC)\n"
	@printf "$(GREEN)✓ Development environment ready. Use 'make up-base' to start.$(NC)\n"

build-gpu: ## Build the GPU Docker image (base + GPU support)
	@printf "$(BLUE)Building GPU image...$(NC)\n"
	@docker build -f docker/Dockerfile.gpu -t $(GPU_IMAGE) . || \
		(printf "$(RED)✗ Failed to build GPU image$(NC)\n"; exit 1)
	@printf "$(GREEN)✓ GPU image built: $(GPU_IMAGE)$(NC)\n"

build-api: ## Build the API Docker image (slim runtime for production)
	@printf "$(BLUE)Building API image...$(NC)\n"
	@docker build -f docker/Dockerfile.api -t $(API_IMAGE) . || \
		(printf "$(RED)✗ Failed to build API image$(NC)\n"; exit 1)
	@printf "$(GREEN)✓ API image built: $(API_IMAGE)$(NC)\n"

build-all: build-base build-gpu build-api ## Build all Docker images
	@printf "$(GREEN)✓ All Docker images built successfully$(NC)\n"

##@ Development
up-base: ## Start development environment with JupyterLab
	@printf "$(BLUE)Starting development environment...$(NC)\n"
	@COMPOSE_PROFILES=dev docker compose up -d --build exolife-dev || \
		(printf "$(RED)✗ Failed to start development environment$(NC)\n"; exit 1)
	@printf "$(GREEN)✓ Development environment started$(NC)\n"
	@printf "$(BLUE)Access JupyterLab at: http://localhost:8888$(NC)\n"

up-gpu: ## Start GPU environment
	@printf "$(BLUE)Starting GPU environment...$(NC)\n"
	@COMPOSE_PROFILES=train docker compose up -d --build exolife-train || \
		(printf "$(RED)✗ Failed to start GPU environment$(NC)\n"; exit 1)
	@printf "$(GREEN)✓ GPU environment started$(NC)\n"

up-api: ## Start API service
	@printf "$(BLUE)Starting API service...$(NC)\n"
	@COMPOSE_PROFILES=api docker compose up -d --build exolife-api || \
		(printf "$(RED)✗ Failed to start API service$(NC)\n"; exit 1)
	@printf "$(GREEN)✓ API service started$(NC)\n"
	@printf "$(BLUE)API available at: http://localhost:8080$(NC)\n"

up-all: up-base up-api up-gpu ## Start all services
	@printf "$(GREEN)✓ All services started$(NC)\n"

down-base: ## Stop development services only
	@printf "$(BLUE)Stopping development services...$(NC)\n"
	@COMPOSE_PROFILES=dev docker compose down
	@printf "$(GREEN)✓ Development services stopped$(NC)\n"

down-gpu: ## Stop GPU services only
	@printf "$(BLUE)Stopping GPU services...$(NC)\n"
	@COMPOSE_PROFILES=train docker compose down
	@printf "$(GREEN)✓ GPU services stopped$(NC)\n"

down-api: ## Stop API services only
	@printf "$(BLUE)Stopping API services...$(NC)\n"
	@COMPOSE_PROFILES=api docker compose down
	@printf "$(GREEN)✓ API services stopped$(NC)\n"

down-all: down-base down-api down-gpu
	@printf "$(GREEN)✓ All services stopped$(NC)\n"

restart: down-all up-all ## Restart development environment

logs: ## Show logs from all services (use SERVICE=name for specific service)
	@printf "$(BLUE)Showing service logs...$(NC)\n"
	@docker compose logs -f $(SERVICE)

shell: build-base ## Open interactive shell in base container
	$(call check_image,$(BASE_IMAGE),build-base)
	@printf "$(BLUE)Opening shell in base container...$(NC)\n"
	@$(call docker_run_base,bash)

jupyter: build-base ## Start JupyterLab in standalone container
	$(call check_image,$(BASE_IMAGE),build-base)
	@printf "$(BLUE)Starting JupyterLab...$(NC)\n"
	@docker run --rm -it \
		-p 8888:8888 \
		-v "$(shell pwd):/workspace" \
		-w /workspace \
		$(BASE_IMAGE) entry.dev.sh

##@ Testing
test: build-base ## Run all tests
	$(call check_image,$(BASE_IMAGE),build-base)
	@printf "$(BLUE)Running all tests...$(NC)\n"
	@$(call docker_run_base,bash -c "pip install -e . && python -m pytest $(TEST_DIR) -v") || \
		(printf "$(RED)✗ Tests failed$(NC)\n"; exit 1)
	@printf "$(GREEN)✓ All tests passed$(NC)\n"

test-verbose: build-base ## Run tests with verbose output
	$(call check_image,$(BASE_IMAGE),build-base)
	@printf "$(BLUE)Running tests with verbose output...$(NC)\n"
	@$(call docker_run_base,bash -c "pip install -e . && python -m pytest $(TEST_DIR) -v -s")

test-coverage: build-base ## Run tests with coverage report
	$(call check_image,$(BASE_IMAGE),build-base)
	@printf "$(BLUE)Running tests with coverage...$(NC)\n"
	@$(call docker_run_base,bash -c "pip install -e . && python -m pytest $(TEST_DIR) --cov=exolife --cov-report=html --cov-report=term --cov-report=xml")
	@printf "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)\n"

test-unit: build-base ## Run unit tests only
	$(call check_image,$(BASE_IMAGE),build-base)
	@printf "$(BLUE)Running unit tests...$(NC)\n"
	@$(call docker_run_base,bash -c "pip install -e . && python -m pytest $(TEST_DIR) -v -k 'not integration'")

test-integration: build-base ## Run integration tests only
	$(call check_image,$(BASE_IMAGE),build-base)
	@printf "$(BLUE)Running integration tests...$(NC)\n"
	@$(call docker_run_base,bash -c "pip install -e . && python -m pytest $(TEST_DIR) -v -k 'integration'")

test-watch: build-base ## Run tests in watch mode (requires pytest-xdist)
	$(call check_image,$(BASE_IMAGE),build-base)
	@printf "$(BLUE)Running tests in watch mode...$(NC)\n"
	@$(call docker_run_base,bash -c "pip install -e . && python -m pytest $(TEST_DIR) -f"

##@ Code Quality
lint: build-base ## Check code style and quality
	$(call check_image,$(BASE_IMAGE),build-base)
	@printf "$(BLUE)Checking code style with ruff...$(NC)\n"
	@$(call docker_run_base,bash -c "pip install -e . && ruff check --diff .") || \
		(printf "$(RED)✗ Linting failed$(NC)\n"; exit 1)
	@printf "$(BLUE)Checking code formatting with black...$(NC)\n"
	@$(call docker_run_base,bash -c "pip install -e . && black --check --diff .") || \
		(printf "$(RED)✗ Code formatting check failed$(NC)\n"; exit 1)
	@printf "$(GREEN)✓ Code quality checks passed$(NC)\n"

format: build-base ## Format code with black and ruff
	$(call check_image,$(BASE_IMAGE),build-base)
	@printf "$(BLUE)Formatting code with ruff...$(NC)\n"
	@$(call docker_run_base,bash -c "pip install -e . && ruff check --fix --unsafe-fixes .")
	@printf "$(BLUE)Formatting code with black...$(NC)\n"
	@$(call docker_run_base,bash -c "pip install -e . && black .")
	@printf "$(GREEN)✓ Code formatted$(NC)\n"

security-check: build-base ## Run security checks with bandit
	$(call check_image,$(BASE_IMAGE),build-base)
	@printf "$(BLUE)Running security checks...$(NC)\n"
	@$(call docker_run_base,bash -c "pip install bandit[toml] && bandit -r package/ -f json") || \
		(printf "$(YELLOW)⚠ Security issues found$(NC)\n")

quality-check: lint test ## Run comprehensive quality checks (lint + tests)
	@printf "$(GREEN)✓ All quality checks passed$(NC)\n"

ci-local: clean format lint test ## Simulate CI pipeline locally
	@printf "$(GREEN)✓ Local CI pipeline completed successfully$(NC)\n"

##@ Data Analysis
analyze-data: build-base ## Analyze processed data
	$(call check_image,$(BASE_IMAGE),build-base)
	@printf "$(BLUE)Analyzing processed data...$(NC)\n"
	@$(call docker_run_base,bash -c "pip install -e . && python scripts/analyze_processed_data.py")
	@printf "$(GREEN)✓ Data analysis completed$(NC)\n"

##@ MLflow & CLI

mlflow: ## Start MLflow UI
	@printf "$(BLUE)Starting MLflow UI at http://localhost:5000$(NC)\n"
	@docker run --rm -p 5000:5000 -v "$(shell pwd)/mlruns:/mlflow/mlruns" \
		ghcr.io/mlflow/mlflow:v2.16.0 \
		mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri file:///mlflow/mlruns

cli-help: build-base ## Show ExoLife CLI help
	$(call check_image,$(BASE_IMAGE),build-base)
		@$(call docker_run_base,bash -c "pip install -e . && exolife --help")

##@ Quick Commands
quick-test: build-base ## Quick test run (fastest tests only)
	$(call check_image,$(BASE_IMAGE),build-base)
	@$(call docker_run_base,bash -c "pip install -e . && python -m pytest tests/ -x -q --tb=short")

quick-lint: build-base ## Quick lint check (fastest checks only)
	$(call check_image,$(BASE_IMAGE),build-base)
	@$(call docker_run_base,bash -c "pip install -e . && ruff check . && black --check .")

##@ Cleanup
clean: clean-pyc clean-build ## Clean Python cache and build artifacts
	@printf "$(GREEN)✓ Cleanup completed$(NC)\n"

clean-pyc: ## Remove Python cache files
	@printf "$(BLUE)Cleaning Python cache files...$(NC)\n"
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name "__pycache__" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name "*.orig" -delete 2>/dev/null || true
	@printf "$(GREEN)✓ Python cache files cleaned$(NC)\n"

clean-build: ## Remove build artifacts
	@printf "$(BLUE)Cleaning build artifacts...$(NC)\n"
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "docs/_build" -exec rm -rf {} + 2>/dev/null || true
	@printf "$(GREEN)✓ Build artifacts cleaned$(NC)\n"

clean-data: ## Remove data directories (CAUTION: This will delete all data!)
	@printf "$(RED)⚠ WARNING: This will delete all data files!$(NC)\n"
	@printf "$(YELLOW)Continue? [y/N]: $(NC)" && read ans && [ $${ans:-N} = y ]
	@printf "$(BLUE)Cleaning data directories...$(NC)\n"
	@rm -rf $(DATA_DIR)/raw/* 2>/dev/null || true
	@rm -rf $(DATA_DIR)/processed/* 2>/dev/null || true
	@rm -rf $(DATA_DIR)/interim/* 2>/dev/null || true
	@if [ -d "$(DATA_DIR)/merged" ]; then rm -rf $(DATA_DIR)/merged/* 2>/dev/null || true; fi
	@printf "$(GREEN)✓ Data directories cleaned$(NC)\n"

clean-docker: ## Remove Docker containers, images and volumes
	@printf "$(YELLOW)⚠ WARNING: This will remove all $(PROJECT_NAME) Docker resources!$(NC)\n"
	@printf "$(YELLOW)Continue? [y/N]: $(NC)" && read ans && [ $${ans:-N} = y ]
	@printf "$(BLUE)Cleaning Docker resources...$(NC)\n"
	@docker compose down --volumes --remove-orphans 2>/dev/null || true
	@docker rmi $(BASE_IMAGE) $(GPU_IMAGE) $(API_IMAGE) 2>/dev/null || true
	@docker volume prune -f 2>/dev/null || true
	@printf "$(GREEN)✓ Docker resources cleaned$(NC)\n"

clean-all: clean clean-docker ## Clean everything including Docker resources

##@ Status & Info

env-info: ## Show environment information
	@printf "$(BLUE)Environment Information:$(NC)\n"
	@printf "Project: $(PROJECT_NAME) (Docker-based)\n"
	@printf "Python version: $(PYTHON_VERSION)\n"
	@printf "Package directory: $(PACKAGE_DIR)\n"
	@printf "Test directory: $(TEST_DIR)\n"
	@printf "Data directory: $(DATA_DIR)\n"
	@printf "Docker images:\n"
	@printf "  Base: $(BASE_IMAGE)\n"
	@printf "  GPU: $(GPU_IMAGE)\n"
	@printf "  API: $(API_IMAGE)\n"

docker-status: ## Show Docker status
	@printf "$(BLUE)Docker Status:$(NC)\n"
	@printf "Running containers:\n"
	@docker compose ps || printf "  No containers running\n"
	@printf "\nAvailable images:\n"
	@docker images --filter reference="$(DOCKER_REGISTRY)/*" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" || printf "  No images found\n"

##@ Additional Tools
dist: build-base ## Build distribution packages
	$(call check_image,$(BASE_IMAGE),build-base)
	@printf "$(BLUE)Building distribution packages...$(NC)\n"
	@$(call docker_run_base,bash -c "pip install build && python -m build")
	@printf "$(GREEN)✓ Distribution packages built in dist/$(NC)\n"

benchmark: build-base ## Run performance benchmarks
	$(call check_image,$(BASE_IMAGE),build-base)
	@printf "$(BLUE)Running performance benchmarks...$(NC)\n"
	@$(call docker_run_base,bash -c "pip install -e . && python -m pytest tests/ -k benchmark -v")

memory-usage: ## Check memory usage of containers
	@printf "$(BLUE)Container memory usage:$(NC)\n"
	@docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.CPUPerc}}"

##@ Help
help: ## Display this help message
	@printf "$(BLUE)ExoLife Makefile$(NC)\n"
	@printf "$(BLUE)=====================================$(NC)\n\n"
	@printf "$(GREEN)Quick Start:$(NC)\n"
	@printf "  make build-base   # Build base image\n"
	@printf "  make up-base       # Start development environment\n"
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(BLUE)<target>$(NC)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) }' $(MAKEFILE_LIST)