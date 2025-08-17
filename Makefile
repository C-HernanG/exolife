.PHONY: install-env install-dev kernel test lint format clean clean-data update-env update-exolife activate help

install-env:
	conda env create -f environment.yml
	git config commit.template .gitmessage

install-dev:
	pip install -e .
	git config commit.template .gitmessage

kernel:
	python -m ipykernel install --user --name=exolife --display-name="Python (exolife)"
	@echo "Kernel 'exolife' installed. Use it in Jupyter notebooks."

update-env:
	conda env update -f environment.yml --prune

update-exolife:
	pip install -e . --upgrade

activate:
	@echo "Run: conda activate exolife"

test:
	pytest

lint:
	ruff check --diff .
	black --check --diff .

format:
	ruff check --fix .
	black .

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

clean-data:
	rm -rf data/raw/*
	rm -rf data/processed/*
	rm -rf data/interim/*
	rm -rf data/merged/*

help:
	@echo "Available targets:"
	@echo "  install-env - Install conda environment from environment.yml"
	@echo "  install-dev - Install development dependencies"
	@echo "  kernel    - Install Jupyter kernel for this environment"
	@echo "  update-env   - Update conda environment"
	@echo "  update-exolife - Update ExoLife package"
	@echo "  activate - Show activation command"
	@echo "  test     - Run pytest"
	@echo "  lint     - Run linting checks (flake8, black, isort)"
	@echo "  format   - Format code with black and isort"
	@echo "  clean    - Remove Python cache files and artifacts"
	@echo "  clean-data - Remove data directories"
	@echo "  help     - Show this help message"