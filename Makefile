.PHONY: install-env install-dev kernel test lint format clean update activate help

install-env:
	conda env create -f environment.yml

install-dev:
	pip install -e .

kernel:
	python -m ipykernel install --user --name=exolife --display-name="Python (exolife)"
	@echo "Kernel 'exolife' installed. Use it in Jupyter notebooks."

update:
	conda env update -f environment.yml --prune

activate:
	@echo "Run: conda activate exolife"

test:
	pytest

lint:
	flake8 src tests
	black --check src tests
	isort --check src tests

format:
	black src tests
	isort src tests

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

help:
	@echo "Available targets:"
	@echo "  install-env - Install conda environment from environment.yml"
	@echo "  install-dev - Install development dependencies"
	@echo "  kernel    - Install Jupyter kernel for this environment"
	@echo "  update   - Update conda environment"
	@echo "  activate - Show activation command"
	@echo "  test     - Run pytest"
	@echo "  lint     - Run linting checks (flake8, black, isort)"
	@echo "  format   - Format code with black and isort"
	@echo "  clean    - Remove Python cache files and artifacts"
	@echo "  help     - Show this help message"