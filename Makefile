# Makefile for ML Project

# Python virtual environment path
VENV_PATH = .venv
PYTHON = $(VENV_PATH)/bin/python
PIP = $(VENV_PATH)/bin/pip

.PHONY: install test clean lint format notebook

# Install dependencies
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

# Run tests
test:
	$(PYTHON) -m pytest tests/ -v

# Clean cache and temporary files
clean:
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

# Lint code
lint:
	$(PYTHON) -m flake8 src/ tests/

# Format code
format:
	$(PYTHON) -m black src/ tests/

# Setup development environment
setup: install
	$(PYTHON) -m pre-commit install

# Run jupyter lab
notebook:
	$(PYTHON) -m jupyter lab

# Create virtual environment (if needed)
venv:
	python3 -m venv $(VENV_PATH)
	$(PIP) install --upgrade pip