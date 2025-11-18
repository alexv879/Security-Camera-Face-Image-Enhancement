.PHONY: help install install-dev test lint format clean docker-build docker-run docs

help:
	@echo "Face Enhancement AI - Makefile Commands"
	@echo "========================================"
	@echo "install          Install package and dependencies"
	@echo "install-dev      Install package with development dependencies"
	@echo "test             Run tests with pytest"
	@echo "test-cov         Run tests with coverage report"
	@echo "lint             Run linters (flake8, mypy)"
	@echo "format           Format code with black and isort"
	@echo "clean            Clean build artifacts and cache files"
	@echo "docker-build     Build Docker image"
	@echo "docker-run       Run Docker container"
	@echo "docker-api       Run API server in Docker"
	@echo "run-api          Run API server locally"
	@echo "run-cli          Run CLI tool (specify ARGS)"
	@echo "pre-commit       Run pre-commit hooks"

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/face_enhancement --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	flake8 src tests --max-line-length=100
	mypy src --ignore-missing-imports

format:
	black src tests examples
	isort src tests examples

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build -t face-enhancement-ai:latest .

docker-run:
	docker run --rm -it --gpus all -v $(PWD)/data:/app/data face-enhancement-ai:latest

docker-api:
	docker-compose up face-enhancement-api

run-api:
	uvicorn face_enhancement.api.server:app --host 0.0.0.0 --port 8000 --reload

run-cli:
	face-enhance $(ARGS)

pre-commit:
	pre-commit run --all-files

# Examples
example-enhance:
	face-enhance enhance data/input/example.jpg --output data/output/enhanced.jpg

example-batch:
	face-enhance batch data/input --output-dir data/output

example-info:
	face-enhance info

# Development
dev-setup: install-dev
	@echo "Development environment ready!"
	@echo "Run 'make test' to run tests"
	@echo "Run 'make run-api' to start API server"

# CI/CD
ci: format lint test-cov
	@echo "CI checks passed!"
