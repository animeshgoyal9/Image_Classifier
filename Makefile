# DocShield Makefile

.PHONY: help test test-unit test-integration test-coverage lint format clean install-dev

# Default target
help:
	@echo "DocShield Development Commands:"
	@echo ""
	@echo "Testing:"
	@echo "  test              Run all tests with coverage"
	@echo "  test-unit         Run only unit tests"
	@echo "  test-integration  Run only integration tests"
	@echo "  test-coverage     Run tests with detailed coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint              Run linting checks"
	@echo "  format            Format code with black and ruff"
	@echo "  type-check        Run type checking with mypy"
	@echo ""
	@echo "Development:"
	@echo "  install-dev       Install development dependencies"
	@echo "  clean             Clean up generated files"
	@echo "  generate-data     Generate synthetic test data"

# Testing targets
test:
	python run_tests.py

test-unit:
	python run_tests.py unit

test-integration:
	python run_tests.py integration

test-coverage:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing --cov-report=xml

# Code quality targets
lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	ruff format src/ tests/

type-check:
	mypy src/

# Development targets
install-dev:
	pip install -e ".[dev]"

clean:
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf src/docshield/__pycache__/
	rm -rf src/docshield/*/__pycache__/
	rm -rf tests/__pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

generate-data:
	python -m src.docshield.data.synth_dummy_data --output_dir data/train --num_samples 10
	python -m src.docshield.data.synth_dummy_data --output_dir data/val --num_samples 5

# CI/CD targets
ci-test:
	pytest tests/ -v --cov=src --cov-report=xml --cov-report=term-missing --junitxml=test-results.xml

# Docker targets
docker-build:
	docker-compose build

docker-test:
	docker-compose run --rm api pytest tests/ -v

# Quick development setup
setup-dev: install-dev generate-data
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify everything is working."

# Application targets
run-api:
	uvicorn src.docshield.api.main:app --host 0.0.0.0 --port 8001 --reload

run-ui:
	streamlit run src/docshield/ui/app.py --server.port 8501 --server.address 0.0.0.0

run-app: run-api run-ui
	@echo "Starting both API and UI..."
	@echo "API will be available at: http://localhost:8001"
	@echo "UI will be available at: http://localhost:8501"

# Training targets
train-synthetic:
	python train_model.py --create-synthetic --epochs 5

train-model:
	python train_model.py --epochs 10

train-quick:
	python train_model.py --create-synthetic --epochs 3
