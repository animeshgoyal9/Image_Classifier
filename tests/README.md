# DocShield Test Suite

This directory contains comprehensive tests for the DocShield document authentication system.

## Test Structure

### Test Files

- **`test_datasets.py`** - Tests for dataset loading and management
  - `DocClassificationDataset` class functionality
  - File listing and class mapping
  - Error handling for invalid directories

- **`test_transforms.py`** - Tests for image transformation pipelines
  - Albumentations-based transforms
  - Configuration parsing
  - Image preprocessing

- **`test_inference.py`** - Tests for model inference functionality
  - Model factory and creation
  - Forward pass validation
  - Training mode testing

- **`test_api.py`** - Tests for the FastAPI endpoints
  - Health and version endpoints
  - Prediction endpoint with various file types
  - Error handling and security

- **`test_pdf_to_image.py`** - Tests for PDF conversion utilities
  - pdf2image and PyMuPDF fallback
  - Multi-page PDF handling
  - Error handling

- **`test_synth_dummy_data.py`** - Tests for synthetic data generation
  - Image generation with text overlays
  - Dataset structure creation
  - Command-line interface

### Configuration Files

- **`conftest.py`** - Common fixtures and test configuration
- **`__init__.py`** - Package initialization
- **`pytest.ini`** - Pytest configuration

## Running Tests

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Basic Test Commands

```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_datasets.py

# Run specific test class
python -m pytest tests/test_api.py::TestAPIEndpoints

# Run specific test method
python -m pytest tests/test_api.py::TestAPIEndpoints::test_health_endpoint
```

### Using the Test Runner

```bash
# Run all tests with coverage
python run_tests.py

# Run only unit tests
python run_tests.py unit

# Run only integration tests
python run_tests.py integration

# Run specific test pattern
python run_tests.py specific test_api
```

### Using Make

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration

# Run with detailed coverage
make test-coverage

# Clean up test artifacts
make clean
```

## Test Categories

### Unit Tests
- Test individual functions and classes in isolation
- Use mocks for external dependencies
- Fast execution
- Marked with `@pytest.mark.unit`

### Integration Tests
- Test interactions between components
- May require external dependencies
- Slower execution
- Marked with `@pytest.mark.integration`

### API Tests
- Test FastAPI endpoints
- Use TestClient for HTTP requests
- Mock model inference
- Marked with `@pytest.mark.api`

### Model Tests
- Test deep learning model functionality
- May require PyTorch
- Test training and inference
- Marked with `@pytest.mark.model`

## Test Fixtures

Common fixtures are defined in `conftest.py`:

- `tmp_path` - Temporary directory for file operations
- `sample_image` - Sample PIL Image for testing
- `sample_tensor` - Sample PyTorch tensor
- `mock_model` - Mock neural network model
- `sample_config` - Sample configuration dictionary
- `sample_dataset_structure` - Sample dataset directory structure

## Coverage

To generate coverage reports:

```bash
# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html

# Generate XML coverage report (for CI)
pytest tests/ --cov=src --cov-report=xml

# View coverage in terminal
pytest tests/ --cov=src --cov-report=term-missing
```

## Continuous Integration

The test suite is designed to work with CI/CD pipelines:

```bash
# CI test command
make ci-test
```

This generates:
- XML coverage report
- JUnit XML test results
- Terminal output with missing coverage

## Troubleshooting

### Missing Dependencies

If tests fail due to missing dependencies:

```bash
# Install all dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"
```

### PyTorch Issues

If PyTorch tests fail:

```bash
# Install PyTorch for your platform
pip install torch torchvision

# Or skip PyTorch tests
pytest tests/ -m "not model"
```

### FastAPI Issues

If API tests fail:

```bash
# Install FastAPI
pip install fastapi uvicorn

# Or skip API tests
pytest tests/ -m "not api"
```

## Adding New Tests

When adding new tests:

1. Follow the naming convention: `test_*.py`
2. Use descriptive test method names
3. Add appropriate markers (`@pytest.mark.unit`, etc.)
4. Use fixtures from `conftest.py` when possible
5. Mock external dependencies
6. Test both success and error cases

Example:

```python
import pytest
from unittest.mock import Mock

@pytest.mark.unit
def test_new_functionality():
    """Test description."""
    # Arrange
    mock_dependency = Mock()
    
    # Act
    result = function_under_test(mock_dependency)
    
    # Assert
    assert result == expected_value
```
