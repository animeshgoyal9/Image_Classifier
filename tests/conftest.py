"""Common test configuration and fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from PIL import Image

# Try to import torch, but don't fail if it's not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@pytest.fixture(scope="session")
def tmp_path():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = Image.new('RGB', (224, 224), color='red')
    return img


@pytest.fixture
def sample_image_array():
    """Create a sample image as numpy array for testing."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    if TORCH_AVAILABLE:
        return torch.randn(1, 3, 224, 224)
    else:
        return None


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.parameters.return_value = [Mock(requires_grad=True)]
    model.train.return_value = None
    model.eval.return_value = None
    model.training = False
    return model


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return {
        "name": "efficientnet",
        "num_classes": 6,
        "pretrained": True,
        "dropout": 0.1,
        "head_width": 128
    }


@pytest.fixture
def sample_train_config():
    """Create a sample training configuration."""
    return {
        "dataset": {
            "train_dir": "data/train",
            "val_dir": "data/val",
            "num_classes": 6
        },
        "model": {
            "name": "efficientnet",
            "pretrained": True,
            "dropout": 0.2
        },
        "optim": {
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 20,
            "batch_size": 32
        },
        "seed": 42
    }


@pytest.fixture
def sample_augmentation_config():
    """Create a sample augmentation configuration."""
    return {
        "train": [
            {"name": "RandomResizedCrop", "params": {"height": 224, "width": 224}},
            {"name": "HorizontalFlip", "params": {"p": 0.5}},
            {"name": "Normalize", "params": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
        ],
        "val": [
            {"name": "Resize", "params": {"height": 224, "width": 224}},
            {"name": "Normalize", "params": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
        ]
    }


@pytest.fixture
def sample_dataset_structure(tmp_path):
    """Create a sample dataset structure for testing."""
    # Create class directories
    classes = ["ssn_real", "ssn_fake", "dl_real", "dl_fake", "bankstmt_real", "bankstmt_fake"]
    
    for class_name in classes:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        
        # Create sample images
        for i in range(5):
            img = Image.new('RGB', (100, 100), color=np.random.randint(0, 255, 3))
            img.save(class_dir / f"sample_{i}.jpg")
    
    return tmp_path


@pytest.fixture
def sample_pdf_content():
    """Create sample PDF content for testing."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"


@pytest.fixture
def mock_predict_response():
    """Create a mock prediction response."""
    return {
        "document_type": "dl",
        "label": "real",
        "confidence": 0.983,
        "top_k": [
            {"label": "dl_real", "prob": 0.983},
            {"label": "dl_fake", "prob": 0.017}
        ],
        "explanations": {"saliency_png_base64": "mock_base64_data"},
        "model_version": "v1.0.0"
    }


# Markers for different test types
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "api: marks tests as API tests")
    config.addinivalue_line("markers", "model: marks tests as model tests")


# Skip tests if dependencies are not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip tests based on available dependencies."""
    skip_torch = pytest.mark.skip(reason="PyTorch not available")
    skip_fastapi = pytest.mark.skip(reason="FastAPI not available")
    skip_albumentations = pytest.mark.skip(reason="Albumentations not available")
    
    for item in items:
        # Skip PyTorch tests if torch is not available
        if "torch" in str(item.fspath) and not hasattr(item, "skip_torch"):
            try:
                import torch
            except ImportError:
                item.add_marker(skip_torch)
        
        # Skip FastAPI tests if fastapi is not available
        if "api" in str(item.fspath) and not hasattr(item, "skip_fastapi"):
            try:
                import fastapi
            except ImportError:
                item.add_marker(skip_fastapi)
        
        # Skip Albumentations tests if albumentations is not available
        if "transforms" in str(item.fspath) and not hasattr(item, "skip_albumentations"):
            try:
                import albumentations
            except ImportError:
                item.add_marker(skip_albumentations)
