"""Tests for the inference module."""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path

# Try to import dependencies, skip tests if not available
try:
    import torch
    from src.docshield.models.factory import create_model
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    create_model = None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.skip(reason="torchvision not available")
@pytest.mark.unit
class TestModelFactory:
    """Test the model factory functionality."""

    def test_create_model_efficientnet(self):
        """Test creating EfficientNet model."""
        config = {
            "name": "efficientnet",
            "num_classes": 6,
            "pretrained": True,
            "dropout": 0.1,
            "head_width": 128
        }
        
        model, feature_dim = create_model(config)
        
        assert model is not None
        assert isinstance(feature_dim, int)
        assert feature_dim > 0

    def test_create_model_vit(self):
        """Test creating Vision Transformer model."""
        config = {
            "name": "vit",
            "num_classes": 6,
            "pretrained": True,
            "dropout": 0.1,
            "head_width": 128
        }
        
        model, feature_dim = create_model(config)
        
        assert model is not None
        assert isinstance(feature_dim, int)
        assert feature_dim > 0

    def test_create_model_unsupported_name(self):
        """Test creating model with unsupported name."""
        config = {
            "name": "unsupported_model",
            "num_classes": 6
        }
        
        with pytest.raises(ValueError, match="Unsupported model name"):
            create_model(config)

    def test_create_model_default_config(self):
        """Test creating model with default configuration."""
        config = {}
        
        model, feature_dim = create_model(config)
        
        assert model is not None
        assert isinstance(feature_dim, int)

    def test_create_model_different_num_classes(self):
        """Test creating model with different number of classes."""
        config = {
            "name": "efficientnet",
            "num_classes": 10
        }
        
        model, feature_dim = create_model(config)
        
        assert model is not None
        # Test that the model can handle the specified number of classes
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            output = model(dummy_input)
            assert output.shape[1] == 10

    def test_create_model_with_dropout(self):
        """Test creating model with dropout."""
        config = {
            "name": "efficientnet",
            "num_classes": 6,
            "dropout": 0.5
        }
        
        model, feature_dim = create_model(config)
        
        assert model is not None
        # Test that dropout is applied during training
        model.train()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            output1 = model(dummy_input)
            output2 = model(dummy_input)
            # With dropout, outputs should be different
            assert not torch.allclose(output1, output2)

    def test_create_model_efficientnet_aliases(self):
        """Test creating EfficientNet with different name aliases."""
        aliases = ["efficientnet", "efficientnet-b0", "efficientnet_b0"]
        
        for alias in aliases:
            config = {
                "name": alias,
                "num_classes": 6
            }
            
            model, feature_dim = create_model(config)
            assert model is not None
            assert isinstance(feature_dim, int)

    def test_create_model_vit_aliases(self):
        """Test creating ViT with different name aliases."""
        aliases = ["vit", "vit-base", "vision_transformer"]
        
        for alias in aliases:
            config = {
                "name": alias,
                "num_classes": 6
            }
            
            model, feature_dim = create_model(config)
            assert model is not None
            assert isinstance(feature_dim, int)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.skip(reason="torchvision not available")
@pytest.mark.unit
class TestModelInference:
    """Test model inference functionality."""

    def test_model_forward_pass(self):
        """Test that models can perform forward pass."""
        config = {
            "name": "efficientnet",
            "num_classes": 6
        }
        
        model, _ = create_model(config)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(dummy_input)
            
            assert output.shape == (1, 6)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

    def test_model_output_probabilities(self):
        """Test that model outputs can be converted to probabilities."""
        config = {
            "name": "efficientnet",
            "num_classes": 6
        }
        
        model, _ = create_model(config)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            logits = model(dummy_input)
            probabilities = torch.softmax(logits, dim=1)
            
            # Check that probabilities sum to 1
            assert torch.allclose(probabilities.sum(dim=1), torch.ones(1), atol=1e-6)
            # Check that all probabilities are between 0 and 1
            assert (probabilities >= 0).all()
            assert (probabilities <= 1).all()

    def test_model_batch_inference(self):
        """Test that models can handle batch inference."""
        config = {
            "name": "efficientnet",
            "num_classes": 6
        }
        
        model, _ = create_model(config)
        model.eval()
        
        # Create batch input
        batch_input = torch.randn(4, 3, 224, 224)
        
        with torch.no_grad():
            output = model(batch_input)
            
            assert output.shape == (4, 6)

    def test_model_different_input_sizes(self):
        """Test that models can handle different input sizes."""
        config = {
            "name": "efficientnet",
            "num_classes": 6
        }
        
        model, _ = create_model(config)
        model.eval()
        
        # Test different input sizes
        sizes = [(224, 224), (256, 256), (384, 384)]
        
        for height, width in sizes:
            dummy_input = torch.randn(1, 3, height, width)
            
            with torch.no_grad():
                output = model(dummy_input)
                assert output.shape == (1, 6)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.skip(reason="torchvision not available")
@pytest.mark.unit
class TestModelTraining:
    """Test model training functionality."""

    def test_model_training_mode(self):
        """Test that models can be set to training mode."""
        config = {
            "name": "efficientnet",
            "num_classes": 6
        }
        
        model, _ = create_model(config)
        
        # Set to training mode
        model.train()
        assert model.training
        
        # Set to evaluation mode
        model.eval()
        assert not model.training

    def test_model_parameters_require_grad(self):
        """Test that model parameters require gradients for training."""
        config = {
            "name": "efficientnet",
            "num_classes": 6
        }
        
        model, _ = create_model(config)
        
        # Check that parameters require gradients
        for param in model.parameters():
            assert param.requires_grad

    def test_model_loss_computation(self):
        """Test that models can compute loss."""
        config = {
            "name": "efficientnet",
            "num_classes": 6
        }
        
        model, _ = create_model(config)
        
        # Create dummy data
        dummy_input = torch.randn(2, 3, 224, 224)
        dummy_labels = torch.randint(0, 6, (2,))
        
        # Forward pass
        outputs = model(dummy_input)
        
        # Compute loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, dummy_labels)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_model_gradient_flow(self):
        """Test that gradients can flow through the model."""
        config = {
            "name": "efficientnet",
            "num_classes": 6
        }
        
        model, _ = create_model(config)
        
        # Create dummy data
        dummy_input = torch.randn(2, 3, 224, 224)
        dummy_labels = torch.randint(0, 6, (2,))
        
        # Forward pass
        outputs = model(dummy_input)
        
        # Compute loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, dummy_labels)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients were computed
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        assert has_gradients


@pytest.fixture
def tmp_path():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)
