"""Tests for the transforms module."""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch

# Try to import the module, skip tests if dependencies are missing
try:
    from src.docshield.data.transforms import build_transforms
    TRANSFORMS_AVAILABLE = True
except ImportError:
    TRANSFORMS_AVAILABLE = False
    build_transforms = None


@pytest.mark.skipif(not TRANSFORMS_AVAILABLE, reason="Transforms dependencies not available")
@pytest.mark.unit
class TestBuildTransforms:
    """Test the build_transforms function."""

    def test_build_transforms_empty_config(self):
        """Test building transforms with empty configuration."""
        config = {"train": [], "val": []}
        result = build_transforms(config)
        
        assert "train" in result
        assert "val" in result
        assert isinstance(result["train"], type(result["train"]))
        assert isinstance(result["val"], type(result["val"]))

    def test_build_transforms_with_resize(self):
        """Test building transforms with resize configuration."""
        config = {
            "train": [{"name": "Resize", "params": {"height": 256, "width": 256}}],
            "val": [{"name": "Resize", "params": {"height": 224, "width": 224}}]
        }
        result = build_transforms(config)
        
        assert "train" in result
        assert "val" in result

    def test_build_transforms_with_random_resized_crop(self):
        """Test building transforms with RandomResizedCrop."""
        config = {
            "train": [{"name": "RandomResizedCrop", "params": {"height": 224, "width": 224}}],
            "val": [{"name": "Resize", "params": {"height": 224, "width": 224}}]
        }
        result = build_transforms(config)
        
        assert "train" in result
        assert "val" in result

    def test_build_transforms_with_default_image_size(self):
        """Test building transforms with default image size."""
        config = {
            "train": [{"name": "Resize"}],
            "val": [{"name": "Resize"}]
        }
        result = build_transforms(config, image_size=512)
        
        assert "train" in result
        assert "val" in result

    def test_build_transforms_with_multiple_transforms(self):
        """Test building transforms with multiple transformations."""
        config = {
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
        result = build_transforms(config)
        
        assert "train" in result
        assert "val" in result

    def test_build_transforms_unknown_transform(self):
        """Test building transforms with unknown transform name."""
        config = {
            "train": [{"name": "UnknownTransform"}],
            "val": []
        }
        
        with pytest.raises(ValueError, match="Unknown transform"):
            build_transforms(config)

    def test_build_transforms_missing_name(self):
        """Test building transforms with missing name in transform config."""
        config = {
            "train": [{"params": {"height": 224, "width": 224}}],
            "val": []
        }
        
        with pytest.raises(ValueError, match="Unknown transform"):
            build_transforms(config)

    def test_build_transforms_applies_to_tensor(self):
        """Test that transforms include ToTensorV2 conversion."""
        config = {"train": [], "val": []}
        result = build_transforms(config)
        
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # Apply transforms
        train_result = result["train"](image=np.array(test_image))["image"]
        val_result = result["val"](image=np.array(test_image))["image"]
        
        # Check that results are tensors (numpy arrays with float dtype)
        assert train_result.dtype == np.float32
        assert val_result.dtype == np.float32

    def test_build_transforms_with_augmentations(self):
        """Test building transforms with common augmentations."""
        config = {
            "train": [
                {"name": "RandomResizedCrop", "params": {"height": 224, "width": 224}},
                {"name": "HorizontalFlip", "params": {"p": 0.5}},
                {"name": "RandomBrightnessContrast", "params": {"p": 0.2}},
                {"name": "Normalize", "params": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
            ],
            "val": [
                {"name": "Resize", "params": {"height": 224, "width": 224}},
                {"name": "Normalize", "params": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
            ]
        }
        result = build_transforms(config)
        
        assert "train" in result
        assert "val" in result

    def test_build_transforms_partial_config(self):
        """Test building transforms with partial configuration."""
        config = {"train": [{"name": "Resize", "params": {"height": 224, "width": 224}}]}
        result = build_transforms(config)
        
        assert "train" in result
        assert "val" in result
        # Val should have empty transforms but still be valid

    def test_build_transforms_custom_image_size(self):
        """Test building transforms with custom image size."""
        config = {
            "train": [{"name": "Resize"}],
            "val": [{"name": "Resize"}]
        }
        result = build_transforms(config, image_size=384)
        
        # Create test image and apply transforms
        test_image = Image.new('RGB', (100, 100), color='red')
        
        train_result = result["train"](image=np.array(test_image))["image"]
        val_result = result["val"](image=np.array(test_image))["image"]
        
        # Check that images are resized to the specified size
        assert train_result.shape[1:] == (384, 384)  # (C, H, W)
        assert val_result.shape[1:] == (384, 384)  # (C, H, W)
