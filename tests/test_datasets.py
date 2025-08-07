"""Tests for the datasets module."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from PIL import Image
import numpy as np

# Try to import the module, skip tests if dependencies are missing
try:
    from src.docshield.data.datasets import DocClassificationDataset, list_image_files, create_class_to_idx
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    DocClassificationDataset = None
    list_image_files = None
    create_class_to_idx = None


@pytest.mark.skipif(not DATASETS_AVAILABLE, reason="Datasets dependencies not available")
@pytest.mark.unit
class TestListImageFiles:
    """Test the list_image_files function."""

    def test_list_image_files_empty_directory(self, tmp_path):
        """Test listing files from an empty directory."""
        result = list_image_files(tmp_path)
        assert result == []

    def test_list_image_files_with_images(self, tmp_path):
        """Test listing files with image files."""
        # Create test directory structure
        class_dir = tmp_path / "test_class"
        class_dir.mkdir()
        
        # Create dummy image files
        (class_dir / "img1.jpg").touch()
        (class_dir / "img2.png").touch()
        (class_dir / "img3.jpeg").touch()
        (class_dir / "ignore.txt").touch()  # Should be ignored
        
        result = list_image_files(tmp_path)
        expected = [
            (class_dir / "img1.jpg", "test_class"),
            (class_dir / "img2.png", "test_class"),
            (class_dir / "img3.jpeg", "test_class"),
        ]
        assert sorted(result) == sorted(expected)

    def test_list_image_files_with_pdfs(self, tmp_path):
        """Test listing files with PDF files."""
        class_dir = tmp_path / "test_class"
        class_dir.mkdir()
        
        (class_dir / "doc1.pdf").touch()
        (class_dir / "doc2.pdf").touch()
        
        result = list_image_files(tmp_path)
        expected = [
            (class_dir / "doc1.pdf", "test_class"),
            (class_dir / "doc2.pdf", "test_class"),
        ]
        assert sorted(result) == sorted(expected)

    def test_list_image_files_multiple_classes(self, tmp_path):
        """Test listing files with multiple classes."""
        # Create multiple class directories
        (tmp_path / "class1").mkdir()
        (tmp_path / "class2").mkdir()
        
        (tmp_path / "class1" / "img1.jpg").touch()
        (tmp_path / "class2" / "img2.png").touch()
        
        result = list_image_files(tmp_path)
        expected = [
            (tmp_path / "class1" / "img1.jpg", "class1"),
            (tmp_path / "class2" / "img2.png", "class2"),
        ]
        assert sorted(result) == sorted(expected)


@pytest.mark.skipif(not DATASETS_AVAILABLE, reason="Datasets dependencies not available")
@pytest.mark.unit
class TestDocClassificationDataset:
    """Test the DocClassificationDataset class."""

    def test_dataset_initialization(self, tmp_path):
        """Test dataset initialization with valid directory."""
        # Create test data
        class_dir = tmp_path / "test_class"
        class_dir.mkdir()
        (class_dir / "img1.jpg").touch()
        
        # Mock transform
        mock_transform = Mock()
        
        dataset = DocClassificationDataset(tmp_path, mock_transform)
        assert len(dataset) == 1
        assert dataset.class_to_idx == {"test_class": 0}

    def test_dataset_initialization_empty_directory(self, tmp_path):
        """Test dataset initialization with empty directory."""
        mock_transform = Mock()
        
        with pytest.raises(ValueError, match="No image or PDF files found"):
            DocClassificationDataset(tmp_path, mock_transform)

    def test_dataset_initialization_with_custom_class_mapping(self, tmp_path):
        """Test dataset initialization with custom class mapping."""
        # Create test data
        class_dir = tmp_path / "test_class"
        class_dir.mkdir()
        (class_dir / "img1.jpg").touch()
        
        mock_transform = Mock()
        custom_mapping = {"test_class": 5}
        
        dataset = DocClassificationDataset(tmp_path, mock_transform, custom_mapping)
        assert dataset.class_to_idx == {"test_class": 5}

    def test_dataset_getitem(self, tmp_path):
        """Test dataset __getitem__ method."""
        # Create test data with actual image
        class_dir = tmp_path / "test_class"
        class_dir.mkdir()
        
        # Create a real image file
        img_path = class_dir / "test.jpg"
        img = Image.new('RGB', (100, 100), color='red')
        img.save(img_path)
        
        # Mock transform to return a tensor
        mock_transform = Mock(return_value=np.random.randn(3, 224, 224))
        
        dataset = DocClassificationDataset(tmp_path, mock_transform)
        
        # Test getting an item
        image, label = dataset[0]
        assert label == 0
        mock_transform.assert_called_once()

    def test_dataset_len(self, tmp_path):
        """Test dataset __len__ method."""
        # Create test data
        class_dir = tmp_path / "test_class"
        class_dir.mkdir()
        
        for i in range(5):
            (class_dir / f"img{i}.jpg").touch()
        
        mock_transform = Mock()
        dataset = DocClassificationDataset(tmp_path, mock_transform)
        
        assert len(dataset) == 5

    def test_dataset_ignores_unknown_classes(self, tmp_path):
        """Test that dataset ignores classes not in class_to_idx."""
        # Create test data
        (tmp_path / "known_class").mkdir()
        (tmp_path / "unknown_class").mkdir()
        
        (tmp_path / "known_class" / "img1.jpg").touch()
        (tmp_path / "unknown_class" / "img2.jpg").touch()
        
        mock_transform = Mock()
        custom_mapping = {"known_class": 0}
        
        dataset = DocClassificationDataset(tmp_path, mock_transform, custom_mapping)
        assert len(dataset) == 1  # Only known_class should be included


@pytest.mark.skipif(not DATASETS_AVAILABLE, reason="Datasets dependencies not available")
@pytest.mark.unit
class TestCreateClassToIdx:
    """Test the create_class_to_idx function."""

    def test_create_class_to_idx(self, tmp_path):
        """Test creating class to index mapping."""
        # Create test directory structure
        (tmp_path / "class1").mkdir()
        (tmp_path / "class2").mkdir()
        (tmp_path / "class3").mkdir()
        
        # Add some files
        (tmp_path / "class1" / "img1.jpg").touch()
        (tmp_path / "class2" / "img2.jpg").touch()
        (tmp_path / "class3" / "img3.jpg").touch()
        
        result = create_class_to_idx(tmp_path)
        expected = {"class1": 0, "class2": 1, "class3": 2}
        assert result == expected

    def test_create_class_to_idx_empty_directory(self, tmp_path):
        """Test creating class mapping from empty directory."""
        result = create_class_to_idx(tmp_path)
        assert result == {}


@pytest.fixture
def tmp_path():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)
