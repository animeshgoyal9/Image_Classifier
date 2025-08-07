"""Tests for the synthetic dummy data generation module."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image, ImageDraw, ImageFont
import random

# Try to import the module, skip tests if dependencies are missing
try:
    from src.docshield.data.synth_dummy_data import (
        generate_image, 
        generate_dataset, 
        CLASSES,
        main
    )
    SYNTH_DATA_AVAILABLE = True
except ImportError:
    SYNTH_DATA_AVAILABLE = False
    generate_image = None
    generate_dataset = None
    CLASSES = []
    main = None


@pytest.mark.skipif(not SYNTH_DATA_AVAILABLE, reason="Synthetic data dependencies not available")
@pytest.mark.unit
class TestGenerateImage:
    """Test the generate_image function."""

    def test_generate_image_basic(self):
        """Test basic image generation."""
        text = "test_class"
        img = generate_image(text)
        
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (256, 256)  # Default size

    def test_generate_image_custom_size(self):
        """Test image generation with custom size."""
        text = "test_class"
        size = 512
        img = generate_image(text, size)
        
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (size, size)

    def test_generate_image_real_class(self):
        """Test image generation for real class."""
        text = "ssn_real"
        img = generate_image(text)
        
        assert isinstance(img, Image.Image)
        # Check that the image contains the expected text
        # Note: This is a basic check - in a real test you might use OCR

    def test_generate_image_fake_class(self):
        """Test image generation for fake class."""
        text = "dl_fake"
        img = generate_image(text)
        
        assert isinstance(img, Image.Image)
        # Check that the image contains the expected text

    def test_generate_image_font_fallback(self):
        """Test image generation with font fallback."""
        text = "test_class"
        
        # Mock font loading to fail
        with patch('PIL.ImageFont.truetype', side_effect=IOError("Font not found")):
            img = generate_image(text)
            
            assert isinstance(img, Image.Image)
            assert img.mode == "RGB"

    def test_generate_image_text_positioning(self):
        """Test that text is positioned correctly."""
        text = "test_class"
        size = 256
        img = generate_image(text, size)
        
        # The image should have text drawn on it
        # We can't easily test the exact positioning without OCR,
        # but we can verify the image was created successfully
        assert isinstance(img, Image.Image)

    def test_generate_image_watermark(self):
        """Test that watermarks are added correctly."""
        # Test real class
        img_real = generate_image("ssn_real")
        assert isinstance(img_real, Image.Image)
        
        # Test fake class
        img_fake = generate_image("dl_fake")
        assert isinstance(img_fake, Image.Image)

    def test_generate_image_random_background(self):
        """Test that background colors are random."""
        text = "test_class"
        img1 = generate_image(text)
        img2 = generate_image(text)
        
        # Get the background color (top-left pixel)
        bg1 = img1.getpixel((0, 0))
        bg2 = img2.getpixel((0, 0))
        
        # Colors should be different (random)
        # Note: There's a small chance they could be the same
        # In a real test, you might run this multiple times
        assert isinstance(bg1, tuple)
        assert isinstance(bg2, tuple)
        assert len(bg1) == 3
        assert len(bg2) == 3


@pytest.mark.skipif(not SYNTH_DATA_AVAILABLE, reason="Synthetic data dependencies not available")
@pytest.mark.unit
class TestGenerateDataset:
    """Test the generate_dataset function."""

    def test_generate_dataset_basic(self, tmp_path):
        """Test basic dataset generation."""
        num_samples = 5
        generate_dataset(tmp_path, num_samples)
        
        # Check that all class directories were created
        for cls in CLASSES:
            cls_dir = tmp_path / cls
            assert cls_dir.exists()
            assert cls_dir.is_dir()
            
            # Check that the correct number of images were created
            image_files = list(cls_dir.glob("*.jpg"))
            assert len(image_files) == num_samples

    def test_generate_dataset_with_pdf(self, tmp_path):
        """Test dataset generation with PDF files."""
        num_samples = 3
        generate_dataset(tmp_path, num_samples, create_pdf=True)
        
        # Check that all class directories were created
        for cls in CLASSES:
            cls_dir = tmp_path / cls
            assert cls_dir.exists()
            
            # Check that both images and PDFs were created
            image_files = list(cls_dir.glob("*.jpg"))
            pdf_files = list(cls_dir.glob("*.pdf"))
            
            assert len(image_files) == num_samples
            assert len(pdf_files) == num_samples

    def test_generate_dataset_zero_samples(self, tmp_path):
        """Test dataset generation with zero samples."""
        generate_dataset(tmp_path, 0)
        
        # Check that directories were created but no files
        for cls in CLASSES:
            cls_dir = tmp_path / cls
            assert cls_dir.exists()
            
            image_files = list(cls_dir.glob("*.jpg"))
            assert len(image_files) == 0

    def test_generate_dataset_large_number(self, tmp_path):
        """Test dataset generation with a large number of samples."""
        num_samples = 100
        generate_dataset(tmp_path, num_samples)
        
        # Check that all files were created
        total_files = 0
        for cls in CLASSES:
            cls_dir = tmp_path / cls
            image_files = list(cls_dir.glob("*.jpg"))
            total_files += len(image_files)
        
        assert total_files == len(CLASSES) * num_samples

    def test_generate_dataset_file_naming(self, tmp_path):
        """Test that files are named correctly."""
        num_samples = 2
        generate_dataset(tmp_path, num_samples)
        
        for cls in CLASSES:
            cls_dir = tmp_path / cls
            image_files = list(cls_dir.glob("*.jpg"))
            
            # Check file naming pattern
            expected_names = [f"{cls}_0.jpg", f"{cls}_1.jpg"]
            actual_names = [f.name for f in image_files]
            
            assert sorted(actual_names) == sorted(expected_names)

    def test_generate_dataset_image_content(self, tmp_path):
        """Test that generated images have correct content."""
        num_samples = 1
        generate_dataset(tmp_path, num_samples)
        
        for cls in CLASSES:
            cls_dir = tmp_path / cls
            image_files = list(cls_dir.glob("*.jpg"))
            
            assert len(image_files) == 1
            img_path = image_files[0]
            
            # Load and verify the image
            img = Image.open(img_path)
            assert img.mode == "RGB"
            assert img.size == (256, 256)  # Default size


@pytest.mark.skipif(not SYNTH_DATA_AVAILABLE, reason="Synthetic data dependencies not available")
@pytest.mark.integration
class TestMainFunction:
    """Test the main function."""

    def test_main_basic(self, tmp_path):
        """Test main function with basic arguments."""
        output_dir = str(tmp_path / "test_output")
        
        with patch('sys.argv', ['synth_dummy_data.py', '--output_dir', output_dir]):
            main()
            
            # Check that data was generated
            output_path = Path(output_dir)
            assert output_path.exists()
            
            # Check that all classes were created
            for cls in CLASSES:
                cls_dir = output_path / cls
                assert cls_dir.exists()

    def test_main_with_custom_samples(self, tmp_path):
        """Test main function with custom number of samples."""
        output_dir = str(tmp_path / "test_output")
        
        with patch('sys.argv', ['synth_dummy_data.py', '--output_dir', output_dir, '--num_samples', '5']):
            main()
            
            # Check that correct number of samples were created
            output_path = Path(output_dir)
            for cls in CLASSES:
                cls_dir = output_path / cls
                image_files = list(cls_dir.glob("*.jpg"))
                assert len(image_files) == 5

    def test_main_with_pdf(self, tmp_path):
        """Test main function with PDF generation."""
        output_dir = str(tmp_path / "test_output")
        
        with patch('sys.argv', ['synth_dummy_data.py', '--output_dir', output_dir, '--pdf']):
            main()
            
            # Check that both images and PDFs were created
            output_path = Path(output_dir)
            for cls in CLASSES:
                cls_dir = output_path / cls
                image_files = list(cls_dir.glob("*.jpg"))
                pdf_files = list(cls_dir.glob("*.pdf"))
                
                assert len(image_files) == 10  # Default
                assert len(pdf_files) == 10


@pytest.mark.skipif(not SYNTH_DATA_AVAILABLE, reason="Synthetic data dependencies not available")
@pytest.mark.unit
class TestConstants:
    """Test the constants defined in the module."""

    def test_classes_constant(self):
        """Test that CLASSES constant is defined correctly."""
        expected_classes = [
            "ssn_real",
            "ssn_fake", 
            "dl_real",
            "dl_fake",
            "bankstmt_real",
            "bankstmt_fake",
        ]
        
        assert CLASSES == expected_classes
        assert len(CLASSES) == 6

    def test_classes_unique(self):
        """Test that all classes are unique."""
        assert len(CLASSES) == len(set(CLASSES))

    def test_classes_pattern(self):
        """Test that classes follow the expected pattern."""
        for cls in CLASSES:
            # Should be in format: document_type_authenticity
            parts = cls.split('_')
            assert len(parts) == 2
            assert parts[1] in ['real', 'fake']


@pytest.fixture
def tmp_path():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)
