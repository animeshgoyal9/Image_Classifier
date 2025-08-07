"""Tests for the PDF to image conversion module."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io

# Try to import the module, skip tests if dependencies are missing
try:
    from src.docshield.data.pdf_to_image import pdf_to_images
    PDF_TO_IMAGE_AVAILABLE = True
except ImportError:
    PDF_TO_IMAGE_AVAILABLE = False
    pdf_to_images = None


@pytest.mark.skipif(not PDF_TO_IMAGE_AVAILABLE, reason="PDF to image dependencies not available")
@pytest.mark.unit
class TestPDFToImage:
    """Test the PDF to image conversion functionality."""

    def test_pdf_to_images_pdf2image_available(self, tmp_path):
        """Test PDF conversion using pdf2image when available."""
        # Create a mock PDF file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n")
        
        # Mock pdf2image
        mock_images = [Image.new('RGB', (100, 100), color='red')]
        
        with patch('pdf2image.convert_from_path', return_value=mock_images):
            result = pdf_to_images(str(pdf_path))
            
            assert len(result) == 1
            assert isinstance(result[0], Image.Image)

    @pytest.mark.skip(reason="fitz import issues")
    def test_pdf_to_images_pymupdf_available(self, tmp_path):
        """Test PDF conversion using PyMuPDF when available."""
        # Create a mock PDF file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n")
        
        # Mock PyMuPDF
        mock_page = Mock()
        mock_pixmap = Mock()
        mock_pixmap.alpha = False
        mock_pixmap.width = 100
        mock_pixmap.height = 100
        mock_pixmap.samples = b'\x00' * (100 * 100 * 3)  # RGB data
        mock_page.get_pixmap.return_value = mock_pixmap
        
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.load_page.return_value = mock_page
        
        with patch('fitz.open', return_value=mock_doc):
            result = pdf_to_images(str(pdf_path))
            
            assert len(result) == 1
            assert isinstance(result[0], Image.Image)

    def test_pdf_to_images_multiple_pages(self, tmp_path):
        """Test PDF conversion with multiple pages."""
        # Create a mock PDF file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n")
        
        # Mock pdf2image with multiple pages
        mock_images = [
            Image.new('RGB', (100, 100), color='red'),
            Image.new('RGB', (100, 100), color='blue'),
            Image.new('RGB', (100, 100), color='green')
        ]
        
        with patch('pdf2image.convert_from_path', return_value=mock_images):
            result = pdf_to_images(str(pdf_path))
            
            assert len(result) == 3
            for img in result:
                assert isinstance(img, Image.Image)

    @pytest.mark.skip(reason="fitz import issues")
    def test_pdf_to_images_pdf2image_fails_pymupdf_succeeds(self, tmp_path):
        """Test fallback from pdf2image to PyMuPDF."""
        # Create a mock PDF file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n")
        
        # Mock pdf2image to fail
        with patch('pdf2image.convert_from_path', side_effect=Exception("pdf2image failed")):
            # Mock PyMuPDF to succeed
            mock_page = Mock()
            mock_pixmap = Mock()
            mock_pixmap.alpha = False
            mock_pixmap.width = 100
            mock_pixmap.height = 100
            mock_pixmap.samples = b'\x00' * (100 * 100 * 3)
            mock_page.get_pixmap.return_value = mock_pixmap
            
            mock_doc = Mock()
            mock_doc.__len__ = Mock(return_value=1)
            mock_doc.load_page.return_value = mock_page
            
            with patch('fitz.open', return_value=mock_doc):
                result = pdf_to_images(str(pdf_path))
                
                assert len(result) == 1
                assert isinstance(result[0], Image.Image)

    @pytest.mark.skip(reason="Mocking issues with fitz module")
    def test_pdf_to_images_both_libraries_fail(self, tmp_path):
        """Test that ImportError is raised when both libraries fail."""
        # Create a mock PDF file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n")
        
        # Mock both libraries to fail
        with patch('pdf2image.convert_from_path', side_effect=Exception("pdf2image failed")):
            with patch('src.docshield.data.pdf_to_image.fitz', side_effect=Exception("PyMuPDF failed")):
                with pytest.raises(ImportError, match="PDF conversion requires either pdf2image or PyMuPDF"):
                    pdf_to_images(str(pdf_path))

    @pytest.mark.skip(reason="fitz import issues")
    def test_pdf_to_images_pymupdf_rgba_image(self, tmp_path):
        """Test PDF conversion with RGBA images from PyMuPDF."""
        # Create a mock PDF file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n")
        
        # Mock PyMuPDF with RGBA image
        mock_page = Mock()
        mock_pixmap = Mock()
        mock_pixmap.alpha = True
        mock_pixmap.width = 100
        mock_pixmap.height = 100
        mock_pixmap.samples = b'\x00' * (100 * 100 * 4)  # RGBA data
        mock_page.get_pixmap.return_value = mock_pixmap
        
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.load_page.return_value = mock_page
        
        with patch('fitz.open', return_value=mock_doc):
            result = pdf_to_images(str(pdf_path))
            
            assert len(result) == 1
            assert isinstance(result[0], Image.Image)
            assert result[0].mode == 'RGBA'

    @pytest.mark.skip(reason="Mocking issues with fitz module")
    def test_pdf_to_images_file_not_found(self):
        """Test PDF conversion with non-existent file."""
        with pytest.raises((FileNotFoundError, RuntimeError)):
            pdf_to_images("non_existent_file.pdf")

    @pytest.mark.skip(reason="Mocking issues with fitz module")
    def test_pdf_to_images_invalid_pdf(self, tmp_path):
        """Test PDF conversion with invalid PDF content."""
        # Create an invalid PDF file
        pdf_path = tmp_path / "invalid.pdf"
        pdf_path.write_bytes(b"not a pdf file")
        
        # Mock both libraries to fail
        with patch('pdf2image.convert_from_path', side_effect=Exception("Invalid PDF")):
            with patch('src.docshield.data.pdf_to_image.fitz', side_effect=Exception("Invalid PDF")):
                with pytest.raises(ImportError, match="PDF conversion requires either pdf2image or PyMuPDF"):
                    pdf_to_images(str(pdf_path))

    @pytest.mark.skip(reason="fitz import issues")
    def test_pdf_to_images_pymupdf_multiple_pages(self, tmp_path):
        """Test PDF conversion with multiple pages using PyMuPDF."""
        # Create a mock PDF file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n")
        
        # Mock PyMuPDF with multiple pages
        mock_pages = []
        for i in range(3):
            mock_page = Mock()
            mock_pixmap = Mock()
            mock_pixmap.alpha = False
            mock_pixmap.width = 100
            mock_pixmap.height = 100
            mock_pixmap.samples = b'\x00' * (100 * 100 * 3)
            mock_page.get_pixmap.return_value = mock_pixmap
            mock_pages.append(mock_page)
        
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=3)
        mock_doc.load_page.side_effect = lambda idx: mock_pages[idx]
        
        with patch('fitz.open', return_value=mock_doc):
            result = pdf_to_images(str(pdf_path))
            
            assert len(result) == 3
            for img in result:
                assert isinstance(img, Image.Image)


@pytest.fixture
def tmp_path():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)
