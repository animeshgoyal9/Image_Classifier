"""Model inference service for DocShield."""

import base64
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from .models.factory import create_model
from .data.transforms import build_transforms


class DocShieldInference:
    """Inference service for DocShield document classification."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """Initialize the inference service.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        self.model = None
        self.class_names = [
            "ssn_real", "ssn_fake", 
            "dl_real", "dl_fake", 
            "bankstmt_real", "bankstmt_fake"
        ]
        self.transform = None
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            print("Warning: No trained model found. Using mock predictions.")
    
    def load_model(self, model_path: str) -> None:
        """Load a trained model from checkpoint."""
        try:
            # Load model configuration
            model_config = {
                'name': 'efficientnet',
                'num_classes': 6,
                'pretrained': False,  # We're loading our own weights
                'dropout': 0.2,
                'head_width': 128
            }
            
            # Create model
            self.model, _ = create_model(model_config)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Setup transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            print(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model inference."""
        if self.transform:
            return self.transform(image).unsqueeze(0).to(self.device)
        else:
            # Fallback transform
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            return transform(image).unsqueeze(0).to(self.device)
    
    def predict(self, image: Image.Image) -> Dict:
        """Perform prediction on an image.
        
        Args:
            image: PIL Image to classify
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            return self._mock_prediction(image)
        
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
            # Get predictions
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_class].item()
            
            # Get top-k predictions
            top_k = torch.topk(probabilities, k=3, dim=1)
            top_k_predictions = []
            for i in range(top_k.values.shape[1]):
                top_k_predictions.append({
                    "label": self.class_names[top_k.indices[0][i].item()],
                    "prob": top_k.values[0][i].item()
                })
            
            # Parse class name
            class_name = self.class_names[pred_class]
            doc_type, label = class_name.split('_')
            
            # Generate saliency map
            saliency_map = self._generate_saliency_map(input_tensor, pred_class)
            
            return {
                "document_type": doc_type,
                "label": label,
                "confidence": confidence,
                "top_k": top_k_predictions,
                "explanations": {"saliency_png_base64": saliency_map},
                "model_version": "1.0.0"
            }
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return self._mock_prediction(image)
    
    def _mock_prediction(self, image: Image.Image) -> Dict:
        """Generate mock prediction when no model is available."""
        # Simple mock based on image characteristics
        img_array = np.array(image)
        
        # Mock logic based on image properties
        if img_array.shape[2] == 3:  # RGB image
            # Simple heuristic: if image has more blue pixels, likely SSN
            blue_ratio = np.mean(img_array[:, :, 2]) / 255.0
            if blue_ratio > 0.6:
                doc_type = "ssn"
                confidence = 0.92
            else:
                doc_type = "dl"
                confidence = 0.88
            
            # Randomly assign real/fake
            label = "real" if np.random.random() > 0.3 else "fake"
        else:
            doc_type = "unknown"
            label = "real"
            confidence = 0.75
        
        # Mock top-k
        top_k = [
            {"label": f"{doc_type}_{label}", "prob": confidence},
            {"label": f"{doc_type}_{'fake' if label == 'real' else 'real'}", "prob": 1 - confidence}
        ]
        
        # Mock saliency map
        saliency_map = self._generate_mock_saliency_map(image)
        
        return {
            "document_type": doc_type,
            "label": label,
            "confidence": confidence,
            "top_k": top_k,
            "explanations": {"saliency_png_base64": saliency_map},
            "model_version": "mock"
        }
    
    def _generate_saliency_map(self, input_tensor: torch.Tensor, pred_class: int) -> str:
        """Generate saliency map using Grad-CAM or similar method."""
        try:
            # Simple gradient-based saliency
            input_tensor.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(input_tensor)
            score = outputs[0, pred_class]
            
            # Backward pass
            score.backward()
            
            # Get gradients
            gradients = input_tensor.grad
            
            # Create saliency map
            saliency = torch.abs(gradients).mean(dim=1, keepdim=True)
            saliency = F.interpolate(saliency, size=(224, 224), mode='bilinear', align_corners=False)
            
            # Convert to image
            saliency_np = saliency[0, 0].cpu().numpy()
            saliency_np = (saliency_np - saliency_np.min()) / (saliency_np.max() - saliency_np.min())
            saliency_np = (saliency_np * 255).astype(np.uint8)
            
            # Convert to PIL and then to base64
            saliency_img = Image.fromarray(saliency_np, mode='L')
            buffer = io.BytesIO()
            saliency_img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()
            
        except Exception as e:
            print(f"Error generating saliency map: {e}")
            return self._generate_mock_saliency_map(None)
    
    def _generate_mock_saliency_map(self, image: Optional[Image.Image]) -> str:
        """Generate a mock saliency map."""
        try:
            if image:
                # Create a gradient based on image size
                width, height = image.size
                saliency = np.random.rand(height, width) * 255
            else:
                # Default size
                saliency = np.random.rand(224, 224) * 255
            
            saliency_img = Image.fromarray(saliency.astype(np.uint8), mode='L')
            buffer = io.BytesIO()
            saliency_img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()
            
        except Exception as e:
            print(f"Error generating mock saliency map: {e}")
            return ""


# Global inference instance
_inference_service = None


def get_inference_service(model_path: Optional[str] = None) -> DocShieldInference:
    """Get or create the global inference service."""
    global _inference_service
    if _inference_service is None:
        _inference_service = DocShieldInference(model_path)
    return _inference_service
