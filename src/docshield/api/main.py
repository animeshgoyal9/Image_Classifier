"""FastAPI application for DocShield document classification."""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import base64
import io
from PIL import Image
import numpy as np
import json
import os
from pathlib import Path

from ..inference import get_inference_service

# Create FastAPI app
app = FastAPI(
    title="DocShield API",
    description="Document Authentication via Deep Learning",
    version="1.0.0"
)

# Initialize inference service
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.ckpt")
inference_service = get_inference_service(MODEL_PATH)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class PredictionResult(BaseModel):
    document_type: str
    label: str
    confidence: float
    top_k: List[Dict[str, Any]]
    explanations: Dict[str, Any]
    model_version: str

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "DocShield API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/version")
async def get_version():
    """Get API version."""
    return {"version": "1.0.0"}

@app.post("/predict", response_model=PredictionResult)
async def predict_document(file: UploadFile = File(...)):
    """
    Predict document authenticity.
    
    Args:
        file: Uploaded image or PDF file
        
    Returns:
        Prediction result with confidence and explanations
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "application/pdf"]
    
    # Debug: Log the content type
    print(f"DEBUG: Received file '{file.filename}' with content_type: '{file.content_type}'")
    
    # More flexible validation - check both content_type and file extension
    file_extension = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
    valid_extensions = ['jpg', 'jpeg', 'png', 'pdf']
    
    if file.content_type not in allowed_types and file_extension not in valid_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Received: {file.content_type}, filename: {file.filename}. Allowed: {', '.join(allowed_types)}"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Process image
        if file.content_type.startswith("image"):
            # Open image with PIL
            img = Image.open(io.BytesIO(content))
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Run inference
            result = inference_service.predict(img)
            
            return PredictionResult(**result)
        else:
            # For PDFs, we would need to convert to image first
            # For now, return a placeholder
            return PredictionResult(
                document_type="unknown",
                label="real",
                confidence=0.5,
                top_k=[{"label": "unknown_real", "prob": 0.5}],
                explanations={"saliency_png_base64": ""},
                model_version="1.0.0"
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": [
            {"name": "efficientnet", "description": "EfficientNet-B0 model"},
            {"name": "vit", "description": "Vision Transformer model"}
        ]
    }

@app.get("/stats")
async def get_stats():
    """Get API statistics."""
    return {
        "total_requests": 0,
        "successful_predictions": 0,
        "average_response_time": 0.0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
