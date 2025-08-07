"""Streamlit UI for DocShield document classification."""

import streamlit as st
import requests
import base64
import io
from PIL import Image
import tempfile
import os
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="DocShield - Document Authentication",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 1rem;
    }
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f9f9f9;
    }
    .result-box {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        background-color: #f8f9fa;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = "http://localhost:8001"

def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def predict_document(file_bytes, filename):
    """Send document to API for prediction."""
    try:
        # Determine MIME type based on file extension
        import mimetypes
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type is None:
            # Fallback for common image types
            if filename.lower().endswith(('.jpg', '.jpeg')):
                mime_type = "image/jpeg"
            elif filename.lower().endswith('.png'):
                mime_type = "image/png"
            elif filename.lower().endswith('.pdf'):
                mime_type = "application/pdf"
            else:
                mime_type = "application/octet-stream"
        
        files = {"file": (filename, file_bytes, mime_type)}
        response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def display_prediction_result(result):
    """Display prediction results in a nice format."""
    if not result:
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Prediction Results")
        
        # Document type and label
        doc_type = result.get("document_type", "Unknown")
        label = result.get("label", "Unknown")
        confidence = result.get("confidence", 0.0)
        
        st.write(f"**Document Type:** {doc_type.upper()}")
        st.write(f"**Classification:** {label.upper()}")
        
        # Confidence indicator
        confidence_pct = confidence * 100
        if confidence_pct >= 80:
            confidence_class = "confidence-high"
        elif confidence_pct >= 60:
            confidence_class = "confidence-medium"
        else:
            confidence_class = "confidence-low"
        
        st.markdown(f"**Confidence:** <span class='{confidence_class}'>{confidence_pct:.1f}%</span>", 
                   unsafe_allow_html=True)
        
        # Progress bar for confidence
        st.progress(confidence)
        
        # Top-k predictions
        top_k = result.get("top_k", [])
        if top_k:
            st.subheader("🏆 Top Predictions")
            for i, pred in enumerate(top_k[:3]):
                prob_pct = pred.get("prob", 0) * 100
                st.write(f"{i+1}. **{pred.get('label', 'Unknown')}**: {prob_pct:.1f}%")
    
    with col2:
        st.subheader("🔍 Explanations")
        
        # Display saliency map if available
        explanations = result.get("explanations", {})
        saliency_data = explanations.get("saliency_png_base64")
        
        if saliency_data:
            try:
                # Decode base64 image
                image_data = base64.b64decode(saliency_data)
                image = Image.open(io.BytesIO(image_data))
                st.image(image, caption="Saliency Map", use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display saliency map: {str(e)}")
        else:
            st.info("No saliency map available for this prediction.")
        
        # Model version
        model_version = result.get("model_version", "Unknown")
        st.caption(f"Model Version: {model_version}")

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ DocShield</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Document Authentication via Deep Learning</p>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API status
        api_status = check_api_health()
        if api_status:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Not Available")
            st.info("Please start the API server first:")
            st.code("uvicorn src.docshield.api.main:app --host 0.0.0.0 --port 8001")
            return
        
        # Model selection (placeholder for future)
        st.subheader("🤖 Model Settings")
        model_name = st.selectbox(
            "Select Model",
            ["efficientnet", "vit"],
            index=0
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Minimum confidence required for prediction"
        )
        
        # Top-k predictions
        top_k = st.slider(
            "Top-K Predictions",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of top predictions to show"
        )
        
        st.divider()
        
        # About section
        st.subheader("ℹ️ About")
        st.markdown("""
        DocShield is an AI-powered document authentication system that can detect fraudulent documents such as:
        
        • Social Security Numbers (SSN)
        • Driver's Licenses (DL)
        • Bank Statements
        
        Upload an image or PDF to get started!
        """)
    
    # Main content area
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload Document",
        type=["jpg", "jpeg", "png", "pdf"],
        help="Upload an image or PDF file for classification"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display uploaded file
        st.subheader("📄 Uploaded Document")
        
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.1f} KB",
            "File type": uploaded_file.type
        }
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**File Details:**")
            for key, value in file_details.items():
                st.write(f"{key}: {value}")
        
        with col2:
            # Display image preview
            if uploaded_file.type.startswith("image"):
                # Reset file pointer to beginning
                uploaded_file.seek(0)
                image = Image.open(uploaded_file)
                st.image(image, caption="Document Preview", use_container_width=True)
            else:
                st.info("📄 PDF file uploaded")
        
        # Prediction button
        if st.button("🔍 Analyze Document", type="primary"):
            with st.spinner("Analyzing document..."):
                # Reset file pointer to beginning and get file bytes
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                
                # Make prediction
                result = predict_document(file_bytes, uploaded_file.name)
                
                if result:
                    # Check confidence threshold
                    confidence = result.get("confidence", 0.0)
                    if confidence >= confidence_threshold:
                        st.success("✅ Analysis Complete!")
                        display_prediction_result(result)
                    else:
                        st.warning(f"⚠️ Low confidence ({confidence:.1%}). Consider reviewing manually.")
                        display_prediction_result(result)
                else:
                    st.error("❌ Failed to analyze document. Please try again.")
    
    # Example section
    with st.expander("📖 How to Use"):
        st.markdown("""
        ### Getting Started
        
        1. **Start the API Server** (if not already running):
           ```bash
           uvicorn src.docshield.api.main:app --host 0.0.0.0 --port 8001
           ```
        
        2. **Upload a Document**:
           - Supported formats: JPG, PNG, PDF
           - Maximum file size: 10MB
           - Single page PDFs recommended
        
        3. **View Results**:
           - Document classification (real/fake)
           - Confidence score
           - Saliency map (if available)
           - Top predictions
        
        ### Understanding Results
        
        - **High Confidence (≥80%)**: Reliable prediction
        - **Medium Confidence (60-79%)**: Moderate reliability
        - **Low Confidence (<60%)**: Manual review recommended
        
        ### Supported Document Types
        
        - **SSN**: Social Security Number cards
        - **DL**: Driver's License documents
        - **Bank Statements**: Financial documents
        """)
    
    # Footer
    st.divider()
    st.markdown(
        "<p style='text-align: center; color: #666;'>🛡️ DocShield - Secure Document Authentication</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
