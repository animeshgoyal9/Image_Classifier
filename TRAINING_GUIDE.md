# DocShield Training Guide
## How to Train the Model with New Data

---

## 🚀 Quick Start Options

### **1. Quick Training (3 epochs, synthetic data)**
```bash
make train-quick
```
- **Time**: ~2-3 minutes
- **Use case**: Testing, validation, quick experiments
- **Data**: Automatically creates synthetic data

### **2. Standard Training (10 epochs, existing data)**
```bash
make train-model
```
- **Time**: ~10-15 minutes
- **Use case**: Production training with current data
- **Data**: Uses existing data in `data/synthetic/`

### **3. Resume Training (continue from existing model)**
```bash
make train-resume
```
- **Time**: ~15-20 minutes
- **Use case**: Continue training from last checkpoint
- **Data**: Uses existing data + previous model weights

---

## 📁 Preparing Your Own Data

### **Data Structure Requirements**

Your data must be organized in this structure:
```
data/
├── your_dataset/
│   ├── ssn_real/          # Real Social Security Number cards
│   │   ├── real_ssn_001.jpg
│   │   ├── real_ssn_002.jpg
│   │   └── ...
│   ├── ssn_fake/          # Fake Social Security Number cards
│   │   ├── fake_ssn_001.jpg
│   │   ├── fake_ssn_002.jpg
│   │   └── ...
│   ├── dl_real/           # Real Driver's License documents
│   ├── dl_fake/           # Fake Driver's License documents
│   ├── bankstmt_real/     # Real Bank Statements
│   └── bankstmt_fake/     # Fake Bank Statements
```

### **Image Requirements**
- **Format**: JPG, JPEG, PNG
- **Size**: Any size (will be resized to 224x224)
- **Channels**: RGB (3 channels)
- **Quality**: Clear, readable documents

### **Data Balance**
- **Minimum**: 10 images per class
- **Recommended**: 100+ images per class
- **Balance**: Try to have equal numbers of real/fake for each document type

---

## 🎯 Training Commands

### **Custom Data Training**
```bash
# Train with your own data
python train.py synthetic --data-dir data/your_dataset --epochs 20 --output-dir models/your_model

# Train with more epochs for better accuracy
python train.py synthetic --data-dir data/your_dataset --epochs 50 --output-dir models/your_model

# Train with custom learning rate (modify src/docshield/train/train_synthetic.py)
python train.py synthetic --data-dir data/your_dataset --epochs 30 --output-dir models/your_model
```

### **Resume Training from Checkpoint**
```bash
# Continue training from existing model
python train.py synthetic --resume-from models/best.ckpt --epochs 15

# Continue with your data
python train.py synthetic --resume-from models/best.ckpt --data-dir data/your_dataset --epochs 20
```

### **Makefile Commands**
```bash
# Quick training
make train-quick

# Standard training
make train-model

# Resume training
make train-resume

# Custom training (modify Makefile first)
make train-custom
```

---

## ⚙️ Training Configuration

### **Model Architecture**
- **Base Model**: EfficientNet-B0 (pretrained on ImageNet)
- **Head**: Custom classification head with dropout
- **Classes**: 6 classes (3 document types × 2 labels)

### **Training Parameters**
```python
# Default settings in src/docshield/train/train_synthetic.py
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
batch_size = 16
```

### **Data Augmentation**
The model uses Albumentations for data augmentation:
- **Resize**: 224x224 pixels
- **Normalize**: ImageNet mean/std values
- **Additional**: Random crops, flips, brightness/contrast adjustments

---

## 📊 Monitoring Training

### **Training Output**
During training, you'll see:
```
Epoch 1/10, Batch 0/25, Loss: 1.7912, Acc: 16.67%
Epoch 1/10, Batch 10/25, Loss: 1.2345, Acc: 45.83%
...
Epoch 1/10 - Loss: 1.1234, Accuracy: 52.34%
```

### **Expected Performance**
- **Epoch 1-5**: Loss should decrease, accuracy should increase
- **Epoch 5-10**: Gradual improvement in accuracy
- **Final Accuracy**: 85-95% on synthetic data, 70-90% on real data

### **Troubleshooting**
- **Low Accuracy**: Increase epochs, add more data, check data quality
- **Overfitting**: Reduce epochs, add more data, increase dropout
- **Slow Training**: Reduce batch size, use GPU if available

---

## 🔄 Using the Trained Model

### **1. Update Model Path**
```bash
export MODEL_PATH=models/your_model/best.ckpt
```

### **2. Restart API Server**
```bash
uvicorn src.docshield.api.main:app --host 0.0.0.0 --port 8001 --reload
```

### **3. Test the Model**
```bash
# Test with curl
curl -X POST -F "file=@test_image.jpg" http://localhost:8001/predict

# Or use the Streamlit UI
streamlit run src/docshield/ui/app.py --server.port 8501
```

---

## 🎨 Advanced Training Options

### **Hyperparameter Tuning**
Edit `src/docshield/train/train_synthetic.py` to modify:
```python
# Learning rate
optimizer = optim.AdamW(model.parameters(), lr=5e-5)  # Lower for fine-tuning

# Batch size
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model architecture
model_config = {
    'name': 'efficientnet',
    'num_classes': 6,
    'pretrained': True,
    'dropout': 0.3,  # Increase for regularization
    'head_width': 256  # Larger head
}
```

### **Transfer Learning**
```python
# Freeze base layers
for param in model.backbone.parameters():
    param.requires_grad = False

# Train only the head
for param in model.head.parameters():
    param.requires_grad = True
```

### **Multi-GPU Training**
```python
# Use multiple GPUs
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

---

## 📈 Performance Optimization

### **GPU Acceleration**
```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA version if needed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### **Data Loading Optimization**
```python
# Increase num_workers for faster data loading
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

# Use pin_memory for GPU training
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                         num_workers=4, pin_memory=True)
```

---

## 🛠️ Troubleshooting

### **Common Issues**

**1. "No module named 'albumentations'"**
```bash
pip install albumentations
```

**2. "CUDA out of memory"**
```bash
# Reduce batch size in src/docshield/train/train_synthetic.py
batch_size = 8  # Instead of 16
```

**3. "Data directory does not exist"**
```bash
# Create the data directory structure
mkdir -p data/your_dataset/{ssn_real,ssn_fake,dl_real,dl_fake,bankstmt_real,bankstmt_fake}
```

**4. "Model not improving"**
- Check data quality and balance
- Increase training epochs
- Adjust learning rate
- Add more diverse data

---

## 📋 Training Checklist

Before starting training:
- [ ] Data organized in correct structure
- [ ] Images are clear and readable
- [ ] Balanced classes (similar number of samples per class)
- [ ] GPU available (optional but recommended)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Sufficient disk space for model checkpoints

After training:
- [ ] Model saved successfully
- [ ] Accuracy is reasonable (>70%)
- [ ] Test model with sample images
- [ ] Update API server with new model
- [ ] Document training parameters and results

---

*For additional support or questions, refer to the main documentation or contact the development team.*
