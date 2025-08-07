#!/usr/bin/env python3
"""Simple training script for DocShield model."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from pathlib import Path

from docshield.models.factory import create_model
from docshield.data.transforms import build_transforms
from docshield.train.utils import save_checkpoint, load_checkpoint, set_seed


def create_synthetic_dataset(output_dir: str, num_samples: int = 100):
    """Create a synthetic dataset for training."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create class directories
    classes = ["ssn_real", "ssn_fake", "dl_real", "dl_fake", "bankstmt_real", "bankstmt_fake"]
    for class_name in classes:
        (output_path / class_name).mkdir(exist_ok=True)
    
    print(f"Creating synthetic dataset with {num_samples} samples per class...")
    
    for class_name in classes:
        class_dir = output_path / class_name
        for i in range(num_samples):
            # Create synthetic image based on class
            if "ssn" in class_name:
                # Blue-tinted image for SSN
                img_array = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
                img_array[:, :, 2] = np.random.randint(150, 255, (224, 224), dtype=np.uint8)  # More blue
            elif "dl" in class_name:
                # Green-tinted image for DL
                img_array = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
                img_array[:, :, 1] = np.random.randint(150, 255, (224, 224), dtype=np.uint8)  # More green
            else:
                # Red-tinted image for bank statements
                img_array = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
                img_array[:, :, 0] = np.random.randint(150, 255, (224, 224), dtype=np.uint8)  # More red
            
            # Add some text-like patterns
            if "fake" in class_name:
                # Add noise for fake documents
                noise = np.random.randint(0, 50, img_array.shape, dtype=np.uint8)
                img_array = np.clip(img_array + noise, 0, 255)
            
            # Save image
            img = Image.fromarray(img_array)
            img.save(class_dir / f"{class_name}_{i:03d}.jpg")
    
    print(f"Dataset created at {output_path}")


class SimpleDataset:
    """Simple dataset for training."""
    
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = ["ssn_real", "ssn_fake", "dl_real", "dl_fake", "bankstmt_real", "bankstmt_fake"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            # Convert PIL to numpy for Albumentations
            import numpy as np
            img_np = np.array(img)
            transformed = self.transform(image=img_np)
            img = transformed['image']
        
        return img, label


def train_model(data_dir: str, output_dir: str, epochs: int = 10, resume_from: str = None):
    """Train a DocShield model."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model_config = {
        'name': 'efficientnet',
        'num_classes': 6,
        'pretrained': True,
        'dropout': 0.2,
        'head_width': 128
    }
    model, _ = create_model(model_config)
    model.to(device)
    
    # Load existing model if resuming
    start_epoch = 0
    if resume_from and Path(resume_from).exists():
        print(f"Loading existing model from {resume_from}")
        checkpoint = load_checkpoint(model, resume_from)
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Setup transforms
    transform = build_transforms({
        'train': [
            {'name': 'Resize', 'params': {'height': 224, 'width': 224}},
            {'name': 'Normalize', 'params': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}
        ],
        'val': [
            {'name': 'Resize', 'params': {'height': 224, 'width': 224}},
            {'name': 'Normalize', 'params': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}
        ]
    })['train']
    
    # Create datasets
    train_dataset = SimpleDataset(data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    
    print(f"Training on {len(train_dataset)} samples")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    model.train()
    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%")
        
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "best.ckpt"
    save_checkpoint(model, model_path)
    
    print(f"Model saved to {model_path}")
    return str(model_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DocShield model")
    parser.add_argument("--data-dir", type=str, default="data/synthetic", 
                       help="Directory containing training data")
    parser.add_argument("--output-dir", type=str, default="models", 
                       help="Directory to save trained model")
    parser.add_argument("--epochs", type=int, default=10, 
                       help="Number of training epochs")
    parser.add_argument("--create-synthetic", action="store_true", 
                       help="Create synthetic dataset for training")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Path to existing model checkpoint to resume training")
    
    args = parser.parse_args()
    
    if args.create_synthetic:
        create_synthetic_dataset(args.data_dir)
    
    if Path(args.data_dir).exists():
        model_path = train_model(args.data_dir, args.output_dir, args.epochs, args.resume_from)
        print(f"\nTraining complete! Model saved to: {model_path}")
        print(f"To use the trained model, set MODEL_PATH={model_path}")
    else:
        print(f"Data directory {args.data_dir} does not exist!")
        print("Use --create-synthetic to create a synthetic dataset for training.")
