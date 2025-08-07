#!/usr/bin/env python3
"""Training script for DocShield with REAL document data (no synthetic images)."""

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


class RealDocumentDataset:
    """Dataset for real document images."""
    
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = ["ssn_real", "ssn_fake", "dl_real", "dl_fake", "bankstmt_real", "bankstmt_fake"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        print(f"Loading real document data from: {self.data_dir}")
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                # Support multiple image formats
                image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
                for img_path in image_files:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                print(f"  {class_name}: {len(image_files)} images")
            else:
                print(f"  Warning: {class_name} directory not found")
        
        print(f"Total images loaded: {len(self.samples)}")
        
        # Check class balance
        class_counts = {}
        for _, label in self.samples:
            class_name = self.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("\nClass distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            # Convert PIL to numpy for Albumentations
            img_np = np.array(img)
            transformed = self.transform(image=img_np)
            img = transformed['image']
        
        return img, label


def train_model(data_dir: str, output_dir: str, epochs: int = 10, resume_from: str = None):
    """Train a DocShield model with real document data."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Validate data directory
    if not Path(data_dir).exists():
        raise ValueError(f"Data directory {data_dir} does not exist!")
    
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
    
    # Setup transforms with data augmentation
    transform = build_transforms({
        'train': [
            {'name': 'Resize', 'params': {'height': 224, 'width': 224}},
            {'name': 'HorizontalFlip', 'params': {'p': 0.5}},
            {'name': 'RandomBrightnessContrast', 'params': {'p': 0.3}},
            {'name': 'Normalize', 'params': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}
        ],
        'val': [
            {'name': 'Resize', 'params': {'height': 224, 'width': 224}},
            {'name': 'Normalize', 'params': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}
        ]
    })['train']
    
    # Create dataset
    train_dataset = RealDocumentDataset(data_dir, transform=transform)
    
    if len(train_dataset) == 0:
        raise ValueError("No images found in the data directory!")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    
    print(f"Training on {len(train_dataset)} real document images")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    model.train()
    best_accuracy = 0.0
    
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
                current_acc = 100. * correct / total
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%")
        
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
        # Save best model
        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            best_model_path = output_path / "best.ckpt"
            save_checkpoint(model, best_model_path)
            print(f"New best model saved! Accuracy: {best_accuracy:.2f}%")
    
    # Save final model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_model_path = output_path / "final.ckpt"
    save_checkpoint(model, final_model_path)
    
    print(f"\nTraining complete!")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print(f"Best model saved to: {best_model_path}")
    print(f"Final model saved to: {final_model_path}")
    
    return str(best_model_path)


def validate_data_structure(data_dir: str):
    """Validate that the data directory has the correct structure."""
    data_path = Path(data_dir)
    required_classes = ["ssn_real", "ssn_fake", "dl_real", "dl_fake", "bankstmt_real", "bankstmt_fake"]
    
    print(f"Validating data structure in: {data_path}")
    
    missing_classes = []
    total_images = 0
    
    for class_name in required_classes:
        class_dir = data_path / class_name
        if not class_dir.exists():
            missing_classes.append(class_name)
            print(f"  ❌ {class_name}: Directory not found")
        else:
            # Count images
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
            total_images += len(image_files)
            print(f"  ✅ {class_name}: {len(image_files)} images")
    
    if missing_classes:
        print(f"\n❌ Missing directories: {missing_classes}")
        print("Please create the missing directories and add images.")
        return False
    
    if total_images == 0:
        print("\n❌ No images found in any directory!")
        print("Please add document images to the appropriate directories.")
        return False
    
    print(f"\n✅ Data structure is valid! Total images: {total_images}")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DocShield model with REAL document data")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing real document images")
    parser.add_argument("--output-dir", type=str, default="models/real_model", 
                       help="Directory to save trained model")
    parser.add_argument("--epochs", type=int, default=20, 
                       help="Number of training epochs")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Path to existing model checkpoint to resume training")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate data structure without training")
    
    args = parser.parse_args()
    
    # Validate data structure
    if not validate_data_structure(args.data_dir):
        print("Data validation failed. Please fix the issues above.")
        sys.exit(1)
    
    if args.validate_only:
        print("Data validation passed! Ready for training.")
        sys.exit(0)
    
    # Train the model
    try:
        model_path = train_model(args.data_dir, args.output_dir, args.epochs, args.resume_from)
        print(f"\n🎉 Training complete! Model saved to: {model_path}")
        print(f"To use the trained model, set MODEL_PATH={model_path}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)
