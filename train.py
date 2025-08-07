#!/usr/bin/env python3
"""Training script wrapper for DocShield.

This script provides easy access to training functionality from the root directory.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Main training script wrapper."""
    if len(sys.argv) < 2:
        print("DocShield Training Script")
        print("=" * 30)
        print("Usage:")
        print("  python train.py synthetic [options]  # Train with synthetic data")
        print("  python train.py real [options]       # Train with real data")
        print("  python train.py validate             # Validate data structure")
        print("  python train.py advanced [options]   # Advanced training with YAML config")
        print("")
        print("Examples:")
        print("  python train.py synthetic --epochs 5")
        print("  python train.py real --data-dir data/my_data --epochs 20")
        print("  python train.py validate --data-dir data/real_documents")
        print("  python train.py advanced --config configs/train.yaml")
        return
    
    command = sys.argv[1]
    
    if command == "synthetic":
        # Call synthetic training script
        script_path = Path(__file__).parent / "src" / "docshield" / "train" / "train_synthetic.py"
        args = sys.argv[2:]
        subprocess.run([sys.executable, str(script_path)] + args)
    
    elif command == "real":
        # Call real data training script
        script_path = Path(__file__).parent / "src" / "docshield" / "train" / "train_real_data.py"
        args = sys.argv[2:]
        subprocess.run([sys.executable, str(script_path)] + args)
    
    elif command == "validate":
        # Call validation
        script_path = Path(__file__).parent / "src" / "docshield" / "train" / "train_real_data.py"
        args = sys.argv[2:] + ["--validate-only"]
        subprocess.run([sys.executable, str(script_path)] + args)
    
    elif command == "advanced":
        # Call advanced training script with YAML config
        script_path = Path(__file__).parent / "src" / "docshield" / "train" / "train_advanced.py"
        args = sys.argv[2:]
        subprocess.run([sys.executable, str(script_path)] + args)
    
    else:
        print(f"Unknown command: {command}")
        print("Use 'synthetic', 'real', or 'validate'")

if __name__ == "__main__":
    main()
