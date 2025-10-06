#!/usr/bin/env python3
"""
Quick COCO Training Script
Run this in Google Colab to train YOLOv8 on COCO dataset only
"""

import os
import sys
from pathlib import Path

def main():
    print("🚀 Quick COCO Training Setup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("yolo_datasets/coco_yolo").exists():
        print("❌ COCO YOLO dataset not found!")
        print("Please run the dataset conversion first:")
        print("python data_pipeline/convert_to_yolo.py")
        return
    
    # Check COCO dataset structure
    coco_train_images = Path("yolo_datasets/coco_yolo/train/images")
    coco_train_labels = Path("yolo_datasets/coco_yolo/train/labels")
    coco_val_images = Path("yolo_datasets/coco_yolo/val/images")
    coco_val_labels = Path("yolo_datasets/coco_yolo/val/labels")
    
    print(f"📊 Checking COCO dataset structure:")
    print(f"   Train images: {coco_train_images.exists()} ({len(list(coco_train_images.glob('*.jpg'))) if coco_train_images.exists() else 0} files)")
    print(f"   Train labels: {coco_train_labels.exists()} ({len(list(coco_train_labels.glob('*.txt'))) if coco_train_labels.exists() else 0} files)")
    print(f"   Val images: {coco_val_images.exists()} ({len(list(coco_val_images.glob('*.jpg'))) if coco_val_images.exists() else 0} files)")
    print(f"   Val labels: {coco_val_labels.exists()} ({len(list(coco_val_labels.glob('*.txt'))) if coco_val_labels.exists() else 0} files)")
    
    if not all([coco_train_images.exists(), coco_train_labels.exists(), coco_val_images.exists(), coco_val_labels.exists()]):
        print("❌ COCO dataset structure incomplete!")
        return
    
    # Run COCO training
    print("\n🚂 Starting COCO training...")
    os.system("python training/train_coco_only.py --model n --epochs 50 --batch 16 --validate")

if __name__ == "__main__":
    main()
