#!/usr/bin/env python3
"""
Simplified YOLOv8 Training Script - COCO Only
Train YOLOv8 on COCO dataset first, then expand to other datasets
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_coco.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class COCOTrainer:
    """Simplified trainer for COCO dataset only"""
    
    def __init__(self, output_dir="training_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Check for GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def create_coco_yaml(self):
        """Create YAML config for COCO dataset"""
        coco_yaml = {
            'path': 'yolo_datasets/coco_yolo',
            'train': 'train/images',
            'val': 'val/images',
            'test': 'val/images',  # Use val as test for now
            'nc': 80,
            'names': [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush'
            ]
        }
        
        yaml_path = self.output_dir / "coco_data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(coco_yaml, f, default_flow_style=False)
        
        logger.info(f"COCO YAML config created: {yaml_path}")
        return yaml_path
    
    def train_model(self, model_size='n', epochs=100, batch_size=16, imgsz=640):
        """Train YOLOv8 model on COCO dataset"""
        
        # Create data config
        data_yaml = self.create_coco_yaml()
        
        # Initialize model
        model_name = f'yolov8{model_size}.pt'
        logger.info(f"Initializing {model_name} model...")
        
        model = YOLO(model_name)
        
        # Training arguments
        train_args = {
            'data': str(data_yaml),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'device': self.device,
            'project': str(self.output_dir),
            'name': f'yolov8{model_size}_coco_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'save': True,
            'save_period': 10,
            'val': True,
            'plots': True,
            'verbose': True,
            'patience': 50,
            'optimizer': 'AdamW',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'cos_lr': True,
            'close_mosaic': 10,
            'resume': False
        }
        
        logger.info("Starting training...")
        logger.info(f"Training arguments: {train_args}")
        
        try:
            # Train the model
            results = model.train(**train_args)
            
            logger.info("Training completed successfully!")
            logger.info(f"Best model saved to: {results.save_dir}")
            
            # Export model
            best_model_path = results.save_dir / 'weights' / 'best.pt'
            if best_model_path.exists():
                logger.info("Exporting model to different formats...")
                best_model = YOLO(str(best_model_path))
                
                # Export to ONNX for deployment
                best_model.export(format='onnx', dynamic=False, simplify=True)
                logger.info("Model exported to ONNX format")
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def validate_model(self, model_path):
        """Validate trained model"""
        logger.info(f"Validating model: {model_path}")
        
        model = YOLO(model_path)
        data_yaml = self.output_dir / "coco_data.yaml"
        
        # Run validation
        results = model.val(data=str(data_yaml))
        
        logger.info("Validation Results:")
        logger.info(f"mAP@0.5: {results.box.map50:.4f}")
        logger.info(f"mAP@0.5:0.95: {results.box.map:.4f}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 on COCO dataset')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--output', type=str, default='training_output', help='Output directory')
    parser.add_argument('--validate', action='store_true', help='Run validation after training')
    
    args = parser.parse_args()
    
    print("🚀 Starting COCO-Only YOLOv8 Training")
    print("=" * 50)
    print(f"📊 Model: YOLOv8{args.model}")
    print(f"📁 Output: {args.output}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📦 Batch Size: {args.batch}")
    print(f"🖼️  Image Size: {args.imgsz}")
    
    # Initialize trainer
    trainer = COCOTrainer(output_dir=args.output)
    
    try:
        # Train model
        results = trainer.train_model(
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz
        )
        
        # Validate if requested
        if args.validate:
            best_model = results.save_dir / 'weights' / 'best.pt'
            if best_model.exists():
                trainer.validate_model(str(best_model))
        
        print("\n✅ Training completed successfully!")
        print(f"📁 Results saved to: {results.save_dir}")
        print(f"🎯 Best model: {results.save_dir}/weights/best.pt")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
