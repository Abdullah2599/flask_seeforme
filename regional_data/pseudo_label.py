"""
Pseudo-Labeling Pipeline for Pakistan Regional Data
Uses CLIP and other vision models to automatically generate labels for collected images.

Features:
- CLIP-based object detection and classification
- Automatic bounding box generation using SAM (Segment Anything)
- Quality filtering and confidence scoring
- YOLO format label generation
"""

import os
import json
import torch
import clip
from PIL import Image
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import logging
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt
import matplotlib.patches as patches

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PseudoLabeler:
    def __init__(self, images_dir="regional_data/pakistan", output_dir="regional_data/pakistan_yolo"):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create YOLO structure
        for split in ['train', 'val']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Load models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.clip_model = None
        self.clip_preprocess = None
        self.blip_processor = None
        self.blip_model = None
        self.object_detector = None
        
        # Pakistan-specific class mappings
        self.pakistan_classes = self.load_pakistan_classes()
        self.confidence_threshold = 0.3
        
    def load_pakistan_classes(self):
        """Load Pakistan-specific classes"""
        classes_file = self.images_dir / 'pakistan_classes.txt'
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        else:
            # Default Pakistan classes if file doesn't exist
            return [
                'pakistani_rupee_note', 'rupee_coin', 'roti', 'naan', 'biryani',
                'chai_cup', 'shalwar_kameez', 'dupatta', 'rickshaw', 'chingchi',
                'prayer_mat', 'tasbih', 'hookah', 'truck_art', 'henna_mehndi'
            ]
    
    def load_models(self):
        """Load all required models"""
        logger.info("Loading models...")
        
        try:
            # Load CLIP
            logger.info("Loading CLIP model...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            
            # Load BLIP for image captioning
            logger.info("Loading BLIP model...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model.to(self.device)
            
            # Load object detection pipeline
            logger.info("Loading object detection model...")
            self.object_detector = pipeline(
                "object-detection",
                model="facebook/detr-resnet-50",
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("All models loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def classify_with_clip(self, image, candidate_classes):
        """Classify image using CLIP"""
        try:
            # Preprocess image
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Create text prompts
            text_prompts = [f"a photo of {cls.replace('_', ' ')}" for cls in candidate_classes]
            text_inputs = clip.tokenize(text_prompts).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                logits_per_image, logits_per_text = self.clip_model(image_input, text_inputs)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
            # Get top predictions
            top_indices = np.argsort(probs)[::-1]
            results = []
            
            for idx in top_indices[:3]:  # Top 3 predictions
                if probs[idx] > self.confidence_threshold:
                    results.append({
                        'class': candidate_classes[idx],
                        'confidence': float(probs[idx])
                    })
            
            return results
            
        except Exception as e:
            logger.warning(f"CLIP classification failed: {e}")
            return []
    
    def generate_caption(self, image):
        """Generate image caption using BLIP"""
        try:
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)
            
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            logger.warning(f"Caption generation failed: {e}")
            return ""
    
    def detect_objects(self, image):
        """Detect objects and get bounding boxes"""
        try:
            # Convert PIL to numpy array for object detection
            image_array = np.array(image)
            
            # Run object detection
            detections = self.object_detector(image_array)
            
            # Filter and process detections
            processed_detections = []
            for detection in detections:
                if detection['score'] > 0.5:  # Confidence threshold
                    box = detection['box']
                    processed_detections.append({
                        'label': detection['label'],
                        'confidence': detection['score'],
                        'box': [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
                    })
            
            return processed_detections
            
        except Exception as e:
            logger.warning(f"Object detection failed: {e}")
            return []
    
    def map_to_pakistan_class(self, detected_label, image_caption, clip_results):
        """Map detected objects to Pakistan-specific classes"""
        # Simple mapping based on keywords
        mapping_rules = {
            'cup': ['chai_cup', 'tea_cup'],
            'bowl': ['food_bowl', 'curry_bowl'],
            'person': ['person_in_traditional_dress'],
            'car': ['rickshaw', 'local_transport'],
            'motorcycle': ['motorcycle_70cc'],
            'bus': ['local_bus'],
            'book': ['quran', 'religious_book'],
            'bottle': ['water_bottle', 'drink_bottle'],
            'chair': ['plastic_chair', 'charpoy'],
            'bed': ['charpoy', 'traditional_bed']
        }
        
        # Check direct mapping
        if detected_label in mapping_rules:
            candidates = mapping_rules[detected_label]
            # Use CLIP to choose best candidate
            if clip_results:
                for result in clip_results:
                    if result['class'] in candidates:
                        return result['class'], result['confidence']
            return candidates[0], 0.7  # Default confidence
        
        # Check if it's already a Pakistan class
        if detected_label.replace(' ', '_').lower() in self.pakistan_classes:
            return detected_label.replace(' ', '_').lower(), 0.8
        
        # Use caption and CLIP results for better mapping
        caption_lower = image_caption.lower()
        for pak_class in self.pakistan_classes:
            class_keywords = pak_class.replace('_', ' ').split()
            if any(keyword in caption_lower for keyword in class_keywords):
                return pak_class, 0.6
        
        # Use CLIP results as fallback
        if clip_results:
            return clip_results[0]['class'], clip_results[0]['confidence']
        
        # Default to generic class
        return 'unknown_object', 0.3
    
    def convert_to_yolo_format(self, box, image_width, image_height):
        """Convert bounding box to YOLO format"""
        x_min, y_min, x_max, y_max = box
        
        # Calculate center coordinates and dimensions
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
        
        # Normalize to image dimensions
        center_x /= image_width
        center_y /= image_height
        width /= image_width
        height /= image_height
        
        return center_x, center_y, width, height
    
    def process_image(self, image_path, category_hint=None):
        """Process single image and generate pseudo-labels"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image_width, image_height = image.size
            
            # Get category-specific classes if hint provided
            if category_hint:
                candidate_classes = [cls for cls in self.pakistan_classes 
                                   if category_hint.lower() in cls.lower()]
                if not candidate_classes:
                    candidate_classes = self.pakistan_classes[:20]  # Top 20 classes
            else:
                candidate_classes = self.pakistan_classes[:20]
            
            # Run CLIP classification
            clip_results = self.classify_with_clip(image, candidate_classes)
            
            # Generate caption
            caption = self.generate_caption(image)
            
            # Detect objects
            detections = self.detect_objects(image)
            
            # Process detections
            yolo_labels = []
            
            if detections:
                # Use detected objects
                for detection in detections:
                    # Map to Pakistan class
                    pak_class, confidence = self.map_to_pakistan_class(
                        detection['label'], caption, clip_results
                    )
                    
                    if pak_class in self.pakistan_classes and confidence > self.confidence_threshold:
                        class_id = self.pakistan_classes.index(pak_class)
                        
                        # Convert to YOLO format
                        center_x, center_y, width, height = self.convert_to_yolo_format(
                            detection['box'], image_width, image_height
                        )
                        
                        yolo_labels.append({
                            'class_id': class_id,
                            'class_name': pak_class,
                            'confidence': confidence,
                            'bbox': [center_x, center_y, width, height]
                        })
            
            else:
                # No objects detected, use CLIP classification for whole image
                if clip_results:
                    best_result = clip_results[0]
                    if best_result['confidence'] > self.confidence_threshold:
                        class_id = self.pakistan_classes.index(best_result['class'])
                        
                        # Use whole image as bounding box
                        yolo_labels.append({
                            'class_id': class_id,
                            'class_name': best_result['class'],
                            'confidence': best_result['confidence'],
                            'bbox': [0.5, 0.5, 1.0, 1.0]  # Whole image
                        })
            
            return yolo_labels, caption
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return [], ""
    
    def save_yolo_labels(self, image_path, labels, split='train'):
        """Save labels in YOLO format"""
        # Copy image to output directory
        image_filename = image_path.name
        output_image_path = self.output_dir / split / 'images' / image_filename
        
        # Copy image
        import shutil
        shutil.copy2(image_path, output_image_path)
        
        # Create label file
        label_filename = image_filename.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = self.output_dir / split / 'labels' / label_filename
        
        # Write labels
        with open(label_path, 'w') as f:
            for label in labels:
                bbox = label['bbox']
                f.write(f"{label['class_id']} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        
        return output_image_path, label_path
    
    def process_all_images(self, train_split=0.8):
        """Process all collected images and generate pseudo-labels"""
        logger.info("Starting pseudo-labeling process...")
        
        if not self.load_models():
            raise RuntimeError("Failed to load required models")
        
        # Find all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(self.images_dir.rglob(ext)))
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process images
        processed_count = 0
        labeled_count = 0
        processing_stats = {'train': 0, 'val': 0}
        
        for i, image_path in enumerate(tqdm(image_files, desc="Processing images")):
            try:
                # Determine category hint from path
                category_hint = image_path.parent.name if image_path.parent.name != 'pakistan' else None
                
                # Process image
                labels, caption = self.process_image(image_path, category_hint)
                
                if labels:
                    # Determine split
                    split = 'train' if i < len(image_files) * train_split else 'val'
                    
                    # Save labels
                    output_image_path, label_path = self.save_yolo_labels(image_path, labels, split)
                    
                    labeled_count += 1
                    processing_stats[split] += 1
                
                processed_count += 1
                
                # Save progress periodically
                if processed_count % 100 == 0:
                    self.save_processing_stats(processed_count, labeled_count, processing_stats)
                
            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")
                continue
        
        # Final statistics
        self.save_processing_stats(processed_count, labeled_count, processing_stats)
        self.create_data_yaml()
        
        logger.info(f"Pseudo-labeling completed!")
        logger.info(f"Processed: {processed_count} images")
        logger.info(f"Labeled: {labeled_count} images")
        logger.info(f"Train: {processing_stats['train']}, Val: {processing_stats['val']}")
        
        return processing_stats
    
    def save_processing_stats(self, processed, labeled, split_stats):
        """Save processing statistics"""
        stats = {
            'processed_images': processed,
            'labeled_images': labeled,
            'split_distribution': split_stats,
            'pakistan_classes': self.pakistan_classes,
            'total_classes': len(self.pakistan_classes),
            'confidence_threshold': self.confidence_threshold
        }
        
        stats_path = self.output_dir / 'pseudo_labeling_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def create_data_yaml(self):
        """Create data.yaml for YOLO training"""
        data_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(self.pakistan_classes),
            'names': self.pakistan_classes
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            import yaml
            yaml.dump(data_config, f, default_flow_style=False)
        
        logger.info(f"Data YAML created: {yaml_path}")
        return yaml_path
    
    def visualize_sample_labels(self, num_samples=5):
        """Visualize sample labeled images"""
        logger.info("Creating sample visualizations...")
        
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Get sample images
        train_images = list((self.output_dir / 'train' / 'images').glob('*.jpg'))[:num_samples]
        
        for img_path in train_images:
            try:
                # Load image
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w = image.shape[:2]
                
                # Load labels
                label_path = self.output_dir / 'train' / 'labels' / f"{img_path.stem}.txt"
                
                if label_path.exists():
                    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                    ax.imshow(image)
                    
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                center_x, center_y, width, height = map(float, parts[1:5])
                                
                                # Convert to pixel coordinates
                                x = (center_x - width/2) * w
                                y = (center_y - height/2) * h
                                w_box = width * w
                                h_box = height * h
                                
                                # Draw bounding box
                                rect = patches.Rectangle(
                                    (x, y), w_box, h_box,
                                    linewidth=2, edgecolor='red', facecolor='none'
                                )
                                ax.add_patch(rect)
                                
                                # Add label
                                class_name = self.pakistan_classes[class_id]
                                ax.text(x, y-5, class_name, color='red', fontsize=10,
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
                    
                    ax.set_title(f"Sample: {img_path.name}")
                    ax.axis('off')
                    
                    # Save visualization
                    viz_path = viz_dir / f"sample_{img_path.stem}.png"
                    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
                    plt.close()
                
            except Exception as e:
                logger.warning(f"Failed to visualize {img_path}: {e}")
        
        logger.info(f"Sample visualizations saved to: {viz_dir}")

def main():
    """Main function to run pseudo-labeling"""
    labeler = PseudoLabeler()
    
    print("🏷️  Starting Pakistan Regional Data Pseudo-Labeling")
    print("=" * 50)
    
    try:
        # Process all images
        stats = labeler.process_all_images()
        
        # Create visualizations
        labeler.visualize_sample_labels()
        
        print("\n📊 Pseudo-Labeling Summary:")
        print("=" * 30)
        print(f"✅ Train Images: {stats['train']}")
        print(f"✅ Val Images: {stats['val']}")
        print(f"🏷️  Total Classes: {len(labeler.pakistan_classes)}")
        print(f"📁 Output Directory: {labeler.output_dir}")
        
        print("\n🎯 Ready for training integration!")
        print("   - Merge with main dataset using merge_datasets.py")
        print("   - Or train separately for Pakistan-specific model")
        
    except Exception as e:
        print(f"\n❌ Pseudo-labeling failed: {e}")
        raise

if __name__ == "__main__":
    main()
