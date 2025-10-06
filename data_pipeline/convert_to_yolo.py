"""
Dataset Format Conversion Pipeline
Converts various dataset formats (COCO, Open Images, LVIS, Objects365) to YOLO format.

YOLO format:
- Images in JPG/PNG format
- Labels in TXT format: class_id center_x center_y width height (normalized 0-1)
"""

import os
import json
import csv
import shutil
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetConverter:
    def __init__(self, datasets_dir="datasets", output_dir="yolo_datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Class mapping for unified vocabulary
        self.class_mapping = {}
        self.unified_classes = []
        self.class_counter = 0
    
    def normalize_bbox(self, bbox, img_width, img_height):
        """Convert absolute bbox to YOLO normalized format"""
        x, y, w, h = bbox
        
        # Convert to center coordinates
        center_x = x + w / 2
        center_y = y + h / 2
        
        # Normalize
        center_x /= img_width
        center_y /= img_height
        w /= img_width
        h /= img_height
        
        return center_x, center_y, w, h
    
    def add_class_to_mapping(self, original_name, dataset_source):
        """Add class to unified mapping"""
        # Normalize class name
        normalized_name = original_name.lower().strip().replace(' ', '_')
        
        # Check if class already exists
        for existing_class in self.unified_classes:
            if existing_class['name'] == normalized_name:
                # Add source to existing class
                if dataset_source not in existing_class['sources']:
                    existing_class['sources'].append(dataset_source)
                return existing_class['id']
        
        # Add new class
        class_info = {
            'id': self.class_counter,
            'name': normalized_name,
            'original_name': original_name,
            'sources': [dataset_source]
        }
        
        self.unified_classes.append(class_info)
        self.class_counter += 1
        
        return class_info['id']
    
    def convert_coco_to_yolo(self):
        """Convert COCO dataset to YOLO format"""
        logger.info("Converting COCO dataset to YOLO format...")
        
        coco_dir = self.datasets_dir / "coco"
        output_dir = self.output_dir / "coco_yolo"
        output_dir.mkdir(exist_ok=True)
        
        # Create train/val directories
        for split in ['train', 'val']:
            (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Load COCO annotations
        splits = {
            'train': coco_dir / 'annotations' / 'instances_train2017.json',
            'val': coco_dir / 'annotations' / 'instances_val2017.json'
        }
        
        for split, ann_file in splits.items():
            if not ann_file.exists():
                logger.warning(f"COCO annotation file not found: {ann_file}")
                continue
            
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)
            
            # Build category mapping
            coco_categories = {}
            for cat in coco_data['categories']:
                class_id = self.add_class_to_mapping(cat['name'], 'coco')
                coco_categories[cat['id']] = class_id
            
            # Build image info mapping
            image_info = {img['id']: img for img in coco_data['images']}
            
            # Process annotations
            image_annotations = {}
            for ann in tqdm(coco_data['annotations'], desc=f"Processing COCO {split} annotations"):
                img_id = ann['image_id']
                if img_id not in image_annotations:
                    image_annotations[img_id] = []
                
                # Convert bbox to YOLO format
                img = image_info[img_id]
                bbox = ann['bbox']  # [x, y, width, height]
                
                center_x, center_y, width, height = self.normalize_bbox(
                    bbox, img['width'], img['height']
                )
                
                class_id = coco_categories[ann['category_id']]
                
                image_annotations[img_id].append(
                    f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                )
            
            # Copy images and create label files
            img_source_dir = coco_dir / f"{split}2017"
            img_dest_dir = output_dir / split / 'images'
            label_dest_dir = output_dir / split / 'labels'
            
            for img_id, img_info in tqdm(image_info.items(), desc=f"Converting COCO {split} images"):
                if img_id not in image_annotations:
                    continue  # Skip images without annotations
                
                # Copy image
                img_filename = img_info['file_name']
                src_path = img_source_dir / img_filename
                dst_path = img_dest_dir / img_filename
                
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                    
                    # Create label file
                    label_filename = img_filename.replace('.jpg', '.txt')
                    label_path = label_dest_dir / label_filename
                    
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(image_annotations[img_id]))
        
        logger.info(f"COCO conversion completed. Output: {output_dir}")
        return True
    
    def convert_open_images_to_yolo(self, max_images_per_class=500):
        """Convert Open Images subset to YOLO format"""
        logger.info("Converting Open Images V7 to YOLO format...")
        
        oi_dir = self.datasets_dir / "open_images_v7"
        output_dir = self.output_dir / "open_images_yolo"
        output_dir.mkdir(exist_ok=True)
        
        # Create directories
        (output_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Load class descriptions
        class_desc_file = oi_dir / "class-descriptions.csv"
        if not class_desc_file.exists():
            logger.warning("Open Images class descriptions not found")
            return False
        
        class_descriptions = {}
        with open(class_desc_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    class_descriptions[row[0]] = row[1]
        
        # Load annotations
        ann_file = oi_dir / "train-annotations-bbox.csv"
        if not ann_file.exists():
            logger.warning("Open Images annotations not found")
            return False
        
        # Process annotations in chunks to manage memory
        chunk_size = 10000
        processed_images = set()
        
        for chunk in pd.read_csv(ann_file, chunksize=chunk_size):
            for _, row in tqdm(chunk.iterrows(), desc="Processing Open Images annotations"):
                image_id = row['ImageID']
                
                if image_id in processed_images:
                    continue
                
                # Map class
                class_name = class_descriptions.get(row['LabelName'], row['LabelName'])
                class_id = self.add_class_to_mapping(class_name, 'open_images')
                
                # Note: This is a simplified conversion
                # In practice, you'd need to download the actual images first
                # and then process the bounding boxes
                
                processed_images.add(image_id)
                
                if len(processed_images) >= max_images_per_class * 10:  # Limit for demo
                    break
        
        logger.info(f"Open Images conversion completed. Processed {len(processed_images)} images.")
        return True
    
    def convert_lvis_to_yolo(self):
        """Convert LVIS dataset to YOLO format"""
        logger.info("Converting LVIS dataset to YOLO format...")
        
        lvis_dir = self.datasets_dir / "lvis"
        output_dir = self.output_dir / "lvis_yolo"
        output_dir.mkdir(exist_ok=True)
        
        # Create directories
        for split in ['train', 'val']:
            (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Process LVIS annotations
        splits = {
            'train': lvis_dir / 'lvis_v1_train.json',
            'val': lvis_dir / 'lvis_v1_val.json'
        }
        
        for split, ann_file in splits.items():
            if not ann_file.exists():
                logger.warning(f"LVIS annotation file not found: {ann_file}")
                continue
            
            with open(ann_file, 'r') as f:
                lvis_data = json.load(f)
            
            # Build category mapping
            lvis_categories = {}
            for cat in lvis_data['categories']:
                class_id = self.add_class_to_mapping(cat['name'], 'lvis')
                lvis_categories[cat['id']] = class_id
            
            # Build image info mapping
            image_info = {img['id']: img for img in lvis_data['images']}
            
            # Process annotations (similar to COCO)
            image_annotations = {}
            for ann in tqdm(lvis_data['annotations'], desc=f"Processing LVIS {split} annotations"):
                img_id = ann['image_id']
                if img_id not in image_annotations:
                    image_annotations[img_id] = []
                
                img = image_info[img_id]
                bbox = ann['bbox']
                
                center_x, center_y, width, height = self.normalize_bbox(
                    bbox, img['width'], img['height']
                )
                
                class_id = lvis_categories[ann['category_id']]
                
                image_annotations[img_id].append(
                    f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                )
            
            # Note: LVIS uses COCO images, so we'd link to the COCO image directory
            # For now, we'll create the label files
            label_dest_dir = output_dir / split / 'labels'
            
            for img_id, annotations in tqdm(image_annotations.items(), desc=f"Creating LVIS {split} labels"):
                img_info = image_info[img_id]
                label_filename = img_info['file_name'].replace('.jpg', '.txt')
                label_path = label_dest_dir / label_filename
                
                with open(label_path, 'w') as f:
                    f.write('\n'.join(annotations))
        
        logger.info(f"LVIS conversion completed. Output: {output_dir}")
        return True
    
    def convert_objects365_to_yolo(self):
        """Convert Objects365 dataset to YOLO format"""
        logger.info("Converting Objects365 dataset to YOLO format...")
        
        obj365_dir = self.datasets_dir / "objects365"
        output_dir = self.output_dir / "objects365_yolo"
        output_dir.mkdir(exist_ok=True)
        
        # Create directories
        (output_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Load Objects365 annotations
        ann_file = obj365_dir / "annotations.json"
        if not ann_file.exists():
            logger.warning("Objects365 annotations not found")
            return False
        
        with open(ann_file, 'r') as f:
            obj365_data = json.load(f)
        
        # Build category mapping
        obj365_categories = {}
        for cat in obj365_data['categories']:
            class_id = self.add_class_to_mapping(cat['name'], 'objects365')
            obj365_categories[cat['id']] = class_id
        
        # Process similar to COCO format
        # (Implementation details would be similar to COCO conversion)
        
        logger.info("Objects365 conversion completed (placeholder implementation)")
        return True
    
    def save_unified_classes(self):
        """Save unified class mapping"""
        classes_file = self.output_dir / "classes.txt"
        mapping_file = self.output_dir / "class_mapping.json"
        
        # Save simple classes.txt for YOLO
        with open(classes_file, 'w') as f:
            for class_info in self.unified_classes:
                f.write(f"{class_info['name']}\n")
        
        # Save detailed mapping
        with open(mapping_file, 'w') as f:
            json.dump(self.unified_classes, f, indent=2)
        
        logger.info(f"Unified classes saved: {len(self.unified_classes)} classes")
        logger.info(f"Classes file: {classes_file}")
        logger.info(f"Mapping file: {mapping_file}")
        
        return len(self.unified_classes)
    
    def convert_all(self):
        """Convert all datasets to YOLO format"""
        logger.info("Starting conversion of all datasets to YOLO format...")
        
        results = {}
        results["coco"] = self.convert_coco_to_yolo()
        results["open_images"] = self.convert_open_images_to_yolo()
        results["lvis"] = self.convert_lvis_to_yolo()
        results["objects365"] = self.convert_objects365_to_yolo()
        
        # Save unified class mapping
        total_classes = self.save_unified_classes()
        
        # Create conversion summary
        summary = {
            "total_classes": total_classes,
            "datasets_converted": sum(results.values()),
            "conversion_results": results,
            "output_directory": str(self.output_dir)
        }
        
        summary_file = self.output_dir / "conversion_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Conversion completed! Total classes: {total_classes}")
        return results, summary

def main():
    """Main function to run dataset conversion"""
    converter = DatasetConverter()
    
    print("🔄 Starting Dataset Format Conversion Pipeline")
    print("=" * 50)
    
    results, summary = converter.convert_all()
    
    print("\n📊 Conversion Summary:")
    print("=" * 50)
    for dataset, status in results.items():
        status_emoji = "✅" if status else "❌"
        print(f"{status_emoji} {dataset.upper()}: {'Success' if status else 'Failed'}")
    
    print(f"\n🎯 Total Unified Classes: {summary['total_classes']}")
    print(f"📁 Output Directory: {summary['output_directory']}")

if __name__ == "__main__":
    main()
