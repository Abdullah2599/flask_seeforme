"""
Dataset Merging Pipeline
Intelligently merges multiple YOLO-format datasets into a unified training dataset.

Features:
- Handles class conflicts and duplicates
- Creates balanced train/val/test splits
- Generates data.yaml configuration
- Manages large datasets efficiently
"""

import os
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter
import yaml
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetMerger:
    def __init__(self, yolo_datasets_dir="yolo_datasets", merged_output_dir="merged_dataset"):
        self.yolo_datasets_dir = Path(yolo_datasets_dir)
        self.merged_output_dir = Path(merged_output_dir)
        self.merged_output_dir.mkdir(exist_ok=True)
        
        # Create output structure
        for split in ['train', 'val', 'test']:
            (self.merged_output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.merged_output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        self.class_stats = defaultdict(lambda: {'count': 0, 'sources': set()})
        self.merged_classes = []
        self.dataset_info = {}
    
    def load_dataset_classes(self, dataset_path):
        """Load class information from a dataset"""
        classes_file = dataset_path / "classes.txt"
        mapping_file = dataset_path / "class_mapping.json"
        
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                return json.load(f)
        elif classes_file.exists():
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
                return [{'id': i, 'name': cls, 'sources': ['unknown']} for i, cls in enumerate(classes)]
        else:
            return []
    
    def analyze_datasets(self):
        """Analyze all available datasets"""
        logger.info("Analyzing available datasets...")
        
        dataset_dirs = [d for d in self.yolo_datasets_dir.iterdir() if d.is_dir()]
        
        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name.replace('_yolo', '')
            logger.info(f"Analyzing dataset: {dataset_name}")
            
            # Load class information
            classes = self.load_dataset_classes(dataset_dir)
            
            # Count images and labels
            image_count = 0
            label_count = 0
            
            for split in ['train', 'val', 'test']:
                split_dir = dataset_dir / split
                if split_dir.exists():
                    images_dir = split_dir / 'images'
                    labels_dir = split_dir / 'labels'
                    
                    if images_dir.exists():
                        image_count += len(list(images_dir.glob('*')))
                    if labels_dir.exists():
                        label_count += len(list(labels_dir.glob('*.txt')))
            
            self.dataset_info[dataset_name] = {
                'path': dataset_dir,
                'classes': classes,
                'num_classes': len(classes),
                'num_images': image_count,
                'num_labels': label_count
            }
            
            # Update class statistics
            for class_info in classes:
                class_name = class_info['name']
                self.class_stats[class_name]['sources'].update(class_info.get('sources', [dataset_name]))
        
        logger.info(f"Found {len(self.dataset_info)} datasets")
        return self.dataset_info
    
    def resolve_class_conflicts(self):
        """Resolve class naming conflicts and create unified vocabulary"""
        logger.info("Resolving class conflicts and creating unified vocabulary...")
        
        # Group similar classes
        class_groups = defaultdict(list)
        
        for class_name in self.class_stats.keys():
            # Simple grouping by normalized name
            normalized = class_name.lower().replace('_', ' ').replace('-', ' ').strip()
            class_groups[normalized].append(class_name)
        
        # Create unified class list
        unified_id = 0
        class_mapping = {}  # old_name -> new_id
        
        for normalized_name, variants in class_groups.items():
            if len(variants) == 1:
                # No conflict
                original_name = variants[0]
                self.merged_classes.append({
                    'id': unified_id,
                    'name': original_name,
                    'normalized': normalized_name,
                    'variants': variants,
                    'sources': list(self.class_stats[original_name]['sources'])
                })
                class_mapping[original_name] = unified_id
            else:
                # Conflict - choose most common variant or merge
                most_common = max(variants, key=lambda x: self.class_stats[x]['count'])
                
                # Merge sources
                all_sources = set()
                for variant in variants:
                    all_sources.update(self.class_stats[variant]['sources'])
                
                self.merged_classes.append({
                    'id': unified_id,
                    'name': most_common,
                    'normalized': normalized_name,
                    'variants': variants,
                    'sources': list(all_sources)
                })
                
                # Map all variants to same ID
                for variant in variants:
                    class_mapping[variant] = unified_id
            
            unified_id += 1
        
        logger.info(f"Created unified vocabulary with {len(self.merged_classes)} classes")
        logger.info(f"Resolved {sum(len(cls['variants']) > 1 for cls in self.merged_classes)} conflicts")
        
        return class_mapping
    
    def count_class_instances(self):
        """Count instances of each class across all datasets"""
        logger.info("Counting class instances...")
        
        for dataset_name, dataset_info in self.dataset_info.items():
            dataset_path = dataset_info['path']
            
            for split in ['train', 'val', 'test']:
                labels_dir = dataset_path / split / 'labels'
                if not labels_dir.exists():
                    continue
                
                for label_file in labels_dir.glob('*.txt'):
                    try:
                        with open(label_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    if class_id < len(dataset_info['classes']):
                                        class_name = dataset_info['classes'][class_id]['name']
                                        self.class_stats[class_name]['count'] += 1
                    except Exception as e:
                        logger.warning(f"Error reading {label_file}: {e}")
    
    def create_balanced_splits(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Create balanced train/val/test splits"""
        logger.info("Creating balanced dataset splits...")
        
        # Collect all image-label pairs
        all_samples = []
        
        for dataset_name, dataset_info in self.dataset_info.items():
            dataset_path = dataset_info['path']
            
            for split in ['train', 'val', 'test']:
                images_dir = dataset_path / split / 'images'
                labels_dir = dataset_path / split / 'labels'
                
                if not (images_dir.exists() and labels_dir.exists()):
                    continue
                
                for img_file in images_dir.glob('*'):
                    label_file = labels_dir / f"{img_file.stem}.txt"
                    if label_file.exists():
                        all_samples.append({
                            'image': img_file,
                            'label': label_file,
                            'dataset': dataset_name
                        })
        
        # Shuffle samples
        random.shuffle(all_samples)
        
        # Calculate split sizes
        total_samples = len(all_samples)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        splits = {
            'train': all_samples[:train_size],
            'val': all_samples[train_size:train_size + val_size],
            'test': all_samples[train_size + val_size:]
        }
        
        logger.info(f"Split sizes - Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
        
        return splits
    
    def copy_and_convert_samples(self, splits, class_mapping):
        """Copy samples to merged dataset with class ID conversion"""
        logger.info("Copying and converting samples to merged dataset...")
        
        for split_name, samples in splits.items():
            logger.info(f"Processing {split_name} split ({len(samples)} samples)...")
            
            for i, sample in enumerate(tqdm(samples, desc=f"Copying {split_name}")):
                # Generate new filename to avoid conflicts
                new_filename = f"{sample['dataset']}_{i:06d}"
                
                # Copy image
                img_ext = sample['image'].suffix
                new_img_path = self.merged_output_dir / split_name / 'images' / f"{new_filename}{img_ext}"
                shutil.copy2(sample['image'], new_img_path)
                
                # Convert and copy label
                new_label_path = self.merged_output_dir / split_name / 'labels' / f"{new_filename}.txt"
                self.convert_label_file(sample['label'], new_label_path, sample['dataset'], class_mapping)
    
    def convert_label_file(self, src_label_path, dst_label_path, dataset_name, class_mapping):
        """Convert label file with new class IDs"""
        try:
            dataset_classes = self.dataset_info[dataset_name]['classes']
            
            with open(src_label_path, 'r') as src_f, open(dst_label_path, 'w') as dst_f:
                for line in src_f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_class_id = int(parts[0])
                        
                        # Get original class name
                        if old_class_id < len(dataset_classes):
                            class_name = dataset_classes[old_class_id]['name']
                            new_class_id = class_mapping.get(class_name, old_class_id)
                            
                            # Write converted line
                            parts[0] = str(new_class_id)
                            dst_f.write(' '.join(parts) + '\n')
        except Exception as e:
            logger.warning(f"Error converting {src_label_path}: {e}")
    
    def create_data_yaml(self):
        """Create data.yaml configuration file for YOLO training"""
        logger.info("Creating data.yaml configuration...")
        
        data_config = {
            'path': str(self.merged_output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.merged_classes),
            'names': [cls['name'] for cls in self.merged_classes]
        }
        
        yaml_path = self.merged_output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        logger.info(f"data.yaml created: {yaml_path}")
        return yaml_path
    
    def save_merge_metadata(self, class_mapping):
        """Save detailed merge metadata"""
        metadata = {
            'merged_classes': self.merged_classes,
            'class_mapping': class_mapping,
            'dataset_info': {k: {**v, 'path': str(v['path'])} for k, v in self.dataset_info.items()},
            'class_statistics': {k: {**v, 'sources': list(v['sources'])} for k, v in self.class_stats.items()},
            'total_classes': len(self.merged_classes),
            'merge_timestamp': str(Path().cwd())
        }
        
        metadata_path = self.merged_output_dir / 'merge_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save simple classes.txt
        classes_path = self.merged_output_dir / 'classes.txt'
        with open(classes_path, 'w') as f:
            for cls in self.merged_classes:
                f.write(f"{cls['name']}\n")
        
        logger.info(f"Merge metadata saved: {metadata_path}")
        return metadata_path
    
    def merge_all_datasets(self):
        """Complete dataset merging pipeline"""
        logger.info("Starting complete dataset merging pipeline...")
        
        # Step 1: Analyze datasets
        self.analyze_datasets()
        
        # Step 2: Count class instances
        self.count_class_instances()
        
        # Step 3: Resolve class conflicts
        class_mapping = self.resolve_class_conflicts()
        
        # Step 4: Create balanced splits
        splits = self.create_balanced_splits()
        
        # Step 5: Copy and convert samples
        self.copy_and_convert_samples(splits, class_mapping)
        
        # Step 6: Create configuration files
        yaml_path = self.create_data_yaml()
        metadata_path = self.save_merge_metadata(class_mapping)
        
        # Create summary
        summary = {
            'total_classes': len(self.merged_classes),
            'total_datasets': len(self.dataset_info),
            'total_samples': sum(len(samples) for samples in splits.values()),
            'output_directory': str(self.merged_output_dir),
            'data_yaml': str(yaml_path),
            'metadata_file': str(metadata_path)
        }
        
        summary_path = self.merged_output_dir / 'merge_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Dataset merging completed successfully!")
        logger.info(f"Total classes: {summary['total_classes']}")
        logger.info(f"Total samples: {summary['total_samples']}")
        logger.info(f"Output directory: {summary['output_directory']}")
        
        return summary

def main():
    """Main function to run dataset merging"""
    merger = DatasetMerger()
    
    print("🔀 Starting Dataset Merging Pipeline")
    print("=" * 50)
    
    summary = merger.merge_all_datasets()
    
    print("\n📊 Merge Summary:")
    print("=" * 50)
    print(f"✅ Total Classes: {summary['total_classes']}")
    print(f"✅ Total Datasets: {summary['total_datasets']}")
    print(f"✅ Total Samples: {summary['total_samples']}")
    print(f"📁 Output Directory: {summary['output_directory']}")
    print(f"⚙️  Data Config: {summary['data_yaml']}")
    
    print("\n🎯 Ready for training with YOLOv8!")

if __name__ == "__main__":
    main()
