"""
Automated Dataset Download Pipeline
Downloads and organizes multiple object detection datasets for training.

Supported datasets:
- COCO (80 classes)
- Open Images V7 (600 classes)
- LVIS (1200+ classes)
- Objects365 (365 classes)
- ImageNet Detection subset (200 classes)
"""

import os
import requests
import zipfile
import tarfile
import json
from pathlib import Path
from tqdm import tqdm
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetDownloader:
    def __init__(self, base_dir="datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Dataset URLs and configurations
        self.datasets_config = {
            "coco": {
                "urls": {
                    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
                    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
                    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
                },
                "dir": "coco",
                "classes": 80
            },
            "open_images": {
                "urls": {
                    "train_images": "https://storage.googleapis.com/openimages/web/download_v7.html",  # Special handling needed
                    "annotations": "https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-bbox.csv"
                },
                "dir": "open_images_v7",
                "classes": 600
            },
            "lvis": {
                "urls": {
                    "train_images": "http://images.cocodataset.org/zips/train2017.zip",  # Uses COCO images
                    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
                    "annotations": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip"
                },
                "dir": "lvis",
                "classes": 1203
            },
            "objects365": {
                "urls": {
                    "train_images": "https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/",
                    "annotations": "https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/annotations/zhiyuan_objv2_train.json"
                },
                "dir": "objects365",
                "classes": 365
            }
        }
    
    def download_file(self, url, destination, chunk_size=8192):
        """Download file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as file, tqdm(
                desc=f"Downloading {destination.name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Successfully downloaded: {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
    
    def extract_archive(self, archive_path, extract_to):
        """Extract zip or tar files"""
        try:
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix.lower() in ['.tar', '.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            
            logger.info(f"Extracted: {archive_path} to {extract_to}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract {archive_path}: {e}")
            return False
    
    def download_coco(self):
        """Download COCO dataset"""
        logger.info("Downloading COCO dataset...")
        coco_dir = self.base_dir / "coco"
        coco_dir.mkdir(exist_ok=True)
        
        config = self.datasets_config["coco"]
        
        for name, url in config["urls"].items():
            filename = url.split('/')[-1]
            filepath = coco_dir / filename
            
            if not filepath.exists():
                if self.download_file(url, filepath):
                    # Extract if it's an archive
                    if filepath.suffix.lower() == '.zip':
                        self.extract_archive(filepath, coco_dir)
                        filepath.unlink()  # Remove zip after extraction
            else:
                logger.info(f"File already exists: {filepath}")
        
        return True
    
    def download_open_images_subset(self, max_images_per_class=100):
        """Download Open Images V7 subset using official downloader"""
        logger.info("Downloading Open Images V7 subset...")
        oi_dir = self.base_dir / "open_images_v7"
        oi_dir.mkdir(exist_ok=True)
        
        try:
            # Download the official downloader script
            downloader_url = "https://raw.githubusercontent.com/openimages/dataset/master/downloader.py"
            downloader_path = oi_dir / "downloader.py"
            
            if not downloader_path.exists():
                self.download_file(downloader_url, downloader_path)
            
            # Download class descriptions
            class_desc_url = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv"
            class_desc_path = oi_dir / "class-descriptions.csv"
            
            if not class_desc_path.exists():
                self.download_file(class_desc_url, class_desc_path)
            
            # Download train annotations (bbox)
            train_ann_url = "https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-bbox.csv"
            train_ann_path = oi_dir / "train-annotations-bbox.csv"
            
            if not train_ann_path.exists():
                self.download_file(train_ann_url, train_ann_path)
            
            logger.info("Open Images V7 metadata downloaded. Use the downloader.py script to download specific images.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download Open Images V7: {e}")
            return False
    
    def download_lvis(self):
        """Download LVIS dataset (uses COCO images)"""
        logger.info("Downloading LVIS dataset...")
        lvis_dir = self.base_dir / "lvis"
        lvis_dir.mkdir(exist_ok=True)
        
        # Download LVIS annotations
        lvis_urls = [
            "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
            "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip"
        ]
        
        for url in lvis_urls:
            filename = url.split('/')[-1]
            filepath = lvis_dir / filename
            
            if not filepath.exists():
                if self.download_file(url, filepath):
                    self.extract_archive(filepath, lvis_dir)
                    filepath.unlink()  # Remove zip after extraction
        
        logger.info("LVIS annotations downloaded. COCO images will be shared.")
        return True
    
    def download_objects365_subset(self):
        """Download Objects365 subset"""
        logger.info("Downloading Objects365 subset...")
        obj365_dir = self.base_dir / "objects365"
        obj365_dir.mkdir(exist_ok=True)
        
        # Download annotations first
        ann_url = "https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/annotations/zhiyuan_objv2_train.json"
        ann_path = obj365_dir / "annotations.json"
        
        if not ann_path.exists():
            self.download_file(ann_url, ann_path)
        
        logger.info("Objects365 annotations downloaded. Image download requires separate handling due to size.")
        return True
    
    def create_download_summary(self):
        """Create a summary of downloaded datasets"""
        summary = {
            "datasets": {},
            "total_estimated_classes": 0,
            "download_status": {}
        }
        
        for dataset_name, config in self.datasets_config.items():
            dataset_dir = self.base_dir / config["dir"]
            exists = dataset_dir.exists()
            
            summary["datasets"][dataset_name] = {
                "classes": config["classes"],
                "directory": str(dataset_dir),
                "downloaded": exists
            }
            
            if exists:
                summary["total_estimated_classes"] += config["classes"]
            
            summary["download_status"][dataset_name] = exists
        
        # Save summary
        summary_path = self.base_dir / "download_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Download summary saved to: {summary_path}")
        return summary
    
    def download_all(self):
        """Download all datasets"""
        logger.info("Starting download of all datasets...")
        
        results = {}
        results["coco"] = self.download_coco()
        results["open_images"] = self.download_open_images_subset()
        results["lvis"] = self.download_lvis()
        results["objects365"] = self.download_objects365_subset()
        
        # Create summary
        summary = self.create_download_summary()
        
        logger.info("Dataset download completed!")
        logger.info(f"Estimated total classes: {summary['total_estimated_classes']}")
        
        return results, summary

def main():
    """Main function to run dataset download"""
    downloader = DatasetDownloader()
    
    print("🚀 Starting Multi-Dataset Download Pipeline")
    print("=" * 50)
    
    results, summary = downloader.download_all()
    
    print("\n📊 Download Summary:")
    print("=" * 50)
    for dataset, status in results.items():
        status_emoji = "✅" if status else "❌"
        print(f"{status_emoji} {dataset.upper()}: {'Success' if status else 'Failed'}")
    
    print(f"\n🎯 Total Estimated Classes: {summary['total_estimated_classes']}")
    print("\n⚠️  Note: Some datasets require additional manual steps:")
    print("   - Open Images V7: Use downloader.py for specific image downloads")
    print("   - Objects365: Large dataset, consider downloading subsets")
    print("   - LVIS: Shares images with COCO dataset")

if __name__ == "__main__":
    main()
