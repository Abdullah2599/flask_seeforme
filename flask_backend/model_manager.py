"""
Model Manager for Custom Multi-Dataset YOLOv8 Models
Handles model loading, versioning, and inference optimization.

Features:
- Automatic model selection (best available)
- Model versioning and metadata management
- GPU/CPU optimization
- Inference caching for similar frames
- Fallback mechanisms
"""

import os
import json
import torch
import hashlib
import time
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
import psutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, models_dir="models", cache_size=100):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model storage
        self.current_model = None
        self.model_metadata = {}
        self.available_models = {}
        
        # Inference cache
        self.cache_size = cache_size
        self.inference_cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = 300  # 5 minutes TTL
        
        # Performance monitoring
        self.inference_stats = {
            'total_inferences': 0,
            'cache_hits': 0,
            'avg_inference_time': 0.0,
            'gpu_available': torch.cuda.is_available()
        }
        
        # Device optimization
        self.device = self._setup_device()
        self.memory_threshold = 0.8  # 80% memory usage threshold
        
        # Load available models
        self._scan_available_models()
        self._load_best_model()
    
    def _setup_device(self) -> str:
        """Setup optimal device for inference"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"Found {gpu_count} GPU(s)")
            
            # Select GPU with most free memory
            best_gpu = 0
            max_free_memory = 0
            
            for i in range(gpu_count):
                torch.cuda.set_device(i)
                free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_gpu = i
            
            device = f"cuda:{best_gpu}"
            logger.info(f"Using GPU {best_gpu} with {max_free_memory/1e9:.1f}GB free memory")
            
            # Set memory fraction to avoid OOM
            torch.cuda.set_per_process_memory_fraction(self.memory_threshold, best_gpu)
            
        else:
            device = "cpu"
            logger.warning("No GPU available, using CPU (inference will be slower)")
        
        return device
    
    def _scan_available_models(self):
        """Scan for available model files and their metadata"""
        logger.info("Scanning for available models...")
        
        # Look for .pt files
        model_files = list(self.models_dir.glob("*.pt"))
        
        for model_file in model_files:
            try:
                # Try to load metadata
                metadata_file = model_file.with_suffix('.json')
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                else:
                    # Create basic metadata
                    metadata = {
                        'model_name': model_file.stem,
                        'model_path': str(model_file),
                        'created_date': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
                        'model_size_mb': model_file.stat().st_size / (1024 * 1024),
                        'num_classes': 'unknown',
                        'map_score': 0.0,
                        'training_dataset': 'unknown',
                        'version': '1.0.0'
                    }
                
                self.available_models[model_file.stem] = {
                    'path': model_file,
                    'metadata': metadata
                }
                
                logger.info(f"Found model: {model_file.stem}")
                
            except Exception as e:
                logger.warning(f"Error processing model {model_file}: {e}")
        
        logger.info(f"Found {len(self.available_models)} available models")
    
    def _load_best_model(self):
        """Load the best available model based on criteria"""
        if not self.available_models:
            logger.warning("No custom models found, using default YOLOv8n")
            self._load_default_model()
            return
        
        # Scoring criteria (weighted)
        best_model = None
        best_score = -1
        
        for model_name, model_info in self.available_models.items():
            metadata = model_info['metadata']
            
            # Calculate score based on multiple factors
            score = 0
            
            # mAP score (40% weight)
            map_score = float(metadata.get('map_score', 0))
            score += map_score * 0.4
            
            # Number of classes (30% weight) - more classes is better
            num_classes = metadata.get('num_classes', 80)
            if isinstance(num_classes, (int, float)):
                score += min(float(num_classes) / 2000, 1.0) * 0.3
            
            # Recency (20% weight) - newer models preferred
            try:
                created_date = datetime.fromisoformat(metadata.get('created_date', '2020-01-01'))
                days_old = (datetime.now() - created_date).days
                recency_score = max(0, 1 - days_old / 365)  # Decay over a year
                score += recency_score * 0.2
            except:
                pass
            
            # Model size penalty (10% weight) - smaller is better for deployment
            model_size_mb = float(metadata.get('model_size_mb', 1000))
            size_score = max(0, 1 - model_size_mb / 1000)  # Penalty for models > 1GB
            score += size_score * 0.1
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        if best_model:
            logger.info(f"Selected best model: {best_model} (score: {best_score:.3f})")
            self._load_model(best_model)
        else:
            logger.warning("No suitable model found, using default")
            self._load_default_model()
    
    def _load_default_model(self):
        """Load default YOLOv8 model as fallback"""
        try:
            logger.info("Loading default YOLOv8n model...")
            self.current_model = YOLO("yolov8n.pt")
            self.current_model.to(self.device)
            
            self.model_metadata = {
                'model_name': 'yolov8n_default',
                'num_classes': 80,
                'class_names': list(self.current_model.names.values()),
                'version': 'default',
                'map_score': 0.0,
                'is_custom': False
            }
            
            logger.info("Default model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load default model: {e}")
            raise RuntimeError("Could not load any model")
    
    def _load_model(self, model_name: str):
        """Load specific model by name"""
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not found")
        
        try:
            model_info = self.available_models[model_name]
            model_path = model_info['path']
            
            logger.info(f"Loading model: {model_name}")
            
            # Load model
            self.current_model = YOLO(str(model_path))
            self.current_model.to(self.device)
            
            # Update metadata
            self.model_metadata = {
                **model_info['metadata'],
                'class_names': list(self.current_model.names.values()),
                'is_custom': True,
                'loaded_at': datetime.now().isoformat()
            }
            
            logger.info(f"Model {model_name} loaded successfully")
            logger.info(f"Classes: {self.model_metadata.get('num_classes', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            logger.info("Falling back to default model")
            self._load_default_model()
    
    def _calculate_image_hash(self, image: Image.Image) -> str:
        """Calculate hash for image caching"""
        # Convert to numpy array and calculate hash
        img_array = np.array(image.resize((64, 64)))  # Resize for consistent hashing
        img_bytes = img_array.tobytes()
        return hashlib.md5(img_bytes).hexdigest()
    
    def _cleanup_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.inference_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
        
        # Also limit cache size
        if len(self.inference_cache) > self.cache_size:
            # Remove oldest entries
            sorted_items = sorted(self.cache_timestamps.items(), key=lambda x: x[1])
            items_to_remove = len(sorted_items) - self.cache_size
            
            for i in range(items_to_remove):
                key = sorted_items[i][0]
                self.inference_cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
    
    def _check_memory_usage(self):
        """Check system memory usage and optimize if needed"""
        if self.device.startswith('cuda'):
            # Check GPU memory
            gpu_memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if gpu_memory_used > self.memory_threshold:
                logger.warning(f"High GPU memory usage: {gpu_memory_used:.1%}")
                torch.cuda.empty_cache()
        
        # Check system memory
        system_memory = psutil.virtual_memory()
        if system_memory.percent > 85:
            logger.warning(f"High system memory usage: {system_memory.percent:.1f}%")
            # Clear some cache
            self.inference_cache.clear()
            self.cache_timestamps.clear()
    
    def predict(self, image: Image.Image, confidence_threshold: float = 0.25, 
                use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Run inference on image with caching and optimization
        
        Args:
            image: PIL Image
            confidence_threshold: Minimum confidence for detections
            use_cache: Whether to use inference caching
            
        Returns:
            List of detection dictionaries
        """
        start_time = time.time()
        
        try:
            # Check cache if enabled
            cache_key = None
            if use_cache:
                cache_key = self._calculate_image_hash(image)
                if cache_key in self.inference_cache:
                    self.inference_stats['cache_hits'] += 1
                    logger.debug("Cache hit for inference")
                    return self.inference_cache[cache_key]
            
            # Check memory usage
            self._check_memory_usage()
            
            # Run inference
            results = self.current_model(image, conf=confidence_threshold, verbose=False)[0]
            
            # Process results
            detections = []
            for box in results.boxes:
                if box.conf.item() >= confidence_threshold:
                    # Get class name
                    class_id = int(box.cls.item())
                    class_name = self.current_model.names.get(class_id, f"class_{class_id}")
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    detection = {
                        "label": class_name,
                        "confidence": round(float(box.conf.item()), 3),
                        "box": [round(x1), round(y1), round(x2), round(y2)],
                        "class_id": class_id
                    }
                    detections.append(detection)
            
            # Cache result if enabled
            if use_cache and cache_key:
                self.inference_cache[cache_key] = detections
                self.cache_timestamps[cache_key] = time.time()
                self._cleanup_cache()
            
            # Update statistics
            inference_time = time.time() - start_time
            self.inference_stats['total_inferences'] += 1
            
            # Update average inference time (exponential moving average)
            alpha = 0.1
            self.inference_stats['avg_inference_time'] = (
                alpha * inference_time + 
                (1 - alpha) * self.inference_stats['avg_inference_time']
            )
            
            logger.debug(f"Inference completed in {inference_time:.3f}s, found {len(detections)} objects")
            
            return detections
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            # Try to recover by clearing cache and retrying once
            if use_cache:
                self.inference_cache.clear()
                self.cache_timestamps.clear()
                torch.cuda.empty_cache() if self.device.startswith('cuda') else None
                
                try:
                    return self.predict(image, confidence_threshold, use_cache=False)
                except:
                    pass
            
            raise RuntimeError(f"Model inference failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information"""
        return {
            'model_name': self.model_metadata.get('model_name', 'unknown'),
            'version': self.model_metadata.get('version', 'unknown'),
            'num_classes': self.model_metadata.get('num_classes', 0),
            'is_custom': self.model_metadata.get('is_custom', False),
            'map_score': self.model_metadata.get('map_score', 0.0),
            'training_dataset': self.model_metadata.get('training_dataset', 'unknown'),
            'loaded_at': self.model_metadata.get('loaded_at', 'unknown'),
            'device': self.device,
            'inference_stats': self.inference_stats,
            'available_models': list(self.available_models.keys())
        }
    
    def get_supported_classes(self) -> List[str]:
        """Get list of supported class names"""
        return self.model_metadata.get('class_names', [])
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different available model"""
        try:
            if model_name == 'default':
                self._load_default_model()
                return True
            elif model_name in self.available_models:
                self._load_model(model_name)
                return True
            else:
                logger.error(f"Model {model_name} not available")
                return False
        except Exception as e:
            logger.error(f"Failed to switch to model {model_name}: {e}")
            return False
    
    def add_model(self, model_path: str, metadata: Dict[str, Any] = None) -> bool:
        """Add a new model to the manager"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Copy to models directory if not already there
            if model_path.parent != self.models_dir:
                dest_path = self.models_dir / model_path.name
                import shutil
                shutil.copy2(model_path, dest_path)
                model_path = dest_path
            
            # Save metadata
            if metadata:
                metadata_path = model_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # Rescan models
            self._scan_available_models()
            
            logger.info(f"Added model: {model_path.stem}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add model: {e}")
            return False
    
    def clear_cache(self):
        """Clear inference cache"""
        self.inference_cache.clear()
        self.cache_timestamps.clear()
        logger.info("Inference cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.inference_cache),
            'cache_hits': self.inference_stats['cache_hits'],
            'cache_hit_rate': (
                self.inference_stats['cache_hits'] / max(self.inference_stats['total_inferences'], 1)
            ),
            'avg_inference_time': self.inference_stats['avg_inference_time']
        }

# Global model manager instance
_model_manager = None

def get_model_manager() -> ModelManager:
    """Get global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
