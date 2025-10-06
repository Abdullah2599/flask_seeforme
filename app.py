from flask import Flask, request, jsonify
from PIL import Image
import torch
import io
import sys
import os
from datetime import datetime
import logging

# Add flask_backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'flask_backend'))

from model_manager import get_model_manager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize model manager
try:
    model_manager = get_model_manager()
    logger.info("Model manager initialized successfully")
except Exception as e:
    logger.error(f"Error initializing model manager: {e}")
    raise RuntimeError(f"Error loading model manager: {e}")

@app.route('/')
def index():
    return "YOLOv8 Flask Backend is running."

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded. Please send image with key 'image'."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to read image. {str(e)}"}), 400

    try:
        # Run inference using model manager
        detections = model_manager.predict(
            image, 
            confidence_threshold=0.25,
            use_cache=True
        )

        # Remove class_id from response to maintain API compatibility
        clean_detections = []
        for detection in detections:
            clean_detection = {
                "label": detection["label"],
                "confidence": detection["confidence"],
                "box": detection["box"]
            }
            clean_detections.append(clean_detection)

        return jsonify({
            "success": True,
            "detections": clean_detections,
            "count": len(clean_detections)
        })

    except Exception as e:
        return jsonify({"error": f"Inference failed. {str(e)}"}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get current model information and capabilities"""
    try:
        info = model_manager.get_model_info()
        return jsonify({
            "success": True,
            "model_info": info
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get model info: {str(e)}"}), 500

@app.route('/model/classes', methods=['GET'])
def supported_classes():
    """Get list of supported object classes"""
    try:
        classes = model_manager.get_supported_classes()
        return jsonify({
            "success": True,
            "classes": classes,
            "count": len(classes)
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get classes: {str(e)}"}), 500

@app.route('/model/switch', methods=['POST'])
def switch_model():
    """Switch to a different available model"""
    try:
        data = request.get_json()
        if not data or 'model_name' not in data:
            return jsonify({"error": "model_name is required"}), 400
        
        model_name = data['model_name']
        success = model_manager.switch_model(model_name)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Switched to model: {model_name}",
                "model_info": model_manager.get_model_info()
            })
        else:
            return jsonify({"error": f"Failed to switch to model: {model_name}"}), 400
            
    except Exception as e:
        return jsonify({"error": f"Model switch failed: {str(e)}"}), 500

@app.route('/cache/stats', methods=['GET'])
def cache_stats():
    """Get inference cache statistics"""
    try:
        stats = model_manager.get_cache_stats()
        return jsonify({
            "success": True,
            "cache_stats": stats
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get cache stats: {str(e)}"}), 500

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear inference cache"""
    try:
        model_manager.clear_cache()
        return jsonify({
            "success": True,
            "message": "Cache cleared successfully"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to clear cache: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed system info"""
    try:
        model_info = model_manager.get_model_info()
        cache_stats = model_manager.get_cache_stats()
        
        return jsonify({
            "success": True,
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model": {
                "name": model_info.get('model_name', 'unknown'),
                "classes": model_info.get('num_classes', 0),
                "device": model_info.get('device', 'unknown'),
                "is_custom": model_info.get('is_custom', False)
            },
            "performance": {
                "total_inferences": model_info.get('inference_stats', {}).get('total_inferences', 0),
                "avg_inference_time": model_info.get('inference_stats', {}).get('avg_inference_time', 0),
                "cache_hit_rate": cache_stats.get('cache_hit_rate', 0)
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "status": "unhealthy",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    print("Starting Enhanced YOLOv8 Flask Backend")
    print("=" * 50)
    
    # Print model info
    try:
        info = model_manager.get_model_info()
        print(f"Model: {info.get('model_name', 'unknown')}")
        print(f"Classes: {info.get('num_classes', 0)}")
        print(f"Device: {info.get('device', 'unknown')}")
        print(f"Custom Model: {info.get('is_custom', False)}")
        
        if info.get('is_custom'):
            print(f"mAP Score: {info.get('map_score', 0):.3f}")
            print(f"Version: {info.get('version', 'unknown')}")
        
        print("\nAvailable Endpoints:")
        print("   POST /predict - Object detection")
        print("   GET  /model/info - Model information")
        print("   GET  /model/classes - Supported classes")
        print("   POST /model/switch - Switch model")
        print("   GET  /cache/stats - Cache statistics")
        print("   POST /cache/clear - Clear cache")
        print("   GET  /health - Health check")
        
    except Exception as e:
        print(f"Warning: Could not get model info: {e}")
    
    print("\nServer starting on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)