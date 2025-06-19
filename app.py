from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import torch
import io

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv8 model (nano for CPU, faster inference)
try:
    model = YOLO("yolov8n.pt")  # Pretrained on COCO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
except Exception as e:
    raise RuntimeError(f"Error loading YOLO model: {e}")

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
        results = model(image)[0]  # Run inference on the first result
        detections = []

        for box in results.boxes:
            label = model.names[int(box.cls)]
            conf = float(box.conf)
            x1, y1, x2, y2 = map(float, box.xyxy[0])

            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                "box": [round(x1), round(y1), round(x2), round(y2)]
            })

        return jsonify({
            "success": True,
            "detections": detections,
            "count": len(detections)
        })

    except Exception as e:
        return jsonify({"error": f"Inference failed. {str(e)}"}), 500

# Error handler for general server errors
@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Server error occurred."}), 500

# Error handler for 404
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found."}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
