# 🚀 Deployment Guide

Complete guide for deploying the Advanced Multi-Dataset Object Detection System to various platforms.

## 📋 Pre-Deployment Checklist

### ✅ **Model Requirements**
- [ ] Trained model file (`yolov8_custom_model.pt`) < 500MB
- [ ] Classes file (`classes.txt`) with all supported classes
- [ ] Model metadata file with performance metrics
- [ ] Evaluation report confirming >0.5 mAP@0.5

### ✅ **Code Requirements**
- [ ] All dependencies listed in `requirements.txt`
- [ ] Environment variables configured
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Health check endpoint working

### ✅ **Performance Requirements**
- [ ] Inference time <500ms on CPU
- [ ] Memory usage <2GB
- [ ] Model loads successfully
- [ ] API endpoints respond correctly

## 🌐 Platform-Specific Deployment

### 1. **Railway Deployment**

#### **Setup**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init
```

#### **Configuration Files**

**`railway.json`**
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "gunicorn --bind 0.0.0.0:$PORT app:app",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 300
  }
}
```

**`Procfile`**
```
web: gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 300 app:app
```

#### **Environment Variables**
```bash
# Set in Railway dashboard
FLASK_ENV=production
FLASK_DEBUG=False
MODEL_DIR=models
CACHE_SIZE=50
INFERENCE_TIMEOUT=30
PORT=5000
```

#### **Deployment Steps**
```bash
# Deploy to Railway
railway up

# Check deployment status
railway status

# View logs
railway logs
```

### 2. **Render Deployment**

#### **Configuration Files**

**`render.yaml`**
```yaml
services:
  - type: web
    name: yolov8-detection
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn --bind 0.0.0.0:$PORT app:app
    plan: free
    healthCheckPath: /health
    envVars:
      - key: FLASK_ENV
        value: production
      - key: MODEL_DIR
        value: models
      - key: CACHE_SIZE
        value: 50
```

#### **Deployment Steps**
1. Connect GitHub repository to Render
2. Configure build settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT app:app`
3. Set environment variables
4. Deploy automatically on git push

### 3. **Google Cloud Run**

#### **Dockerfile**
```dockerfile
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 8080

# Set environment variables
ENV FLASK_ENV=production
ENV PORT=8080

# Start command
CMD exec gunicorn --bind :$PORT --workers 2 --timeout 300 app:app
```

#### **Deployment Commands**
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/yolov8-detection

# Deploy to Cloud Run
gcloud run deploy yolov8-detection \
  --image gcr.io/PROJECT_ID/yolov8-detection \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300
```

### 4. **Heroku Deployment**

#### **Configuration Files**

**`Procfile`**
```
web: gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 300 app:app
```

**`runtime.txt`**
```
python-3.8.17
```

**`Aptfile`**
```
libsm6
libxext6
libxrender-dev
libglib2.0-0
libgomp1
```

#### **Deployment Steps**
```bash
# Install Heroku CLI and login
heroku login

# Create Heroku app
heroku create your-app-name

# Add buildpacks
heroku buildpacks:add --index 1 heroku-community/apt
heroku buildpacks:add --index 2 heroku/python

# Set environment variables
heroku config:set FLASK_ENV=production
heroku config:set MODEL_DIR=models
heroku config:set CACHE_SIZE=50

# Deploy
git push heroku main
```

### 5. **AWS EC2 Deployment**

#### **Setup Script**
```bash
#!/bin/bash
# EC2 setup script

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3 python3-pip python3-venv nginx

# Clone repository
git clone https://github.com/Abdullah2599/flask_seeforme.git
cd flask_seeforme

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install gunicorn

# Create systemd service
sudo tee /etc/systemd/system/yolov8-detection.service > /dev/null <<EOF
[Unit]
Description=YOLOv8 Detection API
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/flask_seeforme
Environment="PATH=/home/ubuntu/flask_seeforme/venv/bin"
ExecStart=/home/ubuntu/flask_seeforme/venv/bin/gunicorn --workers 3 --bind unix:yolov8-detection.sock -m 007 app:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start and enable service
sudo systemctl start yolov8-detection
sudo systemctl enable yolov8-detection

# Configure Nginx
sudo tee /etc/nginx/sites-available/yolov8-detection > /dev/null <<EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        include proxy_params;
        proxy_pass http://unix:/home/ubuntu/flask_seeforme/yolov8-detection.sock;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/yolov8-detection /etc/nginx/sites-enabled
sudo nginx -t
sudo systemctl restart nginx
```

## 🔧 Production Optimizations

### **Model Optimization**

#### **Model Compression**
```python
# Quantize model for faster inference
from ultralytics import YOLO

model = YOLO('yolov8_custom_model.pt')
model.export(format='onnx', int8=True)  # INT8 quantization
model.export(format='tflite')  # TensorFlow Lite
```

#### **Model Caching**
```python
# Implement model warming
def warm_up_model():
    """Warm up model with dummy inference"""
    dummy_image = Image.new('RGB', (640, 640), color='red')
    model.predict(dummy_image)
    logger.info("Model warmed up successfully")
```

### **Performance Tuning**

#### **Gunicorn Configuration**
```python
# gunicorn_config.py
bind = "0.0.0.0:5000"
workers = 2  # CPU cores * 2
worker_class = "sync"
worker_connections = 1000
timeout = 300
keepalive = 2
max_requests = 1000
max_requests_jitter = 100
preload_app = True
```

#### **Nginx Configuration**
```nginx
# nginx.conf
upstream app_server {
    server unix:/path/to/yolov8-detection.sock fail_timeout=0;
}

server {
    listen 80;
    client_max_body_size 10M;
    
    location / {
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Host $http_host;
        proxy_redirect off;
        proxy_pass http://app_server;
        
        # Timeout settings
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
        proxy_read_timeout 300;
    }
}
```

### **Monitoring and Logging**

#### **Application Monitoring**
```python
# monitoring.py
import psutil
import logging
from datetime import datetime

def log_system_metrics():
    """Log system performance metrics"""
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    logger.info(f"System Metrics - CPU: {cpu_percent}%, "
                f"Memory: {memory.percent}%, "
                f"Disk: {disk.percent}%")
```

#### **Error Tracking**
```python
# error_tracking.py
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

sentry_sdk.init(
    dsn="YOUR_SENTRY_DSN",
    integrations=[FlaskIntegration()],
    traces_sample_rate=1.0
)
```

## 🔒 Security Considerations

### **API Security**
```python
# security.py
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # Your prediction logic
    pass
```

### **Input Validation**
```python
# validation.py
from PIL import Image
import magic

def validate_image(file):
    """Validate uploaded image file"""
    # Check file size
    if len(file.read()) > 10 * 1024 * 1024:  # 10MB limit
        return False, "File too large"
    
    file.seek(0)  # Reset file pointer
    
    # Check file type
    file_type = magic.from_buffer(file.read(1024), mime=True)
    if not file_type.startswith('image/'):
        return False, "Invalid file type"
    
    file.seek(0)  # Reset file pointer
    
    # Try to open with PIL
    try:
        img = Image.open(file)
        img.verify()
        return True, "Valid image"
    except Exception as e:
        return False, f"Corrupted image: {e}"
```

## 📊 Monitoring and Maintenance

### **Health Checks**
```python
# health_checks.py
@app.route('/health')
def health_check():
    """Comprehensive health check"""
    checks = {
        'model_loaded': model_manager.current_model is not None,
        'gpu_available': torch.cuda.is_available(),
        'memory_usage': psutil.virtual_memory().percent < 90,
        'disk_space': psutil.disk_usage('/').percent < 90,
    }
    
    healthy = all(checks.values())
    status_code = 200 if healthy else 503
    
    return jsonify({
        'status': 'healthy' if healthy else 'unhealthy',
        'checks': checks,
        'timestamp': datetime.now().isoformat()
    }), status_code
```

### **Performance Monitoring**
```python
# metrics.py
import time
from functools import wraps

def monitor_performance(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logger.info(f"{func.__name__} executed in {execution_time:.3f}s")
        return result
    return wrapper
```

### **Automated Backups**
```bash
#!/bin/bash
# backup_script.sh

# Backup models and configuration
tar -czf "backup_$(date +%Y%m%d_%H%M%S).tar.gz" \
    models/ \
    *.yaml \
    *.json \
    logs/

# Upload to cloud storage (example for AWS S3)
aws s3 cp backup_*.tar.gz s3://your-backup-bucket/
```

## 🚨 Troubleshooting

### **Common Issues**

#### **Out of Memory Errors**
```python
# Solution: Reduce batch size and clear cache
torch.cuda.empty_cache()
model_manager.clear_cache()
```

#### **Model Loading Failures**
```python
# Solution: Implement fallback mechanism
try:
    model = YOLO('custom_model.pt')
except Exception as e:
    logger.error(f"Failed to load custom model: {e}")
    model = YOLO('yolov8n.pt')  # Fallback to default
```

#### **Slow Inference**
```python
# Solution: Optimize model and use caching
model.export(format='onnx')  # Convert to ONNX
# Enable inference caching
use_cache = True
```

### **Performance Debugging**
```python
# profiling.py
import cProfile
import pstats

def profile_inference():
    """Profile inference performance"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run inference
    result = model.predict(image)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions
```

## 📈 Scaling Strategies

### **Horizontal Scaling**
- **Load Balancer**: Distribute requests across multiple instances
- **Auto Scaling**: Scale based on CPU/memory usage
- **Container Orchestration**: Use Kubernetes for container management

### **Vertical Scaling**
- **GPU Acceleration**: Use GPU instances for faster inference
- **Memory Optimization**: Increase RAM for larger models
- **CPU Optimization**: Use high-performance CPU instances

### **Caching Strategies**
- **Redis**: Distributed caching for multiple instances
- **CDN**: Cache static assets and model files
- **Database Caching**: Cache frequent database queries

---

**🎯 Ready for Production!**

This deployment guide ensures your Advanced Multi-Dataset Object Detection System runs reliably in production environments with optimal performance and security.
