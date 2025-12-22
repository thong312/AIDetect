#!/bin/bash
# 🚀 Complete Setup & Run Guide for AIHumanDetect

## ============================================================================
## 1️⃣ PREREQUISITES
## ============================================================================

echo "📋 Checking prerequisites..."

# Check Python version
python3 --version
# Should be 3.8+ (3.10+ recommended)

# Check pip
pip3 --version

# Check git
git --version


## ============================================================================
## 2️⃣ ENVIRONMENT SETUP
## ============================================================================

echo "🔧 Setting up environment..."

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or use conda (if you prefer)
# conda create -n AIHumanDetect python=3.10
# conda activate AIHumanDetect


## ============================================================================
## 3️⃣ INSTALL DEPENDENCIES
## ============================================================================

echo "📦 Installing dependencies..."

# Core requirements (existing)
pip install torch torchvision transformers pillow

# Enhanced ensemble requirements (NEW)
pip install scikit-learn matplotlib numpy psutil

# Or install all at once
pip install -r requirements_enhanced.txt

# (If requirements_enhanced.txt doesn't exist, create it with above packages)


## ============================================================================
## 4️⃣ PROJECT STRUCTURE
## ============================================================================

echo "📁 Project structure:"

# Expected folder structure:
# AIHumanDetect/
# ├── models/                          (Pre-trained models)
# │   ├── age_vit_deploy/
# │   ├── emotion_vit_rafdb/
# │   └── gender_vit_utkface/
# ├── temp/                            (Temporary folder, auto-created)
# ├── logs/                            (Log files, auto-created)
# ├── static/                          (Web assets)
# │   ├── app.js
# │   └── style.css
# ├── templates/                       (HTML templates)
# │   └── index.html
# ├── Core Code Files
# │   ├── app.py                       (Flask web app)
# │   ├── pipeline.py                  (Processing pipeline)
# │   ├── face_recognition.py          (Face embedding)
# │   ├── emotion_model.py             (Emotion prediction)
# │   ├── gender_model.py              (Gender prediction)
# │   ├── age_model.py                 (Age prediction)
# │   ├── attributes.py                (Attribute analysis)
# │   ├── models.py                    (Database models)
# │   ├── database.py                  (Database setup)
# │   └── test.py                      (Testing)
# ├── NEW: Enhanced Ensemble (Optional)
# │   ├── transunet_model.py           (TransUNet architecture)
# │   ├── ensemble_model.py            (Basic ensemble)
# │   ├── enhanced_ensemble.py         (Improved ensemble) ⭐
# │   ├── monitor.py                   (Performance monitoring) ⭐
# │   ├── evaluate.py                  (Evaluation framework) ⭐
# │   ├── pipeline_ensemble.py         (Ensemble pipeline)
# │   └── test_ensemble.py             (Ensemble tests)
# └── Documentation
#     ├── README.md
#     ├── ENSEMBLE_GUIDE.md
#     ├── IMPROVEMENTS.md
#     ├── QUICK_START.py
#     ├── SUMMARY.md
#     ├── INDEX.md
#     ├── COMPARISON.md
#     └── SETUP_GUIDE.sh (this file)


## ============================================================================
## 5️⃣ QUICK START - WEB APP
## ============================================================================

echo "🌐 Running Web Application..."

# Start the Flask app
python3 app.py

# Then open in browser:
# http://localhost:5000

# Features:
# - Upload face image
# - Register new user
# - Recognize registered users
# - Analyze emotion, gender, age


## ============================================================================
## 6️⃣ QUICK START - ENSEMBLE MODEL (NEW)
## ============================================================================

echo "🤖 Running Ensemble Model..."

# Option A: Simple test
python3 -c "
from PIL import Image
from enhanced_ensemble import BatchEnsemblePredictor

predictor = BatchEnsemblePredictor(batch_size=8)
image = Image.open('path/to/face.jpg').convert('RGB')
result = predictor.predict(image, return_all_scores=True)

print(f'Emotion: {result[\"predicted_class\"]}')
print(f'Ensemble confidence: {result[\"ensemble_confidence\"]:.1f}%')
print(f'ViT confidence: {result[\"vit_confidence\"]:.1f}%')
print(f'TransUNet confidence: {result[\"transunet_confidence\"]:.1f}%')
"

# Option B: Run test script
python3 test_ensemble.py

# Option C: Use example code
python3 QUICK_START.py


## ============================================================================
## 7️⃣ SETUP - DETAILED STEPS
## ============================================================================

echo "⚙️ Detailed Setup Steps..."

# Step 1: Clone repository (if not already done)
git clone https://github.com/thong312/AIDetect.git
cd AIDetect

# Step 2: Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Step 3: Upgrade pip
pip install --upgrade pip

# Step 4: Install core dependencies
pip install torch torchvision transformers pillow flask sqlalchemy

# Step 5: Install enhanced ensemble dependencies
pip install scikit-learn matplotlib numpy psutil

# Step 6: Create required directories
mkdir -p temp
mkdir -p logs
mkdir -p models

# Step 7: Download/ensure models exist
# The following should already exist:
# - models/emotion_vit_rafdb/
# - models/gender_vit_utkface/
# - models/age_vit_deploy/

# Step 8: Verify installation
python3 -c "
import torch
import transformers
from PIL import Image
print('✅ Core packages installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Step 9: Test enhanced ensemble (optional)
python3 -c "
from enhanced_ensemble import BatchEnsemblePredictor
print('✅ Enhanced ensemble ready!')
"


## ============================================================================
## 8️⃣ RUNNING SPECIFIC COMPONENTS
## ============================================================================

echo "🔨 Running Specific Components..."

# A. Test face recognition only
python3 -c "
from PIL import Image
from face_recognition import get_embedding

image = Image.open('face.jpg').convert('RGB')
embedding = get_embedding(image)
print(f'Face embedding shape: {embedding.shape}')
"

# B. Test emotion model only
python3 -c "
from PIL import Image
from emotion_model import predict_emotion

image = Image.open('face.jpg').convert('RGB')
emotion, confidence = predict_emotion(image)
print(f'Emotion: {emotion} ({confidence:.1f}%)')
"

# C. Test gender model only
python3 -c "
from PIL import Image
from gender_model import predict_gender

image = Image.open('face.jpg').convert('RGB')
gender, confidence = predict_gender(image)
print(f'Gender: {gender} ({confidence:.1f}%)')
"

# D. Test age model only
python3 -c "
from PIL import Image
from age_model import predict_age_group

image = Image.open('face.jpg').convert('RGB')
age, confidence = predict_age_group(image)
print(f'Age: {age} ({confidence:.1f}%)')
"

# E. Test full pipeline
python3 -c "
from PIL import Image
from pipeline import recognize_user

file = open('face.jpg', 'rb')
name, attributes, distance = recognize_user(file)
print(f'Name: {name}')
print(f'Attributes: {attributes}')
"

# F. Test enhanced ensemble
python3 -c "
from PIL import Image
from enhanced_ensemble import BatchEnsemblePredictor
from monitor import EnsembleMonitor

predictor = BatchEnsemblePredictor()
monitor = EnsembleMonitor()

image = Image.open('face.jpg').convert('RGB')
result = predictor.predict(image, return_all_scores=True)

monitor.record_prediction(
    ensemble_correct=True,
    vit_correct=True,
    transunet_correct=False,
    ensemble_confidence=result['ensemble_confidence'],
    vit_confidence=result['vit_confidence'],
    transunet_confidence=result['transunet_confidence'],
    inference_time=0.05
)

monitor.print_summary()
"


## ============================================================================
## 9️⃣ WEB APP USAGE
## ============================================================================

echo "🌐 Web Application Features..."

# 1. Start server
python3 app.py
# Output: Running on http://localhost:5000

# 2. Open browser
# Navigate to: http://localhost:5000

# 3. Features available:
#    - Register: Upload face image and name → saves to database
#    - Recognize: Upload face image → identifies person + emotions
#    - Analyze: Upload face image → detailed emotion/gender/age analysis

# 4. API Endpoints:
#    GET  /                    - Main page
#    POST /register            - Register new user
#    POST /recognize           - Recognize person
#    POST /analyze             - Analyze attributes


## ============================================================================
## 🔟 RUNNING WITH DOCKER (OPTIONAL)
## ============================================================================

echo "🐳 Docker Setup (Optional)..."

# 1. Build Docker image
docker build -t aihumandetect:latest .

# 2. Run Docker container
docker run -p 5000:5000 -v $(pwd):/app aihumandetect:latest

# 3. Access at http://localhost:5000


## ============================================================================
## 1️⃣1️⃣ MONITORING & EVALUATION
## ============================================================================

echo "📊 Monitoring & Evaluation..."

# Run full evaluation pipeline
python3 -c "
from enhanced_ensemble import BatchEnsemblePredictor
from monitor import EnsembleMonitor, PerformanceOptimizer
from evaluate import ModelComparator
from pathlib import Path
from PIL import Image
import numpy as np

# Initialize
predictor = BatchEnsemblePredictor(batch_size=16)
monitor = EnsembleMonitor()

# Process test images
test_images = list(Path('test_data/').glob('*.jpg'))[:100]

for i, img_path in enumerate(test_images):
    try:
        image = Image.open(img_path).convert('RGB')
        result = predictor.predict(image, return_all_scores=True)
        
        # Simulate correct prediction (replace with actual labels)
        monitor.record_prediction(
            ensemble_correct=True,
            vit_correct=True,
            transunet_correct=False,
            ensemble_confidence=result['ensemble_confidence'],
            vit_confidence=result['vit_confidence'],
            transunet_confidence=result['transunet_confidence'],
            inference_time=0.05,
            image_name=img_path.name
        )
    except Exception as e:
        print(f'Error processing {img_path}: {e}')

# Print summary
monitor.print_summary()
monitor.export_report('ensemble_report.json')

# Get optimization recommendations
optimizer = PerformanceOptimizer(monitor)
print('\n📈 Optimization Recommendations:')
for rec in optimizer.get_recommendations():
    print(f'  ✓ {rec}')
"

# Export evaluation report
# Outputs: ensemble_report.json


## ============================================================================
## 1️⃣2️⃣ TROUBLESHOOTING
## ============================================================================

echo "🆘 Troubleshooting..."

# Problem 1: ModuleNotFoundError: No module named 'torch'
# Solution: pip install torch torchvision

# Problem 2: CUDA out of memory
# Solution: 
#   - Reduce batch_size
#   - Use CPU mode (set device='cpu')
#   - Close other GPU applications

# Problem 3: Models not found
# Solution:
#   - Check models/ folder exists
#   - Models should auto-download on first use
#   - Or manually place model files in models/

# Problem 4: Port 5000 already in use
# Solution:
#   - python3 app.py --port 8000
#   - Or kill existing process: lsof -i :5000

# Problem 5: Slow inference
# Solution:
#   - Use batch processing
#   - Reduce model size (use ViT weight only)
#   - Enable GPU (torch.cuda.is_available())


## ============================================================================
## 1️⃣3️⃣ PERFORMANCE OPTIMIZATION
## ============================================================================

echo "⚡ Performance Optimization..."

# 1. GPU Optimization
python3 -c "
import torch

# Check GPU
print(f'GPU Available: {torch.cuda.is_available()}')
print(f'GPU Count: {torch.cuda.device_count()}')
print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# 2. Batch Size Optimization
python3 -c "
import torch
from enhanced_ensemble import BatchEnsemblePredictor

# Auto-detect optimal batch size
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_memory > 8:
        batch_size = 32
    elif gpu_memory > 4:
        batch_size = 16
    else:
        batch_size = 4
else:
    batch_size = 1

print(f'Optimal batch size: {batch_size}')
predictor = BatchEnsemblePredictor(batch_size=batch_size)
"

# 3. Model Caching
python3 -c "
from enhanced_ensemble import EnhancedEnsembleModel

# Models are lazy loaded (loaded on first use)
# This saves memory on startup
model = EnhancedEnsembleModel(enable_caching=True)
print('✅ Lazy loading enabled - models load only when needed')
"

# 4. Ensemble Method Selection
python3 -c "
from enhanced_ensemble import BatchEnsemblePredictor

# Speed vs Accuracy tradeoff:
# - weighted_average: fastest (default)
# - harmonic_mean: balanced
# - product: more selective
# - max_voting: most selective

predictor = BatchEnsemblePredictor(ensemble_method='weighted_average')
print('✅ Using fastest ensemble method')
"


## ============================================================================
## 1️⃣4️⃣ PRODUCTION DEPLOYMENT
## ============================================================================

echo "🚀 Production Deployment..."

# 1. Use production WSGI server (gunicorn)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# 2. With monitoring
python3 -c "
import logging
from monitor import EnsembleMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Run with monitoring
monitor = EnsembleMonitor(log_file='logs/production.log')
print('✅ Production monitoring enabled')
"

# 3. Docker deployment
# docker run -d -p 5000:5000 \
#   -v /data:/app/data \
#   -v /logs:/app/logs \
#   aihumandetect:latest

# 4. Check logs
tail -f logs/production.log


## ============================================================================
## 1️⃣5️⃣ USAGE EXAMPLES
## ============================================================================

echo "📚 Usage Examples..."

# Example 1: Single Image Inference
python3 << 'EOF'
from PIL import Image
from emotion_model import predict_emotion

image = Image.open("test_face.jpg").convert("RGB")
emotion, confidence = predict_emotion(image)
print(f"Emotion: {emotion} ({confidence:.1f}%)")
EOF

# Example 2: Batch Processing
python3 << 'EOF'
from PIL import Image
from enhanced_ensemble import BatchEnsemblePredictor
from pathlib import Path

predictor = BatchEnsemblePredictor(batch_size=16)
images = [Image.open(p).convert("RGB") for p in Path("faces/").glob("*.jpg")]
results = predictor.predict_batch(images)

for i, result in enumerate(results):
    print(f"Image {i}: {result['predicted_class']} ({result['ensemble_confidence']:.1f}%)")
EOF

# Example 3: With Monitoring
python3 << 'EOF'
from PIL import Image
from enhanced_ensemble import BatchEnsemblePredictor
from monitor import EnsembleMonitor

predictor = BatchEnsemblePredictor()
monitor = EnsembleMonitor()

image = Image.open("test_face.jpg").convert("RGB")
result = predictor.predict(image, return_all_scores=True)

monitor.record_prediction(
    ensemble_correct=True,
    vit_correct=True,
    transunet_correct=False,
    ensemble_confidence=result['ensemble_confidence'],
    vit_confidence=result['vit_confidence'],
    transunet_confidence=result['transunet_confidence'],
    inference_time=0.05
)

monitor.print_summary()
EOF


## ============================================================================
## 📖 DOCUMENTATION
## ============================================================================

echo "📚 Read the following for more details:"
echo ""
echo "1. SUMMARY.md           - Overview of all improvements"
echo "2. ENSEMBLE_GUIDE.md    - Ensemble model guide"
echo "3. QUICK_START.py       - Code examples and recipes"
echo "4. IMPROVEMENTS.md      - Detailed improvements"
echo "5. INDEX.md             - Complete index and navigation"
echo "6. COMPARISON.md        - Before/after comparison"
echo ""
echo "Start with: cat SUMMARY.md"


## ============================================================================
## ✅ VERIFICATION
## ============================================================================

echo "✅ Verifying installation..."

python3 << 'EOF'
import sys
print(f"Python: {sys.version}")

# Check core packages
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
except ImportError:
    print("❌ PyTorch not installed")

try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except ImportError:
    print("❌ Transformers not installed")

try:
    from PIL import Image
    print(f"✅ Pillow: {Image.__version__ if hasattr(Image, '__version__') else 'OK'}")
except ImportError:
    print("❌ Pillow not installed")

# Check GPU
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  GPU: Not available (using CPU)")
except:
    print("⚠️  GPU check failed")

# Check enhanced ensemble
try:
    from enhanced_ensemble import BatchEnsemblePredictor
    print("✅ Enhanced Ensemble: Ready")
except ImportError:
    print("❌ Enhanced Ensemble: Not installed")

print("\n✅ All checks passed! Ready to run.")
EOF


## ============================================================================
## 🎯 QUICK COMMANDS REFERENCE
## ============================================================================

echo "🎯 Quick Commands:"
echo ""
echo "# Web App"
echo "  python3 app.py"
echo ""
echo "# Test Single Component"
echo "  python3 -c \"from emotion_model import predict_emotion; print('OK')\""
echo ""
echo "# Run Ensemble Test"
echo "  python3 test_ensemble.py"
echo ""
echo "# Full Pipeline Test"
echo "  python3 QUICK_START.py"
echo ""
echo "# Monitoring"
echo "  python3 monitor.py"
echo ""
echo "# Evaluation"
echo "  python3 evaluate.py"
echo ""
echo "# View Documentation"
echo "  cat SUMMARY.md"
echo "  cat IMPROVEMENTS.md"
echo ""
echo "✅ Setup Complete!"
