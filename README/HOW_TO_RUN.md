# 🎯 How to Run AIHumanDetect - Complete Guide

## ⚡ TL;DR (30 seconds)

```bash
# 1. Install
pip install torch torchvision transformers pillow flask scikit-learn matplotlib numpy psutil

# 2. Run Web App
python3 app.py

# 3. Open browser
# http://localhost:5000
```

---

## 📋 Table of Contents

1. [Installation](#installation)
2. [Running Web App](#running-web-app)
3. [Running Ensemble Model](#running-ensemble-model)
4. [Running Tests](#running-tests)
5. [Examples](#examples)
6. [GPU Setup](#gpu-setup)
7. [Troubleshooting](#troubleshooting)
8. [Next Steps](#next-steps)

---

## Installation

### Option 1: Quick Install (Recommended)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements_enhanced.txt
```

### Option 2: Manual Install

```bash
# Core ML libraries
pip install torch>=2.0.0 torchvision>=0.15.0 transformers>=4.35.0

# Image processing
pip install Pillow>=10.0.0

# Web framework
pip install Flask>=2.0.0 SQLAlchemy>=2.0.0

# Enhanced features
pip install scikit-learn>=1.3.0 matplotlib>=3.8.0 numpy>=1.24.0 psutil>=5.9.0
```

### Verify Installation

```bash
python3 -c "
import torch
import transformers
print('✅ Installation successful!')
print(f'PyTorch: {torch.__version__}')
print(f'GPU Available: {torch.cuda.is_available()}')
"
```

---

## Running Web App

### Start Server

```bash
python3 app.py
```

**Output:**
```
 * Running on http://127.0.0.1:5000
 * Press CTRL+C to quit
```

### Open Browser

Go to: **http://localhost:5000**

### Features

1. **Register Page**
   - Upload face image
   - Enter name
   - Click "Register"
   - System saves face embedding to database

2. **Recognize Page**
   - Upload face image
   - Click "Recognize"
   - Shows name, emotion, gender, age
   - Shows confidence scores

3. **Analyze Page**
   - Upload any face image
   - Shows detailed emotion probabilities
   - Shows gender and age classifications
   - No database lookup needed

---

## Running Ensemble Model

### Quick Test

```bash
python3 -c "
from PIL import Image
from enhanced_ensemble import BatchEnsemblePredictor

# Load predictor
predictor = BatchEnsemblePredictor(batch_size=8)

# Load and predict
image = Image.open('face.jpg').convert('RGB')
result = predictor.predict(image, return_all_scores=True)

# Show results
print(f'Emotion Class: {result[\"predicted_class\"]}')
print(f'Ensemble Confidence: {result[\"ensemble_confidence\"]:.1f}%')
print(f'ViT Confidence: {result[\"vit_confidence\"]:.1f}%')
print(f'TransUNet Confidence: {result[\"transunet_confidence\"]:.1f}%')
"
```

### Run Test Script

```bash
python3 test_ensemble.py
```

This runs comprehensive tests on:
- Model loading
- Inference
- Memory usage
- Performance metrics

### Batch Processing (Fast)

```bash
python3 -c "
from PIL import Image
from pathlib import Path
from enhanced_ensemble import BatchEnsemblePredictor

# Setup
predictor = BatchEnsemblePredictor(batch_size=16)

# Load multiple images
images = [
    Image.open(p).convert('RGB') 
    for p in Path('faces/').glob('*.jpg')
][:100]

# Process all at once (much faster!)
results = predictor.predict_batch(images)

# Show results
for i, result in enumerate(results):
    print(f'Image {i}: Confidence {result[\"ensemble_confidence\"]:.1f}%')
"
```

---

## Running Tests

### Test Individual Models

```bash
# Test Emotion Model
python3 -c "
from PIL import Image
from emotion_model import predict_emotion

image = Image.open('face.jpg').convert('RGB')
emotion, confidence = predict_emotion(image)
print(f'Emotion: {emotion} ({confidence:.1f}%)')
"

# Test Gender Model
python3 -c "
from PIL import Image
from gender_model import predict_gender

image = Image.open('face.jpg').convert('RGB')
gender, confidence = predict_gender(image)
print(f'Gender: {gender} ({confidence:.1f}%)')
"

# Test Age Model
python3 -c "
from PIL import Image
from age_model import predict_age_group

image = Image.open('face.jpg').convert('RGB')
age, confidence = predict_age_group(image)
print(f'Age: {age} ({confidence:.1f}%)')
"
```

### Test Ensemble

```bash
# Basic ensemble test
python3 test_ensemble.py

# Or detailed testing
python3 -c "
from enhanced_ensemble import EnhancedEnsembleModel
import torch

model = EnhancedEnsembleModel(num_classes=7)
dummy = torch.randn(1, 3, 224, 224)
output, vit_out, tu_out = model(dummy)

print(f'Output shape: {output.shape}')
print(f'ViT output shape: {vit_out.shape}')
print(f'TransUNet output shape: {tu_out.shape}')
print('✅ Ensemble model works!')
"
```

---

## Examples

### Example 1: Single Image Classification

```python
from PIL import Image
from enhanced_ensemble import BatchEnsemblePredictor

# Create predictor
predictor = BatchEnsemblePredictor()

# Load image
image = Image.open('my_face.jpg').convert('RGB')

# Predict
result = predictor.predict(image, return_all_scores=True)

# Print results
emotion_id = result['predicted_class']
emotion_names = {
    0: 'Anger', 1: 'Disgust', 2: 'Fear',
    3: 'Happiness', 4: 'Sadness', 5: 'Surprise', 6: 'Neutral'
}

print(f"Detected emotion: {emotion_names[emotion_id]}")
print(f"Confidence: {result['ensemble_confidence']:.1f}%")
```

### Example 2: Batch Processing with Monitoring

```python
from PIL import Image
from pathlib import Path
from enhanced_ensemble import BatchEnsemblePredictor
from monitor import EnsembleMonitor

# Setup
predictor = BatchEnsemblePredictor(batch_size=16)
monitor = EnsembleMonitor()

# Load images
images = [
    Image.open(p).convert('RGB')
    for p in Path('test_faces/').glob('*.jpg')
]

# Process batch
for i, image in enumerate(images):
    result = predictor.predict(image, return_all_scores=True)
    
    # Record metrics (assuming we have ground truth labels)
    monitor.record_prediction(
        ensemble_correct=True,  # Replace with actual comparison
        vit_correct=True,
        transunet_correct=False,
        ensemble_confidence=result['ensemble_confidence'],
        vit_confidence=result['vit_confidence'],
        transunet_confidence=result['transunet_confidence'],
        inference_time=0.05,
        image_name=f"image_{i}.jpg"
    )

# Print summary
monitor.print_summary()
monitor.export_report('results.json')
```

### Example 3: Model Evaluation

```python
from enhanced_ensemble import BatchEnsemblePredictor
from evaluate import ModelComparator
import numpy as np

# Predictions from different models
y_true = np.array([0, 1, 2, 0, 1, 2, 3, 4, 5, 6])
y_pred_vit = np.array([0, 1, 2, 0, 1, 1, 3, 4, 5, 6])
y_pred_transunet = np.array([0, 1, 2, 0, 1, 2, 3, 4, 4, 6])
y_pred_ensemble = np.array([0, 1, 2, 0, 1, 2, 3, 4, 5, 6])

# Compare models
comparator = ModelComparator()
comparator.add_model_results("ViT", y_true, y_pred_vit)
comparator.add_model_results("TransUNet", y_true, y_pred_transunet)
comparator.add_model_results("Ensemble", y_true, y_pred_ensemble)

# Print comparison
comparator.print_comparison()
```

### Example 4: Performance Optimization

```python
from enhanced_ensemble import BatchEnsemblePredictor
from monitor import EnsembleMonitor, PerformanceOptimizer

# Setup monitoring
monitor = EnsembleMonitor()
# ... make predictions and record metrics ...
monitor.print_summary()

# Get optimization tips
optimizer = PerformanceOptimizer(monitor)
print(f"Bottleneck: {optimizer.get_bottleneck()}")

print("\nOptimization recommendations:")
for recommendation in optimizer.get_recommendations():
    print(f"  ✓ {recommendation}")
```

---

## GPU Setup

### Check GPU

```bash
python3 -c "
import torch
print(f'GPU Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

### Optimal Batch Size by GPU Memory

```python
import torch
from enhanced_ensemble import BatchEnsemblePredictor

if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    if gpu_memory > 8:
        batch_size = 32
    elif gpu_memory > 4:
        batch_size = 16
    elif gpu_memory > 2:
        batch_size = 8
    else:
        batch_size = 4
else:
    batch_size = 1  # CPU only

predictor = BatchEnsemblePredictor(batch_size=batch_size)
print(f"Using batch size: {batch_size}")
```

### Force CPU (if GPU issues)

```python
# All PyTorch operations use CPU
import torch
torch.cuda.is_available = lambda: False

# Or specify in code
from enhanced_ensemble import BatchEnsemblePredictor
predictor = BatchEnsemblePredictor()  # Uses CPU if CUDA unavailable
```

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'torch'"

**Solution:**
```bash
pip install torch torchvision transformers
```

### Problem: "CUDA out of memory"

**Solution:**
```python
# Option 1: Reduce batch size
predictor = BatchEnsemblePredictor(batch_size=4)

# Option 2: Unload models
predictor.ensemble_model.unload_models()
import torch
torch.cuda.empty_cache()

# Option 3: Use CPU only
# (Restart Python and don't use GPU)
```

### Problem: "No module named 'face_recognition'"

**Solution:**
```bash
pip install face-recognition
# Or check face_recognition.py is in project root
```

### Problem: "Port 5000 already in use"

**Solution:**
```bash
# Option 1: Use different port
python3 app.py --port 8000

# Option 2: Kill existing process
# Linux/Mac:
lsof -i :5000 | grep LISTEN | awk '{print $2}' | xargs kill -9

# Windows:
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Problem: "Models not found"

**Solution:**
```bash
# Models auto-download on first use
# Or manually create models folder:
mkdir -p models

# Models should be in:
# models/emotion_vit_rafdb/
# models/gender_vit_utkface/
# models/age_vit_deploy/
```

### Problem: Slow inference

**Solution:**
```python
# Option 1: Use batch processing
results = predictor.predict_batch(images, batch_size=32)

# Option 2: Reduce model weight
predictor = BatchEnsemblePredictor(
    vit_weight=0.8,      # Use more ViT (faster)
    transunet_weight=0.2  # Use less TransUNet (slower)
)

# Option 3: Use CPU instead of slow GPU
```

---

## Next Steps

### 1. Explore Web Interface
```bash
python3 app.py
# Try register, recognize, and analyze features
```

### 2. Test Ensemble Model
```bash
python3 test_ensemble.py
# Verify ensemble works on your system
```

### 3. Read Documentation
```bash
cat SUMMARY.md          # Quick overview (5 min)
cat QUICK_START.py      # Code examples (20 min)
cat INDEX.md            # Complete guide (10 min)
```

### 4. Customize Configuration
```python
# Adjust weights
predictor = BatchEnsemblePredictor(
    vit_weight=0.7,
    transunet_weight=0.3,
    ensemble_method='harmonic_mean'
)

# Tune batch size
predictor = BatchEnsemblePredictor(batch_size=32)
```

### 5. Monitor Performance
```python
from monitor import EnsembleMonitor

monitor = EnsembleMonitor()
# ... make predictions ...
monitor.print_summary()
monitor.export_report('report.json')
```

### 6. Evaluate Models
```python
from evaluate import ModelComparator

comparator = ModelComparator()
comparator.add_model_results("Model", y_true, y_pred)
comparator.print_comparison()
```

---

## File Reference

| File | Purpose | Run Command |
|------|---------|-------------|
| **app.py** | Web application | `python3 app.py` |
| **test_ensemble.py** | Test ensemble | `python3 test_ensemble.py` |
| **test.py** | Test original system | `python3 test.py` |
| **QUICK_START.py** | Code examples | `python3 QUICK_START.py` |
| **setup_guide.py** | Interactive setup | `python3 setup_guide.py` |

---

## Quick Commands

```bash
# Install dependencies
pip install -r requirements_enhanced.txt

# Run web app
python3 app.py

# Test ensemble
python3 test_ensemble.py

# Check GPU
python3 -c "import torch; print(torch.cuda.is_available())"

# View docs
cat SUMMARY.md
cat IMPROVEMENTS.md
cat QUICK_START.py
```

---

## Support

- 📖 See **INDEX.md** for complete navigation
- 💻 See **QUICK_START.py** for code examples
- ⚙️ See **SETUP_GUIDE.sh** for technical setup
- 📊 See **IMPROVEMENTS.md** for architecture details
- 🔍 See **COMPARISON.md** for before/after analysis

---

**You're ready! Pick a method above and get started.** 🚀
