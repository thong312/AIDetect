"""
Quick Start Guide - Using Enhanced Ensemble
Nhanh chóng bắt đầu sử dụng ensemble model đã cải thiện
"""

# =============================================================================
# 🚀 NHANH: Sử dụng Ensemble (2 dòng code)
# =============================================================================

from enhanced_ensemble import BatchEnsemblePredictor
from PIL import Image

# Init
predictor = BatchEnsemblePredictor(num_classes=7)

# Predict
image = Image.open("face.jpg").convert("RGB")
result = predictor.predict(image, return_all_scores=True)
print(f"Emotion: {result['predicted_class']}")
print(f"Ensemble confidence: {result['ensemble_confidence']:.1f}%")
print(f"ViT confidence: {result['vit_confidence']:.1f}%")
print(f"TransUNet confidence: {result['transunet_confidence']:.1f}%")


# =============================================================================
# 📊 MONITORING: Theo dõi performance
# =============================================================================

from monitor import EnsembleMonitor

monitor = EnsembleMonitor(log_file="logs/ensemble.log", enable_detailed_logging=True)

# Process multiple images
for i, image_path in enumerate(image_paths):
    image = Image.open(image_path).convert("RGB")
    result = predictor.predict(image, return_all_scores=True)
    
    # Record metrics
    monitor.record_prediction(
        ensemble_correct=(result['predicted_class'] == true_labels[i]),
        vit_correct=(vit_preds[i] == true_labels[i]),
        transunet_correct=(transunet_preds[i] == true_labels[i]),
        ensemble_confidence=result['ensemble_confidence'],
        vit_confidence=result['vit_confidence'],
        transunet_confidence=result['transunet_confidence'],
        inference_time=0.05,
        image_name=Path(image_path).name
    )

# Print summary
monitor.print_summary()
monitor.export_report("ensemble_report.json")


# =============================================================================
# 🔍 EVALUATION: Đánh giá chi tiết
# =============================================================================

from MLOps.evaluate import ModelComparator, EvaluationMetrics
import numpy as np

# Collect predictions
y_true = np.array(true_labels)
y_pred_vit = np.array(vit_preds)
y_pred_transunet = np.array(transunet_preds)
y_pred_ensemble = np.array(ensemble_preds)

# Compare models
comparator = ModelComparator()
comparator.add_model_results("ViT-B/16", y_true, y_pred_vit)
comparator.add_model_results("TransUNet", y_true, y_pred_transunet)
comparator.add_model_results("Ensemble", y_true, y_pred_ensemble)

comparator.print_comparison()


# =============================================================================
# 🎯 BATCH PROCESSING: Xử lý nhiều images cùng lúc
# =============================================================================

from enhanced_ensemble import BatchEnsemblePredictor
from PIL import Image

predictor = BatchEnsemblePredictor(batch_size=16)

# Load images
images = [Image.open(f"img_{i}.jpg").convert("RGB") for i in range(100)]

# Process all at once (Fast!)
results = predictor.predict_batch(images, return_all_scores=True)

# Results
for i, result in enumerate(results):
    print(f"Image {i}: "
          f"Class {result['predicted_class']} "
          f"({result['ensemble_confidence']:.1f}%)")


# =============================================================================
# ⚙️ CUSTOM CONFIGURATION: Tùy chỉnh
# =============================================================================

# 1. Different ensemble method
predictor = BatchEnsemblePredictor(
    ensemble_method='harmonic_mean',  # Default: 'weighted_average'
    batch_size=8
)

# 2. Adjust weights
predictor = BatchEnsemblePredictor(
    vit_weight=0.7,           # ViT: 70%
    transunet_weight=0.3,     # TransUNet: 30%
)

# 3. Enable calibration
from enhanced_ensemble import EnhancedEnsembleModel

model = EnhancedEnsembleModel(
    use_calibration=True,
    enable_caching=True
)

# 4. Tune batch size based on GPU
import torch
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_memory > 8:
        batch_size = 32
    elif gpu_memory > 4:
        batch_size = 16
    else:
        batch_size = 4
    predictor = BatchEnsemblePredictor(batch_size=batch_size)


# =============================================================================
# 🔧 OPTIMIZATION: Tối ưu performance
# =============================================================================

from monitor import EnsembleMonitor, PerformanceOptimizer

monitor = EnsembleMonitor()
optimizer = PerformanceOptimizer(monitor)

# ... collect metrics ...

# Identify bottleneck
print(f"Bottleneck: {optimizer.get_bottleneck()}")

# Get recommendations
for rec in optimizer.get_recommendations():
    print(f"✓ {rec}")

# Example outputs:
# Bottleneck: TransUNet (125.4ms) is slower
# ✓ TransUNet is much slower. Consider using ViT-only.
# ✓ Consider increasing ViT weight.


# =============================================================================
# 📈 CALIBRATION: Cải thiện độ tin cậy
# =============================================================================

from enhanced_ensemble import ConfidenceCalibrator
import numpy as np

# Create calibrator
calibrator = ConfidenceCalibrator(temperature=1.0)

# Tune on validation set
val_logits = np.random.randn(100, 7)  # Example
val_labels = np.random.randint(0, 7, 100)

calibrator.tune(val_logits, val_labels, lr=0.01, epochs=50)

# Use calibrated predictions
logits = model(images)
calibrated_probs = calibrator.calibrate(logits)


# =============================================================================
# 🎓 CROSS-VALIDATION: Reliable evaluation
# =============================================================================

from MLOps.evaluate import CrossValidator
import numpy as np

validator = CrossValidator(n_splits=5)

# Prepare data
images = [...]  # List of images
labels = np.array([...])  # Ground truth

# Split into folds
folds = validator.stratified_split(images, labels)

# Evaluate each fold
for fold_idx, (train_X, train_y, val_X, val_y) in enumerate(folds):
    print(f"\nFold {fold_idx + 1}/5")
    
    # Evaluate fold
    y_pred = predict_fold(val_X)  # Your prediction function
    result = validator.evaluate_fold(val_y, y_pred)
    validator.fold_results.append(result)

# Final results
cv_results = validator.get_cv_results()
print(f"\nFinal Accuracy: {cv_results['accuracy_mean']:.2f}% "
      f"± {cv_results['accuracy_std']:.2f}%")


# =============================================================================
# 💾 SAVE & LOAD CONFIGURATION
# =============================================================================

import json

# Save config
config = {
    'vit_weight': 0.6,
    'transunet_weight': 0.4,
    'ensemble_method': 'weighted_average',
    'batch_size': 8,
    'use_calibration': True,
    'temperature': 1.5
}

with open('ensemble_config.json', 'w') as f:
    json.dump(config, f, indent=2)

# Load config
with open('ensemble_config.json') as f:
    config = json.load(f)

predictor = BatchEnsemblePredictor(
    vit_weight=config['vit_weight'],
    transunet_weight=config['transunet_weight'],
    ensemble_method=config['ensemble_method'],
    batch_size=config['batch_size']
)


# =============================================================================
# 🚨 ERROR HANDLING
# =============================================================================

from PIL import Image
from enhanced_ensemble import BatchEnsemblePredictor

predictor = BatchEnsemblePredictor()

try:
    image = Image.open("invalid_path.jpg").convert("RGB")
    result = predictor.predict(image)
    print(f"Success: {result['ensemble_confidence']:.1f}%")
except FileNotFoundError:
    print("Error: Image file not found")
except Exception as e:
    print(f"Error: {e}")
    # Model has built-in error handling
    # Returns uniform distribution if model fails


# =============================================================================
# 📊 FULL PIPELINE: Từ A đến Z
# =============================================================================

def full_evaluation_pipeline(image_paths, true_labels):
    """Complete pipeline with monitoring and evaluation"""
    
    # Initialize
    predictor = BatchEnsemblePredictor(batch_size=16)
    monitor = EnsembleMonitor()
    
    # Collect predictions
    vit_preds = []
    transunet_preds = []
    ensemble_preds = []
    all_results = []
    
    # Process images
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        result = predictor.predict(image, return_all_scores=True)
        
        # Store predictions
        vit_pred = result.get('vit_pred', result['predicted_class'])
        transunet_pred = result.get('transunet_pred', result['predicted_class'])
        ensemble_pred = result['predicted_class']
        
        vit_preds.append(vit_pred)
        transunet_preds.append(transunet_pred)
        ensemble_preds.append(ensemble_pred)
        all_results.append(result)
        
        # Monitor
        monitor.record_prediction(
            ensemble_correct=(ensemble_pred == true_labels[i]),
            vit_correct=(vit_pred == true_labels[i]),
            transunet_correct=(transunet_pred == true_labels[i]),
            ensemble_confidence=result['ensemble_confidence'],
            vit_confidence=result.get('vit_confidence', 0),
            transunet_confidence=result.get('transunet_confidence', 0),
            inference_time=0.05,
            image_name=Path(image_path).name
        )
    
    # Evaluate
    from MLOps.evaluate import ModelComparator
    comparator = ModelComparator()
    comparator.add_model_results("ViT", np.array(true_labels), np.array(vit_preds))
    comparator.add_model_results("TransUNet", np.array(true_labels), np.array(transunet_preds))
    comparator.add_model_results("Ensemble", np.array(true_labels), np.array(ensemble_preds))
    
    # Print results
    print("\n" + "="*80)
    monitor.print_summary()
    comparator.print_comparison()
    
    # Export reports
    monitor.export_report("monitoring_report.json")
    
    return {
        'monitor': monitor,
        'comparator': comparator,
        'results': all_results
    }


# Usage
if __name__ == "__main__":
    from pathlib import Path
    
    image_paths = list(Path("data/").glob("*.jpg"))
    true_labels = [...]  # Your labels
    
    results = full_evaluation_pipeline(image_paths, true_labels)
