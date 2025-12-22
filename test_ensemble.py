"""
Test script to demonstrate ensemble model usage
"""

import torch
import time
from PIL import Image
from pathlib import Path

# Test 1: Load models
print("=" * 60)
print("TEST 1: Loading Models")
print("=" * 60)

try:
    from models_service.emotion_model import predict_emotion, predict_emotion_ensemble
    from models_service.transunet_model import TransUNetClassifier
    from models_service.ensemble_model import EnsemblePredictor
    
    print("✓ All imports successful!")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Test 2: Single model inference
print("\n" + "=" * 60)
print("TEST 2: Create Models")
print("=" * 60)

try:
    # Create TransUNet model
    transunet = TransUNetClassifier(num_classes=7, pretrained=True).to(device)
    transunet.eval()
    print("✓ TransUNet loaded successfully!")
    
    # Create ensemble
    predictor = EnsemblePredictor(num_classes=7, device=device)
    print("✓ Ensemble model loaded successfully!")
    
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    exit(1)

# Test 3: Forward pass
print("\n" + "=" * 60)
print("TEST 3: Forward Pass")
print("=" * 60)

try:
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # TransUNet forward pass
    start = time.time()
    with torch.no_grad():
        transunet_output = transunet(dummy_input)
    transunet_time = time.time() - start
    print(f"✓ TransUNet forward pass: {transunet_time:.4f}s")
    print(f"  Output shape: {transunet_output.shape}")
    
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    exit(1)

# Test 4: Inference with actual images (if available)
print("\n" + "=" * 60)
print("TEST 4: Inference with Real Images")
print("=" * 60)

# Find a test image
test_images = list(Path(".").glob("**/*.jpg")) + list(Path(".").glob("**/*.png"))

if test_images:
    test_image = test_images[0]
    print(f"Using test image: {test_image}")
    
    try:
        image = Image.open(test_image).convert("RGB")
        
        # Test ViT only
        print("\n1️⃣ ViT-B/16 (Standalone)")
        start = time.time()
        emotion_vit, conf_vit = predict_emotion(image, use_ensemble=False)
        vit_time = time.time() - start
        print(f"   Emotion: {emotion_vit}")
        print(f"   Confidence: {conf_vit:.2f}%")
        print(f"   Time: {vit_time:.4f}s")
        
        # Test Ensemble
        print("\n2️⃣ Ensemble (ViT + TransUNet)")
        start = time.time()
        emotion_ensemble, conf_ensemble = predict_emotion(image, use_ensemble=True)
        ensemble_time = time.time() - start
        print(f"   Emotion: {emotion_ensemble}")
        print(f"   Confidence: {conf_ensemble:.2f}%")
        print(f"   Time: {ensemble_time:.4f}s")
        
        # Test detailed scores
        print("\n3️⃣ Detailed Scores")
        emotion_det, conf_det, details = predict_emotion_ensemble(image, return_details=True)
        print(f"   ViT Confidence: {details['vit_confidence']:.2f}%")
        print(f"   TransUNet Confidence: {details['transunet_confidence']:.2f}%")
        print(f"   Ensemble Confidence: {details['ensemble_confidence']:.2f}%")
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        print(f"  Make sure you have test images in the workspace")
else:
    print("⚠ No test images found. Skipping inference test.")

# Test 5: Model summary
print("\n" + "=" * 60)
print("TEST 5: Model Configuration")
print("=" * 60)

print("\n1️⃣ TransUNet Architecture:")
print(f"   Total parameters: {sum(p.numel() for p in transunet.parameters()):,}")
print(f"   Trainable parameters: {sum(p.numel() for p in transunet.parameters() if p.requires_grad):,}")

print("\n2️⃣ Ensemble Configuration:")
print(f"   Ensemble method: weighted_average")
print(f"   ViT weight: 0.6")
print(f"   TransUNet weight: 0.4")

print("\n3️⃣ Memory Usage (estimated):")
print(f"   ViT-B/16: ~350 MB")
print(f"   TransUNet: ~800 MB")
print(f"   Ensemble: ~1.1 GB")

# Test 6: Usage examples
print("\n" + "=" * 60)
print("TEST 6: Code Usage Examples")
print("=" * 60)

examples = """
# Example 1: Simple usage (ensemble by default)
from emotion_model import predict_emotion
from PIL import Image

image = Image.open("face.jpg").convert("RGB")
emotion, confidence = predict_emotion(image)
print(f"{emotion}: {confidence:.2f}%")

# Example 2: ViT only (original)
emotion, confidence = predict_emotion(image, use_ensemble=False)

# Example 3: Detailed scores from ensemble
from emotion_model import predict_emotion_ensemble

emotion, conf, details = predict_emotion_ensemble(image, return_details=True)
print(f"ViT: {details['vit_confidence']:.2f}%")
print(f"TransUNet: {details['transunet_confidence']:.2f}%")

# Example 4: Custom ensemble configuration
from ensemble_model import EnsemblePredictor

predictor = EnsemblePredictor(
    vit_weight=0.7,
    transunet_weight=0.3,
    ensemble_method='attention'
)
result = predictor.predict(image, return_all_scores=True)
"""

print(examples)

print("\n" + "=" * 60)
print("✅ ALL TESTS COMPLETED!")
print("=" * 60)
print("\nNext steps:")
print("1. Run inference on your test images")
print("2. Compare ViT vs Ensemble accuracy")
print("3. Adjust weights based on results")
print("4. Consider fine-tuning on your data")
