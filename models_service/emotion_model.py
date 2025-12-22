"""
Emotion prediction with Ensemble Model (ViT-B/16 + TransUNet)
"""

import torch
from transformers import ViTForImageClassification, ViTImageProcessor

MODEL_PATH = "models/emotion_vit_rafdb"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load processor + model
processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
vit_model = ViTForImageClassification.from_pretrained(MODEL_PATH).to(device)

# --- Lazy load ensemble model (only when needed) ---
_ensemble_predictor = None

def get_ensemble_predictor():
    """Lazy load ensemble predictor to avoid circular imports"""
    global _ensemble_predictor
    if _ensemble_predictor is None:
        from models_service.ensemble_model import EnsemblePredictor
        _ensemble_predictor = EnsemblePredictor(
            num_classes=7,
            vit_weight=0.6,
            transunet_weight=0.4,
            ensemble_method='weighted_average',
            device=device
        )
    return _ensemble_predictor


def predict_emotion_vit_only(image_pil):
    """
    Original ViT-B/16 only prediction
    Returns: (emotion_label, confidence)
    """
    inputs = processor(image_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = vit_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

    pred_idx = logits.argmax(-1).item()
    emotion = vit_model.config.id2label[pred_idx]
    confidence = (probs[0][pred_idx].item() * 100)
    
    return emotion, confidence


def predict_emotion_ensemble(image_pil, return_details=False):
    """
    Ensemble prediction (ViT-B/16 + TransUNet)
    Better accuracy by combining both models
    
    Args:
        image_pil: PIL Image
        return_details: if True, return confidence from both models
    
    Returns:
        emotion_label (str), ensemble_confidence (float), 
        details (dict) - optional: ViT and TransUNet confidence
    """
    ensemble_predictor = get_ensemble_predictor()
    result = ensemble_predictor.predict(image_pil, return_all_scores=return_details)
    
    pred_idx = result['predicted_class']
    emotion = vit_model.config.id2label[pred_idx]
    ensemble_confidence = result['ensemble_confidence']
    
    if return_details:
        details = {
            'ensemble_confidence': ensemble_confidence,
            'vit_confidence': result['vit_confidence'],
            'transunet_confidence': result['transunet_confidence']
        }
        return emotion, ensemble_confidence, details
    
    return emotion, ensemble_confidence


def predict_emotion(image_pil, use_ensemble=True):
    """
    Predict emotion from image
    
    Args:
        image_pil: PIL Image
        use_ensemble: if True, use ensemble; if False, use ViT only
    
    Returns:
        emotion_label (str), confidence (float)
    """
    if use_ensemble:
        return predict_emotion_ensemble(image_pil)
    else:
        return predict_emotion_vit_only(image_pil)
