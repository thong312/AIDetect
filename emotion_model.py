import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

MODEL_PATH = "models/emotion_vit_rafdb"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load processor + model
processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
model = ViTForImageClassification.from_pretrained(MODEL_PATH).to(device)

def predict_emotion(image_pil):
    inputs = processor(image_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    pred_idx = logits.argmax(-1).item()
    return model.config.id2label[pred_idx]
