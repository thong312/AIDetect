import torch
from transformers import ViTForImageClassification, ViTImageProcessor

MODEL_PATH = "models/age_vit_deploy"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
model = ViTForImageClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

LABEL_MAP = {
    0: "1-10",
    1: "11-20",
    2: "21-30",
    3: "31-40",
    4: "40+"
}

def predict_age_group(image_pil):
    inputs = processor(image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1).item()
        confidence = (probs[0][pred].item() * 100)

    return LABEL_MAP[pred], confidence
