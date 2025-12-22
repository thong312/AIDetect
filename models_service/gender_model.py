import torch
from transformers import ViTForImageClassification, ViTImageProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = ViTImageProcessor.from_pretrained("models/gender_vit_utkface")
model = ViTForImageClassification.from_pretrained("models/gender_vit_utkface").to(device)

def predict_gender(image_pil):
    """Nhận vào PIL Image, dự đoán giới tính với confidence"""
    inputs = processor(image_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        pred = logits.argmax(-1).item()
        confidence = (probs[0][pred].item() * 100)

    gender = "Nam" if pred == 0 else "Nữ"
    return gender, confidence
