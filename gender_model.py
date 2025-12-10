import torch
from transformers import ViTForImageClassification, ViTImageProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = ViTImageProcessor.from_pretrained("models/gender_vit_utkface")
model = ViTForImageClassification.from_pretrained("models/gender_vit_utkface").to(device)

def predict_gender(image_pil):
    """Nhận vào PIL Image, dự đoán giới tính"""
    inputs = processor(image_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        pred = logits.argmax(-1).item()

    return "Nam" if pred == 0 else "Nữ"
