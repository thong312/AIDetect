# 📚 Hướng dẫn: Kết hợp ViT-B/16 + TransUNet (ResNet-50)

## 🎯 Tổng quan

Bạn đã triển khai thành công **Ensemble Model** kết hợp:
- **ViT-B/16** (Vision Transformer) - mô hình hiện tại
- **TransUNet** với backbone **ResNet-50** - mô hình mới

## 📁 Các file tạo mới

### 1. **transunet_model.py**
Chứa kiến trúc TransUNet với các thành phần:
- `ResNet50Backbone`: Extracts features từ ResNet-50
- `TransformerBlock`: Transformer encoder blocks
- `DecoderBlock`: Upsampling blocks với skip connections
- `TransUNetClassifier`: TransUNet được adapt cho image classification

### 2. **ensemble_model.py**
Kết hợp ViT và TransUNet:
- `EnsembleModel`: Wrapper chính cho cả 2 mô hình
- `EnsemblePredictor`: Interface dễ sử dụng để inference

### 3. **emotion_model.py** (Updated)
Thêm hỗ trợ ensemble:
- `predict_emotion_vit_only()`: Chỉ dùng ViT (cách cũ)
- `predict_emotion_ensemble()`: Dùng ensemble (cách mới)
- `predict_emotion()`: Hàm mặc định (default = ensemble)

## 🚀 Cách sử dụng

### Option A: Sử dụng Ensemble (Recommended)

```python
from emotion_model import predict_emotion

# Sử dụng ensemble mô hình (default)
emotion, confidence = predict_emotion(image_pil)

# Hoặc lấy chi tiết từ cả 2 mô hình
emotion, ensemble_conf, details = predict_emotion_ensemble(image_pil, return_details=True)
print(f"ViT confidence: {details['vit_confidence']:.2f}%")
print(f"TransUNet confidence: {details['transunet_confidence']:.2f}%")
print(f"Ensemble confidence: {details['ensemble_confidence']:.2f}%")
```

### Option B: Sử dụng chỉ ViT (cách cũ)

```python
emotion, confidence = predict_emotion(image_pil, use_ensemble=False)
```

### Option C: Custom Ensemble Configuration

```python
from ensemble_model import EnsemblePredictor

# Tùy chỉnh trọng số và phương pháp kết hợp
predictor = EnsemblePredictor(
    num_classes=7,
    vit_weight=0.7,           # ViT: 70%
    transunet_weight=0.3,     # TransUNet: 30%
    ensemble_method='attention',  # 'weighted_average' | 'attention' | 'max_voting'
    device='cuda'
)

result = predictor.predict(image_pil, return_all_scores=True)
```

## 🔧 Các phương pháp Ensemble

### 1. **Weighted Average** (Default)
```
final_prediction = 0.6 * ViT_output + 0.4 * TransUNet_output
```
- ✅ Đơn giản, nhanh
- ✅ Dễ điều chỉnh trọng số
- ❌ Trọng số cố định

### 2. **Attention-based**
```
weights = learned_attention([ViT_output, TransUNet_output])
final_prediction = weights[0] * ViT + weights[1] * TransUNet
```
- ✅ Tự học trọng số tối ưu
- ✅ Có thể dynamic theo image
- ❌ Phức tạp hơn, chậm hơn

### 3. **Max Voting**
```
final_prediction = max(ViT_output, TransUNet_output)
```
- ✅ Chỉ lấy dự đoán tốt nhất
- ❌ Bỏ đi thông tin từ mô hình kém

## 📊 Hiệu suất dự kiến

| Model | Đặc điểm | Tốc độ |
|-------|----------|-------|
| **ViT-B/16** | Chính xác cao, nhanh | ⚡ Nhanh |
| **TransUNet** | Tốt cho features spatial | 🐢 Chậm hơn |
| **Ensemble** | Chính xác cao nhất | 🟡 Trung bình |

**Kết hợp 2 mô hình thường tăng độ chính xác 2-5%** so với mô hình đơn.

## 💾 Memory Usage

- **ViT-B/16**: ~350 MB VRAM
- **TransUNet**: ~800 MB VRAM
- **Ensemble**: ~1.1 GB VRAM

Nếu memory hạn chế, dùng `vit_weight=0.8, transunet_weight=0.2` (ViT chủ yếu).

## 🎓 Huấn luyện TransUNet (Optional)

Nếu muốn fine-tune TransUNet trên dữ liệu riêng:

```python
from transunet_model import TransUNetClassifier
import torch
import torch.nn as nn

model = TransUNetClassifier(num_classes=7, pretrained=True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(10):
    # ... training code ...
    pass
```

## ⚙️ Các cách điều chỉnh

### 1. Thay đổi trọng số
```python
# ViT more: 70% ViT, 30% TransUNet
emotion, conf = predict_emotion_ensemble(image, vit_weight=0.7, transunet_weight=0.3)
```

### 2. Thay đổi phương pháp
```python
# Sử dụng attention thay vì weighted average
predictor = EnsemblePredictor(ensemble_method='attention')
```

### 3. Sử dụng chỉ TransUNet
```python
# Không dùng ensemble
emotion, conf = predict_emotion(image, use_ensemble=False)
```

## 🔍 Debugging

### Kiểm tra model inference:
```python
from transunet_model import TransUNetClassifier
import torch

model = TransUNetClassifier(num_classes=7, pretrained=True)
dummy = torch.randn(1, 3, 224, 224)
output = model(dummy)
print(f"Output shape: {output.shape}")  # Should be (1, 7)
```

### Kiểm tra ensemble:
```python
from ensemble_model import EnsemblePredictor
from PIL import Image

predictor = EnsemblePredictor(num_classes=7)
img = Image.open("test.jpg").convert("RGB")
result = predictor.predict(img, return_all_scores=True)
print(result)
```

## 📈 Next Steps

1. **Fine-tune TransUNet** trên dữ liệu của bạn để cải thiện độ chính xác
2. **A/B Testing**: So sánh ViT-only vs Ensemble trên test set
3. **Distillation**: Nén ensemble model thành mô hình nhẹ hơn
4. **Multi-task**: Mở rộng ensemble cho gender + age

---

**Lưu ý**: Ensemble mặc định sử dụng **weighted_average** với **ViT:TransUNet = 0.6:0.4**. Bạn có thể tuning các giá trị này dựa trên kết quả trên validation set của bạn.
