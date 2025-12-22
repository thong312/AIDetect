"""
Ensemble model combining ViT-B/16 and TransUNet with ResNet-50 backbone
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTImageProcessor
from models_service.transunet_model import TransUNetClassifier


class EnsembleModel(nn.Module):
    """
    Kết hợp ViT-B/16 + TransUNet (ResNet-50 backbone)
    Sử dụng Weighted Average Ensemble để kết hợp dự đoán từ cả 2 mô hình
    """
    def __init__(self, vit_weight=0.6, transunet_weight=0.4, num_classes=7, device="cpu"):
        super(EnsembleModel, self).__init__()
        
        self.device = device
        self.num_classes = num_classes
        self.vit_weight = vit_weight
        self.transunet_weight = transunet_weight
        
        # Initialize ViT model
        self.vit_model = ViTForImageClassification.from_pretrained(
            "models/emotion_vit_rafdb"
        ).to(device)
        self.vit_model.eval()
        
        # Initialize TransUNet model
        self.transunet_model = TransUNetClassifier(
            in_channels=3, 
            num_classes=num_classes, 
            pretrained=True
        ).to(device)
        self.transunet_model.eval()
        
        # Optional: attention layer to learn weights dynamically
        self.weight_attention = nn.Sequential(
            nn.Linear(num_classes * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        ).to(device)

    def forward_vit(self, x):
        """Get predictions from ViT model"""
        with torch.no_grad():
            outputs = self.vit_model(x)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        return probs

    def forward_transunet(self, x):
        """Get predictions from TransUNet model"""
        with torch.no_grad():
            logits = self.transunet_model(x)
            probs = torch.softmax(logits, dim=-1)
        return probs

    def forward(self, x, method='weighted_average'):
        """
        Forward pass with ensemble logic
        
        Args:
            x: input image tensor
            method: 'weighted_average' | 'attention' | 'max_voting'
        
        Returns:
            ensemble_probs: combined probability distribution
        """
        vit_probs = self.forward_vit(x)
        transunet_probs = self.forward_transunet(x)
        
        if method == 'weighted_average':
            # Simple weighted average
            ensemble_probs = (
                self.vit_weight * vit_probs + 
                self.transunet_weight * transunet_probs
            )
        
        elif method == 'attention':
            # Learn weights dynamically using attention
            combined = torch.cat([vit_probs, transunet_probs], dim=1)
            weights = self.weight_attention(combined)  # (B, 2)
            
            ensemble_probs = (
                weights[:, 0:1] * vit_probs + 
                weights[:, 1:2] * transunet_probs
            )
        
        elif method == 'max_voting':
            # Take maximum probability
            ensemble_probs = torch.max(vit_probs, transunet_probs)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return ensemble_probs, vit_probs, transunet_probs


class EnsemblePredictor:
    """
    Wrapper class for easy inference with ensemble model
    """
    def __init__(self, num_classes=7, vit_weight=0.6, transunet_weight=0.4, 
                 ensemble_method='weighted_average', device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.ensemble_model = EnsembleModel(
            vit_weight=vit_weight,
            transunet_weight=transunet_weight,
            num_classes=num_classes,
            device=device
        )
        self.ensemble_method = ensemble_method
        
        # ViT processor
        self.vit_processor = ViTImageProcessor.from_pretrained("models/emotion_vit_rafdb")

    def predict(self, image_pil, return_all_scores=False):
        """
        Predict using ensemble model
        
        Args:
            image_pil: PIL Image
            return_all_scores: if True, return ViT and TransUNet scores separately
        
        Returns:
            predicted_class_idx: predicted class index
            ensemble_confidence: confidence from ensemble
            detailed_scores: dict with ViT and TransUNet scores (if return_all_scores=True)
        """
        # Preprocess image
        inputs = self.vit_processor(image_pil, return_tensors="pt").to(self.device)
        image_tensor = inputs['pixel_values']
        
        # Forward pass
        ensemble_probs, vit_probs, transunet_probs = self.ensemble_model(
            image_tensor, 
            method=self.ensemble_method
        )
        
        # Get predictions
        pred_idx = ensemble_probs.argmax(-1).item()
        ensemble_confidence = ensemble_probs[0][pred_idx].item() * 100
        
        result = {
            'predicted_class': pred_idx,
            'ensemble_confidence': ensemble_confidence
        }
        
        if return_all_scores:
            vit_conf = vit_probs[0][pred_idx].item() * 100
            transunet_conf = transunet_probs[0][pred_idx].item() * 100
            
            result['vit_confidence'] = vit_conf
            result['transunet_confidence'] = transunet_conf
            result['vit_probs'] = vit_probs[0].cpu().numpy()
            result['transunet_probs'] = transunet_probs[0].cpu().numpy()
        
        return result


if __name__ == "__main__":
    from PIL import Image
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test ensemble model
    predictor = EnsemblePredictor(
        num_classes=7,
        vit_weight=0.6,
        transunet_weight=0.4,
        ensemble_method='weighted_average',
        device=device
    )
    
    print("Ensemble Model loaded successfully!")
    print(f"Device: {device}")
    print(f"Ensemble method: weighted_average")
    print(f"ViT weight: 0.6, TransUNet weight: 0.4")
