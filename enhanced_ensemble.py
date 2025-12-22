"""
Enhanced Ensemble Model with:
- Model caching & lazy loading
- Memory optimization
- Confidence calibration
- Batch processing support
- Better error handling
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, List
from transformers import ViTForImageClassification, ViTImageProcessor
from models_service.transunet_model import TransUNetClassifier
import functools
import warnings


class ConfidenceCalibrator:
    """
    Calibrate confidence scores using temperature scaling
    Improves reliability of confidence estimates
    """
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits"""
        return torch.softmax(logits / self.temperature, dim=-1)

    def tune(self, logits: np.ndarray, labels: np.ndarray, lr: float = 0.01, epochs: int = 50):
        """
        Tune temperature on validation set
        Args:
            logits: model output logits (N, C)
            labels: ground truth labels (N,)
        """
        self.temperature = torch.tensor(1.0, requires_grad=True)
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=50)

        def eval():
            optimizer.zero_grad()
            probs = torch.softmax(torch.from_numpy(logits) / self.temperature, dim=-1)
            loss = nn.CrossEntropyLoss()(
                torch.from_numpy(logits) / self.temperature,
                torch.from_numpy(labels)
            )
            loss.backward()
            return loss

        optimizer.step(eval)
        self.temperature = self.temperature.item()
        print(f"Calibration completed. Optimal temperature: {self.temperature:.4f}")


class AdaptiveWeightScheduler:
    """
    Dynamically adjust ensemble weights based on model performance
    """
    def __init__(self, initial_vit_weight: float = 0.6, initial_transunet_weight: float = 0.4):
        self.vit_weight = initial_vit_weight
        self.transunet_weight = initial_transunet_weight
        self.vit_correct = 0
        self.transunet_correct = 0
        self.total_samples = 0

    def update(self, vit_pred: int, transunet_pred: int, true_label: int):
        """Update weights based on recent predictions"""
        if vit_pred == true_label:
            self.vit_correct += 1
        if transunet_pred == true_label:
            self.transunet_correct += 1
        self.total_samples += 1

        if self.total_samples % 100 == 0:
            vit_acc = self.vit_correct / self.total_samples
            transunet_acc = self.transunet_correct / self.total_samples
            total_acc = vit_acc + transunet_acc

            # Normalize to sum to 1
            if total_acc > 0:
                self.vit_weight = vit_acc / total_acc
                self.transunet_weight = transunet_acc / total_acc

    def get_weights(self) -> Tuple[float, float]:
        return self.vit_weight, self.transunet_weight


class EnhancedEnsembleModel(nn.Module):
    """
    Improved Ensemble Model with:
    - Lazy loading (load models only when needed)
    - Memory efficient inference
    - Confidence calibration
    - Batch processing
    - Better error handling
    """
    def __init__(
        self,
        vit_model_path: str = "models/emotion_vit_rafdb",
        vit_weight: float = 0.6,
        transunet_weight: float = 0.4,
        num_classes: int = 7,
        device: Optional[str] = None,
        use_calibration: bool = True,
        enable_caching: bool = True
    ):
        super(EnhancedEnsembleModel, self).__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.vit_model_path = vit_model_path
        self.num_classes = num_classes
        self.vit_weight = vit_weight
        self.transunet_weight = transunet_weight
        self.use_calibration = use_calibration
        self.enable_caching = enable_caching

        # Lazy loaded models
        self._vit_model = None
        self._vit_processor = None
        self._transunet_model = None

        # Calibration
        self.vit_calibrator = ConfidenceCalibrator() if use_calibration else None
        self.transunet_calibrator = ConfidenceCalibrator() if use_calibration else None

        # Weight scheduler
        self.weight_scheduler = AdaptiveWeightScheduler(vit_weight, transunet_weight)

        # Cache for model outputs
        self._output_cache = {} if enable_caching else None

    @property
    def vit_model(self):
        """Lazy load ViT model"""
        if self._vit_model is None:
            print(f"Loading ViT model from {self.vit_model_path}...")
            self._vit_model = ViTForImageClassification.from_pretrained(
                self.vit_model_path
            ).to(self.device)
            self._vit_model.eval()
        return self._vit_model

    @property
    def vit_processor(self):
        """Lazy load ViT processor"""
        if self._vit_processor is None:
            self._vit_processor = ViTImageProcessor.from_pretrained(self.vit_model_path)
        return self._vit_processor

    @property
    def transunet_model(self):
        """Lazy load TransUNet model"""
        if self._transunet_model is None:
            print("Loading TransUNet model...")
            self._transunet_model = TransUNetClassifier(
                in_channels=3,
                num_classes=self.num_classes,
                pretrained=True
            ).to(self.device)
            self._transunet_model.eval()
        return self._transunet_model

    def forward_vit(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ViT"""
        try:
            with torch.no_grad():
                outputs = self.vit_model(x)
                logits = outputs.logits

            if self.vit_calibrator:
                probs = self.vit_calibrator.calibrate(logits)
            else:
                probs = torch.softmax(logits, dim=-1)

            return probs
        except Exception as e:
            print(f"Error in ViT forward pass: {e}")
            # Return uniform distribution as fallback
            return torch.ones(x.size(0), self.num_classes).to(self.device) / self.num_classes

    def forward_transunet(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TransUNet"""
        try:
            with torch.no_grad():
                logits = self.transunet_model(x)

            if self.transunet_calibrator:
                probs = self.transunet_calibrator.calibrate(logits)
            else:
                probs = torch.softmax(logits, dim=-1)

            return probs
        except Exception as e:
            print(f"Error in TransUNet forward pass: {e}")
            # Return uniform distribution as fallback
            return torch.ones(x.size(0), self.num_classes).to(self.device) / self.num_classes

    def forward(
        self,
        x: torch.Tensor,
        method: str = 'weighted_average',
        use_adaptive_weights: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ensemble forward pass
        Args:
            x: input tensor (B, 3, 224, 224)
            method: 'weighted_average' | 'attention' | 'max_voting' | 'product'
            use_adaptive_weights: use dynamically adjusted weights
        """
        # Get individual model predictions
        vit_probs = self.forward_vit(x)
        transunet_probs = self.forward_transunet(x)

        # Get current weights
        if use_adaptive_weights:
            vit_w, transunet_w = self.weight_scheduler.get_weights()
        else:
            vit_w = self.vit_weight
            transunet_w = self.transunet_weight

        # Ensemble methods
        if method == 'weighted_average':
            ensemble_probs = vit_w * vit_probs + transunet_w * transunet_probs

        elif method == 'product':
            # Geometric mean - more selective
            ensemble_probs = torch.pow(vit_probs, vit_w) * torch.pow(transunet_probs, transunet_w)
            ensemble_probs = ensemble_probs / ensemble_probs.sum(dim=-1, keepdim=True)

        elif method == 'max_voting':
            ensemble_probs = torch.max(vit_probs, transunet_probs)

        elif method == 'harmonic_mean':
            # Better for averaging probabilities
            eps = 1e-8
            ensemble_probs = 2 / (1 / (vit_probs + eps) + 1 / (transunet_probs + eps))
            ensemble_probs = ensemble_probs / ensemble_probs.sum(dim=-1, keepdim=True)

        else:
            raise ValueError(f"Unknown method: {method}")

        return ensemble_probs, vit_probs, transunet_probs

    def update_calibration(self, logits_vit: np.ndarray, logits_transunet: np.ndarray, labels: np.ndarray):
        """Calibrate models on validation set"""
        if self.vit_calibrator:
            print("Calibrating ViT...")
            self.vit_calibrator.tune(logits_vit, labels)

        if self.transunet_calibrator:
            print("Calibrating TransUNet...")
            self.transunet_calibrator.tune(logits_transunet, labels)

    def unload_models(self):
        """Free memory by unloading models"""
        self._vit_model = None
        self._transunet_model = None
        torch.cuda.empty_cache()

    def get_model_info(self) -> Dict:
        """Get information about ensemble configuration"""
        return {
            'device': self.device,
            'vit_weight': self.vit_weight,
            'transunet_weight': self.transunet_weight,
            'num_classes': self.num_classes,
            'use_calibration': self.use_calibration,
            'enable_caching': self.enable_caching,
            'vit_loaded': self._vit_model is not None,
            'transunet_loaded': self._transunet_model is not None
        }


class BatchEnsemblePredictor:
    """
    Optimized batch processing for ensemble predictions
    Handles multiple images at once for better efficiency
    """
    def __init__(
        self,
        num_classes: int = 7,
        vit_weight: float = 0.6,
        transunet_weight: float = 0.4,
        ensemble_method: str = 'weighted_average',
        device: Optional[str] = None,
        use_calibration: bool = True,
        batch_size: int = 8
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.ensemble_model = EnhancedEnsembleModel(
            vit_weight=vit_weight,
            transunet_weight=transunet_weight,
            num_classes=num_classes,
            device=device,
            use_calibration=use_calibration
        )
        self.ensemble_method = ensemble_method
        self.batch_size = batch_size
        self.vit_processor = self.ensemble_model.vit_processor

    def predict(
        self,
        image_pil,
        return_all_scores: bool = False,
        use_adaptive_weights: bool = False
    ) -> Dict:
        """
        Predict single image
        Args:
            image_pil: PIL Image
            return_all_scores: return ViT and TransUNet scores
            use_adaptive_weights: use dynamic weight adjustment
        """
        # Preprocess
        inputs = self.vit_processor(image_pil, return_tensors="pt").to(self.device)
        image_tensor = inputs['pixel_values']

        # Forward pass
        ensemble_probs, vit_probs, transunet_probs = self.ensemble_model(
            image_tensor,
            method=self.ensemble_method,
            use_adaptive_weights=use_adaptive_weights
        )

        # Get predictions
        pred_idx = ensemble_probs.argmax(-1).item()
        ensemble_confidence = ensemble_probs[0][pred_idx].item() * 100

        result = {
            'predicted_class': pred_idx,
            'ensemble_confidence': ensemble_confidence,
            'all_probs': ensemble_probs[0].cpu().numpy()
        }

        if return_all_scores:
            vit_conf = vit_probs[0][pred_idx].item() * 100
            transunet_conf = transunet_probs[0][pred_idx].item() * 100

            result.update({
                'vit_confidence': vit_conf,
                'transunet_confidence': transunet_conf,
                'vit_all_probs': vit_probs[0].cpu().numpy(),
                'transunet_all_probs': transunet_probs[0].cpu().numpy()
            })

        return result

    def predict_batch(
        self,
        image_list: List,
        return_all_scores: bool = False
    ) -> List[Dict]:
        """
        Predict batch of images
        Args:
            image_list: list of PIL Images
            return_all_scores: return all model scores
        """
        results = []

        for i in range(0, len(image_list), self.batch_size):
            batch = image_list[i:i + self.batch_size]

            # Preprocess batch
            inputs = self.vit_processor(
                batch,
                return_tensors="pt"
            ).to(self.device)
            image_tensors = inputs['pixel_values']

            # Forward pass
            ensemble_probs, vit_probs, transunet_probs = self.ensemble_model(
                image_tensors,
                method=self.ensemble_method
            )

            # Process results
            for j in range(len(batch)):
                pred_idx = ensemble_probs[j].argmax(-1).item()
                ensemble_confidence = ensemble_probs[j][pred_idx].item() * 100

                result = {
                    'predicted_class': pred_idx,
                    'ensemble_confidence': ensemble_confidence
                }

                if return_all_scores:
                    result.update({
                        'vit_confidence': vit_probs[j][pred_idx].item() * 100,
                        'transunet_confidence': transunet_probs[j][pred_idx].item() * 100
                    })

                results.append(result)

        return results

    def get_config(self) -> Dict:
        """Get predictor configuration"""
        return {
            'device': self.device,
            'ensemble_method': self.ensemble_method,
            'batch_size': self.batch_size,
            'model_info': self.ensemble_model.get_model_info()
        }
