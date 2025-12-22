"""
TransUNet model with ResNet-50 backbone for segmentation tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet50Backbone(nn.Module):
    """ResNet-50 backbone for feature extraction"""
    def __init__(self, pretrained=True):
        super(ResNet50Backbone, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layers
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer2 = resnet.layer1  # 256 channels
        self.layer3 = resnet.layer2  # 512 channels
        self.layer4 = resnet.layer3  # 1024 channels
        self.layer5 = resnet.layer4  # 2048 channels

    def forward(self, x):
        x1 = self.layer1(x)  # 64, 56, 56
        x2 = self.layer2(x1)  # 256, 56, 56
        x3 = self.layer3(x2)  # 512, 28, 28
        x4 = self.layer4(x3)  # 1024, 14, 14
        x5 = self.layer5(x4)  # 2048, 7, 7
        
        return x1, x2, x3, x4, x5


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: (B, N, C) or (B, C, H, W)
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        else:
            B, N, C = x.shape
            H = W = int(N ** 0.5)
        
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x.transpose(1, 2).view(B, C, H, W)


class DecoderBlock(nn.Module):
    """Decoder block with skip connection"""
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        # 🔥 resize x về đúng spatial size của skip
        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode="bilinear",
            align_corners=True
        )

        x = torch.cat([x, skip], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        return x


class TransUNetClassifier(nn.Module):
    """
    TransUNet adapted for image classification instead of segmentation
    Uses global average pooling + MLP head
    """
    def __init__(self, in_channels=3, num_classes=7, pretrained=True, num_heads=16):
        super(TransUNetClassifier, self).__init__()
        
        # ResNet-50 Encoder
        self.backbone = ResNet50Backbone(pretrained=pretrained)
        
        # Transformer blocks
        # FIXED: Changed default to use the num_heads argument (default=16)
        # 2048 is divisible by 16 (2048 / 16 = 128), but NOT by 12.
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(2048, num_heads=num_heads, dropout=0.1)
            for _ in range(4)
        ])
        
        # Decoder blocks
        self.decoder4 = DecoderBlock(2048 + 1024, 1024)
        self.decoder3 = DecoderBlock(1024 + 512, 512)
        self.decoder2 = DecoderBlock(512 + 256, 256)
        self.decoder1 = DecoderBlock(256 + 64, 64)
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Encoder - ResNet50 backbone
        x1, x2, x3, x4, x5 = self.backbone(x)
        # x1: (B, 64, 56, 56)
        # x2: (B, 256, 56, 56)
        # x3: (B, 512, 28, 28)
        # x4: (B, 1024, 14, 14)
        # x5: (B, 2048, 7, 7)
        
        # Transformer blocks
        transformer_out = x5
        for transformer_block in self.transformer_blocks:
            transformer_out = transformer_block(transformer_out)
        # transformer_out: (B, 2048, 7, 7)
        
        # Decoder with skip connections
        # decoder4: upsample 7x7 -> 14x14, concat with x4 (14x14)
        d4 = self.decoder4(transformer_out, x4)  # 7→14
        d3 = self.decoder3(d4, x3)               # 14→28
        d2 = self.decoder2(d3, x2)               # 28→56
        d1 = self.decoder1(d2, x1)               # 56→56 (KHÔNG 112)

        
        # Global pooling
        pooled = self.global_pool(d1)  # (B, 64, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, 64)
        
        # Classification
        out = self.dropout(pooled)
        out = self.classifier(out)  # (B, num_classes)
        
        return out



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Now it defaults to 16 heads, which works with 2048 channels
    model = TransUNetClassifier(in_channels=3, num_classes=7, num_heads=16).to(device)
    
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}") 
    # Should print: torch.Size([1, 7])
