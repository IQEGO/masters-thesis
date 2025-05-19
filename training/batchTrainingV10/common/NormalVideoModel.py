import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.checkpoint import checkpoint
import copy


class NormalVideoModel(nn.Module):
    """
    Simplified video model using MobileNetV2 feature extractor followed by temporal convolutions
    and MLP for quality assessment. No transformer blocks are used.
    """
    def __init__(self,
                 encoder_dim=512,
                 mlp_dim=1024,
                 output_dim=2,
                 dropout=0.2,
                 use_checkpointing=True):
        super().__init__()

        self.use_checkpointing = use_checkpointing

        # Feature extractor - MobileNetV2
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.feature_extractor = mobilenet.features
        self.feature_extractor.eval()  # Keep BatchNorm in eval mode

        # Freeze feature extractor parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Get feature dimension from MobileNetV2
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            dummy_out = self.feature_extractor(dummy)
        self.feature_dim = dummy_out.shape[1]

        # Projection from feature space to encoder dimension
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Temporal processing with 1D convolutions
        # This replaces the transformer blocks with simpler temporal convolutions
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final layer normalization
        self.norm = nn.LayerNorm(encoder_dim)

        # MLP head for regression
        self.mlp_head = nn.Sequential(
            nn.Linear(encoder_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, mlp_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 4, output_dim),
            nn.Sigmoid()  # Normalized outputs between 0-1
        )

    def _extract_features(self, x):
        """Extract features using MobileNetV2 with optional checkpointing"""
        if self.use_checkpointing and self.training:
            return checkpoint(self.feature_extractor, x, use_reentrant=False)
        else:
            with torch.no_grad():  # Feature extractor is frozen
                return self.feature_extractor(x)

    def forward(self, x):
        B, T, C, H, W = x.shape

        # Reshape for feature extraction
        x = x.view(B * T, C, H, W)

        # Extract features using MobileNetV2
        features = self._extract_features(x)
        features = features.mean(dim=[2, 3])  # Global average pooling → (B*T, F)

        # Project to encoder dimension
        features = self.proj(features)  # → (B*T, encoder_dim)
        features = features.view(B, T, -1)  # → (B, T, encoder_dim)

        # Apply temporal convolutions
        # Reshape for 1D convolution (B, C, T)
        features = features.transpose(1, 2)  # → (B, encoder_dim, T)
        features = self.temporal_conv(features)  # → (B, encoder_dim, T)
        features = features.transpose(1, 2)  # → (B, T, encoder_dim)

        # Apply final normalization
        features = self.norm(features)

        # Global pooling over sequence dimension (mean pooling)
        pooled = features.mean(dim=1)  # → (B, encoder_dim)

        # Apply MLP head
        output = self.mlp_head(pooled)  # → (B, output_dim)

        return output

    def export_to_onnx(self, path, seq_length=10):
        """Export model to ONNX format as copy (without changing original instance)"""
        # Create a copy of the model for export
        model_copy = copy.deepcopy(self)
        model_copy = model_copy.cpu().float()  # Prepare copy for ONNX export - move model to CPU and convert to float32 before export

        # Create dummy input on CPU
        dummy_input = torch.randn(1, seq_length, 3, 224, 224)
        torch.onnx.export(
            model_copy,
            dummy_input,
            path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"Model exported to ONNX format at {path}")
