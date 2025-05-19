import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import math
from torch.utils.checkpoint import checkpoint
import copy

class PretrainedVisionTransformer(nn.Module):
    """
    Video quality assessment model using a pre-trained Vision Transformer (ViT).
    
    This model:
    1. Uses MobileNetV2 to extract features from each frame
    2. Leverages a pre-trained ViT for temporal understanding
    3. Adds a regression head for quality prediction
    
    The pre-trained components are frozen initially and can be fine-tuned
    with a small learning rate if needed.
    """
    def __init__(self,
                 output_dim=2,
                 seq_len=100,
                 finetune_vit=False,
                 dropout=0.1):
        super().__init__()

        raise NotImplementedError("This model does not currently work and need more research.")
        
        self.seq_len = seq_len
        
        # Load pre-trained MobileNetV2 for frame feature extraction
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
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
        
        # Load pre-trained ViT
        vit_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        
        # Extract encoder from ViT (without classification head)
        self.vit_encoder = vit_model.encoder
        self.vit_hidden_dim = vit_model.hidden_dim  # Usually 768
        
        # Freeze ViT parameters if not fine-tuning
        if not finetune_vit:
            for param in self.vit_encoder.parameters():
                param.requires_grad = False
        
        # Projection from MobileNet features to ViT input dimension
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.vit_hidden_dim),
            nn.LayerNorm(self.vit_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Class token for ViT (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.vit_hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Positional embedding for sequence
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, self.vit_hidden_dim))  # +1 for cls token
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Final layer normalization
        self.norm = nn.LayerNorm(self.vit_hidden_dim)
        
        # MLP head for regression
        self.mlp_head = nn.Sequential(
            nn.Linear(self.vit_hidden_dim, self.vit_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.vit_hidden_dim // 2, self.vit_hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.vit_hidden_dim // 4, output_dim),
            nn.Sigmoid()  # Normalized outputs between 0-1
        )
    
    def _extract_features(self, x):
        """Extract features using MobileNetV2"""
        with torch.no_grad():  # Feature extractor is frozen
            features = self.feature_extractor(x)
            features = features.mean(dim=[2, 3])  # Global average pooling → (B*T, F)
        return features
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        
        # Reshape for feature extraction
        x = x.view(B * T, C, H, W)
        
        # Extract features using MobileNetV2
        features = self._extract_features(x)  # → (B*T, F)
        
        # Project to ViT dimension
        features = self.proj(features)  # → (B*T, D)
        features = features.view(B, T, -1)  # → (B, T, D)
        
        # Use only the actual sequence length
        features = features[:, :T, :]  # → (B, T_actual, D)
        
        # Prepare input for ViT with class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # → (B, 1, D)
        x = torch.cat((cls_tokens, features), dim=1)  # → (B, T_actual+1, D)
        
        # Add positional embedding
        x = x + self.pos_embed[:, :T+1]  # → (B, T_actual+1, D)
        
        # ViT expects input in shape (B, T, D)
        encoded = self.vit_encoder(x)  # → (B, T_actual+1, D)
        
        # Use the class token output for classification
        cls_output = encoded[:, 0]  # → (B, D)
        
        # Apply final normalization
        cls_output = self.norm(cls_output)  # → (B, D)
        
        # Apply MLP head
        output = self.mlp_head(cls_output)  # → (B, output_dim)
        
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
