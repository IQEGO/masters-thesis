import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import math
from torch.utils.checkpoint import checkpoint
import copy


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformers."""
    def __init__(self, max_len, dim):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)  # → (1, T, D)


class LinearAttention(nn.Module):
    """Memory-efficient linear attention mechanism."""
    def __init__(self, dim, heads=4, dim_head=64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, n, _ = x.shape  # b=batch_size, n=sequence_length
        h = self.heads

        # Project to queries, keys, values
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, self.dim_head).transpose(1, 2), qkv)

        # Apply softmax to keys for normalization
        k = F.softmax(k, dim=-1)

        # Linear attention (avoid O(n²) memory usage)
        context = torch.einsum('bhnd,bhne->bhde', k, v)
        out = torch.einsum('bhnd,bhde->bhne', q, context)

        # Reshape and project back
        out = out.transpose(1, 2).reshape(b, n, h * self.dim_head)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Efficient transformer block with gradient checkpointing support."""
    def __init__(self, dim, heads=4, dim_head=64, mlp_dim=1024, dropout=0.1, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Attention mechanism
        self.attn = LinearAttention(dim, heads=heads, dim_head=dim_head)

        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def _forward_impl(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)


class BigEfficientVideoModel(nn.Module):
    """
    Efficient video model using MobileNetV2 feature extractor and memory-efficient transformer.
    Designed to handle long video sequences with gradient checkpointing for memory efficiency.
    """
    def __init__(self,
                 encoder_dim=512,
                 heads=8,
                 depth=4,
                 mlp_dim=1024,
                 output_dim=2,
                 seq_len=100,
                 pos_encoding_type='learnable',  # 'sinusoidal', 'learnable', 'none'
                 dropout=0.1,
                 use_checkpointing=True):
        super().__init__()

        if encoder_dim % heads != 0:
            raise ValueError(f"encoder_dim ({encoder_dim}) must be divisible by heads ({heads})")

        self.encoder_dim = encoder_dim
        self.pos_encoding_type = pos_encoding_type
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

        # Positional encoding
        if pos_encoding_type == 'sinusoidal':
            self.pos_encoder = SinusoidalPositionalEncoding(seq_len, encoder_dim)
        elif pos_encoding_type == 'learnable':
            self.pos_embed = nn.Parameter(torch.zeros(seq_len, encoder_dim))
            nn.init.normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=encoder_dim,
                heads=heads,
                dim_head=encoder_dim // heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                use_checkpoint=use_checkpointing
            ) for _ in range(depth)
        ])

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

        # Apply positional encoding
        if self.pos_encoding_type == 'learnable':
            features = features + self.pos_embed[:T].unsqueeze(0)
        elif self.pos_encoding_type == 'sinusoidal':
            features = self.pos_encoder(features)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            features = block(features)

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
