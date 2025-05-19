import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import math
import copy


class SinusoidalPositionalEncoding(nn.Module):
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


class BigFlexibleVideoModel(nn.Module):
    def __init__(self,
                 encoder_dim=768,
                 heads=4,
                 mlp_hidden_dims=[256, 64],
                 output_dim=2,
                 seq_len=16,
                 pos_encoding_type='learnable'  # 'sinusoidal', 'learnable', 'none'
                 ):
        super().__init__()

        if encoder_dim % heads != 0:
            raise ValueError(f"encoder_dim ({encoder_dim}) must be divisible by nhead ({heads})")


        # Extraktor příznaků
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.feature_extractor = mobilenet.features
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Získání dimenze výstupu MobileNetu
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            dummy_out = self.feature_extractor(dummy)
        self.feature_dim = dummy_out.shape[1]

        # Projekce do encoder_dim
        self.proj = nn.Linear(self.feature_dim, encoder_dim)

        # Positional Encoding
        self.pos_encoding_type = pos_encoding_type
        if pos_encoding_type == 'learnable':
            self.pos_embed = nn.Parameter(torch.randn(seq_len, encoder_dim))
        elif pos_encoding_type == 'sinusoidal':
            self.pos_encoder = SinusoidalPositionalEncoding(seq_len, encoder_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=encoder_dim, nhead=heads, batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # MLP hlava
        mlp_layers = []
        dims = [encoder_dim] + mlp_hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(dims[-1], output_dim))
        mlp_layers.append(nn.Sigmoid()) # predictions will be 0-1
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        with torch.no_grad():
            features = self.feature_extractor(x)
            features = features.mean(dim=[2, 3])  # → (B*T, F)

        features = self.proj(features)            # → (B*T, encoder_dim)
        features = features.view(B, T, -1)        # → (B, T, D)

        # Positional encoding – pozor, tvar (B, T, D)
        if self.pos_encoding_type == 'learnable':
            features = features + self.pos_embed[:T].unsqueeze(0)  # (1, T, D) broadcast přes batch
        elif self.pos_encoding_type == 'sinusoidal':
            features = self.pos_encoder(features)  # Sinusová vrstva očekává (B, T, D) ve verzi s batch_first

        encoded = self.encoder(features)          # → (B, T, D)
        seq_repr = encoded[:, -1, :]              # Poslední časový krok pro každou sekvenci

        return self.mlp(seq_repr)
    
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