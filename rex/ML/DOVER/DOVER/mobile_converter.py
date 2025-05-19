import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

## Created with Augment Code

# Load the pretrained weights to analyze their structure
print("Loading pretrained weights for analysis...")
sd = torch.load("pretrained_weights/DOVER-Mobile.pth", map_location="cpu")

# Analyze the structure of the weights
technical_backbone_keys = [k for k in sd.keys() if k.startswith('technical_backbone')]
aesthetic_backbone_keys = [k for k in sd.keys() if k.startswith('aesthetic_backbone')]
technical_head_keys = [k for k in sd.keys() if k.startswith('technical_head')]
aesthetic_head_keys = [k for k in sd.keys() if k.startswith('aesthetic_head')]

# Define the GRN (Global Response Normalization) layer
class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

# Define the LayerNorm for 3D convolutions
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if len(x.shape) == 4:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            elif len(x.shape) == 5:
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

# Define the ConvNeXtV2 3D block
class BlockV23D(nn.Module):
    def __init__(self, dim, inflate_len=1):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=(inflate_len,7,7), padding=(inflate_len // 2,3,3), groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = input + self.drop_path(x)
        return x

# Define the ConvNeXtV2 3D model
class ConvNeXtV23D(nn.Module):
    def __init__(self, in_chans=3, depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], inflate_pattern=None):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()
        
        # Stem layer
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=(2,4,4), stride=(2,4,4)),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        # Downsample layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=(1,2,2), stride=(1,2,2)),
            )
            self.downsample_layers.append(downsample_layer)

        # Stages with custom inflate patterns
        self.stages = nn.ModuleList()
        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                inflate_len = inflate_pattern[i][j] if inflate_pattern else 1
                blocks.append(BlockV23D(dim=dims[i], inflate_len=inflate_len))
            self.stages.append(nn.Sequential(*blocks))

        # Final norm layer
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        
        # Classification head (not used in inference but needed for loading weights)
        self.head = nn.Linear(dims[-1], 1000)
        
    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x

# Define the VQA head
class VQAHead(nn.Module):
    def __init__(self, in_channels=384, hidden_channels=32):
        super().__init__()
        self.fc_hid = nn.Conv3d(in_channels, hidden_channels, kernel_size=1)
        self.fc_last = nn.Conv3d(hidden_channels, 1, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc_hid(x))
        x = self.fc_last(x)
        return x

# Define the DOVER-Mobile model
class DOVERMobile(nn.Module):
    def __init__(self):
        super().__init__()
        # Analyze the weights to determine the inflate patterns
        tech_inflate_pattern = analyze_inflate_pattern('technical_backbone')
        aes_inflate_pattern = analyze_inflate_pattern('aesthetic_backbone')
        
        # Technical backbone
        self.technical_backbone = ConvNeXtV23D(
            in_chans=3,
            depths=[2, 2, 6, 2],
            dims=[48, 96, 192, 384],
            inflate_pattern=tech_inflate_pattern
        )
        
        # Aesthetic backbone
        self.aesthetic_backbone = ConvNeXtV23D(
            in_chans=3,
            depths=[2, 2, 6, 2],
            dims=[48, 96, 192, 384],
            inflate_pattern=aes_inflate_pattern
        )
        
        # Heads
        self.technical_head = VQAHead(in_channels=384, hidden_channels=32)
        self.aesthetic_head = VQAHead(in_channels=384, hidden_channels=32)

    def forward(self, aesthetic_view, technical_view):
        self.eval()
        with torch.no_grad():
            aesthetic_feat = self.aesthetic_backbone(aesthetic_view)
            technical_feat = self.technical_backbone(technical_view)
            
            aesthetic_score = self.aesthetic_head(aesthetic_feat)
            technical_score = self.technical_head(technical_feat)
            
        aesthetic_score_pooled = torch.mean(aesthetic_score, (1,2,3,4))
        technical_score_pooled = torch.mean(technical_score, (1,2,3,4))
        return [aesthetic_score_pooled, technical_score_pooled]

# Function to analyze the inflate pattern from the weights
def analyze_inflate_pattern(prefix):
    pattern = []
    for i in range(4):  # 4 stages
        stage_pattern = []
        for j in range(depths[i]):
            key = f"{prefix}.stages.{i}.{j}.dwconv.weight"
            if key in sd:
                inflate_len = sd[key].shape[2]
                stage_pattern.append(inflate_len)
            else:
                stage_pattern.append(1)  # Default
        pattern.append(stage_pattern)
    return pattern

# Get the depths from the weights
depths = [2, 2, 6, 2]  # Default depths for DOVER-Mobile

# Create the model
model = DOVERMobile()

# Load the pretrained weights
print("Loading pretrained weights...")
# Handle the strict loading by removing the head weights
keys_to_remove = []
for key in sd.keys():
    if key.endswith("head.weight") or key.endswith("head.bias"):
        keys_to_remove.append(key)

for key in keys_to_remove:
    del sd[key]

model.load_state_dict(sd, strict=False)
print("Weights loaded successfully!")

# Create dummy inputs
print("Creating dummy inputs...")
dummy_inputs = (torch.randn(1,3,32,224,224), torch.randn(4,3,32,224,224))

# Export to ONNX
print("Exporting to ONNX...")
torch.onnx.export(
    model, 
    dummy_inputs, 
    "onnx_dover_mobile.onnx", 
    verbose=True, 
    input_names=["aes_view", "tech_view"],
    output_names=["aesthetic_score", "technical_score"],
    dynamic_axes={
        "aes_view": {0: "batch_size", 2: "frames"},
        "tech_view": {0: "batch_size", 2: "frames"}
    },
    opset_version=17
)

print("Successfully exported DOVER-Mobile model to ONNX format!")

# Copy the ONNX file to the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_dover_mobile.onnx")
dst_path = os.path.join(parent_dir, "onnx_dover_mobile.onnx")
import shutil
shutil.copy2(src_path, dst_path)
print(f"Copied ONNX file to {dst_path}")
