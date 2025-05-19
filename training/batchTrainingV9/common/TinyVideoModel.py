import torch
import torch.nn as nn
import copy

class TinyVideoModel(nn.Module):
    """
    A simple baseline model for video quality assessment.
    Uses a small convolutional network followed by fully connected layers.
    """
    def __init__(self, output_dim = 2, sequence_length=10):
        super().__init__()
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        
        # Simple convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # (B*T, 3, 224, 224) → (B*T, 16, 112, 112)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # → (B*T, 32, 56, 56)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # → (B*T, 64, 28, 28)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # → (B*T, 64, 1, 1)
        )
        
        # Fully connected layers for regression
        self.fc = nn.Sequential(
            nn.Linear(64 * sequence_length, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim),
            nn.Sigmoid()  # Normalized outputs between 0-1
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        
        # Process each frame with the CNN
        x = x.view(B * T, C, H, W)
        x = self.conv(x).view(B, T, -1)
        
        # Flatten and pass through FC layers
        x = x.view(B, -1)
        return self.fc(x)
    
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
