import torch
from Palma_model import MultiLayerConvLSTM
import onnx

# Load the model
model = MultiLayerConvLSTM(use_sigmoid=True)
model.load_state_dict(torch.load("/path/to/model.pth", map_location="cpu"))
model.eval()

# Create a sample input (batch size 1, 18 timesteps, 1 channel, 344x315)
dummy_input = torch.randn(1, 18, 1, 344, 315)

# Export to ONNX
torch.onnx.export(
    model,                             # Model to export
    dummy_input,                       # Example input
    "/path/to/output_model.onnx",      # Output ONNX file path
    export_params=True,                 # Export all parameters
    opset_version=11,                   # ONNX opset version
    do_constant_folding=True,           # Optimize constant expressions
    input_names=['input'],              # Input name
    output_names=['output'],            # Output name
    dynamic_axes={
        'input': {0: 'batch_size'},     # Only the batch dimension is dynamic
        'output': {0: 'batch_size'}
    }
)

print("ONNX export completed.")
