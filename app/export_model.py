import os
import torch
from cnn import CNN

# --- Paths ---
pth_model_path = "models/20251203_201015/cnn_best.pth"
onnx_model_path = "cnn_best_single.onnx"

# --- Delete old ONNX files if they exist ---
if os.path.exists(onnx_model_path):
    os.remove(onnx_model_path)
data_file = onnx_model_path + ".data"
if os.path.exists(data_file):
    os.remove(data_file)

# --- Load PyTorch model ---
device = "cpu"
model = CNN()
model.load_state_dict(torch.load(pth_model_path, map_location=device))
model.eval()

# --- Dummy input ---
dummy_input = torch.randn(1, 3, 224, 224)

# --- Export to ONNX (single file) ---
torch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    export_params=True,        # embed weights
    opset_version=18,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

print(f"ONNX export complete: {onnx_model_path}")
