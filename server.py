import torch
import torch.onnx

# Load your PyTorch model
model = torch.load('sciai.pth')
model.eval()

# Dummy input (replace this with a sample input of your model)
dummy_input = torch.tensor([[1, 2, 3, 4, 5]])  # Example input tensor

# Export the PyTorch model to ONNX format
torch.onnx.export(model, dummy_input, 'model.onnx', verbose=True, input_names=['input'], output_names=['output'])

