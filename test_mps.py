import torch

print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)
