import torch
from pathlib import Path

print("Torch Version:", torch.__version__)

model_path = Path("model/best_hybrid_model.pth")

try:
    checkpoint = torch.load(model_path, map_location="cpu")
    print("Model loaded successfully!")
    print(type(checkpoint))
except Exception as e:
    print("Error:", e)