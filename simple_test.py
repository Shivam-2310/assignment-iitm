print("Starting test")
print("Importing torch")
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

print("Importing sentence_transformers")
from sentence_transformers import SentenceTransformer
print("SentenceTransformer imported successfully")

print("All imports successful!")
