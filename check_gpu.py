import torch

print("🧠 Checking GPU availability...\n")

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("✅ GPU detected:", torch.cuda.get_device_name(0))
else:
    print("❌ No GPU detected. Whisper will run on CPU (still works, just slower).")
