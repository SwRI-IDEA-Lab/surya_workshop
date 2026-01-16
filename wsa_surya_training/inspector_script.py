import torch

# Path to your file
path = "./assets/surya.366m.v1.pt"

try:
    checkpoint = torch.load(path, map_location="cpu")
    print(f"\n📁 Checkpoint loaded. Type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"🔑 Top-level keys: {list(checkpoint.keys())[:5]}") # Print first 5 keys
        if "state_dict" in checkpoint:
            print("   -> Found 'state_dict' key.")
        if "model" in checkpoint:
            print("   -> Found 'model' key.")
    else:
        print("   -> This file is not a dictionary. It might be a raw model object.")

except Exception as e:
    print(f"❌ Could not load file: {e}")