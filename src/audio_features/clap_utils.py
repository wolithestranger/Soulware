import os


# ✅ Set HuggingFace + Torch cache paths BEFORE importing model
os.environ["TORCH_HOME"] = "D:/soulware_models/clap/torch_cache"
os.environ["HF_HOME"] = "D:/soulware_models/clap/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/soulware_models/clap/transformers_cache"
os.environ["HF_DATASETS_CACHE"] = "D:/soulware_models/clap/datasets_cache"

from msclap import CLAP


# ✅ Load the CLAP model once at module level
try:
    clap_model = CLAP(version="2023", use_cuda=False)
    print("✅ MSCLAP model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading CLAP model: {e}")
    clap_model = None

def get_clap_embedding(audio_path: str):
    """
    Returns the CLAP embedding vector (1D) for the given audio path.
    Returns None on failure.
    """
    if clap_model is None:
        print("❌ CLAP model not loaded.")
        return None
    try:
        embedding = clap_model.get_audio_embeddings([audio_path])
        return embedding[0]  # return 1D array
    except Exception as e:
        print(f"❌ Failed to get embedding for {audio_path}: {e}")
        return None