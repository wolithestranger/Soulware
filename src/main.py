# main.py
import os
import json
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from generate_feedback import generate_feedback
from audio_features.clap_utils import get_clap_embedding
from audio_features.feature_extractor import extract_features
from music21 import stream, note

# ---------- Config ----------
DEV_PLOTS = True            # Set False for headless/fast runs
ANALYZE_SECONDS = None      # e.g., 90 to analyze only first 90 seconds
SAVE_FEATURES = True        # Save features JSON next to the script
# ----------------------------

# Load audio (keep native SR; mono=True for consistent feature extraction)
filename = "C:/Users/nimbeoviosud/Soulware/audio/kid_a_radiohead.mp3"
y, sr = librosa.load(filename, sr=None, mono=True, duration=ANALYZE_SECONDS)

# ---- CLAP embedding ----
clap_embedding = get_clap_embedding(filename)
if clap_embedding is not None:
    print("🎧 CLAP embedding preview:", clap_embedding[:5])

# ---- Engineered features (single source of truth) ----
features_json, _ = extract_features(y, sr)

# ---- music21 key (second opinion) ----
# Build a pseudo "pitch-strength" stream from chroma to analyze with Krumhansl
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
chroma_mean = chroma_stft.mean(axis=1)
note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

notes_in_song = []
for i, strength in enumerate(chroma_mean):
    n = note.Note(note_names[i])
    n.volume.velocity = float(strength * 127.0)
    notes_in_song.append(n)

melody_stream = stream.Stream(notes_in_song)
m21_key = melody_stream.analyze('Krumhansl')

# Stash the second opinion inside the JSON so downstream sees both
features_json["key_music21"] = f"{m21_key.tonic.name} {m21_key.mode}"
features_json["key_music21_method"] = "music21.Krumhansl"

# Quick console snapshot
print("🧪 Feature snapshot:", {
    "tempo_bpm": features_json["tempo_bpm"],
    "key_librosa": features_json["key_guess"],
    "key_confidence": features_json["key_confidence"],
    "key_music21": features_json["key_music21"],
    "hp_energy_ratio": features_json["hp_energy_ratio"],
})

# (Optional) persist features for reuse/debug/datasets
if SAVE_FEATURES:
    out_json = os.path.splitext(os.path.basename(filename))[0] + "_features.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(features_json, f, indent=2, ensure_ascii=False)
    print(f"💾 Features saved to {out_json}")

# ---- Dev-only visualizations ----
if DEV_PLOTS:
    # Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB'); plt.title('Mel-frequency Spectrogram')
    plt.tight_layout(); plt.show()

    # Waveform
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform"); plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
    plt.tight_layout(); plt.show()

# ---- Derive key/mode/tempo for the LLM call (prefer librosa; fallback to music21) ----
librosa_key = features_json.get("key_guess", "Unknown")
key_parts = librosa_key.split()
m21_key_str = features_json.get("key_music21", "Unknown")
m21_parts = m21_key_str.split()

key_name = key_parts[0] if len(key_parts) > 0 else (m21_parts[0] if len(m21_parts) > 0 else "Unknown")
mode_name = key_parts[1] if len(key_parts) > 1 else (m21_parts[1] if len(m21_parts) > 1 else "Unknown")
tempo_bpm = features_json["tempo_bpm"]

# ---- Call LLM ----
feedback = generate_feedback(
    clap_vector=clap_embedding,
    features_json=features_json,
    key=key_name,
    mode=mode_name,
    tempo=tempo_bpm,
    chords=[],#["F#m", "C#m", "G#m", "C#m"],
    instruments=[],#["classical guitar", "bass", "congas", "bongos", "drums", "saxophone", "light Rhodes"],
    texture_desc=[]#"Dark, sultry, mid-tempo Afrobeat groove",
)

print("🎤 AI Feedback:\n", feedback)
