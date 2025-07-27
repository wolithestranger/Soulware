import librosa
import librosa.display
import matplotlib.pyplot as plt

#from IPython.display import Audio  # This only works in Jupyter, optional

# Load audio
filename = "audio/pretty_angel.wav"  # Replace this with your actual file path
y, sr = librosa.load(filename)

# Print basic info
print(f"Audio loaded: {filename}")
print(f"Sample rate: {sr}")
print(f"Duration: {len(y)/sr:.2f} seconds")

# Plot waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()
