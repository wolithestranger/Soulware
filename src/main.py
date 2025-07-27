import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

#from IPython.display import Audio  # This only works in Jupyter, optional

# Load audio
filename = "audio/pretty_angel.wav"  # Replace this with your actual file path
y, sr = librosa.load(filename)

#y is te audio time series (as a NumPy array)
#sr is the sampling rate (usually 22050 by default)

#Generate Mel Spectogram
S = librosa.feature.melspectrogram(y=y, sr = sr, n_mels = 128, fmax = 8000)
#n_mels = 128 ---number of Mel bands
#fmax = 8000 --- Upper frequency bound (can increase if song has higher frequencies)

#Convert to decibels for easier visualization
S_dB = librosa.power_to_db(S, ref = np.max)


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


#Plot Spectogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis= 'mel', fmax=8000)
plt.colorbar(format ='%+2.0f dB')
plt.title('Mel-frequency Spectogram')
plt.tight_layout()
plt.show()