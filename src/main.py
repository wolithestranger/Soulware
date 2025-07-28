import librosa
import librosa.display
import matplotlib.pyplot as plt

from music21 import analysis, key, pitch, stream, note
import numpy as np

import numpy as np

#from IPython.display import Audio  # This only works in Jupyter, optional

# Load audio
filename = "audio/pretty_angel.wav"  # Replace this with your actual file path
y, sr = librosa.load(filename)

#y is te audio time series (as a NumPy array)
#sr is the sampling rate (usually 22050 by default)

# === Mel Spectogram ===
#Generate Mel Spectogram
S = librosa.feature.melspectrogram(y=y, sr = sr, n_mels = 128, fmax = 8000)
#n_mels = 128 ---number of Mel bands
#fmax = 8000 --- Upper frequency bound (can increase if song has higher frequencies)

#Convert to decibels for easier visualization
S_dB = librosa.power_to_db(S, ref = np.max)

#Plot Spectogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis= 'mel', fmax=8000)
plt.colorbar(format ='%+2.0f dB')
plt.title('Mel-frequency Spectogram')
plt.tight_layout()
plt.show()

#-------------
# === Tempo and Beat Tracking ===
tempo, beats =  librosa.beat.beat_track(y=y, sr=sr)
print(f"Estimated tempo: {tempo.item(): .2f} BPM")

# === MFCCs (Mel-Frequency Cepstral Coefficients) ===
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
print("MFCCs Shape:", mfccs.shape)

# === Plot MFCCs ===
plt.figure(figsize=(10,4))
librosa.display.specshow(mfccs, x_axis='time', sr=sr)
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

#----------------

# === Chroma Features( Pitch Class Energy) ===
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, y_axis='chroma', x_axis = 'time', sr=sr)
plt.colorbar()
plt.title('Chroma Feature')
plt.tight_layout()
plt.show()


# === Key Estimation - Krumhansl-Schmuckler Key-Finding Algorithm ===

chroma_mean = chroma.mean(axis=1)
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 
              'F#', 'G', 'G#', 'A', 'A#', 'B']

notes_in_song = []

for i, strength in enumerate(chroma_mean):
    n = note.Note(note_names[i])
    n.volume.velocity = strength * 127 #--- scale to MIDI velocity
    notes_in_song.append(n)

melody_stream = stream.Stream(notes_in_song)

#Run key analysis
key_detected = melody_stream.analyze('Krumhansl')

print("🎯 Detected key:", key_detected.tonic.name)
print("🎯 Mode:", key_detected.mode)



#print("🎯 Detected key:", key_detected)


#---------------------------------------------------



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


