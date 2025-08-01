import librosa
import librosa.display
import matplotlib.pyplot as plt

from music21 import analysis, key, pitch, stream, note
import numpy as np

from generate_feedback import generate_feedback
from audio_features.clap_utils import get_clap_embedding


#from IPython.display import Audio  # This only works in Jupyter, optional

# Load audio
filename = "C:/Users/nimbeoviosud/Soulware/audio/kid_a_radiohead.mp3"
#"C:/Users/nimbeoviosud/Soulware/audio/pretty_angel.wav"
y, sr = librosa.load(filename)

clap_embedding = get_clap_embedding(filename)

if clap_embedding is not None:
    print("🎧 CLAP embedding preview:", clap_embedding[:5])


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


feedback = generate_feedback(
    key=key_detected.tonic.name,
    mode=key_detected.mode,
    tempo=tempo,
    chords=[],#["F#m", "C#m", "G#m", "C#m"],
    instruments=[],#["classical guitar", "bass", "congas", "bongos", "drums", "saxophone", "light Rhodes"],
    texture_desc=[],#"Dark, sultry, mid-tempo Afrobeat groove",
    clap_vector=clap_embedding
)

print("🎤 AI Feedback:\n", feedback)
