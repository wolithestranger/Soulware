####extracts features to into a json to feed into the LLM

# audio_features/feature_extractor.py
from __future__ import annotations
import numpy as np
import librosa

def _mean_std(x: np.ndarray) -> tuple[float, float]:
    # Works for 1D or 2D arrays; we aggregate across time if needed
    if x.ndim == 1:
        return float(np.mean(x)), float(np.std(x))
    # assume shape=(features, frames) -> aggregate per-feature then average
    mean_per_feat = np.mean(x, axis=1)
    std_per_feat  = np.std(x, axis=1)
    return float(np.mean(mean_per_feat)), float(np.mean(std_per_feat))

def _onset_rate(y: np.ndarray, sr: int) -> float:
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    if len(onsets) < 2:
        return 0.0
    duration = len(y) / sr
    return float(len(onsets) / max(duration, 1e-6))

def _tempogram_peak(y: np.ndarray, sr: int) -> float:
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)
    # Convert the peak lag to BPM
    ac_global = librosa.autocorrelate(oenv, max_size=oenv.shape[0])
    tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, aggregate=None)
    if tempo is None or len(tempo) == 0:
        return 0.0
    # Ensure we return a scalar float
    result = np.median(tempo)
    # Handle both scalar and array cases
    if np.isscalar(result):
        return float(result)
    else:
        return float(result.item())

def _key_from_chroma_template(chroma_mean: np.ndarray) -> tuple[str, float]:
    """
    Simple Krumhansl-like template correlation on 12-dim chroma mean.
    Returns (key_string, confidence [0..1]) where key_string like 'F# minor'.
    """
    # Major/minor profiles (Krumhansl-Kessler variant)
    maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                    2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
    minp = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)
    note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

    c = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-8)
    best_key = None
    best_score = -1.0
    best_mode = "major"

    for i in range(12):
        maj_rot = np.roll(maj, i); maj_rot = maj_rot / (np.linalg.norm(maj_rot) + 1e-8)
        min_rot = np.roll(minp, i); min_rot = min_rot / (np.linalg.norm(min_rot) + 1e-8)
        s_maj = float(np.dot(c, maj_rot))
        s_min = float(np.dot(c, min_rot))
        if s_maj > best_score:
            best_score = s_maj; best_key = f"{note_names[i]}"; best_mode = "major"
        if s_min > best_score:
            best_score = s_min; best_key = f"{note_names[i]}"; best_mode = "minor"

    # crude confidence: rescale [-1,1] to [0,1]
    conf = (best_score + 1.0) / 2.0
    return f"{best_key} {best_mode}", float(conf)

def extract_features(y: np.ndarray, sr: int) -> tuple[dict, np.ndarray]:
    """
    Returns (features_json, feature_vector)
    features_json is compact + human-readable for the LLM.
    feature_vector is an optional numeric vector for a simple learned head later.
    """
    # Tempo & rhythm
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    onset_rate = _onset_rate(y, sr)
    tempogram_bpm = _tempogram_peak(y, sr)

    # Chroma (CQT is more harmony-friendly than STFT chroma)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma_cqt, axis=1)
    chroma_std  = np.std(chroma_cqt, axis=1)
    key_guess, key_conf = _key_from_chroma_template(chroma_mean)

    # MFCCs (mean/std) + deltas (mean/std)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    # To this:
    mfcc_mean = np.mean(mfcc, axis=1)  # Keep as numpy array
    mfcc_std  = np.std(mfcc, axis=1)   # Keep as numpy array  
    dmfcc_mean = np.mean(mfcc_delta, axis=1)  # Keep as numpy array
    dmfcc_std  = np.std(mfcc_delta, axis=1)  

    # Spectral stats
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    spec_bw       = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rms           = librosa.feature.rms(y=y)
    zcr           = librosa.feature.zero_crossing_rate(y)

    centroid_mean, centroid_std = _mean_std(spec_centroid)
    roll_mean, roll_std         = _mean_std(spec_rolloff)
    bw_mean, bw_std             = _mean_std(spec_bw)
    rms_mean, rms_std           = _mean_std(rms)
    zcr_mean, zcr_std           = _mean_std(zcr)

    # Harmonic-percussive ratio
    y_h, y_p = librosa.effects.hpss(y)
    hp_energy_ratio = float(np.sum(np.abs(y_h)) / (np.sum(np.abs(y_p)) + 1e-8))

    features_json = {
        "tempo_bpm": float(tempo),
        "onset_rate_per_s": float(onset_rate),
        "tempogram_peak_bpm": float(tempogram_bpm),
        "key_guess": key_guess,
        "key_confidence": float(key_conf),
        "chroma_cqt_mean": np.round(chroma_mean, 5).tolist(),
        "chroma_cqt_std":  np.round(chroma_std, 5).tolist(),
        # Convert to lists HERE with .tolist()
        "mfcc_mean": np.round(mfcc_mean, 5).tolist(),        # ← Add .tolist()
        "mfcc_std":  np.round(mfcc_std, 5).tolist(),         # ← Add .tolist()
        "delta_mfcc_mean": np.round(dmfcc_mean, 5).tolist(), # ← Add .tolist()
        "delta_mfcc_std":  np.round(dmfcc_std, 5).tolist(),  # ← Add .tolist()
        "spectral_centroid_mean_hz": float(centroid_mean),
        "spectral_centroid_std_hz":  float(centroid_std),
        "rolloff_85_mean_hz": float(roll_mean),
        "rolloff_85_std_hz":  float(roll_std),
        "bandwidth_mean_hz": float(bw_mean),
        "bandwidth_std_hz":  float(bw_std),
        "rms_mean": float(rms_mean),
        "rms_std":  float(rms_std),
        "zcr_mean": float(zcr_mean),
        "zcr_std":  float(zcr_std),
        "hp_energy_ratio": float(hp_energy_ratio),
        "sample_rate": int(sr),
        "duration_s": float(len(y)/sr),
    }

    # Numeric vector if you want it later (ordered, stable)
    feature_vec = np.concatenate([
    np.array([
    float(np.asarray(tempo).item()),
    float(np.asarray(onset_rate).item()),
    float(np.asarray(tempogram_bpm).item())
    ], dtype=np.float32),
    np.asarray(chroma_mean, dtype=np.float32).ravel(),
    np.asarray(chroma_std,  dtype=np.float32).ravel(),
    np.asarray(mfcc_mean,   dtype=np.float32).ravel(),
    np.asarray(mfcc_std,    dtype=np.float32).ravel(),
    np.asarray(dmfcc_mean,  dtype=np.float32).ravel(),
    np.asarray(dmfcc_std,   dtype=np.float32).ravel(),
    np.array([
        centroid_mean, centroid_std,
        roll_mean, roll_std,
        bw_mean, bw_std,
        rms_mean, rms_std,
        zcr_mean, zcr_std,
        hp_energy_ratio
    ], dtype=np.float32)
], axis=0)

    print("feature_vec shape:", feature_vec.shape, "dtype:", feature_vec.dtype)


    return features_json, feature_vec



