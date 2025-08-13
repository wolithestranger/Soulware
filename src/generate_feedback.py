import os
from openai import OpenAI
from dotenv import load_dotenv

import json
from typing import Any, Dict, List, Optional

# Load .env
load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

def _shorten_vector_preview(vec: Optional[List[float]], k: int = 10) -> str:
    if vec is None or len(vec) == 0:  # Changed this line
        return ""
    return ", ".join(f"{x:.4f}" for x in vec[:k])

def _compact_features_json(features_json: Optional[Dict[str, Any]]) -> str:
    """
    Keep only fields that help the model reason, and round numbers so the prompt stays compact.
    """
    if not features_json:
        return "{}"
    
    keep_keys = [
        "tempo_bpm", "onset_rate_per_s", "tempogram_peak_bpm",
        "key_guess", "key_confidence",
        "key_music21", "key_music21_method",
        "chroma_cqt_mean", "chroma_cqt_std",
        "mfcc_mean", "mfcc_std", "delta_mfcc_mean", "delta_mfcc_std",
        "spectral_centroid_mean_hz", "spectral_centroid_std_hz",
        "rolloff_85_mean_hz", "rolloff_85_std_hz",
        "bandwidth_mean_hz", "bandwidth_std_hz",
        "rms_mean", "rms_std", "zcr_mean", "zcr_std",
        "hp_energy_ratio", "duration_s", "sample_rate" 
    ]
    slim = {k: features_json[k] for k in keep_keys if k in features_json}
    # Light rounding for readability
    def _round (v):
        if isinstance(v, float):
            return round(v, 5)
        if isinstance(v, list):
            return [round(x, 5) if isinstance(x, (int, float)) else x for x in v]
        return v
    slim = {k: _round(v) for k, v in slim.items()}
    return json.dumps(slim, ensure_ascii=False)


def generate_feedback(

    key: Optional[str],
    mode: Optional[str],
    tempo: Optional[float],
    chords: Optional[List[str]],
    instruments: Optional[List[str]],
    texture_desc: Optional[List[str]],
    clap_vector: Optional[List[float]] = None,
    features_json: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    temperature: float = 0.85,
    max_tokens: int = 500,
    ) -> str:
    """
    Backwards-compatible. New: pass features_json for smarter analysis.
    """

    chords = chords or []
    instruments = instruments or []
    texture_desc = texture_desc

    clap_preview = _shorten_vector_preview(clap_vector, k=10)
    feat_str =_compact_features_json(features_json)

    # Derive safe key/mode if caller passed None
    key_safe = key or (features_json.get("key_guess", "").split()[0] if features_json else None) or "Unknown"
    mode_safe = mode or (features_json.get("key_guess", "").split()[1] if features_json and "key_guess" in features_json and len(features_json["key_guess"].split()) > 1 else None) or "Unknown"
    tempo_safe = tempo if tempo is not None else (features_json.get("tempo_bpm") if features_json else None)


    
    # # Optional: summarize CLAP vector for prompt context
    # clap_desc = ""
    # if clap_vector is not None:
    #     vector_preview = ", ".join([f"{x:.4f}" for x in clap_vector[:10]])  # First 10 values
    #     #clap_desc = f"\nCLAP audio embedding preview (partial): {vector_preview}"
    #     clap_desc = vector_preview 

    sys_msg = (
        "You are a world-class music producer known for your ability to deconstruct any track by ear and instantly identify its genre, instrumentation, mood, and influences — even from just a few seconds of audio. You’ve worked across every style from Afrobeat and trip-hop to orchestral film scores and underground electronic scenes."
        "Use engineered features for key/tempo/timbre; use CLAP cues for mood/genre/texture. "
        "If there is conflict, prefer features_json for key and tempo, prefer CLAP cues for mood/genre. "
        "Two key detectors may be present: (a) librosa template with confidence, (b) music21 Krumhansl. "
        "If they agree, state the agreement. If they disagree, pick the more plausible key and justify "
        "briefly using chroma patterns and musical context. Be concise but specific. Provide actionable tips."
    )

    user_msg = f"""
Known info:
- key: {key_safe}
- mode: {mode_safe}
- tempo_bpm: {tempo_safe}
- chords (optional): {', '.join(chords) if chords else '—'}
- instruments (optional): {', '.join(instruments) if instruments else '—'}
- texture notes (optional): {', '.join(texture_desc) if texture_desc else '—'}
- CLAP embedding preview (first 10): {clap_preview or '—'}

features_json:
{feat_str}

Respond with five sections:
1) One-paragraph musical summary (vibe, genre cues, references if relevant).
2) Key & tempo validation with a brief confidence note (why).
3) Timbre & mix notes (brightness, punch, dynamics, space).
4) Rhythm & harmony insights (groove, harmonic movement/feel).
5) 2–3 concise production tips tailored to this material.
"""
    
    response = client.chat.completions.create(
        model= "gpt-3.5-turbo", #model or OPENAI_MODEL,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

# Analyze this track’s mood, emotional tone, and possible imagery or meaning based on the following musical features:

# - Key: {key}
# - Mode: {mode}
# - Tempo: {tempo} BPM
# - Texture: {texture_desc}
# - CLAP Audio Embedding (partial): {clap_desc}
# - Chords: {', '.join(chords)}
# - Instrumentation: {', '.join(instruments)}


# I want you to respond like you’re in the studio listening through high-end monitors. Be confident, decisive, and precise — break down what you think the genre, instrumentation, and emotional intention might be. If it reminds you of any artists or musical scenes, say so. End with a 1–2 sentence critique or suggestion from your producer perspective.

# """

    # response = client.chat.completions.create(
    #     model="gpt-3.5-turbo",
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0.85,
    #     max_tokens=300,
    # )

    # return response.choices[0].message.content



#print("🔑 Key:", repr(os.getenv("OPENAI_API_KEY")))


'''
You are a legendary music producer known for your ability to deconstruct any track by ear and instantly identify its genre, instrumentation, mood, and influences — even from just a few seconds of audio. You’ve worked across every style from Afrobeat and trip-hop to orchestral film scores and underground electronic scenes.

Based on the following information, tell me what you're hearing:

- Key: {key}
- Mode: {mode}
- Tempo: {tempo:.2f} BPM
- Texture/Vibe: {texture_desc}
- CLAP Embedding Preview (partial): {clap_vector_summary}
- MFCC Mean Coefficients (first 5): {mfcc_summary}

I want you to respond like you’re in the studio listening through high-end monitors. Be confident, decisive, and precise — break down what you think the genre, instrumentation, and emotional intention might be. If it reminds you of any artists or musical scenes, say so. End with a 1–2 sentence critique or suggestion from your producer perspective.

'''

#original
'''

You are a poetic music analyst who responds with vivid, emotionally intelligent feedback. 

Analyze this track’s mood, emotional tone, and possible imagery or meaning based on the following musical features:

- Key: {key}
- Mode: {mode}
- Tempo: {tempo} BPM
- Texture: {texture_desc}
- CLAP Audio Embedding (partial): {clap_desc}
- Chords: {', '.join(chords)}
- Instrumentation: {', '.join(instruments)}


Respond as if you’re feeling the music. Don’t be overly technical. Use vivid metaphors or imagery, and describe how the music might affect someone emotionally. If relevant, suggest a scene, setting, or story the song might evoke. List the instruments that you hear and how they affect the overall sound. see if you can determine the genre as well.


'''