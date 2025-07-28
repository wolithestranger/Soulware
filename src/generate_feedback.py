import os
from openai import OpenAI
from dotenv import load_dotenv

# Load .env
load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)


def generate_feedback(key, mode, tempo, chords, instruments, texture_desc):
    prompt = f"""
You are a poetic music analyst who responds with vivid, emotionally intelligent feedback. 

Analyze this track’s mood, emotional tone, and possible imagery or meaning based on the following musical features:

- Key: {key}
- Mode: {mode}
- Tempo: {tempo} BPM
- Texture: {texture_desc}
- Chords: {', '.join(chords)}
- Instrumentation: {', '.join(instruments)}

Respond as if you’re feeling the music. Don’t be overly technical. Use vivid metaphors or imagery, and describe how the music might affect someone emotionally. If relevant, suggest a scene, setting, or story the song might evoke.
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.85,
        max_tokens=300,
    )

    return response.choices[0].message.content



#print("🔑 Key:", repr(os.getenv("OPENAI_API_KEY")))
