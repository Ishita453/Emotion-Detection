import sounddevice as sd
import numpy as np
import librosa
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model and label encoder
model = joblib.load("model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# Audio settings
DURATION = 20  # seconds
SR = 22050    # sample rate

def extract_features_live(audio):
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=SR)
        mel = librosa.feature.melspectrogram(y=audio, sr=SR)
        features = np.hstack([
            np.mean(mfccs.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(mel.T, axis=0)
        ])
        return features
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

def record_audio():
    print("🎤 Speak now...")
    audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
    sd.wait()
    print("✅ Recording done.")
    return audio.flatten()

# Real-time loop
while True:
    audio_data = record_audio()
    features = extract_features_live(audio_data)

    if features is not None:
        prediction = model.predict([features])[0]
        emotion = label_encoder.inverse_transform([prediction])[0]
        print(f"🧠 Predicted Emotion: {emotion}")
    else:
        print("❌ Feature extraction failed.")

    choice = input("🔁 Do you want to test again? (y/n): ").strip().lower()
    if choice != 'y':
        print("👋 Exiting...")
        break
