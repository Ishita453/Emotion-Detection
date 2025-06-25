import os
import librosa
import numpy as np

DATA_PATH = r"C:\Users\ishii\Desktop\Emo project\data"

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        features = np.hstack([
            np.mean(mfccs.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(mel.T, axis=0)
        ])
        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def load_data():
    X = []
    y = []
    count = 0
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                try:
                    emotion_code = file.split("-")[2]
                    emotion = EMOTIONS.get(emotion_code)
                    if not emotion:
                        print(f"Unknown emotion code {emotion_code} in file {file}")
                        continue
                    features = extract_features(file_path)
                    if features is not None:
                        X.append(features)
                        y.append(emotion)
                        count += 1
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
    print(f"Total features extracted: {count}")
    return np.array(X), np.array(y)
