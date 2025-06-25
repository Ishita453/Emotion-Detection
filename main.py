from emotion import load_data
from train import train_model

if __name__ == "__main__":
    X, y = load_data()
    if len(X) > 0:
        print(f"Feature vector shape: {X[0].shape}")
        print(f"Sample labels: {y[:5]}")
        model, label_encoder = train_model(X, y)
