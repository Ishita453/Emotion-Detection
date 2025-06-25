from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from emotion import load_data

# Step 1: Load data using load_data (not extract_features!)
X, y = load_data()

# Step 2: Train model
def train_model(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc*100:.2f}%")

    return model, le

model, label_encoder = train_model(X, y)

# Step 3: Save model
joblib.dump(model, "model.joblib")
joblib.dump(label_encoder, "label_encoder.joblib")
