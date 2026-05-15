import pandas as pd
import numpy as np
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

# LOAD DATA
df = pd.read_csv("CEAS_08.csv")

# CLEAN TEXT
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text

# EXTRA FEATURES
def extract_features(text):
    text = str(text)

    num_links = text.count("http")
    num_digits = sum(c.isdigit() for c in text)
    num_special = sum(not c.isalnum() and not c.isspace() for c in text)
    length = len(text)

    return [num_links, num_digits, num_special, length]

# COMBINE SUBJECT + BODY
df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

# CLEAN
df["clean"] = df["text"].apply(clean_text)

# FEATURES + LABELS
X = df["clean"].tolist()
y = df["label"].astype(int).values

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 3)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# EXTRA FEATURES
extra_train = np.array([extract_features(t) for t in X_train])
extra_test = np.array([extract_features(t) for t in X_test])

# COMBINE
X_train_combined = np.hstack([
    X_train_tfidf.toarray(),
    extra_train
])

X_test_combined = np.hstack([
    X_test_tfidf.toarray(),
    extra_test
])

# SCALE
scaler = StandardScaler(with_mean=False)

X_train_scaled = scaler.fit_transform(X_train_combined)
X_test_scaled = scaler.transform(X_test_combined)

# MODEL
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss"
)

model.fit(X_train_scaled, y_train)

# PREDICT
y_pred = model.predict(X_test_scaled)

# RESULTS
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# SAVE MODELS
joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(model, "model.pkl")

print("Models saved successfully.")