import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

from utils import clean_text, extract_features

# LOAD DATA
df = pd.read_csv(r"C:\Users\sande\OneDrive\Desktop\Phishing_Project\CEAS_08.csv")

df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
df["clean"] = df["text"].apply(clean_text)

X = df["clean"].tolist()
y = df["label"].astype(int).values

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,3))
X_train_tfidf = tfidf.fit_transform(X_train)

# EXTRA FEATURES
extra_train = np.array([extract_features(t) for t in X_train])
X_train_combined = np.hstack([X_train_tfidf.toarray(), extra_train])

# SCALE
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train_combined)

# MODEL
model = XGBClassifier(n_estimators=300, eval_metric="logloss")
model.fit(X_train_scaled, y_train)

# TEST
X_test_tfidf = tfidf.transform(X_test)
extra_test = np.array([extract_features(t) for t in X_test])

X_test_combined = np.hstack([X_test_tfidf.toarray(), extra_test])
X_test_scaled = scaler.transform(X_test_combined)

y_pred = model.predict(X_test_scaled)

# METRICS
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# SAVE EVERYTHING
joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(acc, "accuracy.pkl")
joblib.dump(cm, "conf_matrix.pkl")

with open("report.txt", "w") as f:
    f.write(report)

print("✅ Accuracy:", acc)
print("✅ Confusion Matrix:\n", cm)