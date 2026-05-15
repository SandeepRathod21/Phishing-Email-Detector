import streamlit as st
import joblib
import numpy as np
import re

# LOAD MODELS
tfidf = joblib.load("tfidf.pkl")
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

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

# PREDICT FUNCTION
def predict(text):

    clean = clean_text(text)

    # TFIDF
    vec = tfidf.transform([clean]).toarray()

    # EXTRA FEATURES
    extra = np.array([extract_features(text)])

    # COMBINE
    combined = np.hstack([vec, extra])

    # SCALE
    scaled = scaler.transform(combined)

    # PREDICT
    prediction = model.predict(scaled)[0]

    # CONFIDENCE
    confidence = model.predict_proba(scaled)[0]

    return prediction, confidence

# UI
st.title("Phishing Email Detector")

text = st.text_area("Paste email content here")

if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Please enter email text.")
    else:

        label, conf = predict(text)

        if label == 1:
            st.error("Phishing Email Detected")
        else:
            st.success("Legitimate Email")

        st.write("Confidence:")
        st.write(conf)