import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

from utils import clean_text, extract_features, check_urls, analyze_headers

# LOAD
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")
scaler = joblib.load("scaler.pkl")

def predict(text):
    vec = tfidf.transform([clean_text(text)]).toarray()
    extra = np.array([extract_features(text)])
    X = np.hstack([vec, extra])
    X = scaler.transform(X)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    label = "PHISHING" if pred == 1 else "SAFE"
    conf = round(float(prob[pred]) * 100, 2)

    # RULE LAYER
    text_clean = text.replace("\n", " ").replace("\r", " ").lower()
    suspicious = ["usp=sharing", "usp=share", "docs.google.com",
                "drive.google.com", "bit.ly", "tinyurl"]
    for pattern in suspicious:
        if pattern in text_clean:
            label = "PHISHING"
            conf = 99.89
            break

    # WHITELIST - if no URL and no urgency words, force SAFE
    no_url = "http" not in text.lower() and "www" not in text.lower()
    no_urgency = not any(w in text.lower() for w in [
    "urgent", "suspended", "verify", "click here",
    "winner", "claim", "expire", "limited time"
        ])
    if no_url and no_urgency and label == "PHISHING" and conf < 99:
        label = "SAFE"
        conf = round(100 - conf, 2)
    return label, conf

# UI
st.set_page_config(page_title="Created By Sandeep Rathod`", page_icon="📧")

st.markdown("## 📧 Phishing Email Detector")
st.markdown("#### 🔐 AI + Rule-Based Security Engine")

text = st.text_area("Paste Email Content", height=250)

uploaded_file = st.file_uploader("Upload email file")

if uploaded_file:
    text = uploaded_file.read().decode("utf-8", errors="ignore")

# ANALYZE
if st.button("Analyze"):
    if text:
        label, conf = predict(text)

        if label == "PHISHING" and conf >= 90:
            st.error(f"🚨 PHISHING ({conf}%)")
            st.warning("Strong indicators detected - Do NOT click any links or download attachments")
        elif label == "PHISHING" and conf >= 75:
            st.warning(f"⚠️ SUSPICIOUS ({conf}%) - Manual review recommended")
            st.info("Some phishing indicators detected")
        else:
            st.success(f"✅ SAFE ({conf}%)")
            st.info("No strong phishing indicators detected")

        # EXPLAINABILITY
        features = extract_features(text)
        st.write("### 🔍 Risk Indicators")
        st.write({
            "Has URL": features[0],
            "Urgent words": features[1],
            "Verify/Confirm": features[2] + features[3],
            "Click/Login": features[6] + features[4],
            "Risk score": features[8]
        })

    else:
        st.warning("Enter email text")

# DASHBOARD
if st.button("📊 Show Dashboard"):
    acc = joblib.load("accuracy.pkl")
    cm = joblib.load("conf_matrix.pkl")

    st.subheader("📊 Model Performance")
    st.success(f"Accuracy: {round(acc*100,2)}%")

    fig, ax = plt.subplots()
    ax.imshow(cm)

    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i][j], ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    with open("report.txt") as f:
        st.text(f.read())