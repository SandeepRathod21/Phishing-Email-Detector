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
    if check_urls(text) > 0 or analyze_headers(text) >= 2:
        label = "PHISHING"
        conf = max(conf, 85)

    return label, conf

# UI
st.set_page_config(page_title="Phishing Detector", page_icon="📧")

st.markdown("## 📧 Phishing Email Detector")
st.markdown("#### 🔐 AI + Rule-Based Security Engine")

text = st.text_area("Paste Email Content", height=250)

uploaded_file = st.file_uploader("Upload email file")

if uploaded_file:
    text = uploaded_file.read().decode()

# ANALYZE
if st.button("Analyze"):
    if text:
        label, conf = predict(text)

        if label == "PHISHING":
            st.error(f"🚨 PHISHING ({conf}%)")
            st.warning("⚠️ Suspicious links or urgency patterns detected")
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