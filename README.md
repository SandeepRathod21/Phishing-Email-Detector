# 🛡️ Phishing Email Detector

An AI-powered phishing email detection system built using **Python**, **Scikit-learn**, and **Streamlit**.

This project combines **Machine Learning** and **Rule-Based Security Checks** to identify phishing emails and suspicious content with confidence scores and risk indicators.

---

# 🚀 Features

- 🔍 Detects phishing and safe emails
- 🤖 Machine Learning based classification
- 🛡️ Rule-based phishing detection
- 📊 Confidence score prediction
- ⚠️ Risk indicator analysis
- 📁 Upload email text files
- 🌙 Modern Streamlit dark UI
- 📈 Dashboard support
- 🔗 Suspicious URL detection

---

# 🧠 Technologies Used

- Python
- Streamlit
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Joblib

---

# 📂 Project Structure

```bash
Phishing-Email-Detector/
│
├── web_app.py
├── train_model.py
├── utils.py
├── requirements.txt
├── README.md
│
├── model.pkl
├── tfidf.pkl
├── scaler.pkl
│
├── accuracy.pkl
├── conf_matrix.pkl
├── report.txt
│
└── CEAS_08.csv
```
---
# 📊 Dataset

Dataset used:

* CEAS_08 Email Dataset

The dataset contains phishing and legitimate emails used for training and testing the machine learning model.

---
# ⚙️ Installation

> Clone the repository:
```Bash
git clone https://github.com/SandeepRathod21/Phishing-Email-Detector.git
```
> Move into project folder:
```Bash
cd Phishing-Email-Detector
```
> Install dependencies:
```Bash
pip install -r requirements.txt
```
---
# 🏋️ Train the Model

> Run:
```Bash
python train_model.py
```
> This generates:

* model.pkl
* tfidf.pkl
* scaler.pkl
--- 
# ▶️ Run the Application

Start Streamlit app:
```Bash
streamlit run web_app.py
```
---
# 👨‍💻 Author

Sandeep Rathod

Cybersecurity & Al

---
