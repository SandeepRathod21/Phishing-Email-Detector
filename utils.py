import re
import email
from email import policy

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " urltoken ", text)
    text = re.sub(r"\S+@\S+", " emailtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_features(text):
    text = str(text).lower()

    has_url = int("http" in text or "www" in text)
    urgent = int("urgent" in text)
    verify = int("verify" in text)
    confirm = int("confirm" in text)
    login = int("login" in text)
    account = int("account" in text)
    click = int("click" in text)
    suspend = int("suspend" in text or "suspension" in text)

    risk_score = (
        has_url*3 + urgent*2 + verify*2 + confirm*2 +
        click*2 + login*2 + suspend*3
    )

    return [
        has_url, urgent, verify, confirm, login,
        account, click, suspend,
        risk_score, len(text)
    ]

def check_urls(text):
    urls = re.findall(r"http[s]?://\S+", text)
    return sum("login" in u or "verify" in u for u in urls)

def analyze_headers(raw_email):
    try:
        msg = email.message_from_string(raw_email, policy=policy.default)
        sender = msg.get("From", "")
        reply_to = msg.get("Reply-To", "")

        score = 0
        if reply_to and reply_to not in sender:
            score += 2
        if any(x in sender.lower() for x in ["verify","secure","login"]):
            score += 2

        return score
    except:
        return 0