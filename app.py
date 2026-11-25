"""SnareScan: Streamlit demo for the fine-tuned DistilBERT phishing email classifier."""
import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

APP_DIR = Path(__file__).resolve().parent
_default_dir = APP_DIR / "snarescan_model"
if not _default_dir.is_dir() and (APP_DIR / "config.json").is_file():
    _default_dir = APP_DIR          # model files uploaded next to app.py (e.g. GitHub repo root)
MODEL_DIR = Path(os.environ.get("SNARESCAN_MODEL_DIR", _default_dir))
# On Streamlit Community Cloud the model is downloaded from this Hugging Face model repo
# (set it as the secret SNARESCAN_MODEL_REPO, e.g. "yourname/snarescan-distilbert").
MODEL_REPO = os.environ.get("SNARESCAN_MODEL_REPO", "").strip()
MAX_LENGTH = 512

st.set_page_config(page_title="SnareScan", page_icon="🎣", layout="centered")

EXAMPLES = {
    "Phishing example": (
        "Urgent: Your account access will be suspended",
        "Dear Customer,\n\nWe detected unusual sign-in activity on your Northbridge Bank account. "
        "To avoid permanent suspension, please verify your identity within 24 hours by confirming "
        "your username, password and card details at the secure link below:\n\n"
        "http://northbridge-secure-verify.com/login\n\nFailure to respond will result in your account "
        "being locked.\n\nNorthbridge Bank Security Team",
    ),
    "Legitimate example": (
        "Notes from Thursday's project meeting",
        "Hi team,\n\nThanks for joining the meeting yesterday. As agreed, Ana will finish the data "
        "cleaning by Monday, and I will draft the methodology section. Our next check-in is on "
        "Thursday at 3 PM in Room 204.\n\nLet me know if I missed anything.\n\nBest,\nMarco",
    ),
}


def ensure_weights(model_dir):
    """Rebuild model.safetensors from the 20 MB parts (GitHub's website accepts files up to 25 MB)."""
    target = model_dir / "model.safetensors"
    parts = sorted(model_dir.glob("model.safetensors.part*"))
    if target.is_file() or not parts:
        return
    tmp = model_dir / "model.safetensors.tmp"
    with open(tmp, "wb") as out:
        for part in parts:
            out.write(part.read_bytes())
    tmp.replace(target)


@st.cache_resource(show_spinner="Loading the SnareScan model (first load takes about a minute)…")
def load_model():
    model_dir = MODEL_DIR
    if not model_dir.is_dir():
        from huggingface_hub import snapshot_download
        model_dir = Path(snapshot_download(MODEL_REPO))
    ensure_weights(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    info_path = model_dir / "model_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    return tokenizer, model, device, info


def phishing_probability(texts, tokenizer, model, device, batch_size=32):
    probs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())
    return probs


if not MODEL_DIR.is_dir() and not MODEL_REPO:
    st.error(f"Model not found. Put the snarescan_model folder next to app.py, or set the secret "
             f"SNARESCAN_MODEL_REPO to your Hugging Face model repo (looked for: {MODEL_DIR}).")
    st.stop()

tokenizer, model, device, info = load_model()

# ---------------- Sidebar: model details and decision threshold ----------------
with st.sidebar:
    st.header("Model")
    st.write(f"**Base model:** {info.get('base_model', 'distilbert-base-uncased')}")
    st.write(f"**Training data:** {info.get('training_data', 'Enron-v2 + E-PhishLLM (train split)')}")
    if "seed" in info:
        st.write(f"**Seed:** {info['seed']}")
    st.write(f"**Running on:** {device.upper()}")

    st.header("Decision threshold")
    options = {"Default (0.50)": 0.5}
    if "dev_threshold" in info:
        options[f"Validation-tuned ({info['dev_threshold']:.2f})"] = float(info["dev_threshold"])
    choice = st.radio("Classify as phishing when probability ≥", list(options), index=0)
    threshold = options[choice]

    test = info.get("test_at_0.5")
    if test:
        st.header("Held-out test results")
        st.caption(f"This saved model, {info.get('test_n', 3451):,} unseen E-PhishLLM emails, threshold 0.50")
        c1, c2 = st.columns(2)
        c1.metric("Phishing F1", f"{test['f1_binary_phishing']:.4f}")
        c2.metric("Macro-F1", f"{test['f1_macro']:.4f}")
        c1.metric("Precision", f"{test['precision_phishing']:.4f}")
        c2.metric("Recall", f"{test['recall_phishing']:.4f}")
        if "auroc" in test:
            c1.metric("AUROC", f"{test['auroc']:.4f}")

# ---------------- Main page ----------------
st.title("🎣 SnareScan")
st.write("Paste an email and SnareScan predicts whether it is **phishing** or **legitimate**, using a "
         "DistilBERT model fine-tuned on Enron-v2 and E-PhishLLM.")

tab_single, tab_batch = st.tabs(["Check one email", "Check a CSV file"])

with tab_single:
    st.write("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for col, (name, (subject, body)) in zip(cols, EXAMPLES.items()):
        if col.button(name, use_container_width=True):
            st.session_state["subject"] = subject
            st.session_state["body"] = body

    subject = st.text_input("Subject", key="subject")
    body = st.text_area("Body", key="body", height=220)

    if st.button("Analyze email", type="primary", use_container_width=True):
        text = f"{subject.strip()} {body.strip()}".strip()   # same subject + body format used in training
        if not text:
            st.warning("Enter a subject or body first.")
        else:
            prob = phishing_probability([text], tokenizer, model, device)[0]
            if prob >= threshold:
                st.error(f"⚠️ **Phishing**: phishing probability {prob:.1%}")
            else:
                st.success(f"✅ **Legitimate**: phishing probability {prob:.1%}")
            st.progress(min(max(prob, 0.0), 1.0), text=f"Phishing probability (threshold {threshold:.2f})")
            n_tokens = len(tokenizer(text, truncation=False)["input_ids"])
            if n_tokens > MAX_LENGTH:
                st.caption(f"Note: this email has {n_tokens} tokens; only the first {MAX_LENGTH} were read by the model.")

with tab_batch:
    st.write("Upload a CSV with a **text** column, or **subject** and **body** columns. "
             "If it also has a **label** column (0 = legitimate, 1 = phishing), accuracy is shown.")
    upload = st.file_uploader("CSV file", type=["csv"])
    if upload is not None:
        df = pd.read_csv(upload)
        if "text" in df.columns:
            texts = df["text"].fillna("").astype(str).tolist()
        elif {"subject", "body"}.issubset(df.columns):
            texts = (df["subject"].fillna("").astype(str).str.strip() + " " +
                     df["body"].fillna("").astype(str).str.strip()).str.strip().tolist()
        else:
            st.error("The CSV needs a 'text' column, or both 'subject' and 'body' columns.")
            st.stop()
        with st.spinner(f"Classifying {len(texts):,} emails…"):
            probs = phishing_probability(texts, tokenizer, model, device)
        df["phishing_probability"] = probs
        df["prediction"] = ["phishing" if p >= threshold else "legitimate" for p in probs]
        st.write(f"**{(df['prediction'] == 'phishing').sum():,}** of {len(df):,} emails flagged as phishing.")
        if "label" in df.columns:
            correct = (df["label"].astype(int) == (df["prediction"] == "phishing").astype(int)).mean()
            st.write(f"Accuracy against the label column: **{correct:.2%}**")
        st.dataframe(df, use_container_width=True)
        buf = io.StringIO(); df.to_csv(buf, index=False)
        st.download_button("Download results CSV", buf.getvalue(), "snarescan_predictions.csv", "text/csv")

st.caption("Results come from in-domain testing on held-out E-PhishLLM emails and may not reflect real-world accuracy. "
           "Always verify suspicious emails through official channels.")
