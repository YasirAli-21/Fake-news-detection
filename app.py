import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
import pandas as pd
from pathlib import Path

# Page config
st.set_page_config(page_title="Fake News Detection", page_icon="🔎", layout="centered")

# Custom CSS (sidebar gradient + centered layout + nice buttons) ------------
st.markdown("""
<style>
.main > div { display:flex; flex-direction:column; align-items:center; }

section[data-testid="stSidebar"] > div { 
  background: linear-gradient(180deg, #2F3542 0%, #1E272E 100%);
  color: #ecf0f1;
}
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] label { color: #ecf0f1 !important; }

textarea { border-radius:12px !important; border:2px solid #2980B9 !important; }

div.stButton > button:first-child{
  background: linear-gradient(45deg,#2980B9,#6DD5FA);
  color:white; border:none; border-radius:10px; padding:10px 24px;
  transition:0.25s; font-weight:600;
}
div.stButton > button:first-child:hover{ transform:scale(1.04); }

</style>
""", unsafe_allow_html=True)

# ------------ NLP preprocess (same as training) ------------
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# ------------ cached loaders ------------
@st.cache_resource
def load_artifacts():
    mpath = Path("model.pkl")
    vpath = Path("vectorizer.pkl")
    if not mpath.exists() or not vpath.exists():
        raise FileNotFoundError("model.pkl/vectorizer.pkl not found. Run train.py first.")
    model = joblib.load(mpath)
    vectorizer = joblib.load(vpath)
    return model, vectorizer
try:
    model, vectorizer = load_artifacts()
except Exception as e:
    st.error(f"Artifacts not loaded: {e}")
    st.stop()

# ------------ Session history ------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts

# ------------ Sidebar (like your screenshot) ------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Check News", "Past News History"], index=0)

# ------------ Page: Check News ------------
if page == "Check News":
    st.markdown("<div class='title' style='font-size:38px;font-weight:800;color:#ecf0f1;'>🔎 Fake News Detection App</div>", unsafe_allow_html=True)
    user_input = st.text_area("Paste your news text here:", height=180, label_visibility="visible")

    if st.button("Check News"):
        processed = preprocess(user_input)
        vec = vectorizer.transform([processed])

        # Try predict_proba; fallback to decision_function or plain predict
        label = int(model.predict(vec)[0])
        prob = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vec)[0]
            prob = float(proba[label])
        elif hasattr(model, "decision_function"):
            # convert margin to pseudo-probability for display
            import numpy as np
            margin = float(model.decision_function(vec)[0])
            prob = 1/(1+np.exp(-abs(margin)))

        if label == 1:
            st.success(f"✅ This news is **Real**{f'  |  confidence: {prob:.2%}' if prob is not None else ''}")
        else:
            st.error(f"❌ This news is **Fake**{f'  |  confidence: {prob:.2%}' if prob is not None else ''}")

        st.session_state.history.append({
            "text": user_input.strip()[:2000],  # keep size reasonable
            "prediction": "Real" if label == 1 else "Fake",
            "confidence": None if prob is None else round(prob, 4)
        })

# ------------ Page: Past News History ------------
else:
    st.markdown("## 📜 Past News History")
    if len(st.session_state.history) == 0:
        st.info("No checks yet. Go to **Check News** and try some text.")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download History (CSV)", data=csv, file_name="history.csv", mime="text/csv")
        if st.button("Clear History"):
            st.session_state.history = []
            st.success("History cleared.")
