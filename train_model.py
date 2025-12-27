import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# --- NLP setup ---
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

# --- Load & prepare data (keep your files in ./data/) ---
fake = pd.read_csv("data/Fake.csv", low_memory=False)
real = pd.read_csv("data/True.csv", low_memory=False)
fake["label"] = 0
real["label"] = 1
df = pd.concat([fake[["text","label"]], real[["text","label"]]], ignore_index=True).sample(frac=1, random_state=42)

df["clean_text"] = df["text"].apply(preprocess)

# --- Vectorize ---
vectorizer = TfidfVectorizer(max_df=0.7)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

# --- Train ---
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_tr, y_tr)

# --- Quick check ---
pred = model.predict(X_te)
print(f"Accuracy: {accuracy_score(y_te, pred):.4f}")

# --- Save artifacts ---
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Saved: model.pkl & vectorizer.pkl")
