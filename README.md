# 📰 Fake News Detection using Machine Learning & NLP

An end-to-end **Fake News Detection system** that classifies news articles as **Real** or **Fake** using **Natural Language Processing (NLP)** and **Machine Learning**.  
The project includes data preprocessing, model training, and an **interactive Streamlit web application** for real-time predictions.

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Technologies & Tools Used](#-technologies--tools-used)
- [Project Structure](#-project-structure)
- [How the System Works](#-how-the-system-works)
- [How to Run the Project](#-how-to-run-the-project)
- [Application Features](#-application-features)
- [Dataset Information](#-dataset-information)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [Support](#-support)

---

## 🚀 Project Overview

Fake news has become a major challenge in today’s digital world. This project aims to help identify misleading or false news articles by analyzing their textual content using NLP techniques and a supervised machine learning model.

The system:
- Cleans and preprocesses news text
- Converts text into numerical features using **TF-IDF**
- Trains a **Logistic Regression** classifier
- Provides real-time predictions through a **Streamlit-based UI**
- Maintains **prediction history** during the session

---

## 🛠️ Technologies & Tools Used

- **Programming Language:** Python  
- **Machine Learning:** Logistic Regression  
- **Natural Language Processing (NLP):**
  - Text Cleaning
  - Stopwords Removal
  - Stemming
- **Feature Extraction:** TF-IDF Vectorizer  
- **Web Framework:** Streamlit  
- **Libraries & Tools:**
  - Pandas
  - NumPy
  - Scikit-learn
  - NLTK
  - Joblib
- **Development Environment:** PyCharm  
- **Notebook Support:** Jupyter Notebook (for experimentation & analysis)

---

## 📂 Project Structure

Fake-News-Detection/
│
├── data/
│ ├── Fake.csv
│ └── True.csv
│
├── train.py
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md

yaml
Copy code

---

## ⚙️ How the System Works

### 1️⃣ Data Preprocessing
- Converts text to lowercase
- Removes special characters and numbers
- Removes English stopwords
- Applies stemming using Porter Stemmer

### 2️⃣ Feature Extraction
- Uses **TF-IDF Vectorizer** to convert text into numerical form

### 3️⃣ Model Training
- Algorithm: **Logistic Regression**
- Dataset split into training and testing sets
- Model trained and evaluated using accuracy score

### 4️⃣ Model Saving
- Trained model saved as `model.pkl`
- TF-IDF vectorizer saved as `vectorizer.pkl`

### 5️⃣ Web Application
- Streamlit-based interactive UI
- Real-time prediction (Real / Fake)
- Confidence score display (if supported)
- Session-based prediction history

---

## ▶️ How to Run the Project

### 🔹 Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
🔹 Step 2: Create Virtual Environment (Optional)
bash
Copy code
python -m venv venv
venv\Scripts\activate
🔹 Step 3: Install Dependencies
bash
Copy code
pip install -r requirements.txt
🔹 Step 4: Train the Model
bash
Copy code
python train.py
This will generate:

model.pkl

vectorizer.pkl

🔹 Step 5: Run Streamlit App
bash
Copy code
streamlit run app.py
🖥️ Application Features
✅ User-friendly Streamlit interface

📝 Paste or type any news article

🔍 Instant Real/Fake prediction

📊 Confidence score display

🕒 Session-based past prediction history

📥 Download prediction history as CSV

📈 Dataset Information
Fake News Dataset

Real News Dataset

Source: Publicly available Kaggle datasets

Labels:

0 → Fake News

1 → Real News

⚠️ Limitations
Model performance depends on training data

May not generalize well to:

Breaking news

Very short headlines

Regional or unseen writing styles

Session history resets on app reload (Streamlit default behavior)

🔮 Future Improvements
Use advanced models (LSTM, BERT, Transformers)

Add multi-language support

Deploy on Streamlit Cloud / Hugging Face Spaces

Add database for persistent history

Include source credibility analysis

## 👨‍💻 Author <a name="author"></a>
**Yasir Ali** | IT Enthusiast 

[![github](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/YasirAli-21)
[![linkedin](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yasisahito)