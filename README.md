# 🎾 Tennis Match Predictor (Elo-Enhanced)

A machine learning web app that predicts ATP tennis match outcomes using player rankings, Elo ratings (global + surface), and match context (surface & best-of format).  
Built with **Python**, **Streamlit**, and **scikit-learn**.

---

How It Works

build_elo_and_retrain.py calculates Elo ratings for each player and retrains a logistic regression model that predicts the probability Player 1 beats Player 2.

app.py loads that model and the player data to offer an interactive UI with:

✅ Win probability

🎾 Sets prediction (Bo3 or Bo5)

🎯 Total games prediction

📊 Over/Under slider

| Task                  | Command                           |
| --------------------- | --------------------------------- |
| Retrain model         | `python build_elo_and_retrain.py` |
| Launch app            | `streamlit run app.py`            |
| Clear Streamlit cache | `streamlit cache clear`           |
| Freeze deps           | `pip freeze > requirements.txt`   |


## 🚀 Features

- Predict match winners based on two selected players  
- Displays win probability, expected sets (Bo3 / Bo5), and total games (with over/under helper)  
- Uses **Elo-enhanced Logistic Regression** trained on ATP matches (1968–2025)  
- Caches data & models for fast runtime  
- Works locally or on any host (Render, Vercel, Streamlit Cloud, etc.)

---

## 📂 Project Structure
tennis-predictor-app/
│
├── app.py # Streamlit web app (Elo-enhanced version)
├── build_elo_and_retrain.py # Script to compute Elo & retrain the logistic model
│
├── data/ # Data & model storage (recommended)
│ ├── combined_matches_1968_2025.csv
│ ├── players.csv
│ ├── feature_columns.csv
│ ├── model_logreg.joblib
│ ├── model_sets_bo3.joblib
│ ├── model_sets_bo5.joblib
│ ├── model_games.joblib
│ └── elo_current.csv
│
├── requirements.txt
└── README.md

---

## 🧠 Requirements

- Python 3.10 or newer  
- Recommended: a virtual environment (`.venv`)  
- Libraries listed in [`requirements.txt`](./requirements.txt):

- streamlit
pandas
scikit-learn
joblib

---

## ⚙️ Setup Instructions

### 1. Create and activate your virtual environment

**Windows PowerShell:**
```bash
python -m venv .venv
.venv\Scripts\activate

Install dependencies
pip install -r requirements.txt
Run this whenever you update your match dataset (combined_matches_1968_2025.csv):

python build_elo_and_retrain.py --input data/combined_matches_1968_2025.csv

This will output:

elo_current.csv – latest Elo ratings per player

matches_with_elo.csv – historical matches joined with Elo

feature_columns.csv – feature list for app/model

model_logreg.joblib – trained logistic regression model

streamlit run app.py


