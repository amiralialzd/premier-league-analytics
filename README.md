# ⚽ Premier League Match Predictor

> Predicting Premier League 2024/25 match outcomes using machine learning — built from scratch with real match data, engineered features, and a live Streamlit app.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red)](https://premier-league-analytics-2024-25.streamlit.app)

---

## 🚀 Live Demo

**[https://premier-league-analytics-2024-25.streamlit.app](https://premier-league-analytics-2024-25.streamlit.app)**

Select any two Premier League teams → get a predicted result with probability scores.

---

## 📱 What the App Does

- User selects a **home team** and an **away team**
- App looks up each team's end-of-season performance stats
- XGBoost model predicts: **Home Win / Draw / Away Win**
- Results displayed as a probability bar chart

---

## 🔧 Features Engineered

All features were built from raw match data — no pre-built stat sources:

| Feature | Description |
|---|---|
| `home_team_total_wins` | Cumulative wins for home team up to match date |
| `away_team_total_wins` | Cumulative wins for away team up to match date |
| `home_team_win_at_home` | Home team's wins when playing at home |
| `home_team_win_away` | Home team's wins when playing away |
| `away_team_win_at_home` | Away team's wins when playing at home |
| `away_team_win_away` | Away team's wins when playing away |
| `home_team_last5_goals_scored` | Rolling avg goals scored over last 5 matches |
| `home_team_last5_goals_conceded` | Rolling avg goals conceded over last 5 matches |
| `away_team_last5_goals_scored` | Rolling avg goals scored over last 5 matches |
| `away_team_last5_goals_conceded` | Rolling avg goals conceded over last 5 matches |

---

## 🤖 Models and Results

Three classifiers trained and compared:

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 51.3% | 0.44 |
| Random Forest | 48.7% | 0.42 |
| XGBoost (tuned) | 53.0% | 0.45 |

**XGBoost** selected as best performer, tuned with GridSearchCV.

> Note: A naive model always predicting Home Win achieves 41% accuracy. 53% with honest draw prediction is a meaningful result for a single-season dataset with no player-level data.

---

## 💡 Key Finding

**`home_team_last5_goals_conceded`** was the strongest predictor of match outcomes across all 10 features.

Recent defensive form matters more than attacking output — consistent with what football fans already know:

> *"You win games with strikers, but you win championships with defenders."*

The data agreed.

---

## ⚠️ Why Macro F1, Not Accuracy

A model that ignores draws entirely achieves 41% accuracy on this dataset. Optimizing for accuracy produces a model that never predicts draws — inflating the score while hiding a fundamental failure.

**Macro F1** penalizes equally across all three outcome classes (H/D/A), exposing models that sacrifice minority class performance for headline numbers.

---

## 🛠️ Tech Stack

- **Python** · **Pandas** · **NumPy**
- **Scikit-learn** · **XGBoost**
- **Streamlit** · **Plotly**
- **Joblib** · **Git/GitHub**

---

## ▶️ Run Locally

```bash
git clone https://github.com/amiralialzd/premier-league-analytics
cd premier-league-analytics
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Data Source

- **Source:** FBref via `soccerdata` Python library
- **Season:** 2024/25 Premier League
- **Matches:** 380 total

---

*Built by [Amirali Alizadeh](https://linkedin.com/in/amiralializadeh) · Part of LLM/Applied AI Engineer Roadmap*