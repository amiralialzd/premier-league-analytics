"""Re-evaluate the saved model on the same held-out test split used in training.

Run from the repo folder:  python evaluate.py
Needs: model.pkl and premier_league_data.csv (both in this folder).
The split settings (test_size=0.2, stratify=y, random_state=42) must match main.py.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("premier_league_data.csv")
model = joblib.load("model.pkl")

X = df.drop(columns=["home_team", "away_team", "result", "home_form", "away_form", "date"])
y = df["result"]

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
y_test = LabelEncoder().fit(y).transform(y_test)  # A=0, D=1, H=2

pred = model.predict(X_test)
n = len(y_test)
acc = accuracy_score(y_test, pred)
macro_f1 = f1_score(y_test, pred, average="macro")
baseline = (y_test == 2).mean()  # always predict Home Win
margin = 1.96 * np.sqrt(acc * (1 - acc) / n)  # rough 95% interval

print(f"test matches        : {n}")
print(f"accuracy            : {acc:.3f}  (about +/- {margin:.3f})")
print(f"macro F1            : {macro_f1:.3f}")
print(f"always-Home baseline: {baseline:.3f}")
print(classification_report(y_test, pred, target_names=["Away", "Draw", "Home"]))

# Which features does the tuned XGBoost use most? (the pipeline step is named "logistic" in main.py)
xgb_step = model.named_steps["logistic"]
importance = pd.Series(xgb_step.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature importance (tuned XGBoost):")
print(importance.round(3).to_string())