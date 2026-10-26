"""Train regression (cgpa) and classification (chronotype) models on the Kaggle CSV."""
import glob
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# ---------- Load data (folder ki pehli .csv file khud utha leta hai) ----------
csv_files = glob.glob("*.csv")
if not csv_files:
    raise SystemExit("Is folder mein koi .csv file nahi mili!")
print("Using dataset:", csv_files[0])
df = pd.read_csv(csv_files[0])

os.makedirs("models", exist_ok=True)

# ---------- 1. Regression: predict cgpa ----------
REG_FEATURES = [
    "avg_sleep_hours_weekday",
    "avg_bedtime_hour",
    "pre_sleep_screen_time_min",
    "daily_screen_time_hours",
    "caffeine_drinks_per_day",
    "sleep_quality_score",
    "weekly_study_hours",
    "class_attendance_percent",
]
data = df.copy()
data[REG_FEATURES + ["cgpa"]] = data[REG_FEATURES + ["cgpa"]].apply(pd.to_numeric, errors="coerce")
data = data.dropna(subset=["cgpa"])
X, y = data[REG_FEATURES], data["cgpa"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = make_pipeline(SimpleImputer(strategy="median"), LinearRegression())
reg.fit(X_train, y_train)
r2 = r2_score(y_test, reg.predict(X_test))
print(f"Regression  R2 score    : {r2:.4f}")
joblib.dump(
    {"model": reg, "features": REG_FEATURES, "defaults": X.median().round(2).to_dict()},
    "models/regression_model.joblib",
)

# ---------- 2. Classification: predict chronotype ----------
CLF_FEATURES = [
    "avg_bedtime_hour",
    "avg_sleep_hours_weekday",
    "weekend_sleep_delta_hours",
    "pre_sleep_screen_time_min",
    "daily_screen_time_hours",
    "caffeine_drinks_per_day",
    "sleep_quality_score",
    "energy_level",
]
data = df.copy()
data[CLF_FEATURES] = data[CLF_FEATURES].apply(pd.to_numeric, errors="coerce")
data = data.dropna(subset=["chronotype"])
data["chronotype"] = data["chronotype"].astype(str).str.strip().str.title()
counts = data["chronotype"].value_counts()
data = data[data["chronotype"].isin(counts[counts >= 5].index)]  # typo wali rare labels hata do
X, y = data[CLF_FEATURES], data["chronotype"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=200, random_state=42))
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))
print(f"Classification accuracy : {acc:.4f}   (classes: {list(clf.classes_)})")
joblib.dump(
    {"model": clf, "features": CLF_FEATURES, "defaults": X.median().round(2).to_dict()},
    "models/classification_model.joblib",
)
print("Models saved in ./models/")
