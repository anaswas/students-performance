"""Extra models on the SAME dataset: Logistic Regression, SVM, Naive Bayes, Decision Tree.
(Linear Regression aur Random Forest pehle train.py mein ho chuke hain.)"""
import glob

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ---------- Load + preprocess (train.py jaisa hi, taake split same rahe) ----------
csv_files = glob.glob("*.csv")
if not csv_files:
    raise SystemExit("Is folder mein koi .csv file nahi mili!")
df = pd.read_csv(csv_files[0])

FEATURES = [
    "avg_bedtime_hour", "avg_sleep_hours_weekday", "weekend_sleep_delta_hours",
    "pre_sleep_screen_time_min", "daily_screen_time_hours",
    "caffeine_drinks_per_day", "sleep_quality_score", "energy_level",
]
df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=["chronotype"])
df["chronotype"] = df["chronotype"].astype(str).str.strip().str.title()   # "F"/"Female" jaisi ganda labels saaf
counts = df["chronotype"].value_counts()
df = df[df["chronotype"].isin(counts[counts >= 5].index)]
X, y = df[FEATURES], df["chronotype"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
defaults = X.median().round(2).to_dict()

# ---------- Models ----------
# Missing values -> median; Logistic Regression aur SVM ke liye scaling bhi zaroori hai
models = {
    "logistic_regression": ("Logistic Regression",
        make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))),
    "svm": ("SVM",
        make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVC(kernel="rbf"))),
    "naive_bayes": ("Naive Bayes",
        make_pipeline(SimpleImputer(strategy="median"), GaussianNB())),
    "decision_tree": ("Decision Tree",
        make_pipeline(SimpleImputer(strategy="median"), DecisionTreeClassifier(max_depth=5, random_state=42))),
}

results = {}
for key, (label, pipe) in models.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[label] = acc
    joblib.dump({"model": pipe, "features": FEATURES, "defaults": defaults,
                 "label": label, "accuracy": round(acc, 4)}, f"models/{key}.joblib")
    if label == "SVM":
        svm_pred = pred

# Random Forest (train.py se) ko bhi comparison mein dikhao
rf = joblib.load("models/classification_model.joblib")["model"]
results["Random Forest"] = accuracy_score(y_test, rf.predict(X_test))

print("\n===== Accuracy comparison (target: chronotype) =====")
for name, acc in sorted(results.items(), key=lambda kv: -kv[1]):
    print(f"{name:<22}: {acc:.4f}")
print("\nSVM classification report:")
print(classification_report(y_test, svm_pred))
print("Models saved in ./models/")
