# Student Sleep, Screen Time & Academic Performance: ML Web App

A small full-stack machine learning project: scikit-learn models, a FastAPI backend, and an HTML/CSS/JavaScript frontend.

**Dataset:** [Student Sleep, Screen Time & Academic Performance (Kaggle)](https://www.kaggle.com/datasets/aamir28/student-sleep-screen-time-and-academic-performance), 500 synthetic student records, 25 columns.

## Models

| Task | Target | Model | Result |
|---|---|---|---|
| Regression | `cgpa` | Linear Regression | R² = 0.3638 |
| Classification | `chronotype` | Random Forest | Accuracy = 0.73 |
| Classification | `chronotype` | Logistic Regression | Accuracy = 0.72 |
| Classification | `chronotype` | SVM (RBF) | Accuracy = 0.72 |
| Classification | `chronotype` | Naive Bayes (Gaussian) | Accuracy = 0.68 |
| Classification | `chronotype` | Decision Tree (max depth 5) | Accuracy = 0.70 |

All results are measured on a held-out test set (80% train / 20% test, `random_state=42`).

## Preprocessing

- Numeric columns are converted with `pd.to_numeric`; invalid values become missing.
- Missing values are filled with the column median (`SimpleImputer`).
- Rows with a missing target are dropped.
- Chronotype labels are stripped and title-cased; labels with fewer than 5 samples are removed.
- Features are standardised (`StandardScaler`) for Logistic Regression and SVM.

## Project structure

```
train.py            Linear Regression + Random Forest, saves models
train_more.py       Logistic Regression, SVM, Naive Bayes, Decision Tree + accuracy comparison
main.py             FastAPI backend
index.html          Frontend (HTML, CSS, JavaScript)
requirements.txt    Python libraries
models/             Saved models (.joblib)
*.csv               Dataset
```

## Setup and run

1. Install Python 3.10 or newer.
2. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
3. (Optional) Re-train the models. The trained models are already in `models/`.
   ```
   python train.py
   python train_more.py
   ```
4. Start the backend:
   ```
   python -m uvicorn main:app --reload
   ```
5. Open the app in a browser: <http://127.0.0.1:8000>
6. API documentation (Swagger): <http://127.0.0.1:8000/docs>

## API endpoints

All endpoints are `POST` and take a JSON body of numeric fields.

| Endpoint | Returns |
|---|---|
| `/predict/regression` | Predicted CGPA |
| `/predict/classification` | Chronotype (Random Forest) |
| `/predict/logistic_regression` | Chronotype |
| `/predict/svm` | Chronotype |
| `/predict/naive_bayes` | Chronotype |
| `/predict/decision_tree` | Chronotype |

`GET /info` returns the input fields, default values and model accuracies used by the frontend.

Example:
```
curl -X POST http://127.0.0.1:8000/predict/svm -H "Content-Type: application/json" \
  -d '{"avg_bedtime_hour":23.5,"avg_sleep_hours_weekday":7,"weekend_sleep_delta_hours":1,"pre_sleep_screen_time_min":45,"daily_screen_time_hours":6,"caffeine_drinks_per_day":2,"sleep_quality_score":6,"energy_level":5}'
```

## Frontend

One page with three cards: CGPA prediction (regression), chronotype prediction (Random Forest), and a **Compare Models** card that sends one input to all four extra classifiers and shows each model's prediction and accuracy side by side.
