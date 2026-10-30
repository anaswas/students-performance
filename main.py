import os

import joblib
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="Student ML App")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Pehle wale do models (train.py)
reg_bundle = joblib.load("models/regression_model.joblib")
clf_bundle = joblib.load("models/classification_model.joblib")   # Random Forest

# Naye models (train_more.py) - agar file nahi bani to app phir bhi chalti hai
EXTRA = {}
for key in ["logistic_regression", "svm", "naive_bayes", "decision_tree"]:
    path = f"models/{key}.joblib"
    if os.path.exists(path):
        EXTRA[key] = joblib.load(path)


def to_frame(bundle, data: dict) -> pd.DataFrame:
    missing = [f for f in bundle["features"] if f not in data]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing fields: {missing}")
    try:
        row = [float(data[f]) for f in bundle["features"]]
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="All inputs must be numbers")
    return pd.DataFrame([row], columns=bundle["features"])


@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/info")
def info():
    """Frontend isse form fields, default values aur model list leta hai."""
    return {
        "regression": {"features": reg_bundle["features"], "defaults": reg_bundle["defaults"]},
        "classification": {"features": clf_bundle["features"], "defaults": clf_bundle["defaults"]},
        "compare": [
            {"key": k, "label": b["label"], "accuracy": b["accuracy"]} for k, b in EXTRA.items()
        ],
    }


@app.post("/predict/regression")
def predict_regression(data: dict = Body(...)):
    pred = float(reg_bundle["model"].predict(to_frame(reg_bundle, data))[0])
    return {"prediction": round(pred, 2), "target": "cgpa"}


@app.post("/predict/classification")
def predict_classification(data: dict = Body(...)):
    pred = str(clf_bundle["model"].predict(to_frame(clf_bundle, data))[0])
    return {"prediction": pred, "target": "chronotype", "model": "Random Forest"}


def make_endpoint(key):
    def endpoint(data: dict = Body(...)):
        bundle = EXTRA[key]
        pred = str(bundle["model"].predict(to_frame(bundle, data))[0])
        return {"prediction": pred, "target": "chronotype", "model": bundle["label"]}
    return endpoint


# /predict/logistic_regression, /predict/svm, /predict/naive_bayes, /predict/decision_tree
for _key in EXTRA:
    app.post(f"/predict/{_key}")(make_endpoint(_key))
