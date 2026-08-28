import joblib
import pandas as pd
from src.config import MODEL_PATH

def load_model():
    return joblib.load(MODEL_PATH)

def predict_price(input_dict: dict):
    bundle = load_model()
    models, columns = bundle["models"], bundle["columns"]

    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    low = models["low"].predict(df)[0]
    mid = models["mid"].predict(df)[0]
    high = models["high"].predict(df)[0]

    return {"low": low, "mid": mid, "high": high}