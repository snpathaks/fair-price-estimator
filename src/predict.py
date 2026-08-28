import joblib
import shap
import pandas as pd
from src.config import MODEL_PATH, denormalize

def load_model():
    return joblib.load(MODEL_PATH)

def predict_price(input_dict: dict):
    bundle = load_model()
    models, columns = bundle["models"], bundle["columns"]

    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    low_log  = models["low"].predict(df)[0]
    mid_log  = models["mid"].predict(df)[0]
    high_log = models["high"].predict(df)[0]

    # SHAP explainability on the median (mid) model
    explainer = shap.TreeExplainer(models["mid"])
    shap_vals = explainer.shap_values(df)

    return {
        # Raw log-scale values
        "low":  low_log,
        "mid":  mid_log,
        "high": high_log,
        # Human-readable INR prices
        "low_inr":  denormalize(low_log),
        "mid_inr":  denormalize(mid_log),
        "high_inr": denormalize(high_log),
        # SHAP explainability
        "shap_values":   shap_vals[0].tolist(),
        "feature_names": list(columns),
        "base_value":    explainer.expected_value,
    }