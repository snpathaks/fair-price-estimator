import joblib
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
from src.preprocess import prepare_data
from src.config import MODEL_PATH

def train_model():
    X_train, X_test, y_train, y_test = prepare_data()

    models = {}
    for name, alpha in [("low", 0.1), ("mid", 0.5), ("high", 0.9)]:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42
        )
        model.fit(X_train, y_train)
        models[name] = model

    preds_mid = models["mid"].predict(X_test)
    mae = mean_absolute_error(y_test, preds_mid)
    r2 = r2_score(y_test, preds_mid)
    print(f"MAE (log-price scale): {mae:.4f} | R2: {r2:.3f}")

    joblib.dump({"models": models, "columns": X_train.columns.tolist()}, MODEL_PATH)
    print(f"Models saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()