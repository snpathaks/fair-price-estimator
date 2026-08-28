"""
src — Fair Price Estimator core package
========================================
Modules
-------
config      : Paths, column names, constants, and price denormalization helper.
preprocess  : Data loading, cleaning, encoding, and train/test splitting.
train       : Model training (LightGBM quantile regression for P10/P50/P90).
predict     : Inference — loads the saved model bundle and returns price range + SHAP values.

Public API (importable directly from `src`)
-------------------------------------------
    from src import predict_price, prepare_data, train_model, denormalize
"""

from src.config import (
    BASE_DIR,
    DATA_PATH,
    MODEL_PATH,
    TARGET_COL,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    RANDOM_STATE,
    denormalize,
)

from src.preprocess import (
    load_data,
    clean_data,
    encode_features,
    get_train_test_split,
    prepare_data,
)

from src.predict import predict_price

__all__ = [
    # config
    "BASE_DIR",
    "DATA_PATH",
    "MODEL_PATH",
    "TARGET_COL",
    "CATEGORICAL_COLS",
    "NUMERIC_COLS",
    "RANDOM_STATE",
    "denormalize",
    # preprocess
    "load_data",
    "clean_data",
    "encode_features",
    "get_train_test_split",
    "prepare_data",
    # predict
    "predict_price",
]
