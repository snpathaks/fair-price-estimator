import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA_PATH, TARGET_COL, CATEGORICAL_COLS, NUMERIC_COLS, RANDOM_STATE

def load_data():
    return pd.read_csv(DATA_PATH)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=[TARGET_COL])
    for col in NUMERIC_COLS:
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("Unknown")
    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    return df

def get_train_test_split(df: pd.DataFrame):
    feature_cols = NUMERIC_COLS + [c for c in df.columns if c not in NUMERIC_COLS + [TARGET_COL]]
    X = df[feature_cols]
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

def prepare_data():
    df = load_data()
    df = clean_data(df)
    df = encode_features(df)
    return get_train_test_split(df)