from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "used_device_data.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

TARGET_COL = "normalized_used_price"

CATEGORICAL_COLS = ["device_brand", "os", "4g", "5g"]

NUMERIC_COLS = [
    "screen_size",
    "rear_camera_mp",
    "front_camera_mp",
    "internal_memory",
    "ram",
    "battery",
    "weight",
    "release_year",
    "days_used",
    "normalized_new_price",
]

RANDOM_STATE = 42