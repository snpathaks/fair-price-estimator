# Fair Price Estimator 📱

A data-driven resale price estimator for used smartphones, built with **LightGBM Quantile Regression** and **SHAP explainability**.


---

## ✨ Features

- **Price Range** — predicts P10 / P50 / P90 quantiles (not just a single number)
- **Real INR Prices** — output in Indian Rupees (₹), not a meaningless normalized scale
- **Depreciation Badge** — shows % drop from original retail price
- **Confidence Indicator** — tells you how tight/wide the predicted range is
- **SHAP Explainability** — shows which specs pushed the price up or down
- **Market Distribution Chart** — visualizes where your estimate sits vs. all listings

---

## 🗂️ Project Structure

```
fair-price-estimator/
├── app/
│   └── streamlit_app.py      # Streamlit UI
├── data/
│   └── raw/
│       └── used_device_data.csv   # Dataset (Kaggle)
├── models/
│   └── model.pkl             # Trained LightGBM model bundle (3 quantiles)
├── notebooks/
│   └── 01_eda.ipynb          # Exploratory data analysis
├── src/
│   ├── __init__.py           # Package public API
│   ├── config.py             # Paths, constants, denormalize()
│   ├── preprocess.py         # Data loading, cleaning, encoding
│   ├── train.py              # Model training script
│   └── predict.py            # Inference + SHAP values
├── requirements.txt
└── render.yaml               # Render deployment config
```

---

## 🧠 Model

| Detail | Value |
|--------|-------|
| Algorithm | LightGBM (Quantile Regression) |
| Quantiles | P10 (low), P50 (median), P90 (high) |
| Features | 48 (device specs + one-hot encoded brand/OS/connectivity) |
| MAE | 0.1769 (log-price scale) |
| R² | 0.847 |

**Price normalization:** The dataset stores `log(price_INR / 1000)`. The app reverses this with `exp(x) × 1000` to show real Rupees.

---

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/fair-price-estimator.git
cd fair-price-estimator

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Train the model (only needed once)
python -m src.train

# Launch the app
streamlit run app/streamlit_app.py
```

---

## 📦 Dataset

[Used Device Price Dataset](https://www.kaggle.com/datasets/ahsan81/used-handheld-device-data) from Kaggle.

---

## 🛠️ Tech Stack

- **Python 3.14**
- **LightGBM** — gradient boosted quantile regression
- **SHAP** — model explainability
- **Streamlit** — web UI
- **pandas / scikit-learn / matplotlib**
