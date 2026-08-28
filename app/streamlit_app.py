import streamlit as st
import sys, os
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict_price
from src.config import DATA_PATH, TARGET_COL

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Fair Price Estimator", page_icon="◆", layout="centered")

# ---------- MONOCHROME CSS ----------
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    .main {
        background-color: #FFFFFF;
    }
    h1 {
        font-weight: 800;
        letter-spacing: -1px;
        border-bottom: 3px solid #000000;
        padding-bottom: 12px;
    }
    h2, h3 {
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-top: 30px;
    }
    .stButton>button {
        background-color: #000000;
        color: #FFFFFF;
        border-radius: 0px;
        border: none;
        padding: 12px 30px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #333333;
        color: #FFFFFF;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #000000;
        border-radius: 0px;
    }
    .price-box {
        border: 2px solid #000000;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }
    .price-main {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .price-range {
        font-size: 14px;
        color: #555555;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 8px;
    }
    .confidence-label {
        font-size: 12px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #000000;
        font-weight: 700;
        margin-top: 25px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<h1>FAIR PRICE ESTIMATOR</h1>", unsafe_allow_html=True)
st.caption("A data-driven valuation for used devices — built on real resale market data.")

# ---------- SECTION 1: DEVICE INFO ----------
with st.expander("① DEVICE INFO", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        device_brand = st.selectbox("Brand", ["Honor", "Apple", "Samsung", "Xiaomi", "Other"])
        os_type = st.selectbox("Operating System", ["Android", "iOS", "Windows", "Other"])
        release_year = st.number_input("Release Year", value=2020, step=1)
    with col2:
        screen_size = st.number_input("Screen Size (cm)", value=15.0)
        weight = st.number_input("Weight (g)", value=180.0)
        internal_memory = st.number_input("Internal Memory (GB)", value=64.0)

# ---------- SECTION 2: CONDITION ----------
with st.expander("② CONDITION", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        days_used = st.number_input("Days Used", value=200, step=1)
        battery = st.number_input("Battery Capacity (mAh)", value=4000.0)
    with col2:
        ram = st.number_input("RAM (GB)", value=4.0)
        normalized_new_price = st.number_input("Original Price (normalized scale)", value=5.0)

# ---------- SECTION 3: CONNECTIVITY & CAMERA ----------
with st.expander("③ CONNECTIVITY & CAMERA", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        support_4g = st.selectbox("4G Support", ["yes", "no"])
        rear_camera_mp = st.number_input("Rear Camera (MP)", value=13.0)
    with col2:
        support_5g = st.selectbox("5G Support", ["yes", "no"])
        front_camera_mp = st.number_input("Front Camera (MP)", value=8.0)

st.markdown("<br>", unsafe_allow_html=True)

# ---------- PREDICT ----------
if st.button("ESTIMATE FAIR PRICE"):
    input_data = {
        "device_brand": device_brand,
        "os": os_type,
        "screen_size": screen_size,
        "4g": support_4g,
        "5g": support_5g,
        "rear_camera_mp": rear_camera_mp,
        "front_camera_mp": front_camera_mp,
        "internal_memory": internal_memory,
        "ram": ram,
        "battery": battery,
        "weight": weight,
        "release_year": release_year,
        "days_used": days_used,
        "normalized_new_price": normalized_new_price,
    }

    result = predict_price(input_data)
    low, mid, high = result["low"], result["mid"], result["high"]
    spread = high - low

    # ---------- RESULT BOX ----------
    st.markdown(f"""
    <div class="price-box">
        <div class="price-main">{mid:.3f}</div>
        <div class="price-range">RANGE&nbsp;&nbsp;{low:.3f} &nbsp;—&nbsp; {high:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

    # ---------- CONFIDENCE INDICATOR ----------
    if spread < 0.3:
        confidence, fill = "HIGH CONFIDENCE", 90
    elif spread < 0.6:
        confidence, fill = "MODERATE CONFIDENCE", 60
    else:
        confidence, fill = "LOW CONFIDENCE", 30

    st.markdown(f'<div class="confidence-label">{confidence}</div>', unsafe_allow_html=True)
    st.progress(fill)

    # ---------- DISTRIBUTION CHART ----------
    st.markdown("<h3>MARKET DISTRIBUTION</h3>", unsafe_allow_html=True)

    df = pd.read_csv(DATA_PATH)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(df[TARGET_COL], bins=40, color="#000000", alpha=0.85)
    ax.axvline(mid, color="#000000", linestyle="--", linewidth=2)
    ax.axvspan(low, high, color="#000000", alpha=0.1)

    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel("Normalized Used Price", fontsize=10)
    ax.set_title("Where this estimate sits vs. all listings", fontsize=11, fontweight="bold")

    st.pyplot(fig)

    st.caption("Dashed line = predicted price · Shaded band = 10th–90th percentile confidence range")