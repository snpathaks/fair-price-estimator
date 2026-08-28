import streamlit as st
import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import importlib
import math

# Insert at position 0 so project root takes priority over any stale sys.path entries
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
importlib.invalidate_caches()

from src.predict import predict_price
from src.config import DATA_PATH, TARGET_COL, denormalize

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Fair Price Estimator", page_icon="◆", layout="centered")

# ---------- CSS ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    /* ============================================================
       1) KILL EVERY DARK-THEME SURFACE STREAMLIT MIGHT USE
       (this is what caused the solid black bar at the top)
    ============================================================ */
    html, body,
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stBottomBlockContainer"],
    section.main,
    .block-container {
        background-color: #FAFAFA !important;
        color: #111111 !important;
    }
    [data-testid="stHeader"] { background-color: transparent !important; }

    /* ============================================================
       2) TEXT FONT + COLOR — but do NOT touch icon-font spans,
       svgs, or anything inside the primary button (handled separately)
    ============================================================ */
    body, p, span, div, label, li, h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    }
    p, span, div, label, li {
        color: #111111;
    }
    [data-testid="stCaptionContainer"], .stCaption, small {
        color: #666666 !important;
    }

    /* Never touch Streamlit's icon font — this is what breaks the
       expander chevron and turns it into literal "arrow_downward" text */
    [data-testid="stIconMaterial"],
    span[class*="material-icons"],
    span[class*="material-symbols"] {
        font-family: 'Material Symbols Outlined', 'Material Symbols Rounded', 'Material Icons' !important;
        color: #111111 !important;
    }

    h1 {
        font-weight: 800;
        font-size: 2rem;
        letter-spacing: -1.5px;
        border-bottom: 3px solid #111;
        padding-bottom: 12px;
        margin-bottom: 4px;
        color: #111111 !important;
    }
    h2, h3 {
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-top: 28px;
        color: #111111 !important;
    }

    /* ============================================================
       3) PRIMARY BUTTON — set color on every descendant explicitly,
       since a plain inherited color loses to any directly-matched rule
    ============================================================ */
    .stButton>button, .stButton>button * {
        color: #FFFFFF !important;
    }
    .stButton>button {
        background-color: #111111 !important;
        border-radius: 4px;
        border: none;
        padding: 14px 30px;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton>button:hover { background-color: #333333 !important; }

    /* ============================================================
       4) EXPANDER — header bar, label text, icon
    ============================================================ */
    div[data-testid="stExpander"] {
        border: 1.5px solid #E0E0E0;
        border-radius: 6px;
        background: #FFFFFF !important;
        overflow: hidden;
    }
    div[data-testid="stExpander"] summary {
        background: #F5F5F5 !important;
    }
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary span:not([data-testid="stIconMaterial"]) {
        color: #111111 !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
    }

    /* Inputs inside expanders */
    div[data-testid="stExpander"] input,
    div[data-testid="stExpander"] [data-baseweb="select"] {
        background-color: #FFFFFF !important;
        color: #111111 !important;
    }

    /* Selectbox dropdown menu renders in a PORTAL appended to <body>,
       outside the expander — must be targeted separately */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] *,
    ul[role="listbox"],
    ul[role="listbox"] li {
        background-color: #FFFFFF !important;
        color: #111111 !important;
    }
    ul[role="listbox"] li:hover {
        background-color: #F0F0F0 !important;
    }

    /* ============================================================
       5) PRICE RESULT CARD — intentionally white text on black
    ============================================================ */
    .price-card, .price-card * {
        color: #FFFFFF !important;
    }
    .price-card {
        background: #111111 !important;
        border-radius: 8px;
        padding: 32px 28px 24px;
        text-align: center;
        margin: 20px 0 16px;
    }
    .price-label { font-size: 11px; letter-spacing: 3px; text-transform: uppercase; color: #AAAAAA !important; margin-bottom: 8px; }
    .price-main { font-size: 52px; font-weight: 800; letter-spacing: -2px; line-height: 1; }
    .price-sub { font-size: 13px; color: #AAAAAA !important; margin-top: 10px; letter-spacing: 1px; }
    .price-range { display: flex; justify-content: center; gap: 24px; margin-top: 18px; padding-top: 18px; border-top: 1px solid #333; }
    .range-item { text-align: center; }
    .range-val { font-size: 18px; font-weight: 700; }
    .range-lbl { font-size: 10px; color: #888888 !important; letter-spacing: 2px; text-transform: uppercase; margin-top: 2px; }

    .dep-badge {
        display: inline-block;
        background: #F3F3F3 !important;
        border: 1.5px solid #E0E0E0;
        border-radius: 20px;
        padding: 6px 16px;
        font-size: 13px;
        font-weight: 600;
        color: #111111 !important;
        margin-top: 4px;
    }

    .conf-label {
        font-size: 11px;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        font-weight: 700;
        color: #111111 !important;
        margin-top: 20px;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ---------- HEADER ----------
st.markdown("<h1>FAIR PRICE ESTIMATOR</h1>", unsafe_allow_html=True)
st.caption("Data-driven resale valuation for used smartphones · Prices shown in Indian Rupees (₹)")

st.markdown("---")

# ---------- SECTION 1: DEVICE INFO ----------
with st.expander("① DEVICE INFO", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        device_brand = st.selectbox("Brand", ["Honor", "Apple", "Samsung", "Xiaomi", "Other"])
        os_type = st.selectbox("Operating System", ["Android", "iOS", "Windows", "Other"])
        release_year = st.number_input("Release Year", min_value=2010, max_value=2025, value=2020, step=1)
    with col2:
        screen_size = st.number_input("Screen Size (cm)", min_value=5.0, max_value=25.0, value=15.0)
        weight = st.number_input("Weight (g)", min_value=50.0, max_value=500.0, value=180.0)
        internal_memory = st.number_input("Internal Memory (GB)", min_value=4.0, max_value=1024.0, value=64.0)

# ---------- SECTION 2: CONDITION ----------
with st.expander("② CONDITION & PRICING", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        days_used = st.number_input("Days Used", min_value=0, max_value=3650, value=200, step=1)
        battery = st.number_input("Battery Capacity (mAh)", min_value=500.0, max_value=10000.0, value=4000.0)
    with col2:
        ram = st.number_input("RAM (GB)", min_value=1.0, max_value=64.0, value=4.0)
        new_price_inr = st.number_input(
            "Original New Price (₹)",
            min_value=1000, max_value=500000, value=15000, step=1000,
            help="The retail price when this phone was new, in Indian Rupees."
        )
        # Convert INR → normalized log scale for the model
        normalized_new_price = math.log(new_price_inr / 1000)

# ---------- SECTION 3: CONNECTIVITY & CAMERA ----------
with st.expander("③ CONNECTIVITY & CAMERA", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        support_4g = st.selectbox("4G Support", ["yes", "no"])
        rear_camera_mp = st.number_input("Rear Camera (MP)", min_value=0.0, max_value=200.0, value=13.0)
    with col2:
        support_5g = st.selectbox("5G Support", ["yes", "no"])
        front_camera_mp = st.number_input("Front Camera (MP)", min_value=0.0, max_value=100.0, value=8.0)

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

    with st.spinner("Calculating fair price..."):
        result = predict_price(input_data)

    low_inr  = result["low_inr"]
    mid_inr  = result["mid_inr"]
    high_inr = result["high_inr"]
    low_log  = result["low"]
    mid_log  = result["mid"]
    high_log = result["high"]

    # Depreciation %
    depr_pct = ((new_price_inr - mid_inr) / new_price_inr) * 100

    def fmt_inr(v):
        """Format as ₹1,23,456"""
        return f"₹{int(round(v)):,}"

    # ---------- PRICE CARD ----------
    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">ESTIMATED FAIR PRICE</div>
        <div class="price-main">{fmt_inr(mid_inr)}</div>
        <div class="price-sub">Median · Quantile Regression Model</div>
        <div class="price-range">
            <div class="range-item">
                <div class="range-val">{fmt_inr(low_inr)}</div>
                <div class="range-lbl">Low (P10)</div>
            </div>
            <div class="range-item">
                <div class="range-val">{fmt_inr(high_inr)}</div>
                <div class="range-lbl">High (P90)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------- DEPRECIATION BADGE ----------
    dep_label = f"📉 Depreciated {depr_pct:.1f}% from new price ({fmt_inr(new_price_inr)})"
    st.markdown(f'<div class="dep-badge">{dep_label}</div>', unsafe_allow_html=True)

    # ---------- CONFIDENCE ----------
    spread = high_log - low_log
    if spread < 0.3:
        confidence, fill = "HIGH CONFIDENCE", 90
    elif spread < 0.6:
        confidence, fill = "MODERATE CONFIDENCE", 60
    else:
        confidence, fill = "LOW CONFIDENCE", 30

    st.markdown(f'<div class="conf-label">{confidence}</div>', unsafe_allow_html=True)
    st.progress(fill)

    st.markdown("---")

    # ---------- MARKET DISTRIBUTION CHART ----------
    st.markdown("<h3>MARKET DISTRIBUTION</h3>", unsafe_allow_html=True)

    df_raw = pd.read_csv(DATA_PATH)
    inr_prices = df_raw[TARGET_COL].dropna().apply(denormalize)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.hist(inr_prices, bins=50, color="#111111", alpha=0.80, zorder=2)
    ax.axvline(mid_inr, color="#111", linestyle="--", linewidth=2, zorder=3, label=f"Your estimate: {fmt_inr(mid_inr)}")
    ax.axvspan(low_inr, high_inr, color="#555555", alpha=0.15, zorder=1, label="P10–P90 band")

    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{int(x/1000)}K"))
    ax.set_xlabel("Resale Price (₹)", fontsize=10)
    ax.set_title("Where your estimate falls vs. all market listings", fontsize=11, fontweight="bold", pad=10)
    ax.legend(fontsize=9, frameon=False)

    st.pyplot(fig)
    st.caption("Dashed line = your predicted price · Shaded band = P10–P90 confidence range")

    st.markdown("---")

    # ---------- SHAP EXPLAINABILITY ----------
    st.markdown("<h3>WHY THIS PRICE?</h3>", unsafe_allow_html=True)
    st.caption("SHAP values show how each feature pushed the price up or down from the average.")

    shap_vals   = np.array(result["shap_values"])
    feat_names  = result["feature_names"]
    base_val    = result["base_value"]

    # Top 8 features by absolute SHAP impact
    indices = np.argsort(np.abs(shap_vals))[::-1][:8]
    top_names  = [feat_names[i] for i in indices]
    top_shaps  = [shap_vals[i] for i in indices]

    colors = ["#111111" if v > 0 else "#888888" for v in top_shaps]
    labels = [
        f"{n.replace('_', ' ').title()}  ({'+' if v>0 else ''}{v:.3f})"
        for n, v in zip(top_names, top_shaps)
    ]

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.barh(range(len(top_names)), top_shaps[::-1], color=colors[::-1], height=0.6)
    ax2.set_yticks(range(len(top_names)))
    ax2.set_yticklabels([l for l in reversed(labels)], fontsize=9)
    ax2.axvline(0, color="#333", linewidth=1)
    ax2.set_xlabel("SHAP Value (log-price impact)", fontsize=9)
    ax2.set_title("Feature Impact on Predicted Price", fontsize=11, fontweight="bold", pad=10)
    ax2.set_facecolor("#FAFAFA")
    fig2.patch.set_facecolor("#FAFAFA")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    st.pyplot(fig2)
    st.caption("Dark bars = pushed price UP · Grey bars = pushed price DOWN · Values in log-price units")

    st.markdown("---")
    st.caption("Model: LightGBM Quantile Regression (P10/P50/P90) · Dataset: Kaggle Used Device Price Data · Prices in INR")