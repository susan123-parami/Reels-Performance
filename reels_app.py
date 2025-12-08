import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# -------------------------
# Load viral model
# -------------------------
def load_viral_model():
    with open("viral_model.pkl", "rb") as f:
        viral_model, viral_scaler, viral_numeric_cols, viral_feature_cols = pickle.load(f)
    return viral_model, viral_scaler, viral_numeric_cols, viral_feature_cols


# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Reel Viral Predictor", page_icon="🌸", layout="wide")
st.sidebar.write("Student: May Thaw Tar 🌷")
st.sidebar.write("Reel Analysis Dashboard")

st.title("🌸 Reel Viral Predictor")
st.write("Enter reel information below to predict virality.")

# -------------------------
# User Inputs
# -------------------------
col1, col2 = st.columns(2)

with col1:
    length = st.slider("Reel length (seconds)", 1, 60, 10)
    hashtags = st.slider("Number of hashtags", 0, 30, 5)
    creator_size = st.number_input("Creator followers (in thousands)", 0, 5000, 10)

with col2:
    category = st.selectbox("Reel Category", ["Dance", "Food", "Travel", "Education", "Comedy", "Lifestyle"])
    posting_hour = st.slider("Posting hour (0–23)", 0, 23, 14)

col3, col4 = st.columns(2)

with col3:
    niche = st.selectbox("Niche (specific)", ["Dance", "Beauty", "Food", "Gaming", "Fitness", "Other"])
    hook_strength_score = st.slider("Hook Strength (0–10)", 0.0, 10.0, 5.0, step=0.5) / 10.0

with col4:
    music_type = st.selectbox("Music Type", ["original", "trending", "licensed", "other"])
    retention_rate = st.slider("Retention rate (%)", 0, 100, 45) / 100.0


# -------------------------
# Helper: Build model-ready DataFrame
# -------------------------
def build_input_dict():
    d = {
        "duration_sec": float(length),
        "hook_strength_score": float(hook_strength_score),
        "niche": niche,
        "upload_time": int(posting_hour),
        "music_type": music_type,
        "retention_rate": float(retention_rate),
        "first_3_sec_engagement": min(1.0, 0.02 * creator_size + 0.01 * hashtags),
        "views_first_hour": float(creator_size) * 100.0,
        "views_total": float(creator_size) * 500.0,
    }
    return d


# -------------------------
# Prediction (viral only)
# -------------------------
def safe_predict_viral():
    try:
        viral_model, viral_scaler, viral_numeric_cols, viral_feature_cols = load_viral_model()
    except Exception as e:
        st.error(f"Error loading viral_model.pkl: {e}")
        return

    base_input = build_input_dict()
    df = pd.DataFrame([base_input])

    # Determine expected columns
    expected = None
    try:
        if viral_feature_cols is not None:
            expected = list(viral_feature_cols)
    except:
        expected = None

    # Reindex DataFrame if model expects specific features
    if expected:
        X = df.reindex(columns=expected, fill_value=0)
    else:
        X = df.copy()

    # --- FIX: Avoid scaling error ---
    try:
        numeric_cols = list(viral_numeric_cols) if viral_numeric_cols is not None else []
    except:
        numeric_cols = []

    if viral_scaler is not None and len(numeric_cols) > 0:
        present_num = [c for c in numeric_cols if c in X.columns]
        if len(present_num) > 0:
            try:
                X.loc[:, present_num] = viral_scaler.transform(X[present_num])
            except Exception as e:
                st.warning(f"Scaling skipped for viral model: {e}")
    # --------------------------------

    # Predict viral probability
    try:
        if hasattr(viral_model, "predict_proba"):
            prob = viral_model.predict_proba(X)[:, 1][0]
            st.success(f"Probability of going viral: {prob:.2%}")
        else:
            pred = viral_model.predict(X)[0]
            st.success(f"Model output: {pred}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.dataframe(X)


# -------------------------
# BUTTON
# -------------------------
if st.button("Predict Viral"):
    safe_predict_viral()
