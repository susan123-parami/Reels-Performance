import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt



def load_viral_model():
    with open("viral_model.pkl", "rb") as f:
        viral_model, viral_scaler, viral_numeric_cols, viral_feature_cols = pickle.load(f)
    return viral_model, viral_scaler, viral_numeric_cols, viral_feature_cols



def load_likes_model():
    with open("likes_model.pkl", "rb") as f:
        likes_model, likes_scaler, likes_numeric_cols, likes_feature_cols = pickle.load(f)
    return likes_model, likes_scaler, likes_numeric_cols, likes_feature_cols


#design

st.set_page_config(
    page_title="Reel Viral Predictor",
    page_icon="🌸",
    layout="wide"
)

#sidebar

st.sidebar.write("Student: May Thaw Tar 🌷")
st.sidebar.write("Reel Analysis Dashboard")

#Title

st.title("🌸 Reel Viral Predictor")
st.write("Enter reel information below to predict virality & likes.")

#user input

col1, col2 = st.columns(2)

with col1:
    length = st.slider("Reel length (seconds)", 1, 60, 10)
    hashtags = st.slider("Number of hashtags", 0, 30, 5)
    creator_size = st.number_input("Creator followers (in thousands)", 0, 5000, 10)

with col2:
    category = st.selectbox(
        "Reel Category",
        ["Dance", "Food", "Travel", "Education", "Comedy", "Lifestyle"]
    )
    posting_hour = st.slider("Posting hour (0–23)", 0, 23, 14)

# Encode category

category_map = {
    "Dance": 0, "Food": 1, "Travel": 2,
    "Education": 3, "Comedy": 4, "Lifestyle": 5
}
category_encoded = category_map[category]

input_data = [[
    length,
    hashtags,
    posting_hour,
    creator_size,
    category_encoded
]]


#predict button

if st.button("Predict Viral", key="btn_viral"):
    ...

if st.button("Predict Likes", key="btn_likes"):
    ...

    viral_model, viral_scaler, viral_numeric_cols, viral_feature_cols = load_viral_model()
    likes_model, likes_scaler, likes_numeric_cols, likes_feature_cols = load_likes_model()

    viral_prob = viral_model.predict(input_data)[0][1]
    likes_pred = likes_model.predict(input_data)[0]

    st.success(f"Probability of going viral: {viral_prob:.2f}")
    st.success(f"Predicted likes: {int(likes_pred)} likes")