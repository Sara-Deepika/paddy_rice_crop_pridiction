
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Rice Crop Production Prediction",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 Rice Crop Production Prediction")
st.write("Predict Paddy Yield using Machine Learning")

# Model File Path
MODEL_PATH = Path(__file__).parent / "rice_crop_model.pkl"

# Load Model
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("rice_crop_model.pkl file not found!")
        st.stop()

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    return model

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    print("Loaded object:", type(model))

    if isinstance(model, dict):
        print("Available keys:", model.keys())

        for key, value in model.items():
            print(key, type(value))

    return model



# Prediction
if st.button("Predict Paddy Yield"):
    try:
        input_data = pd.DataFrame({
            "Hectares": [hectares]
        })

        prediction = model.predict(input_data)

        st.success("Prediction Completed!")

        st.metric(
            label="Predicted Paddy Yield",
            value=f"{float(prediction[0]):,.2f}"
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.divider()
st.caption("Rice Crop Production Prediction | Machine Learning Project")
