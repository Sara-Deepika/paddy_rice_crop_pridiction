
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Rice Crop Production Prediction",
    page_icon="🌾",
    layout="centered"
)

# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #f5f8f2;
}

h1 {
    color: #247a35;
    text-align: center;
}

.stButton > button {
    background-color: #218838;
    color: white;
    width: 100%;
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #155724;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. TITLE
# ==========================================

st.title("🌾 Rice Crop Production Prediction")

st.write(
    "Predict Paddy Yield using Machine Learning"
)

st.divider()

# ==========================================
# 4. MODEL PATH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "rice_crop_model.pkl"

# ==========================================
# 5. LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model_data = joblib.load(MODEL_PATH)

    # Select the Simple Linear Regression pipeline
    model = model_data["simple_model"]

    return model

model = load_model()

# ==========================================
# 6. USER INPUT
# ==========================================

st.subheader("🌱 Enter Agricultural Details")

hectares = st.number_input(
    "Land Area (Hectares)",
    min_value=0.01,
    max_value=100000.0,
    value=1.0,
    step=0.1
)

st.info(
    "Enter the agricultural land area to predict paddy yield."
)

# ==========================================
# 7. PREDICTION
# ==========================================

if st.button("🌾 Predict Paddy Yield"):

    try:

        input_data = pd.DataFrame({
            "Hectares": [hectares]
        })

        prediction = model.predict(input_data)

        predicted_yield = float(prediction[0])

        st.success(
            "✅ Prediction Completed Successfully!"
        )

        st.metric(
            label="Predicted Paddy Yield",
            value=f"{predicted_yield:,.2f}"
        )

    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )

# ==========================================
# 8. FOOTER
# ==========================================

st.divider()

st.caption(
    "🌾 Rice Crop Production Prediction | Machine Learning Project"
)

st.caption(
    "Developed using Python, Pandas, Scikit-learn and Streamlit"
)
