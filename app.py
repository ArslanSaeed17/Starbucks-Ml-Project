
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Starbucks Calorie Predictor",
    page_icon="☕",
    layout="centered"
)

st.title("☕ Starbucks Calorie Predictor")
st.write(
    "Select a beverage category and preparation type "
    "to estimate the drink's calories."
)

MODEL_PATH = Path(__file__).parent / "model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("model.pkl was not found. Add it to the project folder.")
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

st.info(
    "Choose category and preparation values that match "
    "the categories in your training dataset."
)

category = st.text_input(
    "Beverage category",
    placeholder="e.g. Coffee"
)

preparation = st.text_input(
    "Beverage preparation",
    placeholder="e.g. Short"
)

if st.button("Predict Calories", type="primary"):
    if not category.strip() or not preparation.strip():
        st.warning("Please enter both values.")
    else:
        input_data = pd.DataFrame([{
            "Beverage_category": category.strip(),
            "Beverage_prep": preparation.strip()
        }])

        try:
            prediction = model.predict(input_data)[0]
            st.metric(
                label="Estimated Calories",
                value=f"{max(0, prediction):.1f} kcal"
            )
            st.caption(
                "This is a machine-learning estimate, "
                "not official Starbucks nutritional information."
            )
        except Exception as error:
            st.error(f"Prediction failed: {error}")
