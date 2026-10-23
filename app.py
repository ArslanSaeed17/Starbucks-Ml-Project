
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Starbucks Calories Predictor",
    page_icon="☕"
)

# Load the trained machine learning model
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "starbucks_calories_model.pkl"
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model load nahi hua: {e}")
    st.stop()

st.title("☕ Starbucks Calories Predictor")
st.write("Enter nutritional values to predict calories.")

with st.form("prediction_form"):
    trans_fat = st.number_input("Trans Fat (g)", min_value=0.0)
    saturated_fat = st.number_input("Saturated Fat (g)", min_value=0.0)
    sodium = st.number_input("Sodium (mg)", min_value=0.0)
    carbs = st.number_input("Total Carbohydrates (g)", min_value=0.0)
    cholesterol = st.number_input("Cholesterol (mg)", min_value=0.0)
    fibre = st.number_input("Dietary Fibre (g)", min_value=0.0)
    sugars = st.number_input("Sugars (g)", min_value=0.0)
    protein = st.number_input("Protein (g)", min_value=0.0)

    submitted = st.form_submit_button("Predict Calories")

if submitted:
    feature_names = getattr(model, "feature_names_in_", None)

    if feature_names is None:
        st.error("Model ke feature names available nahi hain.")
    else:
        values = {
            "Trans Fat (g) ": trans_fat,
            "Saturated Fat (g)": saturated_fat,
            " Sodium (mg)": sodium,
            " Total Carbohydrates (g) ": carbs,
            "Cholesterol (mg)": cholesterol,
            " Dietary Fibre (g)": fibre,
            " Sugars (g)": sugars,
            " Protein (g) ": protein
        }

        try:
            features = pd.DataFrame(
                [[values[name] for name in feature_names]],
                columns=feature_names
            )
            prediction = model.predict(features)[0]
            st.success(f"Predicted Calories: {prediction:.1f}")
        except Exception as e:
            st.error(f"Prediction mein error: {e}")
