import streamlit as st
import pickle
import numpy as np

# Load models
model_files = {
    "s1": "model1.pkl",
    "s2": "model2.pkl",
    "s3": "model3.pkl"
}

models = {}
for sector, file in model_files.items():
    try:
        with open(file, 'rb') as f:
            models[sector] = pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model file {file} not found. Please upload it.")
        models[sector] = None

# Streamlit UI
st.title("Sector Traffic vs Power Consumption Prediction App")

# Select sector
sector_choice = st.selectbox("Select a sector", ["s1", "s2", "s3"], index=0)

# Input traffic volume
traffic_volume = st.number_input(f"Enter traffic volume for {sector_choice}", min_value=0.0, format="%.2f")

# Predict button
if st.button("Predict Power Consumption"):
    model = models.get(sector_choice)
    if model:
        prediction = model.predict(np.array([[traffic_volume]]))
        if prediction[0]>=70:
            st.success(f"Predicted power consumption ({sector_choice.replace('s', 'p')}): {prediction[0]:.2f}")
            st.write("POWER MODE INITIALIZED")
            
        elif prediction[0]>=310:
            st.success(f"Predicted power consumption ({sector_choice.replace('s', 'p')}): {prediction[0]:.2f}")
            st.write("PEAK TRAFFIC")
            
        else:
            #st.error("Model not loaded. Please check your model files.")
            st.write("SLEEP MODE INITIALIZED")
    else:
        st.error("Model not loaded, Please check your model files.")
        
