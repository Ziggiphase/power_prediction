import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Sidebar
choice = st.sidebar.selectbox('Select Predict per sector or multiple sector', ('Predict per sector', 'Predict per multiple sector'))

if choice == 'Predict per sector':
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
            st.error(f"⚠️ Model file {file} not found. Please upload it.")
            models[sector] = None

    # Data storage file
    data_file = "data.csv"

    # Load existing data or create a new DataFrame
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
    else:
        df = pd.DataFrame(columns=["Sector", "Traffic Volume", "Power Consumption"])

    # Color mapping for sectors
    sector_colors = {
        "s1": "blue",
        "s2": "green",
        "s3": "red"
    }

    # Streamlit UI
    st.title("📊 Sector Traffic vs Power Consumption Prediction App")

    # Select sector
    sector_choice = st.selectbox("Select a sector", list(model_files.keys()))

    # Input traffic volume
    traffic_volume = st.number_input(
        f"Enter traffic volume for {sector_choice}", 
        min_value=0.0, 
        value=0.0, 
        format="%.2f"
    )

    # Predict button
    if st.button("🚀 Predict Power Consumption"):
        model = models.get(sector_choice)
        
        if model:
            prediction = model.predict(np.array([[traffic_volume]]))[0]
            
            # Display prediction
            st.success(f"🔋 Predicted power consumption ({sector_choice.replace('s', 'p')}): {prediction:.2f}")

            # Save data to CSV
            new_data = pd.DataFrame({"Sector": [sector_choice], "Traffic Volume": [traffic_volume], "Power Consumption": [prediction]})
            df = pd.concat([df, new_data], ignore_index=True)
            df.to_csv(data_file, index=False)

            # Display mode status
            if prediction >= 310:
                st.markdown('<p style="color:#006400; font-size:24px; font-weight:bold;">🔥 PEAK TRAFFIC</p>', unsafe_allow_html=True)
            elif prediction >= 65:
                st.markdown('<p style="color:#008000; font-size:20px;">⚡ ACTIVATE MODE INITIALIZED</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:#FF0000; font-size:20px;">🌙 SLEEP MODE INITIALIZED</p>', unsafe_allow_html=True)
            
else:
    # Load models for multiple sector prediction
    with open("model1.pkl", 'rb') as f:
        model1 = pickle.load(f)
    with open("model2.pkl", 'rb') as f:
        model2 = pickle.load(f)
    with open("model3.pkl", 'rb') as f:
        model3 = pickle.load(f)

    st.title("📊 Multiple Sector Prediction")
    csv = st.file_uploader('Upload CSV file for prediction', type=['csv'])

    if csv is not None:
        csv = pd.read_csv(csv)
    else:
        csv = pd.DataFrame(columns=['S1', 'S2', 'S3'])

    if st.button("🚀 Predict Power Consumption"):
        p1 = model1.predict(pd.DataFrame(csv.iloc[:, 0]))
        p2 = model2.predict(pd.DataFrame(csv.iloc[:, 1]))
        p3 = model3.predict(pd.DataFrame(csv.iloc[:, 2]))

        df = pd.DataFrame({
            'S1': csv.iloc[:, 0], 'P1': p1,
            'S2': csv.iloc[:, 1], 'P2': p2,
            'S3': csv.iloc[:, 2], 'P3': p3
        })
        st.write(df)

        # Line chart for multiple sectors
        st.subheader("📈 Traffic vs Power Consumption")

        # Plot S1 vs P1, S2 vs P2, and S3 vs P3
        st.line_chart(df[['S1', 'P1']], use_container_width=True)
        st.line_chart(df[['S2', 'P2']], use_container_width=True)
        st.line_chart(df[['S3', 'P3']], use_container_width=True)

        # Download CSV Button
        st.sidebar.download_button(
            label="📥 Download Data as CSV",
            data=df.to_csv(index=False),
            file_name="traffic_power_data.csv",
            mime="text/csv"
        )
