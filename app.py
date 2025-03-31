import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

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

        # Display status
        if prediction >= 310:
            st.markdown('<p style="color:#006400; font-size:24px; font-weight:bold;">🔥 PEAK TRAFFIC</p>', unsafe_allow_html=True)
        elif prediction >= 65:
            st.markdown('<p style="color:#008000; font-size:20px;">⚡ ACTIVATE MODE INITIALIZED</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#FF0000; font-size:20px;">🌙 SLEEP MODE INITIALIZED</p>', unsafe_allow_html=True)

# Sidebar: Filter and chart options
st.sidebar.header("⚙️ Visualization Settings")

# Select sector to visualize (or all)
sector_filter = st.sidebar.selectbox("Select sector to view", ["All"] + list(model_files.keys()))

# Select chart type
chart_type = st.sidebar.radio("Choose chart type:", ["Line Chart", "Bar Chart", "Scatter Plot"])

# Select color theme
color_theme = st.sidebar.selectbox("Choose chart color:", ["blue", "green", "red", "purple", "orange"])

# Clear Data Button
if st.sidebar.button("🗑️ Reset Data"):
    df = pd.DataFrame(columns=["Sector", "Traffic Volume", "Power Consumption"])
    df.to_csv(data_file, index=False)
    st.sidebar.success("Data cleared!")
    st.experimental_rerun()  # Refresh page

# Download CSV Button
if not df.empty:
    st.sidebar.download_button(
        label="📥 Download Data as CSV",
        data=df.to_csv(index=False),
        file_name="traffic_power_data.csv",
        mime="text/csv"
    )

# Visualization
st.subheader("📈 Traffic Volume vs Power Consumption Trends")

if not df.empty:
    # Filter data based on selection
    if sector_filter != "All":
        df = df[df["Sector"] == sector_filter]

    fig, ax = plt.subplots()

    # Plot based on chart type
    for sector in df["Sector"].unique():
        sector_data = df[df["Sector"] == sector]

        if chart_type == "Line Chart":
            ax.plot(sector_data["Traffic Volume"], sector_data["Power Consumption"], 
                    marker='o', linestyle='-', color=color_theme, label=sector)
        elif chart_type == "Bar Chart":
            ax.bar(sector_data["Traffic Volume"], sector_data["Power Consumption"], color=color_theme, label=sector)
        elif chart_type == "Scatter Plot":
            ax.scatter(sector_data["Traffic Volume"], sector_data["Power Consumption"], color=color_theme, label=sector)

    ax.set_xlabel("Traffic Volume")
    ax.set_ylabel("Power Consumption")
    ax.set_title(f"Traffic vs Power Consumption ({sector_filter})")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("No data available yet. Make a prediction to start visualizing trends.")
