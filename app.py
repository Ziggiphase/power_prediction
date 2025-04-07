import streamlit as st
import altair as alt
import pickle
import numpy as np
import pandas as pd
import os

# Sidebar
choice = st.sidebar.selectbox('Select Predict per sector or multiple sector', ('Predict per sector', 'Predict per multiple sector'))

# Initialize storage for previous predictions
if 'sector_data' not in st.session_state:
    st.session_state.sector_data = pd.DataFrame(columns=["Sector", "Traffic Volume", "Power Consumption"])

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

            # Save data to session state (persistent storage)
            new_data = pd.DataFrame({"Sector": [sector_choice], "Traffic Volume": [traffic_volume], "Power Consumption": [prediction]})
            st.session_state.sector_data = pd.concat([st.session_state.sector_data, new_data], ignore_index=True)

            # Display mode status
            if prediction >= 310:
                st.markdown('<p style="color:#006400; font-size:24px; font-weight:bold;">🔥 PEAK TRAFFIC</p>', unsafe_allow_html=True)
            elif prediction >= 65:
                st.markdown('<p style="color:#008000; font-size:20px;">⚡ ACTIVATE MODE INITIALIZED</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:#FF0000; font-size:20px;">🌙 SLEEP MODE INITIALIZED</p>', unsafe_allow_html=True)

    # Reset button to clear the session state
    if st.button("🔄 Reset"):
        st.session_state.sector_data = pd.DataFrame(columns=["Sector", "Traffic Volume", "Power Consumption"])
        st.rerun()

    # Create a line chart for each sector using the saved data
    if not st.session_state.sector_data.empty:
        def create_colored_line_chart(df, x_col, y_col, title):
            color_scale = alt.Scale(
                domain=["High (>=310)", "Medium (65-309)", "Low (<65)"],
                range=["#0000FF", "#90EE90", "#FF0000"]  # Dark Blue, Light Green, Red
            )

            # Create a categorical column for color mapping
            df["Category"] = df[y_col].apply(lambda x: 
                "High (>=310)" if x >= 310 else 
                "Medium (65-309)" if x >= 65 else 
                "Low (<65)"
            )

            # Create Altair line chart
            chart = (
                alt.Chart(df)
                .mark_line(strokeWidth=3)
                .encode(
                    x=alt.X(x_col, title="Traffic Volume"),
                    y=alt.Y(y_col, title="Power Consumption"),
                    color=alt.Color("Category:N", scale=color_scale, legend=alt.Legend(title="Power Consumption Levels"))
                )
                .properties(title=title, width=600, height=400)
            )
            return chart

        # Generate charts for each sector
        chart1 = create_colored_line_chart(st.session_state.sector_data[st.session_state.sector_data["Sector"] == "s1"], "Traffic Volume", "Power Consumption", "Sector 1: Traffic vs Power Consumption")
        chart2 = create_colored_line_chart(st.session_state.sector_data[st.session_state.sector_data["Sector"] == "s2"], "Traffic Volume", "Power Consumption", "Sector 2: Traffic vs Power Consumption")
        chart3 = create_colored_line_chart(st.session_state.sector_data[st.session_state.sector_data["Sector"] == "s3"], "Traffic Volume", "Power Consumption", "Sector 3: Traffic vs Power Consumption")

        # Display charts in Streamlit
        st.altair_chart(chart1, use_container_width=True)
        st.altair_chart(chart2, use_container_width=True)
        st.altair_chart(chart3, use_container_width=True)

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
        # Make predictions
        p1 = model1.predict(pd.DataFrame(csv.iloc[:, 0]))
        p2 = model2.predict(pd.DataFrame(csv.iloc[:, 1]))
        p3 = model3.predict(pd.DataFrame(csv.iloc[:, 2]))

        # Add predictions to the DataFrame
        csv['P1'] = p1
        csv['P2'] = p2
        csv['P3'] = p3
        
        st.write(csv)

        # Function to create colored line charts
        def create_colored_line_chart(df, x_col, y_col, title):
            color_scale = alt.Scale(
                domain=["High (>=310)", "Medium (65-309)", "Low (<65)"],
                range=["#0000FF", "#90EE90", "#FF0000"]  # Dark Blue, Light Green, Red
            )

            # Create a categorical column for color mapping
            df["Category"] = df[y_col].apply(lambda x: 
                "High (>=310)" if x >= 310 else 
                "Medium (65-309)" if x >= 65 else 
                "Low (<65)"
            )

            # Create Altair line chart
            chart = (
                alt.Chart(df)
                .mark_line(strokeWidth=3)
                .encode(
                    x=alt.X(x_col, title="Traffic Volume"),
                    y=alt.Y(y_col, title="Power Consumption"),
                    color=alt.Color("Category:N", scale=color_scale, legend=alt.Legend(title="Power Consumption Levels"))
                )
                .properties(title=title, width=600, height=400)
            )
            return chart

        # Generate charts for each sector
        chart1 = create_colored_line_chart(csv, "S1", "P1", "Sector 1: Traffic vs Power Consumption")
        chart2 = create_colored_line_chart(csv, "S2", "P2", "Sector 2: Traffic vs Power Consumption")
        chart3 = create_colored_line_chart(csv, "S3", "P3", "Sector 3: Traffic vs Power Consumption")

        # Display charts in Streamlit
        st.altair_chart(chart1, use_container_width=True)
        st.altair_chart(chart2, use_container_width=True)
        st.altair_chart(chart3, use_container_width=True)

        # Download CSV Button
        st.sidebar.download_button(
            label="📥 Download Data as CSV",
            data=csv.to_csv(index=False),
            file_name="traffic_power_data.csv",
            mime="text/csv"
        )
