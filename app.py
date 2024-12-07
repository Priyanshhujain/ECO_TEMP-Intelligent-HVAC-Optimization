import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import joblib

# Load the pre-trained RandomForest model from the .pkl file
rf_model = joblib.load('C:/Users/hp/OneDrive/Desktop/Ac_optimization/rf_model.pkl')

# Create synthetic training data for the placeholder model (replace with your real model)
X_train = np.random.rand(100, 2) * 50  # Random indoor and outdoor temperatures (features)
y_train = X_train[:, 0] * 0.6 + X_train[:, 1] * 0.4 + np.random.rand(100) * 5  # Simulated desired temperatures (target)
rf_model.fit(X_train, y_train)

# Streamlit UI
st.title("HVAC Temperature Predictor")
st.sidebar.header("Settings")

# Select Prediction Mode (AI or User Preference)
mode = st.sidebar.radio("Choose Prediction Mode", ("AI", "User Preference"))

# Define a default value for user preference temperature
user_preference_temp = None

if mode == "User Preference":
    # User Preference Input
    user_preference_temp = st.sidebar.number_input(
        "Enter your preferred desired temperature (°C)", min_value=16.0, max_value=21.0, value=20.0, step=0.5
    )

# File uploader
uploaded_file = st.sidebar.file_uploader("C:/Users/hp/OneDrive/Desktop/Ac_optimization/balanced_ntc_sensor_data_70.csv", type=["csv"])

# Set the update speed
update_speed = st.sidebar.slider("Update Speed (seconds per step)", 0.1, 5.0, 1.0)

if uploaded_file is not None:
    # Load the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.write(data.head())

    # Check if required columns are present
    if "Indoor Temperature (°C)" in data.columns and "Outdoor Temperature (°C)" in data.columns:
        predicted_temps = []  # List to store predicted temperatures
        indoor_temp_data = []
        outdoor_temp_data = []
        adjusted_temps = []  # For gradual adjustments (User Preference mode)
        timestamps = list(range(1, len(data) + 1))

        # Create a Plotly figure for live chart updates
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", name="Indoor Temperature"))
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", name="Outdoor Temperature"))
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", name="Predicted Temp"))

        chart_placeholder = st.empty()  # Placeholder for the chart
        prediction_placeholder = st.empty()  # Placeholder for the prediction output

        # Iterate through each row in the CSV to make predictions
        for i, row in data.iterrows():
            indoor_temp = row["Indoor Temperature (°C)"]
            outdoor_temp = row["Outdoor Temperature (°C)"]

            # Predict the temperature using the AI model
            predicted_temp = rf_model.predict([[indoor_temp, outdoor_temp]])[0]
            predicted_temps.append(predicted_temp)

            # Gradual adjustment towards user preference (only in "User Preference" mode)
            if mode == "User Preference" and user_preference_temp is not None:
                current_temp = predicted_temp
                # Gradually adjust towards user preference (by 0.2°C at a time)
                while abs(current_temp - user_preference_temp) > 0.1:
                    adjustment = 0.2 if current_temp < user_preference_temp else -0.2
                    current_temp += adjustment
                    adjusted_temps.append(current_temp)
                    indoor_temp_data.append(indoor_temp)
                    outdoor_temp_data.append(outdoor_temp)

                    # Update the chart dynamically
                    fig.data[0].x = timestamps[: len(adjusted_temps)]
                    fig.data[0].y = indoor_temp_data[: len(adjusted_temps)]

                    fig.data[1].x = timestamps[: len(adjusted_temps)]
                    fig.data[1].y = outdoor_temp_data[: len(adjusted_temps)]

                    fig.data[2].x = timestamps[: len(adjusted_temps)]
                    fig.data[2].y = adjusted_temps

                    # Update the chart
                    chart_placeholder.plotly_chart(fig, use_container_width=True)

                    # Show the current adjusted temperature
                    prediction_placeholder.write(
                        f"Adjusting: Row {i + 1}, Current Temp = {current_temp:.2f}°C towards {user_preference_temp:.2f}°C"
                    )
                    time.sleep(update_speed)  # Control update speed
            else:
                # AI Mode: Just append the predicted temperature without gradual adjustment
                adjusted_temps.append(predicted_temp)
                indoor_temp_data.append(indoor_temp)
                outdoor_temp_data.append(outdoor_temp)

                # Update the chart dynamically for AI Mode
                fig.data[0].x = timestamps[: i + 1]
                fig.data[0].y = indoor_temp_data[: i + 1]

                fig.data[1].x = timestamps[: i + 1]
                fig.data[1].y = outdoor_temp_data[: i + 1]

                fig.data[2].x = timestamps[: i + 1]
                fig.data[2].y = adjusted_temps[: i + 1]

                # Update the chart
                chart_placeholder.plotly_chart(fig, use_container_width=True)

                # Show the current predicted temperature
                prediction_placeholder.write(
                    f"Row {i + 1}: Predicted Desired Temperature = {predicted_temp:.2f}°C"
                )

                time.sleep(update_speed)  # Control update speed

        # Add adjusted temperatures to the DataFrame
        data["Final Adjusted Temp"] = adjusted_temps[: len(data)]
        st.write("### Final Adjusted Predictions")
        st.write(data)

        # Allow the user to download predictions
        st.sidebar.download_button(
            label="Download Predictions as CSV",
            data=data.to_csv(index=False),
            file_name="predictions_with_adjusted_temp.csv",
            mime="text/csv"
        )
    else:
        st.error("The uploaded file must contain 'Indoor Temperature (°C)' and 'Outdoor Temperature (°C)' columns.")
else:
    st.info("Please choose a mode and upload a CSV file to begin.")

# Inside Streamlit app
st.title("Test Set Predictions")



# Show the scatter plot of actual vs predicted values
st.subheader("Actual vs Predicted Desired Temperature")
st.pyplot()

# Show the comparison table of actual vs predicted values
st.write("### Predictions vs Actual")


