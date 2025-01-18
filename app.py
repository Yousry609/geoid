import streamlit as st
import numpy as np  # Import NumPy for array handling
import joblib

# Title and description
st.title("Gradient Boosting Model for the Prediction of Geoid Height Difference")
st.write("This app predicts outputs using a pre-trained Gradient Boosting model.")

# Load the saved model
try:
    loaded_model = joblib.load('GradientBoosting.pkl')
except FileNotFoundError:
    st.error("The model file 'GradientBoosting.pkl' was not found. Please ensure it is in the correct directory.")
    st.stop()

# Input sliders for user data
feature1 = st.number_input("Enter value for Latitude (22 to 32):", format="%.8f", min_value=0.0)
feature2 = st.number_input("Enter value for Longitude (24 to 37):", format="%.8f", min_value=0.0)

# Predict button
if st.button("Predict the Height Difference:"):
    # Check if inputs are within valid ranges
    if not (22 <= feature1 <= 32):
        st.error("Invalid Latitude: Latitude must be between 22 and 32. Prediction not allowed.")
    elif not (24 <= feature2 <= 37):
        st.error("Invalid Longitude: Longitude must be between 24 and 37. Prediction not allowed.")
    else:
        # Inputs are valid; proceed with prediction
        new_data = np.array([[feature1, feature2]])

        try:
            # Make predictions using the loaded model
            predictions = loaded_model.predict(new_data)

            # Display the prediction result
            st.success(f"Prediction: {predictions[0]}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
