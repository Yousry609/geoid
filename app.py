import streamlit as st
import numpy as np  # Import NumPy for array handling
import joblib
# Title and description
st.title("Gradient boosting model for the prediction of geoid height difference")
st.write("This app predicts outputs using a pre-trained Gradient Boosting model.")

# Load the saved modelس


# Load the model
loaded_model = joblib.load('GradientBoosting.pkl')

# Input sliders for user data
feature1 = st.number_input("Enter value for Latitude:",format="%.8f", min_value=0.0)
feature2 = st.number_input("Enter value for Longitude:",format="%.8f",  min_value=0.0)

# Predict button
if st.button("Predict the height difference : "):
    new_data = np.array([[feature1, feature2]])

    try:
        # Make predictions using the loaded model
        predictions = loaded_model.predict(new_data)

        # Display the prediction result
        st.success(f"Prediction: {predictions[0]}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
