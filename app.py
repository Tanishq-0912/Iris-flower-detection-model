
import streamlit as st
import pickle
import numpy as np
import joblib

# Load the model
model = joblib.load('best_model.pkl')

st.title("🌸 Iris Flower Classifier")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0)

# Predict
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"🌼 Predicted class: {class_names[prediction]}")
