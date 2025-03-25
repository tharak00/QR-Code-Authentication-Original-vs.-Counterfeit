import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os

# Load pre-trained models
cnn_model = load_model(r"D:\WEB\aids\EDA-Analysis\cnn_qr_model.h5")  # Update this path to the correct location of the file
rf_model = joblib.load(r"D:\WEB\aids\EDA-Analysis\random_forest_model.pkl")  # Ensure this model is available
scaler = joblib.load(r"D:\WEB\aids\EDA-Analysis\scaler.pkl")  # Ensure the scaler is available

# Function to preprocess uploaded image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0  # Normalize
    return image.reshape(1, 128, 128, 1)

# Function to extract traditional ML features
def extract_features(image):
    image_flatten = image.flatten().reshape(1, -1)
    image_scaled = scaler.transform(image_flatten)
    return image_scaled

# Streamlit App UI
st.title("QR Code Authentication: Original vs. Counterfeit")
st.write("Upload a QR code image to check its authenticity.")

uploaded_file = st.file_uploader("Upload a QR Code Image", type=["jpg", "png", "jpeg", "bmp", "tiff", "gif"])

if uploaded_file is not None:
    # Read and display the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded QR Code", use_column_width=True)

    # Preprocess image
    processed_image = preprocess_image(image)
    ml_features = extract_features(processed_image.reshape(128, 128))

    # CNN Prediction
    cnn_pred_prob = cnn_model.predict(processed_image)[0][0]
    cnn_prediction = "First Print (Original)" if cnn_pred_prob < 0.5 else "Second Print (Counterfeit)"

    # Random Forest Prediction
    rf_pred = rf_model.predict(ml_features)[0]
    rf_prediction = "First Print (Original)" if rf_pred == 0 else "Second Print (Counterfeit)"

    # Display Predictions
    st.subheader("Results")
    st.write(f"**CNN Prediction:** {cnn_prediction} (Confidence: {100 * (1 - cnn_pred_prob) if cnn_pred_prob < 0.5 else 100 * cnn_pred_prob:.2f}%)")
    st.write(f"**Random Forest Prediction:** {rf_prediction}")
