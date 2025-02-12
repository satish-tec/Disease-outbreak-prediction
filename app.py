import streamlit as st
import pickle
import os
import numpy as np
import time
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Heart Disease Prediction", layout="wide", page_icon="❤️")

# Function to add a logo
def add_logo():
    logo_path = "images/image.png"
    col1, col2 = st.columns([5, 2])  
    with col1:
        st.title("Heart Disease Prediction 🩺")  
    with col2:
        if os.path.exists(logo_path):
            st.image(logo_path, width=100)  

add_logo()

# Load models with error handling
model_path = 'heart_disease_model.sav'
scaler_path = 'scaler_heart.sav'

if os.path.exists(model_path) and os.path.exists(scaler_path):
    heart_model = pickle.load(open(model_path, 'rb'))
    heart_scaler = pickle.load(open(scaler_path, 'rb'))
else:
    st.error("Error: Model files are missing. Please check 'Saved Models' directory.")
    st.stop()

# Function to predict heart disease
def predict_heart_disease(features):
    features_scaled = heart_scaler.transform([features])
    prediction = heart_model.predict(features_scaled)
    return prediction

# Resize image function
def resize_image(image_path, width=600, height=400):
    if os.path.exists(image_path):
        img = Image.open(image_path)
        img = img.resize((width, height))
        return img
    else:
        return None

# App interface
tabs = st.tabs(["🏠 Home", "❤️ Heart Disease Prediction"])

# Home Tab
with tabs[0]:
    st.header("Welcome to the Heart Disease Prediction System")
    st.markdown("""
    ### About This App  
    - This AI-powered system predicts **heart disease risk** based on medical indicators.  
    - It helps in **early diagnosis** and **preventive measures**.  
    - Provides **instant results** based on patient data.  
    """)

    image_path = "images/view-anatomic-heart-model-educational-purpose-with-stethoscope.jpg"
    resized_img = resize_image(image_path, width=1000, height=450)
    
    if resized_img:
        st.image(resized_img, caption="Your Health Matters")
    else:
        st.warning("Image not found. Please check the file path.")

# Heart Disease Prediction Tab
with tabs[1]:
    st.header("Heart Disease Risk Assessment")
    
    with st.form(key='heart_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=100, step=1)
            sex = st.radio("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
            cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=200, step=1)
            chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, step=1)
            fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
        
        with col2:
            thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, step=1)
            exang = st.radio("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1)
            slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
            ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
            thal = st.selectbox("Thalassemia (1-3)", [1, 2, 3])
        
        diagnose_button = st.form_submit_button(label="🩺 Diagnose")

        if diagnose_button:
            with st.spinner("Processing your data..."):
                time.sleep(1.5)  # Simulate processing time
                
                features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
                prediction = predict_heart_disease(features)

                if prediction == 1:
                    st.error("⚠️ **High Risk**: The person may have heart disease. Consult a doctor immediately.", icon="🚨")
                else:
                    st.success("✅ **Low Risk**: No heart disease detected. Stay healthy!", icon="💖")
