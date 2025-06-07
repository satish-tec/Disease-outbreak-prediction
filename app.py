import streamlit as st
import pickle
import time
import numpy as np
import base64

# Set page configuration
st.set_page_config(
    page_title="MedPredict - Disease Prediction System",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded"
)

# Function to get base64 string of images
def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Get base64 strings for background images (replace with your actual image paths)
home_bg = get_image_base64("Images/5.jpg")
heart_bg = get_image_base64("Images/4.jpg")
diabetes_bg = get_image_base64("Images/1.jpg")
about_bg = get_image_base64("Images/2.jpg")

# Replace the existing CSS block with this updated version
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    :root {{
        --primary: #1a73e8;
        --secondary: #34a853;
        --accent1: #fbbc05;
        --accent2: #ea4335;
        --dark: #202124;
        --light: #f8f9fa;
        --card-bg: rgba(255, 255, 255, 0.92);
        --text-dark: #2c3e50;
    }}
    
    * {{
        font-family: 'Poppins', sans-serif;
    }}
    
    .stApp {{
        background: linear-gradient(135deg, #87CEEB 0%, #B0E2FF 100%);
        background-attachment: fixed;
    }}
    
    /* Home Tab Background */
    [data-testid="stTabs"] [aria-selected="true"]:nth-child(1) + div [data-testid="stVerticalBlock"] {{
        background: url("data:image/jpg;base64,{home_bg}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-blend-mode: overlay;
        background-color: rgba(135, 206, 235, 0.6);
    }}
    
    /* Heart Tab Background */
    [data-testid="stTabs"] [aria-selected="true"]:nth-child(2) + div [data-testid="stVerticalBlock"] {{
        background: url("data:image/jpg;base64,{heart_bg}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-blend-mode: overlay;
        background-color: rgba(135, 206, 235, 0.6);
    }}
    
    /* Diabetes Tab Background */
    [data-testid="stTabs"] [aria-selected="true"]:nth-child(3) + div [data-testid="stVerticalBlock"] {{
        background: url("data:image/jpg;base64,{diabetes_bg}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-blend-mode: overlay;
        background-color: rgba(135, 206, 235, 0.6);
    }}
    
    /* About Tab Background */
    [data-testid="stTabs"] [aria-selected="true"]:nth-child(4) + div [data-testid="stVerticalBlock"] {{
        background: url("data:image/jpg;base64,{about_bg}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-blend-mode: overlay;
        background-color: rgba(135, 206, 235, 0.6);
    }}
    
    /* Tab content styling */
    .stTabs > div > div:last-child > div > div > div > div {{
        padding: 2rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(5px);
        margin: -1rem;
    }}
    
    .header {{
        background: linear-gradient(90deg, #1a73e8 0%, #0d47a1 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }}
    
    .card {{
        background: var(--card-bg);
        border-radius: 20px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.05);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        border: none;
    }}
    
    .card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }}
    
    .feature-card {{
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-left: 4px solid var(--primary);
    }}
    
    .result-card {{
        background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
        border-top: 4px solid var(--primary);
        text-align: center;
    }}
    
    .stButton>button {{
        background: linear-gradient(90deg, var(--primary) 0%, #0d47a1 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.7rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 100%;
    }}
    
    .stButton>button:hover {{
        transform: scale(1.03);
        box-shadow: 0 4px 15px rgba(26, 115, 232, 0.3);
    }}
    
    .input-label {{
        font-weight: 500;
        color: var(--text-dark);
        margin-bottom: 0.3rem;
    }}
    
    .footer {{
        text-align: center;
        padding: 1.5rem;
        color: var(--text-dark);
        font-size: 0.9rem;
        margin-top: 2rem;
        background: rgba(255, 255, 255, 0.85);
        border-radius: 20px 20px 0 0;
        box-shadow: 0 -4px 10px rgba(0,0,0,0.05);
    }}
    
    .risk-indicator {{
        height: 20px;
        border-radius: 10px;
        margin: 1rem 0;
        background: linear-gradient(90deg, #34a853 0%, #fbbc05 50%, #ea4335 100%);
        position: relative;
    }}
    
    .risk-marker {{
        position: absolute;
        top: -5px;
        width: 30px;
        height: 30px;
        background: white;
        border: 3px solid var(--primary);
        border-radius: 50%;
        transform: translateX(-50%);
        transition: left 0.5s ease;
    }}
    
    .stTabs [role="tablist"] {{
        gap: 10px;
        background: rgba(255, 255, 255, 0.8);
        padding: 0.5rem;
        border-radius: 12px;
        backdrop-filter: blur(5px);
        margin-bottom: 1rem;
    }}
    
    .stTabs [role="tab"] {{
        border-radius: 12px !important;
        padding: 0.7rem 1.5rem !important;
        background-color: #e8f0fe !important;
        color: var(--primary) !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }}
    
    .stTabs [role="tab"][aria-selected="true"] {{
        background: linear-gradient(90deg, var(--primary) 0%, #0d47a1 100%) !important;
        color: white !important;
    }}
    
    .stNumberInput, .stSelectbox {{
        margin-bottom: 1.2rem;
    }}
    
    .stMetric {{
        border-left: 4px solid var(--primary);
    }}
    
    .logo-container {{
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 0.5rem;
    }}
    
    .logo-text {{
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00B4DB 0%, #0083B0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    .risk-value {{
        font-size: 2.5rem;
        font-weight: 700;
        margin: 1rem 0;
    }}
    
    .low-risk {{ color: #34a853; }}
    .medium-risk {{ color: #fbbc05; }}
    .high-risk {{ color: #ea4335; }}
    
    .metric-card {{
        background: rgba(255, 255, 255, 0.92);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        border-left: 4px solid var(--primary);
    }}
    
    .custom-card {{
        background: rgba(255, 255, 255, 0.92);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.05);
    }}
    
    .custom-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }}
    
    .heart-card {{
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    }}
    
    .heart-card:hover {{
        box-shadow: 0 8px 25px rgba(234, 67, 53, 0.2);
    }}
    
    .diabetes-card {{
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    }}
    
    .diabetes-card:hover {{
        box-shadow: 0 8px 25px rgba(52, 168, 83, 0.2);
    }}
    
    .tech-card {{
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    }}
    
    .method-card {{
        background: linear-gradient(135deg, #f1f8e9 0%, #dcedc8 100%);
    }}
    
    /* Text contrast adjustments */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--text-dark);
    }}
    
    p, div, span, label {{
        color: var(--text-dark);
    }}
    
    .stMarkdown {{
        color: var(--text-dark);
    }}

    /* Custom Selectbox Styling */
    div[data-baseweb="select"] > div {{
        background-color: #f0f8ff !important;  /* Light blue background */
        border-radius: 8px !important;
        border: 1px solid #1a73e8 !important;
        padding: 1px 12px !important;
    }}
    
    div[data-baseweb="select"] > div:hover {{
        background-color: #f0f8ff !important;
        border-color: #0d47a1 !important;
    }}
    
    div[data-baseweb="select"] > div > div > div {{
        color: #1a237e !important;  /* Dark blue text for contrast */
        font-weight: 500 !important;
    }}
    
    /* Radio Button Styling */
    .stRadio > div {{
        background-color: #f0f8ff !important;
        border-radius: 8px !important;
        border: 1px solid #1a73e8 !important;
        padding: 8px 12px !important;
    }}
    
    .stRadio > div > label {{
        color: #1a237e !important;
        font-weight: 500 !important;
    }}
    
    /* Slider Styling */
    .stSlider > div {{
        background-color: #f0f8ff !important;
        border-radius: 8px !important;
        border: 1px solid #1a73e8 !important;
        padding: 8px 12px !important;
    }}
    
    .stSlider > div > div > div {{
        color: #1a237e !important;
    }}
    
    /* Input Number Styling */
    .stNumberInput > div {{
        background-color: #f0f8ff !important;
        border-radius: 8px !important;
        border: 1px solid #1a73e8 !important;
        padding: 8px 12px !important;
    }}
    
    .stNumberInput input {{
        color: #1a237e !important;
        font-weight: 500 !important;
    }}
    
    /* Focus State */
    div[data-baseweb="select"] > div:focus-within,
    .stRadio > div:focus-within,
    .stSlider > div:focus-within,
    .stNumberInput > div:focus-within {{
        box-shadow: 0 0 0 2px rgba(26, 115, 232, 0.3) !important;
        border-color: #0d47a1 !important;
    }}
</style>
""", unsafe_allow_html=True)

# Load the saved models and scalers
@st.cache_resource
def load_models():
    heart_model = pickle.load(open('Saved Models/heart_disease_model.sav', 'rb'))
    diabetes_model = pickle.load(open('Saved Models/diabetes_model.sav', 'rb'))
    heart_scaler = pickle.load(open('Saved Models/scaler_heart.sav', 'rb'))
    diabetes_scaler = pickle.load(open('Saved Models/scaler_diabetes.sav', 'rb'))
    return heart_model, diabetes_model, heart_scaler, diabetes_scaler

heart_model, diabetes_model, heart_scaler, diabetes_scaler = load_models()

# Function to predict heart disease
def predict_heart_disease(features):
    features_scaled = heart_scaler.transform([features])
    prediction = heart_model.predict(features_scaled)
    probability = heart_model.predict_proba(features_scaled)[0][1]
    return prediction, probability

# Function to predict diabetes
def predict_diabetes(features):
    features_scaled = diabetes_scaler.transform([features])
    prediction = diabetes_model.predict(features_scaled)
    probability = diabetes_model.predict_proba(features_scaled)[0][1]
    return prediction, probability

# App header
with st.container():
    st.markdown('<div class="header">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("Images/Icon.png", width=100)
    with col2:
        st.markdown('<div class="logo-container"><h1 class="logo-text">MedPredict</h1></div>', unsafe_allow_html=True)
        st.caption("AI-powered disease prediction system for proactive healthcare")
    st.markdown('</div>', unsafe_allow_html=True)




if 'tab_index' not in st.session_state:
    st.session_state.tab_index = 0

# Function to trigger tab switch
def switch_tab(tab_name):
    if tab_name == "❤️ Heart Disease":
        st.session_state.tab_index = 1
    elif tab_name == "🩸 Diabetes":
        st.session_state.tab_index = 2
    st.experimental_rerun()

# Tab navigation
tabs = st.tabs(["🏠 Dashboard", "❤️ Heart Disease", "🩸 Diabetes", "ℹ️ About"])

# Dashboard Tab
with tabs[0]:
    st.subheader("Health Assessment Dashboard")
    st.markdown("""
    Welcome to MedPredict, your AI-powered health assessment tool. Our predictive models analyze key health indicators 
    to assess your risk for common diseases. Select a disease category to begin your assessment.
    """)

    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            #st.markdown('<div class="custom-card heart-card">', unsafe_allow_html=True)
            st.markdown("### Heart Disease Risk")
            st.image("https://cdn-icons-png.flaticon.com/512/2489/2489792.png", width=100)
            st.markdown("Assess your risk for cardiovascular diseases based on key health indicators")
            if st.button("Assess Heart Risk", key="heart_btn"):
                switch_tab("❤️ Heart Disease")
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        with st.container():
            #st.markdown('<div class="custom-card diabetes-card">', unsafe_allow_html=True)
            st.markdown("### Diabetes Risk")
            st.image("https://cdn-icons-png.flaticon.com/512/3699/3699540.png", width=100)
            st.markdown("Evaluate your diabetes risk based on metabolic health markers")
            if st.button("Assess Diabetes Risk", key="diabetes_btn"):
                switch_tab("🩸 Diabetes")
            st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Health metrics
    st.subheader("Healthcare Insights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Heart Disease Prevalence", "12.7%", "Global average")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Diabetes Prevalence", "9.3%", "1 in 11 adults")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Early Detection Impact", "Up to 80%", "Risk reduction potential")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Health tips
    st.subheader("Preventive Health Tips")
    tips = [
        {"title": "Balanced Diet", "desc": "Focus on whole foods, fruits, vegetables and lean proteins", "icon": "🥗"},
        {"title": "Regular Exercise", "desc": "Aim for 150 minutes of moderate activity weekly", "icon": "🏃‍♂️"},
        {"title": "Stress Management", "desc": "Practice mindfulness and relaxation techniques", "icon": "🧘‍♀️"},
        {"title": "Regular Checkups", "desc": "Annual health screenings can detect issues early", "icon": "🩺"},
    ]
    for tip in tips:
        with st.expander(f"{tip['icon']} {tip['title']}"):
            st.write(tip['desc'])

# Heart Disease Prediction Tab
with tabs[1]:
    st.subheader("Heart Disease Risk Assessment")
    st.markdown("Complete the form below to assess your cardiovascular health risk")
    
    with st.form(key='heart_form'):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="input-label">Age</div>', unsafe_allow_html=True)
            age = st.slider("", 20, 100, 45, label_visibility="collapsed")
            
            st.markdown('<div class="input-label">Sex</div>', unsafe_allow_html=True)
            sex = st.radio("", ["Male", "Female"], index=0, horizontal=True, label_visibility="collapsed")
            sex = 1 if sex == "Male" else 0
            
            st.markdown('<div class="input-label">Chest Pain Type</div>', unsafe_allow_html=True)
            cp = st.selectbox("", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], 
                             index=0, label_visibility="collapsed")
            cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
            cp = cp_map[cp]
            
            st.markdown('<div class="input-label">Resting Blood Pressure (mm Hg)</div>', unsafe_allow_html=True)
            trestbps = st.slider("", 80, 200, 120, label_visibility="collapsed")
            
            st.markdown('<div class="input-label">Serum Cholesterol (mg/dl)</div>', unsafe_allow_html=True)
            chol = st.slider("", 100, 600, 200, label_visibility="collapsed")
            
        with col2:
            st.markdown('<div class="input-label">Fasting Blood Sugar</div>', unsafe_allow_html=True)
            fbs = st.radio("", ["≤ 120 mg/dl", "> 120 mg/dl"], index=0, horizontal=True, label_visibility="collapsed")
            fbs = 1 if fbs == "> 120 mg/dl" else 0
            
            st.markdown('<div class="input-label">Resting ECG Results</div>', unsafe_allow_html=True)
            restecg = st.selectbox("", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"], 
                                  index=0, label_visibility="collapsed")
            restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
            restecg = restecg_map[restecg]
            
            st.markdown('<div class="input-label">Max Heart Rate Achieved</div>', unsafe_allow_html=True)
            thalach = st.slider("", 70, 220, 150, label_visibility="collapsed")
            
            st.markdown('<div class="input-label">Exercise Induced Angina</div>', unsafe_allow_html=True)
            exang = st.radio("", ["No", "Yes"], index=0, horizontal=True, label_visibility="collapsed")
            exang = 1 if exang == "Yes" else 0
            
            st.markdown('<div class="input-label">ST Depression Induced by Exercise</div>', unsafe_allow_html=True)
            oldpeak = st.slider("", 0.0, 6.0, 1.0, step=0.1, label_visibility="collapsed")
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="input-label">Slope of Peak Exercise ST Segment</div>', unsafe_allow_html=True)
            slope = st.selectbox("", ["Upsloping", "Flat", "Downsloping"], index=0, label_visibility="collapsed")
            slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
            slope = slope_map[slope]
            
        with col4:
            st.markdown('<div class="input-label">Number of Major Vessels (0-3)</div>', unsafe_allow_html=True)
            ca = st.selectbox("", [0, 1, 2, 3], index=0, label_visibility="collapsed")
            
            st.markdown('<div class="input-label">Thalassemia</div>', unsafe_allow_html=True)
            thal = st.selectbox("", ["Normal", "Fixed Defect", "Reversible Defect"], index=0, label_visibility="collapsed")
            thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
            thal = thal_map[thal]
        
        diagnose_button = st.form_submit_button(label="Assess Heart Disease Risk", use_container_width=True)
        
        if diagnose_button:
            with st.spinner('Analyzing cardiovascular health markers...'):
                time.sleep(2)
                features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
                prediction, probability = predict_heart_disease(features)
                risk_percent = int(probability * 100)
                
                with st.container():
                    #st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                    st.subheader("Assessment Result")
                    
                    # Risk visualization
                    #st.markdown(f'<div class="risk-value {"" if risk_percent < 30 else "medium-risk" if risk_percent < 70 else "high-risk"}">{risk_percent}%</div>', unsafe_allow_html=True)
                    st.markdown('<div class="risk-indicator"><div class="risk-marker" style="left: {0}%;"></div></div>'.format(risk_percent), unsafe_allow_html=True)
                    
                    col_low, col_med, col_high = st.columns(3)
                    col_low.caption("Low Risk")
                    col_med.caption("Medium Risk")
                    col_high.caption("High Risk")
                    
                    # Result message
                    if prediction == 1:
                        st.error("### ⚠️ Elevated Heart Disease Risk Detected")
                        st.markdown("Your assessment indicates a higher than average risk for cardiovascular disease. We recommend:")
                        st.markdown("- Consulting with a cardiologist for a comprehensive evaluation")
                        st.markdown("- Implementing lifestyle changes: diet, exercise, stress management")
                        st.markdown("- Regular monitoring of blood pressure and cholesterol levels")
                    else:
                        st.success("### ✅ Normal Cardiovascular Risk")
                        st.markdown("Your assessment indicates a normal risk profile for heart disease. To maintain heart health:")
                        st.markdown("- Continue healthy lifestyle habits")
                        st.markdown("- Schedule regular checkups with your physician")
                        st.markdown("- Monitor key indicators like blood pressure annually")
                    
                    # Disclaimer
                    st.caption("Note: This assessment is for informational purposes only and not a substitute for professional medical advice. Always consult with a healthcare provider for personal health guidance.")
                    st.markdown('</div>', unsafe_allow_html=True)

# Diabetes Prediction Tab
with tabs[2]:
    st.subheader("Diabetes Risk Assessment")
    st.markdown("Complete the form below to evaluate your diabetes risk")
    
    with st.form(key='diabetes_form'):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="input-label">Age</div>', unsafe_allow_html=True)
            age = st.slider("", 18, 100, 45, key="diab_age", label_visibility="collapsed")
            
            st.markdown('<div class="input-label">Pregnancies (if applicable)</div>', unsafe_allow_html=True)
            pregnancies = st.slider("", 0, 15, 0, key="preg", label_visibility="collapsed")
            
            st.markdown('<div class="input-label">Glucose Level (mg/dl)</div>', unsafe_allow_html=True)
            glucose = st.slider("", 50, 300, 100, key="glucose", label_visibility="collapsed")
            
            st.markdown('<div class="input-label">Blood Pressure (mm Hg)</div>', unsafe_allow_html=True)
            blood_pressure = st.slider("", 40, 180, 70, key="bp", label_visibility="collapsed")
            
        with col2:
            st.markdown('<div class="input-label">Skin Thickness (mm)</div>', unsafe_allow_html=True)
            skin_thickness = st.slider("", 0, 100, 20, key="skin", label_visibility="collapsed")
            
            st.markdown('<div class="input-label">Insulin Level (μU/ml)</div>', unsafe_allow_html=True)
            insulin = st.slider("", 0, 900, 80, key="insulin", label_visibility="collapsed")
            
            st.markdown('<div class="input-label">BMI (kg/m²)</div>', unsafe_allow_html=True)
            bmi = st.slider("", 10.0, 50.0, 22.0, step=0.1, key="bmi", label_visibility="collapsed")
            
            st.markdown('<div class="input-label">Diabetes Pedigree Function</div>', unsafe_allow_html=True)
            diabetes_pedigree = st.slider("", 0.0, 2.5, 0.3, step=0.01, key="pedigree", label_visibility="collapsed")
        
        diagnose_button = st.form_submit_button(label="Assess Diabetes Risk", use_container_width=True)
        
        if diagnose_button:
            with st.spinner('Analyzing metabolic health markers...'):
                time.sleep(2)
                features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
                prediction, probability = predict_diabetes(features)
                risk_percent = int(probability * 100)
                
                with st.container():
                    #st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                    st.subheader("Assessment Result")
                    
                    # Risk visualization
                    st.markdown(f'<div class="risk-value {"" if risk_percent < 30 else "medium-risk" if risk_percent < 70 else "high-risk"}">{risk_percent}%</div>', unsafe_allow_html=True)
                    st.markdown('<div class="risk-indicator"><div class="risk-marker" style="left: {0}%;"></div></div>'.format(risk_percent), unsafe_allow_html=True)
                    
                    col_low, col_med, col_high = st.columns(3)
                    col_low.caption("Low Risk")
                    col_med.caption("Medium Risk")
                    col_high.caption("High Risk")
                    
                    # Result message
                    if prediction == 1:
                        st.error("### ⚠️ Elevated Diabetes Risk Detected")
                        st.markdown("Your assessment indicates a higher than average risk for diabetes. We recommend:")
                        st.markdown("- Consulting with an endocrinologist or primary care physician")
                        st.markdown("- Implementing dietary changes to regulate blood sugar")
                        st.markdown("- Increasing physical activity and monitoring glucose levels")
                    else:
                        st.success("### ✅ Normal Diabetes Risk")
                        st.markdown("Your assessment indicates a normal risk profile for diabetes. To maintain metabolic health:")
                        st.markdown("- Continue balanced nutrition and regular physical activity")
                        st.markdown("- Monitor blood sugar levels periodically")
                        st.markdown("- Maintain healthy body weight")
                    
                    # Disclaimer
                    st.caption("Note: This assessment is for informational purposes only and not a substitute for professional medical advice. Always consult with a healthcare provider for personal health guidance.")
                    st.markdown('</div>', unsafe_allow_html=True)
# About Tab
with tabs[3]:
    st.subheader("About MedPredict")
    st.markdown("""
    MedPredict is an AI-powered health assessment platform designed to provide early risk detection for common diseases.\n
    Our predictive models analyze key health indicators to generate personalized risk assessments.
    """)
        
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            #st.markdown('<div class="custom-card tech-card">', unsafe_allow_html=True)
            st.markdown("### Technology")
            st.markdown("""
            - **Machine Learning**: Predictive models trained on clinical datasets
            - **Feature Engineering**: Advanced preprocessing of health indicators
            - **Risk Stratification**: Probability-based risk assessment
            - **Healthcare Analytics**: Evidence-based medical insights
            """)
            #st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            #st.markdown('<div class="custom-card method-card">', unsafe_allow_html=True)
            st.markdown("### Methodology")
            st.markdown("""
            - **Heart Disease Model**: Logistic Regression classifier
            - **Diabetes Model**: Random Forest classifier
            - **Data Sources**: Curated clinical datasets
            - **Validation**: Rigorous cross-validation protocols
            """)
            #st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Contact")
    st.markdown("For inquiries, email us at **solutionscyber@yahoo.com**")

# Footer
st.markdown("""
<div class="footer">
    © 2025 MedPredict Health Analytics | Privacy Policy | Terms of Service
</div>
""", unsafe_allow_html=True)