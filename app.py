import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Dropout Risk Predictor", layout="wide", page_icon="🎓")
st.title("🎓 Student Dropout Prediction")
st.markdown("Easily predict student dropout risk. Use the sidebar to load demo profiles for quick testing without typing 36 fields.")

@st.cache_resource
def load_resources():
    if os.path.exists('best_model.pkl') and os.path.exists('scaler.pkl') and os.path.exists('feature_names.pkl'):
        return joblib.load('best_model.pkl'), joblib.load('scaler.pkl'), joblib.load('feature_names.pkl')
    return None, None, None

model, scaler, feature_names = load_resources()

if model is None:
    st.error("Model files not found. Please train the model first by running `python train_model.py`.")
else:
    st.sidebar.title("Quick Demo Profiles")
    st.sidebar.info("Use these profiles to auto-fill the form below for the presentation.")
    
    # Initialize session state for all features to 0.0 if they don't exist
    if 'profile' not in st.session_state:
        st.session_state.profile = "Custom"
        for f in feature_names:
            st.session_state[f] = 0.0

    profile = st.sidebar.selectbox("Load Student Profile", ["Custom (Blank)", "High Risk (Struggling)", "Low Risk (Excelling)"])
    
    # If the user selects a new profile, we update the session_state variables
    if profile != st.session_state.profile:
        st.session_state.profile = profile
        if profile == "High Risk (Struggling)":
            for f in feature_names:
                st.session_state[f] = 0.0
                if "Tuition fees up to date" in f: st.session_state[f] = 0.0 # No
                elif "Debtor" in f: st.session_state[f] = 1.0 # Yes
                elif "approved" in f: st.session_state[f] = 1.0 # Only 1 unit approved
                elif "enrolled" in f: st.session_state[f] = 6.0 # Enrolled in 6
                elif "grade" in f: st.session_state[f] = 9.5 # Failing grades
                elif "Age at enrollment" in f: st.session_state[f] = 25.0
                
        elif profile == "Low Risk (Excelling)":
            for f in feature_names:
                st.session_state[f] = 0.0
                if "Tuition fees up to date" in f: st.session_state[f] = 1.0 # Yes
                elif "Debtor" in f: st.session_state[f] = 0.0 # No
                elif "approved" in f: st.session_state[f] = 6.0 # 6 units approved
                elif "enrolled" in f: st.session_state[f] = 6.0 # Enrolled in 6
                elif "grade" in f: st.session_state[f] = 16.5 # Good grades
                elif "Scholarship holder" in f: st.session_state[f] = 1.0
                elif "Age at enrollment" in f: st.session_state[f] = 19.0
        else: # Custom
            for f in feature_names:
                if "Age at enrollment" in f: st.session_state[f] = 20.0
                else: st.session_state[f] = 0.0
                
    st.sidebar.divider()
    st.sidebar.markdown(f"**Current Model:** {type(model).__name__}")

    # Top Features we actually want to show to the examiner
    top_features = [
        'Curricular units 2nd sem (approved)',
        'Curricular units 1st sem (approved)',
        'Curricular units 2nd sem (enrolled)',
        'Tuition fees up to date',
        'Curricular units 2nd sem (grade)',
        'Debtor'
    ]
    
    st.header("Key Predictors")
    st.caption("These are the most important features determining dropout risk as found by our model.")
    col1, col2 = st.columns(2)
    
    input_data = {}
    
    friendly_labels = {
        'Curricular units 2nd sem (approved)': 'Classes Passed (2nd Semester)',
        'Curricular units 1st sem (approved)': 'Classes Passed (1st Semester)',
        'Curricular units 2nd sem (enrolled)': 'Classes Enrolled (2nd Semester)',
        'Tuition fees up to date': 'Are Tuition Fees Paid Up To Date?',
        'Curricular units 2nd sem (grade)': 'Average Grade (2nd Semester, out of 20)',
        'Debtor': 'Does the Student Have Outstanding Debt?',
        'Curricular units 1st sem (enrolled)': 'Classes Enrolled (1st Semester)',
        'Curricular units 1st sem (grade)': 'Average Grade (1st Semester, out of 20)',
        'Age at enrollment': 'Age at Enrollment',
        'Scholarship holder': 'Has a Scholarship?',
        'International': 'Is an International Student?'
    }
    
    # Render top features prominently
    for i, f in enumerate(top_features):
        if f not in feature_names: continue
        c = col1 if i % 2 == 0 else col2
        label = friendly_labels.get(f, f)
        if "Tuition fees up to date" in f or "Debtor" in f:
            # We use session state as default
            val = c.selectbox(label, options=[1.0, 0.0], format_func=lambda x: "Yes" if x==1 else "No", key=f)
        else:
            if any(keyword in f.lower() for keyword in ["grade", "rate", "gdp", "inflation"]):
                val = c.number_input(label, value=float(st.session_state[f]), key=f)
            else:
                val = c.number_input(label, value=int(float(st.session_state[f])), step=1, key=f)
        input_data[f] = val

    # Hide all other features in an expander
    with st.expander("Show All Other 30 Attributes (Advanced)"):
        other_features = [f for f in feature_names if f not in top_features]
        cols = st.columns(3)
        for i, f in enumerate(other_features):
            c = cols[i % 3]
            label = friendly_labels.get(f, f)
            if "Scholarship holder" in f or "Gender" in f or "International" in f:
                val = c.selectbox(label, options=[1.0, 0.0], format_func=lambda x: "Yes" if x==1 else "No", key=f)
            else:
                if any(keyword in f.lower() for keyword in ["grade", "rate", "gdp", "inflation"]):
                    val = c.number_input(label, value=float(st.session_state[f]), key=f)
                else:
                    val = c.number_input(label, value=int(float(st.session_state[f])), step=1, key=f)
            input_data[f] = val

    st.write("")
    
    # Predict Button
    if st.button("Predict Dropout Risk", type="primary", use_container_width=True):
        input_df = pd.DataFrame([input_data])
        # Reorder to match scaler's exact expectations
        input_df = input_df[feature_names]
        input_scaled = scaler.transform(input_df)
        
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1] if hasattr(model, 'predict_proba') else None
        
        st.divider()
        st.header("Prediction Results")
        
        if prediction == 1:
            st.error(f"🚨 **Risk of Dropout: HIGH**")
        else:
            st.success(f"✅ **Risk of Dropout: LOW**")
            
        if prob is not None:
            st.metric(label="Calculated Probability of Dropping Out", value=f"{prob * 100:.2f}%")
            st.progress(float(prob))
            
        st.markdown("---")
        st.caption("Note: This is an early warning system. Ethical considerations should be taken before labelling students as at-risk. Predictions should be used to offer support, not to penalize.")
