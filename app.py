import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Lung Cancer Risk AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; background-color: #ff4b4b; color: white; font-weight: bold; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    /* Smaller font for T Stage Definitions block */
    .tstage-defs { font-size: 13px; line-height: 1.45; }
    .tstage-defs h4 { font-size: 14px; margin-bottom: 6px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Load Model ---
@st.cache_resource
def load_model():
    try:
        package = joblib.load('rf_lung_cancer_model.pkl')
        return package['model'], package['features']
    except FileNotFoundError:
        return None, None

model, feature_names = load_model()

# --- 4. Sidebar: Inputs ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=60)
    st.title("Clinical Parameters")
    st.markdown("Please configure the patient's data:")
    st.markdown("---")

    # 1. T Stage
    st.subheader("🧬 Tumor Characteristics")
    # Explicit T1, T2, T3, T4 labels
    t_stage_map = {
        1: "T1",
        2: "T2",
        3: "T3",
        4: "T4"
    }
    t_stage = st.selectbox(
        "T Stage Assessment",
        options=[1, 2, 3, 4],
        format_func=lambda x: t_stage_map[x]
    )

    # 2. Invasions
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        airway = st.radio("Airway Dissemination", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    with col_s2:
        pleural = st.radio("Pleural Invasion", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    st.markdown("---")
    
    # 3. Biomarkers
    st.subheader("🩸 Biomarkers & History")
    
    # Fixed Ki-67 labels: ≥30% is high expression, <30% is low expression
    ki67_map = {1: "High Expression (≥30%)", 2: "Low Expression (<30%)"}
    ki67 = st.selectbox("Ki-67 Expression", options=[1, 2], format_func=lambda x: ki67_map[x])
    
    bmi_map = {1: "Normal (<25)", 2: "Overweight (25-30)", 3: "Obese (>30)"}
    bmi = st.selectbox("BMI Category", options=[1, 2, 3], format_func=lambda x: bmi_map[x])
    
    bilirubin = st.slider("Direct Bilirubin (μmol/L)", 0.0, 50.0, 3.5, 0.1)

    st.markdown("---")
    predict_btn = st.button("🚀 Run Risk Prediction")

# --- 5. Main Dashboard ---

st.title("🫁 Lung Cancer Mortality Risk Calculator")

st.markdown("""
<div style="background-color: #e8f4f8; padding: 20px; border-radius: 10px; margin-bottom: 25px;">
    <h4>Introduction</h4>
    <p>
        This web-based calculator was developed using a <b>XGboost–based machine learning model</b>. 
        By inputting relevant clinical parameters, users can estimate the individualized <b>3-year mortality risk</b> for lung cancer patients.
    </p>
    <p>
        For patients classified as <b>high risk</b>, closer follow-up, nutritional optimization, and comprehensive management of prognostic factors may be warranted to improve long-term outcomes.
    </p>
    <p style="font-size: 0.9em; color: #555; border-top: 1px solid #ccc; padding-top: 10px; margin-top: 10px;">
        <i><b>Disclaimer:</b> This calculator is an experimental research tool based on a single-center retrospective cohort. 
        It has not undergone external validation and should not be used for diagnosis, treatment selection, prognosis counseling, or direct clinical decision-making. The results should be interpreted only for research and exploratory purposes.</i>
    </p>
</div>
""", unsafe_allow_html=True)

# ==================================================

if model is None:
    st.error("⚠️ Model file not found. Please run `model_train.py` first.")
else:
    # Prepare Data
    input_data = pd.DataFrame([{
        'T_Stage': t_stage,
        'Airway_dissemination': airway,
        'Pleural_invasion': pleural,
        'Ki_67': ki67,
        'BMI': bmi,
        'Direct_bilirubin': bilirubin
    }])

    if not predict_btn:
        st.info("👈 Please enter clinical parameters in the sidebar and click **Run Risk Prediction**.")
    else:
        try:
            # Ensure feature order matches model's expected features
            final_input = input_data[feature_names]
            
            # Prediction
            prob_death = model.predict_proba(final_input)[0][1]
            prob_survival = 1 - prob_death
            prob_percent = prob_death * 100
            # clamp progress value to 0-100 and convert to int
            progress_val = int(max(0, min(100, round(prob_percent))))

            # Results
            st.divider()
            st.subheader("📊 Prediction Results")
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("3-Year Mortality Risk", f"{prob_percent:.1f}%", delta_color="inverse")
            m2.metric("Survival Probability", f"{prob_survival*100:.1f}%")
            m3.metric("Risk Level", "High" if prob_percent > 50 else "Low/Moderate", 
                      delta="-Critical" if prob_percent > 50 else "+Stable")
            
            # Visualization
            st.write("**Risk Visualization:**")
            if prob_percent < 30:
                color, msg = "green", "✅ Low Risk Profile"
            elif prob_percent < 60:
                color, msg = "orange", "⚠️ Moderate Risk Profile"
            else:
                color, msg = "red", "🚨 High Risk Profile"

            st.progress(progress_val)
            
            # Interpretation Box
            st.markdown(f"""
            <div style="padding: 15px; border-left: 5px solid {color}; background-color: #f0f2f6;">
                <b>Interpretation:</b> {msg}<br>
                The model estimates a <b>{prob_percent:.1f}%</b> probability of mortality within 3 years.
                {'<br><i>Recommendation: Consider closer follow-up and nutritional optimization.</i>' if prob_percent > 50 else ''}
            </div>
            """, unsafe_allow_html=True)

            # Data Summary
            with st.expander("🔍 View Patient Data"):
                st.dataframe(input_data, hide_index=True)

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# --- 6. T Stage Definitions (English) with smaller font ---
st.markdown("""
<div class="tstage-defs" style="background-color:#f7fbfc; padding:14px; border-radius:8px; margin-top:20px;">
  <h4>T Stage Definitions</h4>
  <p><b>T1</b>: Tumor ≤ 3 cm in greatest dimension; tumor invasion does not extend beyond the lobar bronchus.</p>
  <p><b>T2</b>: Tumor &gt; 3 cm and ≤ 5 cm in greatest dimension; or any one of the following: invasion of the main bronchus but not involving the carina; invasion of the visceral pleura; associated obstructive pneumonia of the hilum or partial/complete lung atelectasis.</p>
  <p><b>T3</b>: Tumor &gt; 5 cm and ≤ 7 cm in greatest dimension; or invasion of any one of the following structures: pleura, chest wall, phrenic nerve, or pericardium; or presence of a separate tumor nodule(s) in the same lobe.</p>
  <p><b>T4</b>: Tumor &gt; 7 cm in greatest dimension; or, regardless of size, invasion of one or more of the following: mediastinum, heart, great vessels, trachea, recurrent laryngeal nerve, esophagus, vertebra, or diaphragm; or presence of separate tumor nodule(s) in different ipsilateral lobes.</p>
</div>
""", unsafe_allow_html=True)
