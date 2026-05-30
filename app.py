"""Heart attack risk predictor — Streamlit app (XGBoost, ROC-AUC)."""

from pathlib import Path

import joblib
import streamlit as st

from preprocess import SMOKING_OPTIONS, row_to_feature_frame

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: #fff;
        box-shadow: 0 8px 32px rgba(15, 52, 96, 0.35);
    }
    .main-header h1 { color: #fff !important; margin-bottom: 0.25rem; font-weight: 700; }
    .main-header p { color: #e8e8e8; opacity: 0.92; margin: 0; font-size: 1.05rem; }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        text-align: center;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #0f3460; }
    .metric-card .label { font-size: 0.85rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f1f5f9 0%, #fff 100%);
    }
</style>
"""


@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACT_DIR / "xgb_model.joblib")
    scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")
    feature_columns = joblib.load(ARTIFACT_DIR / "feature_columns.joblib")
    metrics = joblib.load(ARTIFACT_DIR / "metrics.joblib")
    return model, scaler, feature_columns, metrics


def render_header(roc_auc: float):
    st.markdown(
        f"""
        <div class="main-header">
            <h1>❤️ Heart Attack Risk Predictor</h1>
            <p>Indonesia cardiovascular dataset · XGBoost classifier · Hold-out ROC-AUC <strong>{roc_auc:.4f}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if not (ARTIFACT_DIR / "xgb_model.joblib").exists():
        with st.spinner("Training XGBoost model (first run only)…"):
            from train_model import main as train_main

            train_main()
        st.rerun()

    model, scaler, feature_columns, metrics = load_artifacts()
    roc_auc = metrics["roc_auc"]

    render_header(roc_auc)

    col_metrics = st.columns(3)
    with col_metrics[0]:
        st.markdown(
            f'<div class="metric-card"><div class="value">{roc_auc:.2%}</div><div class="label">ROC-AUC (test)</div></div>',
            unsafe_allow_html=True,
        )
    with col_metrics[1]:
        st.markdown(
            '<div class="metric-card"><div class="value">XGBoost</div><div class="label">Model</div></div>',
            unsafe_allow_html=True,
        )
    with col_metrics[2]:
        st.markdown(
            '<div class="metric-card"><div class="value">158K+</div><div class="label">Training records</div></div>',
            unsafe_allow_html=True,
        )

    st.sidebar.header("About")
    st.sidebar.markdown(
        """
        This app predicts **heart attack risk** from clinical and lifestyle inputs,
        matching the pipeline in your notebook (`Untitled10.ipynb`).

        **Output:** probability of heart attack (class 1) and binary risk label.

        **Metric:** ROC-AUC on the held-out test set (same split as the notebook).
        """
    )
    st.subheader("Patient clinical data")
    st.caption("Enter values below, then click **Analyze risk**.")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=55)
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        obesity = st.selectbox("Obesity", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        family_history = st.selectbox(
            "Family history of heart disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
        )

    with c2:
        cholesterol_level = st.number_input("Total cholesterol", min_value=100, max_value=400, value=210)
        cholesterol_hdl = st.number_input("HDL cholesterol", min_value=20, max_value=120, value=50)
        cholesterol_ldl = st.number_input("LDL cholesterol", min_value=50, max_value=250, value=120)
        waist_circumference = st.number_input("Waist circumference (cm)", min_value=60, max_value=150, value=90)
        smoking_status = st.selectbox("Smoking status", SMOKING_OPTIONS)

    with c3:
        blood_pressure_systolic = st.number_input(
            "Systolic blood pressure (mmHg)", min_value=80, max_value=220, value=130
        )
        fasting_blood_sugar = st.number_input("Fasting blood sugar", min_value=50, max_value=300, value=100)
        previous_heart_disease = st.selectbox(
            "Previous heart disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
        )

    st.markdown("---")

    if st.button("Analyze risk", type="primary", use_container_width=True):
        row = {
            "age": age,
            "hypertension": hypertension,
            "diabetes": diabetes,
            "cholesterol_level": cholesterol_level,
            "obesity": obesity,
            "waist_circumference": waist_circumference,
            "family_history": family_history,
            "smoking_status": smoking_status,
            "blood_pressure_systolic": blood_pressure_systolic,
            "fasting_blood_sugar": fasting_blood_sugar,
            "cholesterol_hdl": cholesterol_hdl,
            "cholesterol_ldl": cholesterol_ldl,
            "previous_heart_disease": previous_heart_disease,
        }

        features = row_to_feature_frame(row, feature_columns)
        scaled = scaler.transform(features)
        proba = float(model.predict_proba(scaled)[0, 1])
        prediction = int(proba >= 0.5)

        st.subheader("Results")

        r1, r2 = st.columns([1, 1])
        with r1:
            st.metric("Predicted risk probability", f"{proba * 100:.2f}%")
            st.progress(min(proba, 1.0))
        with r2:
            st.metric("Model ROC-AUC (test set)", f"{roc_auc:.4f}")

        if prediction == 1:
            st.error(
                f"**Elevated risk** — The model estimates a **{proba * 100:.1f}%** probability of heart attack. "
                "This is not a medical diagnosis; consult a healthcare professional."
            )
        else:
            st.success(
                f"**Lower risk** — Estimated probability: **{proba * 100:.1f}%**. "
                "Continue regular check-ups and healthy habits."
            )

        with st.expander("Technical details"):
            st.write("**Binary prediction** (threshold 0.5):", "Heart attack risk" if prediction else "No heart attack")
            st.write("**Engineered features:** age_risk, metabolic_syndrome, cholesterol_ratio, bp_hypertension, age_previous_hd")
            st.dataframe(features.T.rename(columns={0: "value"}), use_container_width=True)


if __name__ == "__main__":
    main()
