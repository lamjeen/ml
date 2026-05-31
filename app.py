"""
Heart Attack Risk Predictor — Streamlit app (XGBoost, ROC-AUC).
Pipeline matches Untitled10.ipynb. Deployment loads ml/artifacts/roc_auc_results.pkl only.
"""

import pickle
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from preprocessing import (
    SMOKING_OPTIONS,
    encode_single_row,
    raw_input_to_frame,
)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
ROC_AUC_RESULTS_PATH = ARTIFACTS_DIR / "roc_auc_results.pkl"

st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .hero {
        background: linear-gradient(135deg, #1a1f3c 0%, #2d3561 45%, #c0392b 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: #fff;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 40px rgba(26, 31, 60, 0.25);
    }
    .hero h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .hero p { margin: 0.5rem 0 0; opacity: 0.92; font-size: 1.05rem; }
    .metric-card {
        background: #f8f9fc;
        border: 1px solid #e8ecf4;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #c0392b;
        line-height: 1.2;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #5c6370;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.35rem;
    }
    .risk-low { color: #27ae60 !important; }
    .risk-mid { color: #f39c12 !important; }
    .risk-high { color: #c0392b !important; }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f4f6fb 0%, #eef1f8 100%);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    if not ROC_AUC_RESULTS_PATH.exists():
        return None, None, None, None
    with ROC_AUC_RESULTS_PATH.open("rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_columns = bundle["feature_columns"]
    metrics = {
        "roc_auc": bundle.get("roc_auc"),
        "n_features": len(feature_columns),
    }
    return model, scaler, feature_columns, metrics


def risk_label(probability: float) -> tuple[str, str]:
    if probability < 0.35:
        return "Lower risk", "risk-low"
    if probability < 0.55:
        return "Moderate risk", "risk-mid"
    return "Higher risk", "risk-high"


def render_hero(roc_auc):
    auc_text = f"{roc_auc:.4f}" if roc_auc is not None else "—"
    st.markdown(
        f"""
        <div class="hero">
            <h1>❤️ Heart Attack Risk Predictor</h1>
            <p>Indonesia cardiovascular dataset · XGBoost classifier · Primary metric: <strong>ROC-AUC {auc_text}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_predict(model, scaler, feature_columns, metrics):
    col_metrics = st.columns(3)
    roc_auc = metrics.get("roc_auc")
    auc_display = f"{roc_auc:.4f}" if roc_auc is not None else "—"
    with col_metrics[0]:
        st.markdown(
            f'<div class="metric-card"><div class="value">{auc_display}</div>'
            '<div class="label">Test ROC-AUC</div></div>',
            unsafe_allow_html=True,
        )
    with col_metrics[1]:
        st.markdown(
            f'<div class="metric-card"><div class="value">{metrics.get("n_features", "—")}</div>'
            '<div class="label">Features</div></div>',
            unsafe_allow_html=True,
        )
    with col_metrics[2]:
        st.markdown(
            '<div class="metric-card"><div class="value">XGBoost</div>'
            '<div class="label">Model</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Enter patient characteristics")
    left, right = st.columns(2)

    with left:
        age = st.slider("Age", 25, 90, 55)
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        obesity = st.selectbox("Obesity", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        family_history = st.selectbox("Family history", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        previous_heart_disease = st.selectbox(
            "Previous heart disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
        )
        smoking_status = st.selectbox("Smoking status", SMOKING_OPTIONS)

    with right:
        cholesterol_level = st.slider("Total cholesterol", 100, 350, 200)
        cholesterol_hdl = st.slider("HDL cholesterol", 8, 93, 50)
        cholesterol_ldl = st.slider("LDL cholesterol", 0, 282, 130)
        blood_pressure_systolic = st.slider("Systolic BP (mmHg)", 61, 199, 130)
        fasting_blood_sugar = st.slider("Fasting blood sugar", 70, 230, 110)
        waist_circumference = st.slider("Waist circumference (cm)", 20, 173, 90)

    predict = st.button("Predict heart attack risk", type="primary", use_container_width=True)

    if predict:
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
        raw_df = raw_input_to_frame(row)
        encoded = encode_single_row(raw_df, feature_columns)
        scaled = scaler.transform(encoded)
        prob = float(model.predict_proba(scaled)[0, 1])
        pred_class = int(prob >= 0.5)

        label, css_class = risk_label(prob)

        res1, res2, res3 = st.columns(3)
        with res1:
            st.markdown(
                f'<div class="metric-card"><div class="value {css_class}">{prob*100:.1f}%</div>'
                '<div class="label">Predicted probability</div></div>',
                unsafe_allow_html=True,
            )
        with res2:
            st.markdown(
                f'<div class="metric-card"><div class="value {css_class}">{label}</div>'
                '<div class="label">Risk band</div></div>',
                unsafe_allow_html=True,
            )
        with res3:
            outcome = "Heart attack likely" if pred_class == 1 else "No heart attack (model)"
            st.markdown(
                f'<div class="metric-card"><div class="value">{outcome}</div>'
                '<div class="label">Class @ 0.5 threshold</div></div>',
                unsafe_allow_html=True,
            )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%"},
                title={"text": "Heart attack probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#c0392b"},
                    "steps": [
                        {"range": [0, 35], "color": "rgba(39, 174, 96, 0.25)"},
                        {"range": [35, 55], "color": "rgba(243, 156, 18, 0.25)"},
                        {"range": [55, 100], "color": "rgba(192, 57, 43, 0.25)"},
                    ],
                    "threshold": {
                        "line": {"color": "#1a1f3c", "width": 2},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            )
        )
        gauge.update_layout(height=320, margin=dict(t=40, b=20, l=30, r=30))
        st.plotly_chart(gauge, use_container_width=True)

        st.info(
            "This tool is for educational use only. It does not replace clinical diagnosis. "
            "Probabilities come from an XGBoost model trained on synthetic Indonesia health data."
        )


def page_about():
    st.markdown(
        """
        ### About this app

        This Streamlit site implements the **XGBoost** pipeline from `Untitled10.ipynb`:

        1. Load Indonesia heart attack prediction data  
        2. Drop unused columns and engineer features (`metabolic_syndrome`, `cholesterol_ratio`, etc.)  
        3. Stratified 80/20 split, one-hot encode `smoking_status`, MinMax scaling  
        4. Train `XGBClassifier` with the same hyperparameters as the notebook  
        5. Report **ROC-AUC** on the hold-out test set (target metric)

        **Deploy locally**

        ```bash
        cd ml
        pip install -r requirements.txt
        streamlit run app.py
        ```

        **Deploy on Streamlit Community Cloud**

        - Push the `ml` folder with `artifacts/roc_auc_results.pkl` (model, scaler, features, ROC-AUC).  
        - Set main file: `app.py`  
        - Python 3.10+

        Dataset: ~158k rows, 18 features after engineering, binary target `heart_attack`.
        """
    )


def main():
    model, scaler, feature_columns, metrics = load_artifacts()

    if model is None:
        st.error(
            f"Model bundle not found. Add `{ROC_AUC_RESULTS_PATH.name}` under `artifacts/`."
        )
        st.stop()

    render_hero(metrics.get("roc_auc"))

    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio(
            "Go to",
            ["Predict", "About"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Model: XGBoost")
        if metrics.get("roc_auc"):
            st.caption(f"Test ROC-AUC: **{metrics['roc_auc']:.4f}**")

    if page == "Predict":
        page_predict(model, scaler, feature_columns, metrics)
    else:
        page_about()


if __name__ == "__main__":
    main()
