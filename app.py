"""
Heart Attack Risk Predictor — Streamlit app (XGBoost, ROC-AUC).
Pipeline matches Untitled10.ipynb. Deployment loads ml/artifacts/roc_auc_results.pkl only.
"""

import pickle
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
ROC_AUC_RESULTS_PATH = ARTIFACTS_DIR / "roc_auc_results.pkl"

SMOKING_OPTIONS = ["Never", "Past", "Current"]
SMOKING_MAP = {"Never": 0, "Past": 2, "Current": 3}

# Slider ranges (min, max) — defaults set in page_predict
SLIDER_AGE = (18, 90)
SLIDER_HDL = (20, 100)
SLIDER_LDL = (50, 250)
SLIDER_SYSTOLIC = (90, 200)
SLIDER_FASTING = (60, 200)
SLIDER_WAIST = (60, 150)


def _smoking_to_code(value) -> int:
    if isinstance(value, (int, float)) and not pd.isna(value):
        return int(value)
    return SMOKING_MAP[str(value)]


def encode_features(row: dict, feature_columns: list) -> pd.DataFrame:
    row = {**row, "smoking_status": _smoking_to_code(row["smoking_status"])}
    df = pd.DataFrame([row])
    df["non_hdl_cholesterol"] = df["cholesterol_level"] - df["cholesterol_hdl"]
    df["ldl_hdl_ratio"] = df["cholesterol_ldl"] / df["cholesterol_hdl"]
    df["metabolic_syndrome"] = (
        (df["diabetes"] == 1) & (df["hypertension"] == 1) & (df["obesity"] == 1)
    ).astype(int)
    df["bp_hypertension"] = df["blood_pressure_systolic"] * df["hypertension"]
    df["age_previous_hd"] = df["age"] * df["previous_heart_disease"]
    df["age_smoking"] = df["age"] * df["smoking_status"]
    df["diabetes_hypertension"] = (
        (df["diabetes"] == 1) & (df["hypertension"] == 1)
    ).astype(int)
    df["ldl_with_diabetes"] = df["cholesterol_ldl"] * df["diabetes"]
    df["medication_usage"] = 0
    return df[feature_columns]

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
            '<div class="metric-card"><div class="value">12</div>'
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
        age = st.slider("Age", min_value=SLIDER_AGE[0], max_value=SLIDER_AGE[1], value=55)
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        obesity = st.selectbox("Obesity", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        previous_heart_disease = st.selectbox(
            "Previous heart disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
        )
        smoking_status = st.selectbox("Smoking status", SMOKING_OPTIONS)

    with right:
        cholesterol_level = st.slider("Total cholesterol", min_value=100, max_value=240, value=200)
        cholesterol_hdl = st.slider(
            "HDL cholesterol (mg/dL)",
            min_value=SLIDER_HDL[0],
            max_value=SLIDER_HDL[1],
            value=50,
            help="Values above 60 mg/dL are generally considered protective.",
        )
        cholesterol_ldl = st.slider(
            "LDL cholesterol (mg/dL)",
            min_value=SLIDER_LDL[0],
            max_value=SLIDER_LDL[1],
            value=130,
        )
        blood_pressure_systolic = st.slider(
            "Systolic BP (mmHg)",
            min_value=SLIDER_SYSTOLIC[0],
            max_value=SLIDER_SYSTOLIC[1],
            value=130,
        )
        fasting_blood_sugar = st.slider(
            "Fasting blood sugar (mg/dL)",
            min_value=SLIDER_FASTING[0],
            max_value=SLIDER_FASTING[1],
            value=110,
        )
        waist_circumference = st.slider(
            "Waist circumference (cm)",
            min_value=SLIDER_WAIST[0],
            max_value=SLIDER_WAIST[1],
            value=90,
        )

    predict = st.button("Predict heart attack risk", type="primary", use_container_width=True)

    if predict:
        row = {
            "age": age,
            "hypertension": hypertension,
            "diabetes": diabetes,
            "cholesterol_level": cholesterol_level,
            "obesity": obesity,
            "waist_circumference": waist_circumference,
            "family_history": 0,
            "smoking_status": SMOKING_MAP[smoking_status],
            "blood_pressure_systolic": blood_pressure_systolic,
            "fasting_blood_sugar": fasting_blood_sugar,
            "cholesterol_hdl": cholesterol_hdl,
            "cholesterol_ldl": cholesterol_ldl,
            "previous_heart_disease": previous_heart_disease,
        }
        encoded = encode_features(row, feature_columns)
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

        XGBoost model loaded from `artifacts/roc_auc_results.pkl` (model, scaler, features, ROC-AUC).

        **Run locally**

        ```bash
        cd ml
        pip install -r requirements.txt
        streamlit run app.py
        ```

        **Deploy on Streamlit Community Cloud**

        - Push the `ml` folder with `artifacts/roc_auc_results.pkl` (model, scaler, features, ROC-AUC).  
        - Set main file: `app.py`  
        - Python 3.10+
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
