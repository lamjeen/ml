# Heart Attack Risk Predictor (Streamlit)

Deploy the app with `app.py` and `artifacts/roc_auc_results.pkl` only.

## Quick start

```bash
cd ml
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

## Streamlit Cloud

1. Push this folder to GitHub (include `artifacts/roc_auc_results.pkl`).
2. Main file: `app.py`, Python 3.10+.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI + inference |
| `artifacts/roc_auc_results.pkl` | Model, scaler, features, ROC-AUC |
| `requirements.txt` | Python dependencies |
