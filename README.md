# Heart Attack Risk Predictor (Streamlit)

Streamlit deployment of the XGBoost pipeline from `Untitled10.ipynb`, with **ROC-AUC** as the primary evaluation metric.

## Quick start

```bash
cd v2
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Open http://localhost:8501

## Expected metrics

On the hold-out test set (same split as the notebook):

- **ROC-AUC ≈ 0.812**

## Streamlit Cloud

1. Upload this folder to GitHub.
2. Include `heart_attack_prediction_indonesia.csv` and run `train_model.py` once, then commit `artifacts/`.
3. Create app with main file `app.py` and Python 3.10+.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI |
| `preprocessing.py` | Feature engineering & encoding |
| `train_model.py` | Train XGBoost and save artifacts |
| `artifacts/` | Model, scaler, metrics (generated) |
