"""Train XGBoost and persist artifacts for the Streamlit app."""

import json
from pathlib import Path

import joblib
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

from preprocessing import XGB_PARAMS, load_and_prepare, split_encode_scale

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare()
    x_train, x_test, y_train, y_test, scaler, feature_columns = split_encode_scale(df)

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(x_train, y_train)

    y_probs = model.predict_proba(x_test)[:, 1]
    y_pred = model.predict(x_test)
    roc_auc = float(roc_auc_score(y_test, y_probs))
    fpr, tpr, _ = roc_curve(y_test, y_probs)

    joblib.dump(model, ARTIFACTS_DIR / "xgb_model.joblib")
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    joblib.dump(feature_columns, ARTIFACTS_DIR / "feature_columns.joblib")

    metrics = {
        "roc_auc": roc_auc,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": len(feature_columns),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "xgb_params": XGB_PARAMS,
    }
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"ROC-AUC (test): {roc_auc:.4f}")
    print(f"Artifacts saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
