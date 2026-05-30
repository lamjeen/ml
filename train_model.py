"""Train XGBoost model and save artifacts for Streamlit deployment."""

from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

DATA_PATH = Path(__file__).parent / "heart_attack_prediction_indonesia.csv"
ARTIFACT_DIR = Path(__file__).parent / "artifacts"

COLUMNS_TO_DROP = [
    "alcohol_consumption",
    "blood_pressure_diastolic",
    "triglycerides",
    "physical_activity",
    "air_pollution_exposure",
    "participated_in_free_screening",
    "medication_usage",
    "EKG_results",
    "sleep_hours",
    "stress_level",
    "dietary_habits",
    "income_level",
    "region_Urban",
    "region",
    "gender",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_risk"] = out["age"] ** 2
    out["metabolic_syndrome"] = (
        (out["diabetes"] == 1) & (out["hypertension"] == 1) & (out["obesity"] == 1)
    ).astype(int)
    out["cholesterol_ratio"] = out["cholesterol_level"] / (out["cholesterol_hdl"] + 1)
    out["bp_hypertension"] = out["blood_pressure_systolic"] * out["hypertension"]
    out["age_previous_hd"] = out["age"] * out["previous_heart_disease"]
    return out


def preprocess(df: pd.DataFrame):
    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])
    df = engineer_features(df)

    categorical_features = df.select_dtypes(include=["object", "string"]).columns.tolist()
    x = df.drop("heart_attack", axis=1)
    y = df["heart_attack"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    x_train_processed = pd.get_dummies(x_train, columns=categorical_features, drop_first=True)
    x_test_processed = pd.get_dummies(x_test, columns=categorical_features, drop_first=True)
    x_test_processed = x_test_processed.reindex(columns=x_train_processed.columns, fill_value=0)

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train_processed)
    x_test_scaled = scaler.transform(x_test_processed)

    return x_train_scaled, x_test_scaled, y_train, y_test, scaler, list(x_train_processed.columns)


def main():
    df = pd.read_csv(DATA_PATH)
    x_train, x_test, y_train, y_test, scaler, feature_columns = preprocess(df)

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.3,
        min_child_weight=3,
        reg_alpha=0.2,
        reg_lambda=2,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(x_train, y_train)

    y_probs = model.predict_proba(x_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_probs)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "xgb_model.joblib")
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")
    joblib.dump(feature_columns, ARTIFACT_DIR / "feature_columns.joblib")
    joblib.dump({"roc_auc": float(roc_auc)}, ARTIFACT_DIR / "metrics.joblib")

    print(f"Saved artifacts to {ARTIFACT_DIR}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()
