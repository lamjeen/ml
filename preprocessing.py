"""Data preprocessing aligned with Untitled10.ipynb."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "gamma": 0.3,
    "min_child_weight": 3,
    "reg_alpha": 0.2,
    "reg_lambda": 2,
    "random_state": 42,
    "eval_metric": "logloss",
}

RAW_FEATURE_COLUMNS = [
    "age",
    "hypertension",
    "diabetes",
    "cholesterol_level",
    "obesity",
    "waist_circumference",
    "family_history",
    "smoking_status",
    "blood_pressure_systolic",
    "fasting_blood_sugar",
    "cholesterol_hdl",
    "cholesterol_ldl",
    "previous_heart_disease",
    "metabolic_syndrome",
    "cholesterol_ratio",
    "bp_hypertension",
    "age_previous_hd",
]

SMOKING_OPTIONS = ["Never", "Past", "Current"]


def default_data_path() -> Path:
    here = Path(__file__).resolve().parent
    local = here / "heart_attack_prediction_indonesia.csv"
    if local.exists():
        return local
    return here.parent / "ml" / "heart_attack_prediction_indonesia.csv"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["metabolic_syndrome"] = (
        (out["diabetes"] == 1) & (out["hypertension"] == 1) & (out["obesity"] == 1)
    ).astype(int)
    out["cholesterol_ratio"] = out["cholesterol_level"] / (out["cholesterol_hdl"] + 1)
    out["bp_hypertension"] = out["blood_pressure_systolic"] * out["hypertension"]
    out["age_previous_hd"] = out["age"] * out["previous_heart_disease"]
    return out


def load_and_prepare(csv_path: Optional[Union[Path, str]] = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path else default_data_path()
    df = pd.read_csv(path)
    drop_cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=drop_cols)
    return engineer_features(df)


def split_encode_scale(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    categorical_features = df.select_dtypes(include=["object", "string"]).columns

    x = df.drop("heart_attack", axis=1)
    y = df["heart_attack"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    x_train_processed = pd.get_dummies(x_train, columns=categorical_features, drop_first=True)
    x_test_processed = pd.get_dummies(x_test, columns=categorical_features, drop_first=True)
    x_test_processed = x_test_processed.reindex(columns=x_train_processed.columns, fill_value=0)

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train_processed)
    x_test_scaled = scaler.transform(x_test_processed)

    return (
        x_train_scaled,
        x_test_scaled,
        y_train,
        y_test,
        scaler,
        list(x_train_processed.columns),
    )


def raw_input_to_frame(row: dict) -> pd.DataFrame:
    """Build a single-row frame from UI inputs (before one-hot encoding)."""
    metabolic = int(
        row["diabetes"] == 1 and row["hypertension"] == 1 and row["obesity"] == 1
    )
    cholesterol_ratio = row["cholesterol_level"] / (row["cholesterol_hdl"] + 1)
    bp_hypertension = row["blood_pressure_systolic"] * row["hypertension"]
    age_previous_hd = row["age"] * row["previous_heart_disease"]

    data = {
        "age": row["age"],
        "hypertension": row["hypertension"],
        "diabetes": row["diabetes"],
        "cholesterol_level": row["cholesterol_level"],
        "obesity": row["obesity"],
        "waist_circumference": row["waist_circumference"],
        "family_history": row["family_history"],
        "smoking_status": row["smoking_status"],
        "blood_pressure_systolic": row["blood_pressure_systolic"],
        "fasting_blood_sugar": row["fasting_blood_sugar"],
        "cholesterol_hdl": row["cholesterol_hdl"],
        "cholesterol_ldl": row["cholesterol_ldl"],
        "previous_heart_disease": row["previous_heart_disease"],
        "metabolic_syndrome": metabolic,
        "cholesterol_ratio": cholesterol_ratio,
        "bp_hypertension": bp_hypertension,
        "age_previous_hd": age_previous_hd,
    }
    return pd.DataFrame([data])


def encode_single_row(
    raw_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    categorical_features = raw_df.select_dtypes(include=["object", "string"]).columns
    processed = pd.get_dummies(raw_df, columns=categorical_features, drop_first=True)
    processed = processed.reindex(columns=feature_columns, fill_value=0)
    return processed
