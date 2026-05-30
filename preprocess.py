"""Shared preprocessing for training and Streamlit inference."""

import pandas as pd

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

SMOKING_OPTIONS = ["Never", "Past", "Current"]


def engineer_features_from_row(row: dict) -> dict:
    age = row["age"]
    hypertension = row["hypertension"]
    diabetes = row["diabetes"]
    obesity = row["obesity"]
    cholesterol_level = row["cholesterol_level"]
    cholesterol_hdl = row["cholesterol_hdl"]
    blood_pressure_systolic = row["blood_pressure_systolic"]
    previous_heart_disease = row["previous_heart_disease"]

    row["age_risk"] = age**2
    row["metabolic_syndrome"] = int(
        diabetes == 1 and hypertension == 1 and obesity == 1
    )
    row["cholesterol_ratio"] = cholesterol_level / (cholesterol_hdl + 1)
    row["bp_hypertension"] = blood_pressure_systolic * hypertension
    row["age_previous_hd"] = age * previous_heart_disease
    return row


def row_to_feature_frame(row: dict, feature_columns: list) -> pd.DataFrame:
    engineered = engineer_features_from_row(row.copy())
    frame = pd.DataFrame([engineered])
    categorical = [c for c in frame.columns if frame[c].dtype == object]
    processed = pd.get_dummies(frame, columns=categorical, drop_first=True)
    processed = processed.reindex(columns=feature_columns, fill_value=0)
    return processed
