"""
Retrain script for the Student Social Media Addiction project.

This reproduces the cleaning + feature engineering steps from eda_models.ipynb,
but (unlike the notebook) bundles the encoders and the Random Forest model into
a single sklearn Pipeline. That's the key fix needed for deployment: the
notebook only saved `rf_default` with joblib, not the OrdinalEncoder /
OneHotEncoder it depended on -- so a raw reload of that .pkl has no idea how
to turn new user input into the columns the model was trained on. Saving the
whole pipeline solves that in one step.

Run this once, locally, in the same folder as `dataset.csv`:
    python train_model.py

It produces:
    - rf_pipeline.pkl   -> preprocessing + trained RandomForestRegressor
    - metadata.json      -> dropdown options / slider ranges + metrics, for the Streamlit app
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

RANDOM_STATE = 42
ACADEMIC_LEVEL_ORDER = ["High School", "Undergraduate", "Graduate"]

FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Academic_Level",
    "Most_Used_Platform",
    "Affects_Academic_Performance",
    "Mental_Health_Score",
    "Relationship_Status",
    "Usage_Sleep_Ratio",
]
TARGET_COLUMN = "Addicted_Score"

ONE_HOT_COLUMNS = [
    "Gender",
    "Affects_Academic_Performance",
    "Relationship_Status",
    "Most_Used_Platform",
]
ORDINAL_COLUMNS = ["Academic_Level"]
NUMERIC_COLUMNS = ["Age", "Mental_Health_Score", "Usage_Sleep_Ratio"]


def load_and_engineer(path="dataset.csv"):
    df = pd.read_csv(path)

    # Same cleaning as the notebook
    df = df.drop(columns=["Student_ID"])
    df = df.drop_duplicates(keep="first").reset_index(drop=True)

    # Save raw ranges before they get combined away, for the app's sliders
    raw_ranges = {
        "Age": [int(df["Age"].min()), int(df["Age"].max())],
        "Mental_Health_Score": [int(df["Mental_Health_Score"].min()), int(df["Mental_Health_Score"].max())],
        "Avg_Daily_Usage_Hours": [float(df["Avg_Daily_Usage_Hours"].min()), float(df["Avg_Daily_Usage_Hours"].max())],
        "Sleep_Hours_Per_Night": [float(df["Sleep_Hours_Per_Night"].min()), float(df["Sleep_Hours_Per_Night"].max())],
        "Addicted_Score": [float(df["Addicted_Score"].min()), float(df["Addicted_Score"].max())],
    }

    # Same feature engineering as the notebook
    df["Usage_Sleep_Ratio"] = df["Avg_Daily_Usage_Hours"] / df["Sleep_Hours_Per_Night"]
    df = df.drop(columns=["Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night"])
    df = df.drop(columns=["Conflicts_Over_Social_Media"])  # collinear with Mental_Health_Score
    df = df.drop(columns=["Country"])  # showed no relationship with the target in EDA

    dropdown_options = {
        "Gender": sorted(df["Gender"].dropna().unique().tolist()),
        "Academic_Level": ACADEMIC_LEVEL_ORDER,
        "Most_Used_Platform": sorted(df["Most_Used_Platform"].dropna().unique().tolist()),
        "Affects_Academic_Performance": sorted(df["Affects_Academic_Performance"].dropna().unique().tolist()),
        "Relationship_Status": sorted(df["Relationship_Status"].dropna().unique().tolist()),
    }

    return df, raw_ranges, dropdown_options


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal", OrdinalEncoder(categories=[ACADEMIC_LEVEL_ORDER]), ORDINAL_COLUMNS),
            (
                "onehot",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                ONE_HOT_COLUMNS,
            ),
            ("numeric", "passthrough", NUMERIC_COLUMNS),
        ]
    )

    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main():
    df, raw_ranges, dropdown_options = load_and_engineer("dataset.csv")

    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    preds = pipeline.predict(x_test)
    metrics = {
        "MAE": round(float(mean_absolute_error(y_test, preds)), 4),
        "MSE": round(float(mean_squared_error(y_test, preds)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "R2": round(float(r2_score(y_test, preds)), 4),
    }
    print("Random Forest test performance:", metrics)

    joblib.dump(pipeline, "rf_pipeline.pkl")

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "raw_ranges": raw_ranges,
        "dropdown_options": dropdown_options,
        "test_metrics": metrics,
    }
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved rf_pipeline.pkl and metadata.json")


if __name__ == "__main__":
    main()
