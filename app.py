import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Social Media Addiction Predictor",
    page_icon="📱",
    layout="centered",
)

APP_DIR = Path(__file__).parent
PIPELINE_PATH = APP_DIR / "rf_pipeline.pkl"
METADATA_PATH = APP_DIR / "metadata.json"


MODEL_COMPARISON = pd.DataFrame(
    [
        ["Random Forest", 0.0889, 0.0542, 0.2329, 0.9796],
        ["KNN", 0.1234, 0.0757, 0.2752, 0.9715],
        ["Decision Tree (Scaled)", 0.0851, 0.0851, 0.2917, 0.9680],
        ["Decision Tree (Raw)", 0.0851, 0.0851, 0.2917, 0.9680],
        ["RBF SVR (Scaled)", 0.1487, 0.0895, 0.2992, 0.9663],
        ["Linear SVR (Scaled)", 0.1974, 0.1095, 0.3309, 0.9588],
        ["Linear SVR (Raw)", 0.1972, 0.1096, 0.3310, 0.9588],
        ["Decision Tree (Pre-Pruned)", 0.1847, 0.1133, 0.3366, 0.9574],
        ["RBF SVR (Raw)", 0.1589, 0.1140, 0.3376, 0.9571],
    ],
    columns=["Model", "MAE", "MSE", "RMSE", "R2"],
)


@st.cache_resource
def load_artifacts():
    if not PIPELINE_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    pipeline = joblib.load(PIPELINE_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return pipeline, metadata


pipeline, metadata = load_artifacts()

st.title("📱 Student Social Media Addiction Predictor")

tab_overview, tab_predict = st.tabs(["📊 Project Overview", "🔮 Predict My Score"])

# ---------------------------------------------------------------- Overview --
with tab_overview:
    st.subheader("What this project does")
    st.markdown(
        """
This project predicts a Gen Z student's **social media addiction score**
(on a 1–9 scale) from demographic, academic, and usage-related information.

- **Problem type:** Regression
- **Dataset:** Student Social Media Addiction Analysis Dataset (Kaggle)
- **Original shape:** 705 rows × 12 columns (705 students, one row each)
- **Target:** `Addicted_Score`
        """
    )

    st.subheader("Key EDA takeaways")
    st.markdown(
        """
- Daily usage hours and addiction score are positively related — more time on
  social media tracks with a higher addiction score, though the relationship
  isn't perfectly linear.
- Country showed no meaningful relationship with the target, so it was dropped.
- `Mental_Health_Score` and `Conflicts_Over_Social_Media` were highly
  correlated with each other, so only one (`Mental_Health_Score`) was kept to
  avoid redundant/collinear features.
- `Avg_Daily_Usage_Hours` and `Sleep_Hours_Per_Night` were combined into a
  single engineered feature, `Usage_Sleep_Ratio`, to capture the trade-off
  between the two directly.
        """
    )

    st.subheader("Features used by the model")
    st.markdown(
        """
`Age`, `Gender`, `Academic_Level`, `Most_Used_Platform`,
`Affects_Academic_Performance`, `Mental_Health_Score`, `Relationship_Status`,
`Usage_Sleep_Ratio` (derived from daily usage hours ÷ sleep hours)
        """
    )

    st.subheader("Model comparison")
    st.caption("Several regressors were trained and compared on held-out test data:")
    st.dataframe(
        MODEL_COMPARISON.sort_values("RMSE").reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown(
        "**Random Forest** had the lowest RMSE and highest R² (≈0.98), so it "
        "was selected as the deployed model."
    )

    if metadata:
        m = metadata["test_metrics"]
        st.info(
            f"Currently loaded pipeline's held-out performance — "
            f"MAE: {m['MAE']} · RMSE: {m['RMSE']} · R²: {m['R2']}"
        )

with tab_predict:
    if pipeline is None:
        st.error(
            "Model files not found. Run `train_model.py` (with `dataset.csv` in "
            "this folder) first — it produces `rf_pipeline.pkl` and "
            "`metadata.json` that this app needs."
        )
    else:
        st.subheader("Tell us about yourself")
        opts = metadata["dropdown_options"]
        ranges = metadata["raw_ranges"]

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider(
                    "Age",
                    min_value=int(ranges["Age"][0]),
                    max_value=int(ranges["Age"][1]),
                    value=int((ranges["Age"][0] + ranges["Age"][1]) // 2),
                )
                gender = st.selectbox("Gender", opts["Gender"])
                academic_level = st.selectbox("Academic level", opts["Academic_Level"])
                platform = st.selectbox("Most used platform", opts["Most_Used_Platform"])

            with col2:
                affects_academics = st.selectbox(
                    "Does social media affect your academic performance?",
                    opts["Affects_Academic_Performance"],
                )
                relationship_status = st.selectbox(
                    "Relationship status", opts["Relationship_Status"]
                )
                mental_health = st.slider(
                    "Mental health score (1 = poor, 10 = excellent)",
                    min_value=1,
                    max_value=10,
                    value=int((ranges["Mental_Health_Score"][0] + ranges["Mental_Health_Score"][1]) // 2),
                )

            st.markdown("**Usage & sleep**")
            col3, col4 = st.columns(2)
            with col3:
                usage_hours = st.slider(
                    "Average daily social media usage (hours)",
                    min_value=float(ranges["Avg_Daily_Usage_Hours"][0]),
                    max_value=float(ranges["Avg_Daily_Usage_Hours"][1]),
                    value=round(
                        (ranges["Avg_Daily_Usage_Hours"][0] + ranges["Avg_Daily_Usage_Hours"][1]) / 2, 1
                    ),
                    step=0.1,
                )
            with col4:
                sleep_hours = st.slider(
                    "Sleep per night (hours)",
                    min_value=float(ranges["Sleep_Hours_Per_Night"][0]),
                    max_value=float(ranges["Sleep_Hours_Per_Night"][1]),
                    value=round(
                        (ranges["Sleep_Hours_Per_Night"][0] + ranges["Sleep_Hours_Per_Night"][1]) / 2, 1
                    ),
                    step=0.1,
                )

            submitted = st.form_submit_button("Predict my addiction score", use_container_width=True)

        if submitted:
            usage_sleep_ratio = usage_hours / sleep_hours

            input_row = pd.DataFrame(
                [
                    {
                        "Age": age,
                        "Gender": gender,
                        "Academic_Level": academic_level,
                        "Most_Used_Platform": platform,
                        "Affects_Academic_Performance": affects_academics,
                        "Mental_Health_Score": mental_health,
                        "Relationship_Status": relationship_status,
                        "Usage_Sleep_Ratio": usage_sleep_ratio,
                    }
                ]
            )

            prediction = float(pipeline.predict(input_row)[0])
            prediction = max(ranges["Addicted_Score"][0], min(ranges["Addicted_Score"][1], prediction))

            st.divider()
            st.metric("Predicted Addiction Score", f"{prediction:.1f} / 9")

            if prediction < 4:
                st.success("Low predicted addiction risk.")
            elif prediction < 7:
                st.warning("Moderate predicted addiction risk.")
            else:
                st.error("High predicted addiction risk.")

            st.caption(
                "This is an estimate from a model trained on survey data — not a "
                "clinical assessment."
            )
