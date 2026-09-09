# Social Media Addiction Predictor

## What changed from the original notebook

The notebook trained several models and saved only the fitted
`RandomForestRegressor` (`rf_default`) with `joblib.dump(...)`. That's not
enough to deploy: the model expects an already-encoded row (ordinal-encoded
`Academic_Level`, one-hot-encoded `Gender` / `Affects_Academic_Performance` /
`Relationship_Status` / `Most_Used_Platform`, plus the engineered
`Usage_Sleep_Ratio`), and the encoders themselves were never saved.

`train_model.py` fixes that by wrapping the *same* cleaning, feature
engineering, and encoding steps from the notebook, plus the Random Forest,
into one `sklearn.Pipeline`. Saving that single object means the Streamlit
app can just hand it a raw row (`Age=20, Gender="Female", ...`) and get a
prediction back — no manual encoding logic duplicated in the app.

## Setup

1. Put `dataset.csv` (the Kaggle "Student Social Media Addiction" dataset) in
   this folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train and export the pipeline:
   ```bash
   python train_model.py
   ```
   This creates `rf_pipeline.pkl` and `metadata.json`.
4. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

## Files

- `train_model.py` — rebuilds the pipeline (cleaning + encoding + Random
  Forest) from `dataset.csv` and saves `rf_pipeline.pkl` + `metadata.json`.
- `streamlit_app.py` — two-tab app: a project overview (dataset info, EDA
  takeaways, model comparison) and a prediction form.
- `requirements.txt` — dependencies.

## Notes / assumptions

- Dropdown options (platforms, gender values, etc.) and slider ranges are
  read from `metadata.json`, which is generated from your actual
  `dataset.csv` — so they'll always match what the model was trained on,
  even if your data differs slightly from the original Kaggle version.
- The prediction form asks for daily usage hours and sleep hours separately
  (more intuitive than asking for a ratio) and computes `Usage_Sleep_Ratio`
  internally, exactly as the notebook did.
- The model comparison table on the Overview tab is the static result from
  the notebook's evaluation; it isn't recomputed by the app.
