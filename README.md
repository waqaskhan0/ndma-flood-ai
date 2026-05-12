# NDMA Flood AI

Flood-risk dashboard and daily alert pipeline for Pakistan districts.

## Run Locally

```powershell
pip install -r requirements.txt
streamlit run app/app.py
```

Open:

```text
http://localhost:8501
```

## Run Daily Prediction Pipeline

```powershell
python operational_pipeline.py
```

Outputs:

```text
data/processed/latest_predictions.csv
data/processed/active_alerts.csv
data/processed/active_alerts.json
```

## Retrain Model

```powershell
python train_model.py
```

The saved production model is stored in:

```text
models/flood_model_best.pkl
models/scaler.pkl
models/model_info.pkl
```

## Deployment

Deploy `app/app.py` on Streamlit Community Cloud.

The included GitHub Actions workflow runs `operational_pipeline.py` every day at 06:00 Pakistan time and commits the latest alert files back to the repository.
