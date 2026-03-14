# ML Regression Hackathon Project

This project provides an end-to-end machine learning regression system for predictive analytics.

## What This Includes

- Data ingestion from CSV/Excel
- Data validation, imputation, encoding, scaling, and outlier handling
- Feature analysis and visual EDA
- Multiple regression model training and comparison
- Hyperparameter tuning using `RandomizedSearchCV`
- Evaluation with RMSE, MAE, and R2
- Visual explanations (residuals, correlation, feature importance)
- Trained model export (`.pkl`)
- FastAPI inference endpoint for predictions
- Streamlit UI for interactive predictions

## Project Structure

- `src/data.py`: Ingestion, schema checks, sample data generation, outlier removal
- `src/modeling.py`: Model definitions, tuning, metrics, model selection
- `src/visualization.py`: EDA and explanation plots
- `src/train.py`: Main training pipeline
- `src/api.py`: Prediction API using FastAPI
- `src/streamlit_app.py`: Streamlit app for single and batch predictions
- `notebooks/regression_workflow.ipynb`: Notebook walkthrough
- `artifacts/`: Saved outputs (model, metrics, plots)

## Quick Start

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train and generate all artifacts:

```powershell
python -m src.train --target SalePrice
```

If `data/raw/housing_sample.csv` does not exist, a sample dataset is generated automatically.

3. Run the API:

```powershell
uvicorn src.api:app --reload
```

4. Run the Streamlit UI:

```powershell
streamlit run src/streamlit_app.py
```

If `streamlit` is not in PATH:

```powershell
py -3.12 -m streamlit run src/streamlit_app.py
```

5. Test prediction endpoint:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType 'application/json' -Body '{"records":[{"MedInc":5.2,"HouseAge":20,"AveRooms":6.1,"AveBedrms":1.1,"Population":1000,"AveOccup":2.8,"Latitude":34.2,"Longitude":-118.4,"IncomeBand":"high","RegionCluster":"south"}]}'
```

## Customer Spend Pipeline (Kaggle Dataset)

Run the dedicated customer spend workflow (cleaning + feature engineering + training):

```powershell
py -3.12 -m src.train_customer_spend --data-path data/raw/customer_spend_dataset_200k.csv
```

This pipeline predicts `next_month_spend` using engineered monthly customer features, including:

- `avg_spending`
- `transaction_frequency`
- `engagement_indicator`
- `discount_rate`
- `return_rate`
- `spend_momentum`

Customer spend artifacts are saved in `artifacts/customer_spend/`.

## Outputs

After training, outputs are saved under `artifacts/`:

- `model.pkl`
- `model_metadata.json`
- `metrics.json`
- `model_comparison.json`
- `cleaned_train_dataset.csv`
- `test_predictions.csv`
- `plots/`:
  - `target_distribution.png`
  - `correlation_heatmap.png`
  - `model_rmse_comparison.png`
  - `residual_plot.png`
  - `feature_importance.png`

## Notes

- Optional XGBoost support is enabled automatically if `xgboost` is installed.
- You can replace the sample data with your own CSV/Excel and pass `--data-path` and `--target`.
