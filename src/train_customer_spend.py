from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

from src.modeling import build_preprocessor


sns.set_theme(style="whitegrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train customer next-month spend regression model.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/customer_spend_dataset_200k.csv",
        help="Path to customer transaction CSV/Excel.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--remove-outliers",
        action="store_true",
        help="Apply IQR outlier row removal before training (disabled by default).",
    )
    parser.add_argument(
        "--keep-customer-id",
        action="store_true",
        help="Keep Customer_ID as a model feature (dropped by default).",
    )
    parser.add_argument(
        "--no-log-target",
        action="store_true",
        help="Disable log1p/expm1 target transform during training.",
    )
    parser.add_argument(
        "--force-model",
        type=str,
        default="",
        help="Force a single model by name (e.g. XGBRegressor, GradientBoostingRegressor).",
    )
    return parser.parse_args()


def _to_bool_series(series: pd.Series) -> pd.Series:
    mapping = {
        "yes": 1,
        "no": 0,
        "returned": 1,
        "not returned": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
    }
    return series.astype(str).str.strip().str.lower().map(mapping).fillna(0).astype(int)


def load_transactions(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    required_cols = {
        "Customer_ID",
        "Transaction_Date",
        "Purchased_Amount",
        "Quantity_Purchased",
        "Unit_Price",
        "Website_Visits_Last_30_Days",
        "Discount_Used",
        "Return_Status",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned["Transaction_Date"] = pd.to_datetime(cleaned["Transaction_Date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["Transaction_Date", "Customer_ID", "Purchased_Amount"])

    numeric_cols = [
        "Age",
        "Monthly_Salary",
        "Family_Size",
        "Quantity_Purchased",
        "Unit_Price",
        "Purchased_Amount",
        "Website_Visits_Last_30_Days",
    ]

    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    cat_cols = cleaned.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        cleaned[col] = cleaned[col].fillna("Unknown")

    # Drop obvious duplicates from transactional records.
    if "Transaction_ID" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["Transaction_ID"])
    else:
        cleaned = cleaned.drop_duplicates()

    return cleaned


def create_monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["month"] = work["Transaction_Date"].dt.to_period("M").dt.to_timestamp()
    work["discount_flag"] = _to_bool_series(work["Discount_Used"])
    work["return_flag"] = _to_bool_series(work["Return_Status"])
    work["loyalty_flag"] = _to_bool_series(work.get("Loyalty_Member", pd.Series([0] * len(work))))

    agg_map: dict[str, tuple[str, str]] = {
        "monthly_spend": ("Purchased_Amount", "sum"),
        "avg_spending": ("Purchased_Amount", "mean"),
        "transaction_frequency": ("Transaction_ID", "count"),
        "avg_quantity": ("Quantity_Purchased", "mean"),
        "avg_unit_price": ("Unit_Price", "mean"),
        "engagement_indicator": ("Website_Visits_Last_30_Days", "mean"),
        "discount_rate": ("discount_flag", "mean"),
        "return_rate": ("return_flag", "mean"),
        "loyalty_member_flag": ("loyalty_flag", "max"),
    }

    optional_first_cols = [
        "Age",
        "Gender",
        "Location_City",
        "State",
        "Region_Type",
        "Occupation",
        "Monthly_Salary",
        "Family_Size",
        "Marital_Status",
        "Home_Ownership_Status",
        "Purchase_Channel",
        "Payment_Method",
    ]
    for col in optional_first_cols:
        if col in work.columns:
            agg_map[col] = (col, "first")

    monthly = work.groupby(["Customer_ID", "month"], as_index=False).agg(**agg_map)

    monthly = monthly.sort_values(["Customer_ID", "month"])

    monthly["spend_lag_1"] = monthly.groupby("Customer_ID")["monthly_spend"].shift(1)
    monthly["spend_lag_2"] = monthly.groupby("Customer_ID")["monthly_spend"].shift(2)
    monthly["spend_lag_3"] = monthly.groupby("Customer_ID")["monthly_spend"].shift(3)
    monthly["spend_rolling_3"] = (
        monthly.groupby("Customer_ID")["monthly_spend"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    lag_cols = ["spend_lag_1", "spend_lag_2", "spend_lag_3", "spend_rolling_3"]
    for col in lag_cols:
        monthly[col] = monthly[col].fillna(monthly["monthly_spend"])

    monthly["next_month_spend"] = monthly.groupby("Customer_ID")["monthly_spend"].shift(-1)

    # Feature to capture spend trend behavior over time.
    monthly["spend_momentum"] = monthly.groupby("Customer_ID")["monthly_spend"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)

    feature_df = monthly.dropna(subset=["next_month_spend"]).copy()
    feature_df = feature_df.drop(columns=["month"])

    return feature_df


def remove_outliers(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, int]:
    numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != target_col]
    mask = pd.Series(True, index=df.index)

    for col in numeric_cols + [target_col]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or np.isnan(iqr):
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask &= df[col].between(lower, upper)

    cleaned = df.loc[mask].copy()
    return cleaned, int((~mask).sum())


def save_eda_plots(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    sns.histplot(df["monthly_spend"], kde=True, color="#1f77b4")
    plt.title("Monthly Spend Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_spend_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=["number"])
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, linewidths=0.4)
    plt.title("Customer Feature Correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "customer_correlation_heatmap.png", dpi=150)
    plt.close()

    if "Purchase_Category" in df.columns:
        cat_spend = (
            df.groupby("Purchase_Category", as_index=False)["monthly_spend"]
            .mean()
            .sort_values("monthly_spend", ascending=False)
        )
        plt.figure(figsize=(10, 5))
        sns.barplot(data=cat_spend, x="Purchase_Category", y="monthly_spend", palette="crest")
        plt.title("Average Monthly Spend by Purchase Category")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "spend_by_purchase_category.png", dpi=150)
        plt.close()


def train_requested_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor,
    use_log_target: bool,
    force_model: str = "",
) -> tuple[str, Pipeline, pd.DataFrame, np.ndarray]:
    xgb_model = None
    try:
        from xgboost import XGBRegressor

        xgb_model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=350,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    except Exception:
        xgb_model = None

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=140,
            max_depth=16,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=42,
        ),
        "QuantileGradientBoostingRegressor": GradientBoostingRegressor(
            loss="quantile",
            alpha=0.3,
            n_estimators=220,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=42,
        ),
    }

    if xgb_model is not None:
        models["XGBRegressor"] = xgb_model

    if force_model:
        if force_model not in models:
            raise ValueError(
                f"Invalid --force-model '{force_model}'. Available models: {sorted(models.keys())}"
            )
        models = {force_model: models[force_model]}

    rows = []
    best_name = ""
    best_pipeline: Pipeline | None = None
    best_preds: np.ndarray | None = None
    best_selection_score = np.inf
    best_mae = np.inf
    best_r2 = -np.inf

    low_spend_threshold = float(np.quantile(y_test, 0.2))
    low_mask = y_test <= low_spend_threshold

    for name, estimator in models.items():
        base_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        if use_log_target:
            pipeline = TransformedTargetRegressor(
                regressor=base_pipeline,
                func=np.log1p,
                inverse_func=np.expm1,
            )
        else:
            pipeline = base_pipeline
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        mae = float(mean_absolute_error(y_test, preds))
        mse = float(mean_squared_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        low_spend_mae = float(mean_absolute_error(y_test[low_mask], preds[low_mask])) if low_mask.any() else mae
        selection_score = 0.6 * low_spend_mae + 0.4 * mae

        rows.append(
            {
                "model": name,
                "mae": round(mae, 4),
                "mse": round(mse, 4),
                "r2": round(r2, 4),
                "low_spend_mae": round(low_spend_mae, 4),
                "selection_score": round(selection_score, 4),
            }
        )

        if (selection_score < best_selection_score) or (
            np.isclose(selection_score, best_selection_score) and mae < best_mae
        ) or (
            np.isclose(selection_score, best_selection_score) and np.isclose(mae, best_mae) and r2 > best_r2
        ):
            best_selection_score = selection_score
            best_mae = mae
            best_r2 = r2
            best_name = name
            best_pipeline = pipeline
            best_preds = preds

    comparison_df = pd.DataFrame(rows).sort_values(["selection_score", "mae", "r2"], ascending=[True, True, False])
    return best_name, best_pipeline, comparison_df, best_preds


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = project_root / data_path

    artifacts_dir = project_root / "artifacts" / "customer_spend"
    plots_dir = artifacts_dir / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    tx_df = load_transactions(data_path)
    cleaned_tx_df = clean_transactions(tx_df)

    feature_df = create_monthly_features(cleaned_tx_df)
    removed_outliers = 0
    if args.remove_outliers:
        feature_df, removed_outliers = remove_outliers(feature_df, "next_month_spend")

    if not args.keep_customer_id and "Customer_ID" in feature_df.columns:
        feature_df = feature_df.drop(columns=["Customer_ID"])

    save_eda_plots(feature_df, plots_dir)

    X = feature_df.drop(columns=["next_month_spend"])
    y = feature_df["next_month_spend"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
    )

    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
    best_model_name, best_pipeline, comparison_df, y_pred = train_requested_models(
        X_train,
        y_train,
        X_test,
        y_test,
        preprocessor,
        use_log_target=not args.no_log_target,
        force_model=args.force_model.strip(),
    )

    if best_pipeline is None or y_pred is None:
        raise RuntimeError("Model training failed to produce a valid estimator.")

    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    model_rows = comparison_df.to_dict(orient="records")

    metadata = {
        "best_model": best_model_name,
        "target": "next_month_spend",
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "removed_outliers": removed_outliers,
        "source_rows": int(tx_df.shape[0]),
        "monthly_rows_after_feature_engineering": int(feature_df.shape[0]),
        "log_target_enabled": bool(not args.no_log_target),
    }

    metrics = {
        "model": best_model_name,
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "r2": round(r2, 4),
    }

    predictions_df = pd.DataFrame(
        {
            "actual": y_test.values,
            "predicted": y_pred,
            "residual": y_test.values - y_pred,
        }
    )

    cleaned_tx_df.to_csv(artifacts_dir / "cleaned_transactions.csv", index=False)
    feature_df.to_csv(artifacts_dir / "engineered_monthly_features.csv", index=False)
    predictions_df.to_csv(artifacts_dir / "test_predictions.csv", index=False)

    joblib.dump(best_pipeline, artifacts_dir / "model.pkl")
    with (artifacts_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with (artifacts_dir / "model_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    with (artifacts_dir / "model_comparison.json").open("w", encoding="utf-8") as file:
        json.dump(model_rows, file, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Artifacts saved at: {artifacts_dir}")


if __name__ == "__main__":
    main()
