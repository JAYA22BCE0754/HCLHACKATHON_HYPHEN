from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import create_sample_dataset, load_dataset, remove_outliers_iqr, validate_schema
from src.modeling import build_preprocessor, train_and_select_model
from src.visualization import (
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_model_comparison,
    plot_residuals,
    plot_target_distribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate regression models.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/housing_sample.csv",
        help="Path to CSV or Excel dataset.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="SalePrice",
        help="Target column name.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--skip-outlier-removal",
        action="store_true",
        help="Skip IQR outlier removal on train split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = project_root / "artifacts"
    plots_dir = artifacts_dir / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = project_root / data_path

    if not data_path.exists():
        data_path = create_sample_dataset(data_path)
        print(f"Sample dataset generated at: {data_path}")

    df = load_dataset(data_path)
    validate_schema(df, args.target)

    plot_target_distribution(df, args.target, plots_dir / "target_distribution.png")
    plot_correlation_heatmap(df, plots_dir / "correlation_heatmap.png")

    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
    )

    train_df = X_train.copy()
    train_df[args.target] = y_train

    removed_outliers = 0
    if args.skip_outlier_removal:
        cleaned_train_df = train_df.copy()
    else:
        cleaned_train_df, removed_outliers = remove_outliers_iqr(train_df, args.target)

    X_train_clean = cleaned_train_df.drop(columns=[args.target])
    y_train_clean = cleaned_train_df[args.target]

    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train_clean)

    best_run, all_runs = train_and_select_model(
        X_train=X_train_clean,
        y_train=y_train_clean,
        X_test=X_test,
        y_test=y_test,
        preprocessor=preprocessor,
    )

    model_rows = [
        {
            "model": run.name,
            "rmse": round(run.rmse, 4),
            "mae": round(run.mae, 4),
            "r2": round(run.r2, 4),
        }
        for run in all_runs
    ]

    y_pred = best_run.estimator.predict(X_test)

    predictions_df = pd.DataFrame(
        {
            "actual": y_test.values,
            "predicted": y_pred,
            "residual": y_test.values - y_pred,
        }
    )

    cleaned_train_df.to_csv(artifacts_dir / "cleaned_train_dataset.csv", index=False)
    predictions_df.to_csv(artifacts_dir / "test_predictions.csv", index=False)

    joblib.dump(best_run.estimator, artifacts_dir / "model.pkl")

    metadata = {
        "best_model": best_run.name,
        "target": args.target,
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "removed_outliers": removed_outliers,
        "train_rows": int(X_train_clean.shape[0]),
        "test_rows": int(X_test.shape[0]),
    }

    best_metrics = {
        "model": best_run.name,
        "rmse": round(best_run.rmse, 4),
        "mae": round(best_run.mae, 4),
        "r2": round(best_run.r2, 4),
    }

    with (artifacts_dir / "model_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    with (artifacts_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(best_metrics, file, indent=2)

    with (artifacts_dir / "model_comparison.json").open("w", encoding="utf-8") as file:
        json.dump(model_rows, file, indent=2)

    plot_model_comparison(model_rows, plots_dir / "model_rmse_comparison.png")
    plot_residuals(y_test, pd.Series(y_pred, index=y_test.index), plots_dir / "residual_plot.png")
    plot_feature_importance(
        best_run.estimator,
        X_test,
        y_test,
        plots_dir / "feature_importance.png",
    )

    print(json.dumps(best_metrics, indent=2))
    print(f"Artifacts saved to: {artifacts_dir}")


if __name__ == "__main__":
    main()
