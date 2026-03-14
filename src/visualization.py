from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


sns.set_theme(style="whitegrid")


def plot_target_distribution(df: pd.DataFrame, target_col: str, output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    sns.histplot(df[target_col], kde=True, color="#2c7fb8")
    plt.title(f"Target Distribution: {target_col}")
    plt.xlabel(target_col)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        return

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        square=False,
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_model_comparison(model_rows: List[dict], output_path: Path) -> None:
    comp_df = pd.DataFrame(model_rows).sort_values("rmse", ascending=True)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=comp_df, x="model", y="rmse", palette="viridis")
    plt.title("Model Comparison by RMSE")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residuals(y_true: pd.Series, y_pred: pd.Series, output_path: Path) -> None:
    residuals = y_true - y_pred

    plt.figure(figsize=(9, 5))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, color="#1d91c0")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Residual Plot")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
    top_n: int = 12,
) -> None:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=8,
        random_state=42,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance": result.importances_mean,
        }
    ).sort_values("importance", ascending=False)

    importance_df = importance_df.head(top_n)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=importance_df, x="importance", y="feature", palette="mako")
    plt.title("Permutation Feature Importance")
    plt.xlabel("Importance (higher means more impact)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
