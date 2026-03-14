from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ModelRunResult:
    name: str
    estimator: Pipeline
    rmse: float
    mae: float
    r2: float


def _search_iterations(search_space: Dict[str, List[object]], max_iter: int = 6) -> int:
    total = 1
    for values in search_space.values():
        total *= max(1, len(values))
    return min(max_iter, total)


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def model_candidates(with_xgboost: bool = True) -> Dict[str, Tuple[object, Dict[str, List[object]]]]:
    candidates: Dict[str, Tuple[object, Dict[str, List[object]]]] = {
        "LinearRegression": (
            LinearRegression(),
            {},
        ),
        "Ridge": (
            Ridge(random_state=42),
            {
                "model__alpha": [0.01, 0.1, 1.0, 10.0, 50.0],
            },
        ),
        "Lasso": (
            Lasso(random_state=42, max_iter=5000),
            {
                "model__alpha": [0.001, 0.01, 0.1, 1.0],
            },
        ),
        "RandomForestRegressor": (
            RandomForestRegressor(random_state=42),
            {
                "model__n_estimators": [80, 140],
                "model__max_depth": [None, 10, 18],
                "model__min_samples_split": [2, 6],
                "model__min_samples_leaf": [1, 2],
            },
        ),
        "GradientBoostingRegressor": (
            GradientBoostingRegressor(random_state=42),
            {
                "model__n_estimators": [100, 180],
                "model__learning_rate": [0.03, 0.08],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.75, 1.0],
            },
        ),
    }

    if with_xgboost:
        try:
            from xgboost import XGBRegressor

            candidates["XGBRegressor"] = (
                XGBRegressor(
                    random_state=42,
                    objective="reg:squarederror",
                    n_jobs=-1,
                    verbosity=0,
                ),
                {
                    "model__n_estimators": [120, 220],
                    "model__max_depth": [3, 5],
                    "model__learning_rate": [0.03, 0.08],
                    "model__subsample": [0.8, 1.0],
                    "model__colsample_bytree": [0.8, 1.0],
                },
            )
        except Exception:
            pass

    return candidates


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Tuple[float, float, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def train_and_select_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
) -> Tuple[ModelRunResult, List[ModelRunResult]]:
    runs: List[ModelRunResult] = []

    for name, (model, search_space) in model_candidates().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        if search_space:
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=search_space,
                n_iter=_search_iterations(search_space),
                scoring="neg_root_mean_squared_error",
                cv=3,
                random_state=42,
                n_jobs=1,
            )
            search.fit(X_train, y_train)
            fitted = search.best_estimator_
        else:
            fitted = pipeline.fit(X_train, y_train)

        y_pred = fitted.predict(X_test)
        rmse, mae, r2 = evaluate_predictions(y_test, y_pred)
        runs.append(ModelRunResult(name=name, estimator=fitted, rmse=rmse, mae=mae, r2=r2))

    runs.sort(key=lambda item: item.rmse)
    return runs[0], runs
