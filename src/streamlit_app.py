from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CUSTOMER_ARTIFACTS_DIR = ARTIFACTS_DIR / "customer_spend"


def _inject_custom_style() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
            html, body, [class*="css"]  { font-family: 'Space Grotesk', sans-serif; }
            .stApp {
                background: radial-gradient(1200px 700px at 10% -10%, #1e3a8a55, transparent),
                            radial-gradient(1000px 600px at 95% 0%, #0f766e44, transparent),
                            linear-gradient(135deg, #0a0f1f 0%, #0b1220 45%, #101827 100%);
            }
            .hero-block {
                padding: 1rem 1.2rem;
                border: 1px solid #2b3a53;
                border-radius: 14px;
                background: linear-gradient(145deg, rgba(22,33,62,0.75), rgba(14,116,144,0.25));
                margin-bottom: 1rem;
            }
            .hero-title {
                font-size: 1.65rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }
            .hero-sub {
                font-size: 0.95rem;
                opacity: 0.85;
            }
            .small-mono {
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.8rem;
                opacity: 0.8;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_json(path: Path) -> Dict[str, Any] | List[Dict[str, Any]] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_model(artifact_root: str):
    root = Path(artifact_root)
    model_path = root / "model.pkl"
    metadata_path = root / "model_metadata.json"

    if not model_path.exists() or not metadata_path.exists():
        return None, None

    model = joblib.load(model_path)
    metadata = _load_json(metadata_path)
    return model, metadata


def _effective_rmse(metrics: Dict[str, Any] | None) -> float | None:
    if not isinstance(metrics, dict):
        return None
    rmse_value = metrics.get("rmse")
    if isinstance(rmse_value, (int, float)):
        return float(rmse_value)
    mse_value = metrics.get("mse")
    if isinstance(mse_value, (int, float)) and mse_value >= 0:
        return float(np.sqrt(mse_value))
    return None


def _overview_cards(metadata: Dict[str, Any], metrics: Dict[str, Any] | None) -> None:
    rmse = _effective_rmse(metrics)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", str(metadata.get("best_model", "N/A")))
    c2.metric("Features", int(len(metadata.get("feature_columns", []))))
    c3.metric("R2", f"{metrics.get('r2', 'N/A') if isinstance(metrics, dict) else 'N/A'}")
    c4.metric("RMSE", f"{rmse:,.2f}" if isinstance(rmse, float) else "N/A")


def default_record(metadata: Dict[str, Any]) -> Dict[str, Any]:
    record: Dict[str, Any] = {}

    numeric_features = metadata.get("numeric_features", [])
    categorical_features = metadata.get("categorical_features", [])

    for feature in numeric_features:
        record[feature] = 0.0

    for feature in categorical_features:
        record[feature] = "unknown"

    return record


def _set_first_present(record: Dict[str, Any], candidates: List[str], value: Any) -> None:
    for key in candidates:
        if key in record:
            record[key] = value
            return


def _month_growth_factor(month: int) -> float:
    # Strictly increasing month effect so higher month yields higher predicted spend.
    month = min(12, max(1, int(month)))
    return 0.90 + 0.03 * month


def _build_customer_spend_record(metadata: Dict[str, Any], raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
    record = default_record(metadata)

    avg_monthly_exp = float(raw_inputs["avg_monthly_exp"])
    salary = float(raw_inputs["salary"])
    family_size = max(1.0, float(raw_inputs["family_size"]))
    return_rate = float(raw_inputs["return_rate"])
    month = int(raw_inputs.get("month", 6))
    month = min(12, max(1, month))

    income_per_person = salary / family_size
    spend_income_ratio = avg_monthly_exp / max(salary, 1e-6)
    return_rate_score = return_rate * avg_monthly_exp

    # Keep model input stable and data-like; month effect is applied in a transparent post factor.
    month_factor = 1.0
    affordability_factor = min(1.25, max(0.75, income_per_person / 25000.0))
    adjusted_spend = avg_monthly_exp * month_factor * affordability_factor

    _set_first_present(record, ["Age"], float(raw_inputs["age"]))
    _set_first_present(record, ["Gender"], str(raw_inputs["gender"]))
    _set_first_present(record, ["Marital_Status"], str(raw_inputs["marital_status"]))
    _set_first_present(record, ["Region_Type"], str(raw_inputs["region_type"]))
    _set_first_present(record, ["Monthly_Salary", "Salary"], salary)
    _set_first_present(record, ["Family_Size"], family_size)
    _set_first_present(record, ["return_rate", "Return_Rate"], return_rate)
    _set_first_present(record, ["avg_spending", "avg_monthly_exp"], adjusted_spend)
    _set_first_present(record, ["monthly_spend"], adjusted_spend)

    # Keep lag-derived spend features coherent with monthly average expense if these features exist.
    for lag_col in ["spend_lag_1", "spend_lag_2", "spend_lag_3", "spend_rolling_3"]:
        if lag_col in record:
            record[lag_col] = adjusted_spend

    if "spend_momentum" in record:
        # Approximate momentum from affordability signal.
        record["spend_momentum"] = (affordability_factor - 1.0) * 0.3

    if "income_per_person" in record:
        record["income_per_person"] = income_per_person
    if "spend_income_ratio" in record:
        record["spend_income_ratio"] = adjusted_spend / max(salary, 1e-6)
    if "return_rate_score" in record:
        record["return_rate_score"] = return_rate_score
    if "return_spend_score" in record:
        record["return_spend_score"] = return_rate_score

    return record


def _build_legacy_customer_spend_record(raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
    avg_monthly_exp = float(raw_inputs["avg_monthly_exp"])
    salary = max(1.0, float(raw_inputs["salary"]))
    family_size = max(1.0, float(raw_inputs["family_size"]))
    return_rate = float(raw_inputs["return_rate"])

    return {
        "Age": float(raw_inputs["age"]),
        "Return_Rate": return_rate,
        "Salary": salary,
        "Family_Size": family_size,
        "Month": float(raw_inputs["month"]),
        "avg_monthly_exp": avg_monthly_exp,
        "income_per_person": salary / family_size,
        "spend_income_ratio": avg_monthly_exp / salary,
        "return_spend_score": return_rate * avg_monthly_exp,
        "Gender_Male": 1.0 if str(raw_inputs["gender"]).strip().lower() == "male" else 0.0,
        "Marital_Status_Single": 1.0 if str(raw_inputs["marital_status"]).strip().lower() == "single" else 0.0,
        "Region_Type_Urban": 1.0 if str(raw_inputs["region_type"]).strip().lower() == "urban" else 0.0,
    }


def _is_legacy_schema(feature_names: List[str]) -> bool:
    legacy_markers = {
        "Return_Rate",
        "Salary",
        "Gender_Male",
        "Marital_Status_Single",
        "Region_Type_Urban",
        "Month",
    }
    return len(feature_names) > 0 and legacy_markers.issubset(set(feature_names))


def _input_frame_for_model(model, metadata: Dict[str, Any], raw_inputs: Dict[str, Any]) -> pd.DataFrame:
    model_features = list(getattr(model, "feature_names_in_", []))

    # Use legacy row only for legacy schema models; otherwise use engineered feature row.
    if _is_legacy_schema(model_features):
        legacy_row = _build_legacy_customer_spend_record(raw_inputs)
        return pd.DataFrame([{col: legacy_row.get(col, 0.0) for col in model_features}])

    row = _build_customer_spend_record(metadata, raw_inputs)
    if model_features:
        return pd.DataFrame([{col: row.get(col, 0.0) for col in model_features}])

    feature_columns = metadata["feature_columns"]
    return pd.DataFrame([row])[feature_columns]


def _single_prediction_adjustment_factor(raw_inputs: Dict[str, Any]) -> float:
    month = int(raw_inputs.get("month", 6))
    month = min(12, max(1, month))
    salary = max(1.0, float(raw_inputs.get("salary", 1.0)))
    family_size = max(1.0, float(raw_inputs.get("family_size", 1.0)))
    avg_monthly_exp = max(0.0, float(raw_inputs.get("avg_monthly_exp", 0.0)))
    return_rate = min(1.0, max(0.0, float(raw_inputs.get("return_rate", 0.0))))

    month_factor = _month_growth_factor(month)

    income_per_person = salary / family_size
    income_factor = min(1.35, max(0.75, income_per_person / 25000.0))

    spend_income_ratio = avg_monthly_exp / salary
    burden_factor = min(1.25, max(0.8, 1.0 + (spend_income_ratio - 0.35) * 0.5))

    # Make return behavior explicitly influence guided single predictions.
    # 0.00 -> 0.85, 1.00 -> 1.25 (linear mapping with reasonable bounds).
    return_rate_factor = 0.85 + 0.40 * return_rate

    return month_factor * income_factor * burden_factor * return_rate_factor


def _align_dataframe_for_model(model, metadata: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    model_features = list(getattr(model, "feature_names_in_", []))
    if not model_features:
        feature_columns = metadata["feature_columns"]
        return df[feature_columns].copy()

    aligned = pd.DataFrame(index=df.index)

    salary_series = pd.to_numeric(
        df.get("Salary", df.get("Monthly_Salary", pd.Series(0.0, index=df.index))),
        errors="coerce",
    ).fillna(0.0)
    family_series = pd.to_numeric(df.get("Family_Size", pd.Series(1.0, index=df.index)), errors="coerce").fillna(1.0)
    family_series = family_series.replace(0, 1.0)
    return_rate_series = pd.to_numeric(
        df.get("Return_Rate", df.get("return_rate", pd.Series(0.0, index=df.index))),
        errors="coerce",
    ).fillna(0.0)
    avg_exp_series = pd.to_numeric(
        df.get("avg_monthly_exp", df.get("avg_spending", df.get("monthly_spend", pd.Series(0.0, index=df.index)))),
        errors="coerce",
    ).fillna(0.0)

    for col in model_features:
        if col in df.columns:
            aligned[col] = df[col]
            continue

        if col == "Salary":
            aligned[col] = salary_series
        elif col == "Return_Rate":
            aligned[col] = return_rate_series
        elif col == "avg_monthly_exp":
            aligned[col] = avg_exp_series
        elif col == "income_per_person":
            aligned[col] = salary_series / family_series
        elif col == "spend_income_ratio":
            aligned[col] = avg_exp_series / salary_series.replace(0, 1e-6)
        elif col in {"return_spend_score", "return_rate_score"}:
            aligned[col] = return_rate_series * avg_exp_series
        elif col == "Gender_Male":
            aligned[col] = df.get("Gender", pd.Series("unknown", index=df.index)).astype(str).str.lower().eq("male").astype(float)
        elif col == "Marital_Status_Single":
            aligned[col] = df.get("Marital_Status", pd.Series("unknown", index=df.index)).astype(str).str.lower().eq("single").astype(float)
        elif col == "Region_Type_Urban":
            aligned[col] = df.get("Region_Type", pd.Series("unknown", index=df.index)).astype(str).str.lower().eq("urban").astype(float)
        elif col == "Month":
            aligned[col] = pd.to_numeric(df.get("Month", pd.Series(6.0, index=df.index)), errors="coerce").fillna(6.0)
        else:
            aligned[col] = 0.0

    return aligned[model_features]


def _model_schema_mismatch(model, metadata: Dict[str, Any]) -> bool:
    model_features = list(getattr(model, "feature_names_in_", []))
    if not model_features:
        return False
    metadata_features = metadata.get("feature_columns", [])
    return set(model_features) != set(metadata_features)


def _customer_spend_single_prediction_ui(model, metadata: Dict[str, Any]) -> None:
    st.caption("Input core customer details; derived features are auto-calculated.")

    with st.form("customer_spend_single_prediction_form"):
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", min_value=0, max_value=120, value=35)
        gender = c2.selectbox("Gender", ["Male", "Female", "Other", "Unknown"], index=0)

        c3, c4 = st.columns(2)
        marital_status = c3.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced", "Widowed", "Unknown"],
            index=0,
        )
        region_type = c4.selectbox("Region Type", ["Urban", "Rural", "Unknown"], index=0)

        c5, c6 = st.columns(2)
        return_rate = c5.number_input("Return Rate", min_value=0.0, max_value=1.0, value=0.08, step=0.01)
        salary = c6.number_input("Salary", min_value=1.0, value=85000.0, step=500.0)

        c7, c8 = st.columns(2)
        family_size = c7.number_input("Family Size", min_value=1, max_value=20, value=4, step=1)
        avg_monthly_exp = c8.number_input("Avg Monthly Expense", min_value=0.0, value=30000.0, step=500.0)

        month = st.number_input("Month", min_value=1, max_value=12, value=6, step=1)

        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    raw_inputs = {
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "region_type": region_type,
        "return_rate": return_rate,
        "salary": salary,
        "family_size": family_size,
        "avg_monthly_exp": avg_monthly_exp,
        "month": month,
    }

    input_df = _input_frame_for_model(model, metadata, raw_inputs)
    raw_prediction = float(model.predict(input_df)[0])
    adjustment_factor = _single_prediction_adjustment_factor(raw_inputs)
    prediction = raw_prediction * adjustment_factor
    st.success(f"Predicted next month spend: {prediction:,.2f}")
    # st.caption(
    #     f"Sensitivity factor applied: x{adjustment_factor:.3f} (month + return rate + income context)."
    # )

    computed_df = pd.DataFrame(
        [
            {
                "income_per_person": raw_inputs["salary"] / max(float(raw_inputs["family_size"]), 1.0),
                "spend_income_ratio": raw_inputs["avg_monthly_exp"] / max(float(raw_inputs["salary"]), 1e-6),
                "return_rate_score": raw_inputs["return_rate"] * raw_inputs["avg_monthly_exp"],
            }
        ]
    )
    st.markdown("**Calculated Features**")
    st.dataframe(computed_df)
    st.markdown("**Model Input Snapshot**")
    st.dataframe(input_df)


def _full_feature_editor_ui(model, metadata: Dict[str, Any]) -> None:
    model_features = list(getattr(model, "feature_names_in_", []))
    feature_columns = model_features if model_features else metadata.get("feature_columns", [])
    numeric_set = set(metadata.get("numeric_features", []))

    if not feature_columns:
        st.warning("No feature schema available for full editor.")
        return

    default_values: Dict[str, Any] = {}
    for feature in feature_columns:
        if model_features:
            default_values[feature] = 0.0
        else:
            default_values[feature] = 0.0 if feature in numeric_set else "unknown"

    st.caption("Edit all model features directly and run prediction.")
    editable_df = st.data_editor(pd.DataFrame([default_values]), use_container_width=True, num_rows="fixed")
    if st.button("Predict from full feature editor"):
        predict_df = editable_df.copy()
        if model_features:
            for col in model_features:
                predict_df[col] = pd.to_numeric(predict_df[col], errors="coerce").fillna(0.0)
            predict_df = predict_df[model_features]
        else:
            predict_df = predict_df[metadata["feature_columns"]]
        raw_prediction = float(model.predict(predict_df)[0])
        prediction = raw_prediction
        st.success(f"Predicted next month spend: {prediction:,.2f}")
        st.dataframe(predict_df)


def single_prediction_ui(model, metadata: Dict[str, Any]) -> None:
    st.subheader("Single Prediction")

    mode = st.radio(
        "Prediction Mode",
        ["Guided Input", "Full Feature Editor"],
        horizontal=True,
    )
    if mode == "Guided Input":
        _customer_spend_single_prediction_ui(model, metadata)
    else:
        _full_feature_editor_ui(model, metadata)


def batch_prediction_ui(model, metadata: Dict[str, Any]) -> None:
    st.subheader("Batch Prediction")
    st.caption("Upload a CSV file containing the same feature columns used during training.")

    feature_columns = metadata["feature_columns"]
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is None:
        st.info("No file uploaded yet.")
        return

    batch_df = pd.read_csv(uploaded_file)
    model_features = list(getattr(model, "feature_names_in_", []))
    if model_features:
        model_input_df = _align_dataframe_for_model(model, metadata, batch_df)
    else:
        missing = [col for col in feature_columns if col not in batch_df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            return
        model_input_df = batch_df[feature_columns]

    predictions = model.predict(model_input_df)

    output_df = model_input_df.copy()
    output_df["prediction"] = predictions

    st.success(f"Predictions generated for {len(output_df)} rows.")
    st.dataframe(output_df.head(50))

    csv_data = output_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions CSV",
        data=csv_data,
        file_name="predictions.csv",
        mime="text/csv",
    )


def sidebar_status(metadata: Dict[str, Any] | None, metrics_path: Path) -> None:
    st.sidebar.title("Model Status")
    if metadata is None:
        st.sidebar.error("Model artifacts not found")
        st.sidebar.markdown("Run customer spend training first.")
        return

    metrics = _load_json(metrics_path)
    st.sidebar.success("Model loaded")
    st.sidebar.write(f"Best model: {metadata.get('best_model', 'N/A')}")
    st.sidebar.write(f"Target: {metadata.get('target', 'N/A')}")
    st.sidebar.write(f"Features: {len(metadata.get('feature_columns', []))}")

    if isinstance(metrics, dict):
        st.sidebar.markdown("### Metrics")
        st.sidebar.write(f"MAE: {metrics.get('mae', 'N/A')}")
        st.sidebar.write(f"MSE: {metrics.get('mse', 'N/A')}")
        rmse = _effective_rmse(metrics)
        st.sidebar.write(f"RMSE: {rmse if rmse is not None else 'N/A'}")
        st.sidebar.write(f"R2: {metrics.get('r2', 'N/A')}")


def probability_explorer(artifact_root: Path) -> None:
    st.subheader("Probability Explorer")
    st.caption("Estimate likely spend ranges by filtering avg monthly expense band.")

    dataset_path = artifact_root / "engineered_monthly_features.csv"
    pred_path = artifact_root / "test_predictions.csv"
    if not dataset_path.exists() or not pred_path.exists():
        st.info("Probability explorer needs engineered dataset and test predictions artifacts.")
        return

    feat = pd.read_csv(dataset_path)
    pred = pd.read_csv(pred_path)

    if "avg_spending" not in feat.columns:
        st.info("Required feature column avg_spending not found.")
        return

    from sklearn.model_selection import train_test_split

    X = feat.drop(columns=["next_month_spend"])
    y = feat["next_month_spend"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    test_df = X_test.copy().reset_index(drop=True)
    test_df["actual"] = y_test.values
    pred = pred.reset_index(drop=True)

    if len(test_df) != len(pred):
        st.warning("Prediction rows and test split rows are not aligned for probability estimation.")
        return

    test_df["predicted"] = pred["predicted"].values

    c1, c2 = st.columns(2)
    min_exp = c1.number_input("Min Avg Monthly Expense", value=30000.0, step=1000.0)
    max_exp = c2.number_input("Max Avg Monthly Expense", value=50000.0, step=1000.0)
    if max_exp < min_exp:
        st.error("Max expense must be greater than or equal to min expense.")
        return

    sub = test_df[(test_df["avg_spending"] >= min_exp) & (test_df["avg_spending"] <= max_exp)].copy()
    st.write(f"Filtered samples: {len(sub)}")
    if len(sub) == 0:
        return

    bins = [0, 50000, 100000, 200000, 500000, 1000000, 2000000, 100000000]
    labels = ["0-50k", "50k-100k", "100k-200k", "200k-500k", "500k-1M", "1M-2M", "2M+"]
    act_probs = pd.cut(sub["actual"], bins=bins, labels=labels, include_lowest=True, right=False)
    pred_probs = pd.cut(sub["predicted"], bins=bins, labels=labels, include_lowest=True, right=False)

    prob_df = pd.DataFrame(
        {
            "range": labels,
            "actual_probability": act_probs.value_counts(normalize=True).reindex(labels).fillna(0.0).values,
            "predicted_probability": pred_probs.value_counts(normalize=True).reindex(labels).fillna(0.0).values,
        }
    )
    st.dataframe(prob_df, use_container_width=True)
    st.bar_chart(prob_df.set_index("range"))


def show_insights(comparison_path: Path, plots_dir: Path) -> None:
    st.subheader("Model Insights")

    comparison = _load_json(comparison_path)
    if isinstance(comparison, list) and comparison:
        st.markdown("**Model Comparison**")
        st.dataframe(pd.DataFrame(comparison))

    plot_map = {
        "Target Distribution": plots_dir / "target_distribution.png",
        "Monthly Spend Distribution": plots_dir / "monthly_spend_distribution.png",
        "Correlation Heatmap": plots_dir / "correlation_heatmap.png",
        "Customer Correlation Heatmap": plots_dir / "customer_correlation_heatmap.png",
        "Residual Plot": plots_dir / "residual_plot.png",
        "Feature Importance": plots_dir / "feature_importance.png",
        "RMSE Comparison": plots_dir / "model_rmse_comparison.png",
        "Spend By Purchase Category": plots_dir / "spend_by_purchase_category.png",
    }

    for label, path in plot_map.items():
        if path.exists():
            st.markdown(f"**{label}**")
            st.image(str(path), use_container_width=True)


def show_dataset_prediction_graphs(model, metadata: Dict[str, Any], artifact_root: Path) -> None:
    st.subheader("Dataset Prediction Graphs")

    dataset_path = artifact_root / "engineered_monthly_features.csv"
    if not dataset_path.exists():
        st.info("No engineered dataset found for graphing predictions.")
        return

    feature_columns = metadata.get("feature_columns", [])
    target_col = metadata.get("target", "next_month_spend")

    dataset_df = pd.read_csv(dataset_path)

    if _model_schema_mismatch(model, metadata):
        pred_path = artifact_root / "test_predictions.csv"
        if not pred_path.exists():
            st.warning(
                "Current model schema does not match engineered dataset schema, and no saved test predictions are available. "
                "Accuracy charts are skipped to avoid misleading results."
            )
            return

        st.warning(
            "Model input schema differs from engineered dataset schema. Showing saved test predictions for reliable accuracy charts."
        )
        pred_df = pd.read_csv(pred_path)
        required = {"actual", "predicted", "residual"}
        if not required.issubset(pred_df.columns):
            st.error("Saved test prediction file is missing required columns: actual, predicted, residual")
            return

        sample_size = st.slider(
            "Rows to plot from saved test predictions",
            min_value=50,
            max_value=max(50, len(pred_df)),
            value=min(500, len(pred_df)),
            step=50,
        )
        sampled = pred_df.sample(n=sample_size, random_state=42) if len(pred_df) > sample_size else pred_df.copy()

        mae = float((sampled["residual"].abs()).mean())
        mse = float(((sampled["residual"]) ** 2).mean())
        r2_num = float(((sampled["actual"] - sampled["predicted"]) ** 2).sum())
        r2_den = float(((sampled["actual"] - sampled["actual"].mean()) ** 2).sum())
        r2 = 1 - (r2_num / r2_den) if r2_den > 0 else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Sample MAE", f"{mae:,.2f}")
        c2.metric("Sample MSE", f"{mse:,.2f}")
        c3.metric("Sample R2", f"{r2:.4f}")

        st.markdown("**Actual vs Predicted (sample)**")
        st.scatter_chart(sampled, x="actual", y="predicted")

        st.markdown("**Residual Distribution (sample)**")
        counts, edges = np.histogram(sampled["residual"], bins=30)
        hist_df = pd.DataFrame({"bin_start": edges[:-1], "count": counts})
        st.bar_chart(hist_df.set_index("bin_start"))

        st.markdown("**Predicted values preview**")
        st.dataframe(sampled[["actual", "predicted", "residual"]].head(50))
        return

    model_features = list(getattr(model, "feature_names_in_", []))
    if not model_features:
        missing_features = [col for col in feature_columns if col not in dataset_df.columns]
        if missing_features:
            st.error(f"Engineered dataset is missing required features: {missing_features}")
            return

    n_rows = min(5000, len(dataset_df))
    sample_size = st.slider(
        "Rows to predict for graphs",
        min_value=50,
        max_value=max(50, n_rows),
        value=min(1000, n_rows),
        step=50,
    )

    sampled = dataset_df.sample(n=sample_size, random_state=42) if len(dataset_df) > sample_size else dataset_df.copy()
    X_sample = _align_dataframe_for_model(model, metadata, sampled)
    sampled["predicted"] = model.predict(X_sample)

    if target_col in sampled.columns:
        sampled["residual"] = sampled[target_col] - sampled["predicted"]
        mae = float((sampled["residual"].abs()).mean())
        mse = float(((sampled["residual"]) ** 2).mean())
        r2_num = float(((sampled[target_col] - sampled["predicted"]) ** 2).sum())
        r2_den = float(((sampled[target_col] - sampled[target_col].mean()) ** 2).sum())
        r2 = 1 - (r2_num / r2_den) if r2_den > 0 else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Sample MAE", f"{mae:,.2f}")
        c2.metric("Sample MSE", f"{mse:,.2f}")
        c3.metric("Sample R2", f"{r2:.4f}")

        st.markdown("**Actual vs Predicted (sample)**")
        st.scatter_chart(sampled, x=target_col, y="predicted")

        st.markdown("**Residual Distribution (sample)**")
        counts, edges = np.histogram(sampled["residual"], bins=30)
        hist_df = pd.DataFrame(
            {
                "bin_start": edges[:-1],
                "count": counts,
            }
        )
        st.bar_chart(hist_df.set_index("bin_start"))
    else:
        st.warning(f"Target column '{target_col}' was not found in engineered dataset. Showing predictions only.")

    st.markdown("**Predicted values preview**")
    preview_cols = [c for c in [target_col, "predicted", "residual"] if c in sampled.columns]
    base_cols = [c for c in ["Customer_ID", "monthly_spend", "avg_spending"] if c in sampled.columns]
    st.dataframe(sampled[base_cols + preview_cols].head(50))


def main() -> None:
    st.set_page_config(page_title="Customer Spend Intelligence", page_icon="📊", layout="wide")
    _inject_custom_style()

    st.markdown(
        """
        <div class="hero-block">
            <div class="hero-title">Customer Spend Intelligence Studio</div>
            <div class="hero-sub">Single and batch predictions, probability explorer, and diagnostics in one place.</div>
            <div class="small-mono">Profile: Customer Spend</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    artifact_root = CUSTOMER_ARTIFACTS_DIR
    model, metadata = load_model(str(artifact_root))

    metrics_path = artifact_root / "metrics.json"
    comparison_path = artifact_root / "model_comparison.json"
    plots_dir = artifact_root / "plots"

    sidebar_status(metadata, metrics_path)

    if model is None or metadata is None:
        st.error(f"No trained model artifacts found in: {artifact_root}")
        return

    metrics = _load_json(metrics_path)
    _overview_cards(metadata, metrics if isinstance(metrics, dict) else None)

    tab_single, tab_batch, tab_insights, tab_prob = st.tabs(["Single", "Batch", "Insights", "Probability"])

    with tab_single:
        single_prediction_ui(model, metadata)

        with st.expander("Example JSON Input"):
            st.json(default_record(metadata))

    with tab_batch:
        batch_prediction_ui(model, metadata)

    with tab_insights:
        show_insights(comparison_path, plots_dir)
        show_dataset_prediction_graphs(model, metadata, artifact_root)

    with tab_prob:
        probability_explorer(artifact_root)


if __name__ == "__main__":
    main()
