from __future__ import annotations

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd

from src.xgb_utils import add_lag_features, XGB_FEATURE_COLS


def make_submission_id(df: pd.DataFrame) -> pd.Series:
    return (
        df["Store"].astype(str)
        + "_"
        + df["Dept"].astype(str)
        + "_"
        + pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    )


class XGBForecastPipeline(mlflow.pyfunc.PythonModel):
    """Wraps an XGBRegressor for MLflow serving.

    Artifacts payload keys:
        model          – trained XGBRegressor
        feature_cols   – ordered list of feature column names
        train_tail     – last 52 rows per series used to compute lags at test time
        fallback_by_id – {unique_id: last_known_sales}
        global_fallback – median sales used when series is unknown
    """

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        fallback_by_id: dict[str, float] | None = None,
        global_fallback: float = 0.0,
    ):
        self.feature_cols = feature_cols or XGB_FEATURE_COLS
        self.fallback_by_id = fallback_by_id or {}
        self.global_fallback = float(global_fallback)
        self.model = None
        self.train_tail: pd.DataFrame | None = None

    def load_context(self, context):
        payload = joblib.load(context.artifacts["xgb_model_path"])
        self.model = payload["model"]
        self.feature_cols = payload["feature_cols"]
        self.train_tail = payload["train_tail"]
        self.fallback_by_id = payload.get("fallback_by_id", self.fallback_by_id)
        self.global_fallback = float(payload.get("global_fallback", self.global_fallback))

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        test_df = model_input.copy()
        test_df["Date"] = pd.to_datetime(test_df["Date"])

        # Placeholder Weekly_Sales so add_lag_features can run uniformly.
        test_df["Weekly_Sales"] = np.nan
        test_df["_is_test"] = True

        tail = self.train_tail.copy()
        tail["_is_test"] = False

        combined = pd.concat([tail, test_df], ignore_index=True)
        combined = add_lag_features(combined)

        test_rows = combined[combined["_is_test"]].copy()
        test_rows["Id"] = make_submission_id(test_rows)

        available = [c for c in self.feature_cols if c in test_rows.columns]
        X_test = test_rows[available].fillna(0.0).astype(float)

        preds = self.model.predict(X_test)

        result = pd.DataFrame({
            "Id": test_rows["Id"].values,
            "Weekly_Sales": preds.clip(min=0.0),
        })
        return result
