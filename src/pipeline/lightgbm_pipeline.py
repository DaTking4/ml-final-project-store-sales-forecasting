from __future__ import annotations

from typing import Any

import joblib
import mlflow.pyfunc
import pandas as pd

from src.lightgbm_utils import CATEGORICAL_COLS, FEATURE_COLS, prepare_model_frame
from src.pipeline.arima_pipeline import make_submission_id
from src.transforms import apply_shared_features


class LightGBMForecastPipeline(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        fallback_by_id: dict[str, float] | None = None,
        global_fallback: float = 0.0,
    ):
        self.fallback_by_id = fallback_by_id or {}
        self.global_fallback = float(global_fallback)
        self.model: Any = None
        self.lookup: dict[str, pd.DataFrame] = {}
        self.feature_cols = FEATURE_COLS
        self.categorical_cols = CATEGORICAL_COLS

    def load_context(self, context):
        payload = joblib.load(context.artifacts["lightgbm_model_path"])
        self.model = payload["model"]
        self.lookup = payload.get("lookup", {})
        self.feature_cols = payload.get("feature_cols", FEATURE_COLS)
        self.categorical_cols = payload.get("categorical_cols", CATEGORICAL_COLS)
        self.fallback_by_id = payload.get("fallback_by_id", self.fallback_by_id)
        self.global_fallback = float(
            payload.get("global_fallback", self.global_fallback)
        )

    def predict(self, context, model_input):
        test_df = model_input.copy()
        test_df["Date"] = pd.to_datetime(test_df["Date"])

        prepared = apply_shared_features(test_df)

        X = prepare_model_frame(
            prepared,
            self.lookup,
            feature_cols=self.feature_cols,
            categorical_cols=self.categorical_cols,
        )

        preds = pd.Series(self.model.predict(X), index=prepared.index)

        result = prepared[["Store", "Dept", "Date"]].copy()
        result["Id"] = make_submission_id(result)
        result["Weekly_Sales"] = preds

        unique_id = result["Store"].astype(str) + "_" + result["Dept"].astype(str)
        fallback_values = unique_id.map(self.fallback_by_id)
        result["Weekly_Sales"] = (
            result["Weekly_Sales"]
            .fillna(fallback_values)
            .fillna(self.global_fallback)
        )

        return result[["Id", "Weekly_Sales"]]