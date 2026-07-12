from __future__ import annotations

from typing import Any

import joblib
import mlflow.pyfunc
import pandas as pd

from src.pipeline.arima_pipeline import make_submission_id


class SARIMAForecastPipeline(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        fallback_by_id: dict[str, float] | None = None,
        global_fallback: float = 0.0,
    ):
        self.fallback_by_id = fallback_by_id or {}
        self.global_fallback = float(global_fallback)
        self.models: dict[str, Any] = {}

    def load_context(self, context):
        payload = joblib.load(context.artifacts["sarima_model_path"])
        self.models = payload.get("models", {})
        self.fallback_by_id = payload.get("fallback_by_id", self.fallback_by_id)
        self.global_fallback = float(
            payload.get("global_fallback", self.global_fallback)
        )

    def predict(self, context, model_input):
        test_df = model_input.copy()
        test_df["Date"] = pd.to_datetime(test_df["Date"])

        test_keys = test_df[["Store", "Dept", "Date"]].copy()
        test_keys["unique_id"] = (
            test_keys["Store"].astype(str) + "_" + test_keys["Dept"].astype(str)
        )
        test_keys["Id"] = make_submission_id(test_keys)

        pred_parts = []
        for unique_id, group in test_keys.sort_values("Date").groupby("unique_id"):
            group = group.copy()
            model = self.models.get(unique_id)
            if model is None:
                group["Weekly_Sales"] = self.fallback_by_id.get(
                    unique_id, self.global_fallback
                )
            else:
                try:
                    forecast = pd.Series(model.forecast(steps=len(group))).astype(float)
                    group["Weekly_Sales"] = forecast.to_numpy()
                except Exception:
                    group["Weekly_Sales"] = self.fallback_by_id.get(
                        unique_id, self.global_fallback
                    )
            pred_parts.append(group[["Id", "Weekly_Sales"]])

        preds = pd.concat(pred_parts, ignore_index=True)
        preds["Weekly_Sales"] = preds["Weekly_Sales"].fillna(self.global_fallback)
        return preds
