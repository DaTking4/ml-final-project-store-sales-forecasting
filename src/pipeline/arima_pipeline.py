from __future__ import annotations

import os

import joblib
import mlflow.pyfunc
import pandas as pd

from src.arima_utils import is_valid_forecast, load_arima_model


def make_submission_id(df: pd.DataFrame) -> pd.Series:
    return (
        df["Store"].astype(str)
        + "_"
        + df["Dept"].astype(str)
        + "_"
        + pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    )


class ARIMAForecastPipeline(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        fallback_by_id: dict[str, float] | None = None,
        global_fallback: float = 0.0,
    ):
        self.fallback_by_id = fallback_by_id or {}
        self.global_fallback = float(global_fallback)
        self.models_dir: str | None = None
        self.fitted_ids: set[str] = set()

    def load_context(self, context):
        root = context.artifacts["arima_model_dir"]
        manifest = joblib.load(os.path.join(root, "manifest.joblib"))
        self.models_dir = os.path.join(root, "models")
        self.fitted_ids = set(manifest.get("fitted_ids", []))
        self.fallback_by_id = manifest.get("fallback_by_id", self.fallback_by_id)
        self.global_fallback = float(
            manifest.get("global_fallback", self.global_fallback)
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
            # Loaded and discarded one series at a time -- with ~2,660 fitted
            # models at several MB each, holding them all in memory at once
            # is neither necessary for prediction nor safe to assume fits.
            model = (
                load_arima_model(self.models_dir, unique_id)
                if unique_id in self.fitted_ids and self.models_dir is not None
                else None
            )
            if model is None:
                group["Weekly_Sales"] = self.fallback_by_id.get(
                    unique_id,
                    self.global_fallback,
                )
            else:
                try:
                    forecast = pd.Series(model.forecast(steps=len(group))).astype(float)
                    if not is_valid_forecast(forecast.to_numpy(), model.model.endog):
                        raise ValueError(f"Non-finite or implausible forecast for {unique_id}")
                    group["Weekly_Sales"] = forecast.to_numpy()
                except Exception:
                    group["Weekly_Sales"] = self.fallback_by_id.get(
                        unique_id,
                        self.global_fallback,
                    )
            pred_parts.append(group[["Id", "Weekly_Sales"]])

        preds = pd.concat(pred_parts, ignore_index=True)
        preds["Weekly_Sales"] = preds["Weekly_Sales"].fillna(self.global_fallback)
        return preds
