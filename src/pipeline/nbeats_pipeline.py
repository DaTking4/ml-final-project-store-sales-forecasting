import pandas as pd
import mlflow.pyfunc
from neuralforecast import NeuralForecast

from src.training_diagnostics import strip_neuralforecast_callbacks
from src.pipeline.dlinear_pipeline import to_long_format, make_submission_id


class NBEATSForecastPipeline(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        model_col: str = "NBEATS",
        fallback_by_id: dict[str, float] | None = None,
        global_fallback: float = 0.0,
    ):
        self.model_col = model_col
        self.fallback_by_id = fallback_by_id or {}
        self.global_fallback = float(global_fallback)
        self.nf_model = None

    def load_context(self, context):
        self.nf_model = NeuralForecast.load(path=context.artifacts["nf_model_dir"])
        for model in getattr(self.nf_model, "models", []):
            strip_neuralforecast_callbacks(model)

    def predict(self, context, model_input):
        test_df = model_input.copy()
        test_df["Date"] = pd.to_datetime(test_df["Date"])

        test_keys = test_df[["Store", "Dept", "Date"]].copy()
        test_keys["unique_id"] = (
            test_keys["Store"].astype(str) + "_" + test_keys["Dept"].astype(str)
        )
        test_keys["ds"] = test_keys["Date"]
        test_keys["Id"] = make_submission_id(test_keys)

        for model in getattr(self.nf_model, "models", []):
            strip_neuralforecast_callbacks(model)

        forecast_df = self.nf_model.predict()
        forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

        preds = test_keys.merge(
            forecast_df[["unique_id", "ds", self.model_col]],
            on=["unique_id", "ds"],
            how="left",
        )

        preds = preds.rename(columns={self.model_col: "Weekly_Sales"})
        fallback_values = preds["unique_id"].map(self.fallback_by_id)
        preds["Weekly_Sales"] = (
            preds["Weekly_Sales"]
            .fillna(fallback_values)
            .fillna(self.global_fallback)
        )

        return preds[["Id", "Weekly_Sales"]]
