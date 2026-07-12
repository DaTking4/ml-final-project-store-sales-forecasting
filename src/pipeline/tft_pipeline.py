from __future__ import annotations

import mlflow.pyfunc
import pandas as pd
from neuralforecast import NeuralForecast

from src.transforms import apply_shared_features, handle_missing, add_time_features


def make_submission_id(df: pd.DataFrame) -> pd.Series:
    return (
        df["Store"].astype(str)
        + "_"
        + df["Dept"].astype(str)
        + "_"
        + pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    )


def to_long_format(
    df: pd.DataFrame,
    include_target: bool = True,
    futr_cols: list[str] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["unique_id"] = df["Store"].astype(str) + "_" + df["Dept"].astype(str)
    df["ds"] = df["Date"]

    cols = ["unique_id", "ds"]
    if include_target:
        df["y"] = df["Weekly_Sales"].astype(float)
        cols.append("y")

    if futr_cols is not None:
        missing = [col for col in futr_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"to_long_format: expected future exogenous columns missing from input: {missing}"
            )
        cols.extend(futr_cols)

    out = df[cols].sort_values(["unique_id", "ds"]).reset_index(drop=True)
    bool_cols = out.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        out[bool_cols] = out[bool_cols].astype(int)
    return out


def to_static_format(
    df: pd.DataFrame,
    static_cols: list[str] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df["unique_id"] = df["Store"].astype(str) + "_" + df["Dept"].astype(str)

    cols = ["unique_id"]
    if static_cols is not None:
        missing = [col for col in static_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"to_static_format: expected static columns missing from input: {missing}"
            )
        cols.extend(static_cols)

    out = (
        df[cols]
        .sort_values("unique_id")
        .drop_duplicates("unique_id")
        .reset_index(drop=True)
    )
    bool_cols = out.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        out[bool_cols] = out[bool_cols].astype(int)
    return out


def build_store_date_features(
    features_df: pd.DataFrame,
    futr_cols: list[str],
) -> pd.DataFrame:
    """Store+Date future exogenous covariates (calendar, weather, economic,
    markdown, holiday), deduplicated to one row per Store+Date.

    None of `futr_cols` vary by Dept, so this table (built from the complete
    features.csv reference data) can backfill exogenous values for series/dates
    the sparse per-department test rows don't cover.
    """
    df = features_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = handle_missing(df)
    if "IsHoliday" in df.columns:
        df["IsHoliday"] = df["IsHoliday"].astype(int)
    df = add_time_features(df)
    keep_cols = ["Store", "Date"] + [c for c in futr_cols if c in df.columns]
    return df[keep_cols].drop_duplicates(["Store", "Date"]).reset_index(drop=True)


class TFTForecastPipeline(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        model_col: str = "TFT",
        futr_cols: list[str] | None = None,
        static_cols: list[str] | None = None,
        fallback_by_id: dict[str, float] | None = None,
        global_fallback: float = 0.0,
        store_date_features: pd.DataFrame | None = None,
    ):
        self.model_col = model_col
        self.futr_cols = futr_cols or []
        self.static_cols = static_cols or []
        self.fallback_by_id = fallback_by_id or {}
        self.global_fallback = float(global_fallback)
        self.store_date_features = store_date_features
        self.nf_model = None

    def load_context(self, context):
        self.nf_model = NeuralForecast.load(path=context.artifacts["nf_model_dir"])

    def predict(self, context, model_input):
        if self.store_date_features is None:
            raise ValueError(
                "TFTForecastPipeline requires store_date_features to build a "
                "full future frame (see build_store_date_features)."
            )

        test_df = apply_shared_features(model_input.copy())
        bool_cols = test_df.select_dtypes(include="bool").columns
        if len(bool_cols) > 0:
            test_df[bool_cols] = test_df[bool_cols].astype(int)
        test_df["Date"] = pd.to_datetime(test_df["Date"])

        test_keys = test_df[["Store", "Dept", "Date"]].copy()
        test_keys["unique_id"] = (
            test_keys["Store"].astype(str) + "_" + test_keys["Dept"].astype(str)
        )
        test_keys["ds"] = test_keys["Date"]
        test_keys["Id"] = make_submission_id(test_keys)

        static_df = to_static_format(test_df, static_cols=self.static_cols)

        # NeuralForecast requires a full, contiguous h-step future block for
        # every trained series. The real test set only has rows for weeks a
        # department is active, so build the grid the model actually expects
        # and backfill exogenous covariates from the Store+Date reference
        # table instead of the sparse per-department test rows.
        future_grid = self.nf_model.make_future_dataframe()
        future_grid["Store"] = (
            future_grid["unique_id"]
            .str.split("_", n=1)
            .str[0]
            .astype(self.store_date_features["Store"].dtype)
        )
        futr_df = future_grid.merge(
            self.store_date_features.rename(columns={"Date": "ds"}),
            on=["Store", "ds"],
            how="left",
        ).drop(columns="Store")

        missing_mask = futr_df[self.futr_cols].isna().any(axis=1)
        if missing_mask.any():
            missing_ids = futr_df.loc[missing_mask, "unique_id"].unique()
            raise ValueError(
                f"Missing store-date exogenous coverage for {len(missing_ids)} "
                f"series, e.g. {list(missing_ids[:5])}"
            )

        forecast_df = self.nf_model.predict(futr_df=futr_df, static_df=static_df)
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

        # Weekly_Sales cannot be negative; clip any negative forecasts to 0.
        preds["Weekly_Sales"] = preds["Weekly_Sales"].clip(lower=0.0)

        return preds[["Id", "Weekly_Sales"]]
