from __future__ import annotations

import numpy as np
import pandas as pd
import mlflow.pyfunc

from src.transforms import build_store_date_features

__all__ = [
    "make_submission_id",
    "to_long_format",
    "to_static_format",
    "build_series_context",
    "build_store_date_features",
    "TimesFMForecastPipeline",
]


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
        missing = [c for c in futr_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"to_long_format: expected future exogenous columns missing from input: {missing}"
            )
        cols.extend(futr_cols)

    return df[cols].sort_values(["unique_id", "ds"]).reset_index(drop=True)


def to_static_format(
    df: pd.DataFrame,
    static_cols: list[str] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df["unique_id"] = df["Store"].astype(str) + "_" + df["Dept"].astype(str)

    cols = ["unique_id"]
    if static_cols is not None:
        missing = [c for c in static_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"to_static_format: expected static columns missing from input: {missing}"
            )
        cols.extend(static_cols)

    return (
        df[cols]
        .sort_values("unique_id")
        .drop_duplicates("unique_id")
        .reset_index(drop=True)
    )


def build_series_context(
    long_df: pd.DataFrame,
    unique_ids,
    context_length: int,
    futr_cols: list[str],
):
    """Per-series (context target, matching-length history covariates, last date).

    TimesFM has no persistent "fitted" state the way NeuralForecast models do -
    every call to `forecast`/`forecast_with_covariates` needs the raw context
    array handed to it directly, so this needs to be rebuilt from `long_df` on
    every call site rather than stored on a trained model object.
    """
    contexts: dict[str, np.ndarray] = {}
    hist_futr: dict[str, dict[str, np.ndarray]] = {}
    last_ds: dict[str, pd.Timestamp] = {}

    subset = long_df[long_df["unique_id"].isin(unique_ids)].sort_values(["unique_id", "ds"])
    for uid, group in subset.groupby("unique_id"):
        if len(group) > context_length:
            group = group.iloc[-context_length:]
        contexts[uid] = group["y"].to_numpy(dtype=np.float32)
        hist_futr[uid] = {c: group[c].to_numpy(dtype=np.float32) for c in futr_cols}
        last_ds[uid] = group["ds"].iloc[-1]

    return contexts, hist_futr, last_ds


class TimesFMForecastPipeline(mlflow.pyfunc.PythonModel):
    """Wraps a frozen TimesFM checkpoint as an mlflow pyfunc model.

    Unlike the NeuralForecast-based pipelines (DLinear, TFT), TimesFM is not
    fitted to this project's data - it is a zero-shot foundation model, so the
    historical context it needs at inference time has to travel with the
    pipeline object itself (`history_long_df`), not live inside a saved model
    checkpoint. `store_date_features` backfills future exogenous covariates for
    store/dept series the real (sparse) test set doesn't cover for every date -
    see `build_store_date_features` in `src/transforms.py` for why that's needed.
    """

    def __init__(
        self,
        repo_id: str,
        history_long_df: pd.DataFrame,
        static_df: pd.DataFrame,
        store_date_features: pd.DataFrame,
        futr_cols: list[str],
        static_num_cols: list[str],
        static_cat_cols: list[str],
        horizon: int,
        context_length: int = 104,
        use_covariates: bool = True,
        xreg_mode: str = "xreg + timesfm",
        ridge: float = 1.0,
        use_quantile_head: bool = False,
        model_col: str = "TimesFM",
        fallback_by_id: dict[str, float] | None = None,
        global_fallback: float = 0.0,
    ):
        self.repo_id = repo_id
        self.history_long_df = history_long_df
        self.static_df = static_df
        self.store_date_features = store_date_features.rename(columns={"Date": "ds"})
        self.futr_cols = futr_cols
        self.static_num_cols = static_num_cols
        self.static_cat_cols = static_cat_cols
        self.horizon = horizon
        self.context_length = context_length
        self.use_covariates = use_covariates
        self.xreg_mode = xreg_mode
        self.ridge = ridge
        self.use_quantile_head = use_quantile_head
        self.model_col = model_col
        self.fallback_by_id = fallback_by_id or {}
        self.global_fallback = float(global_fallback)
        self.model = None

    def load_context(self, context):
        import timesfm

        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.repo_id)
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=self.context_length,
                max_horizon=self.horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=self.use_quantile_head,
                return_backcast=self.use_covariates,
            )
        )

    def predict(self, context, model_input):
        test_df = model_input.copy()
        test_df["Date"] = pd.to_datetime(test_df["Date"])

        test_keys = test_df[["Store", "Dept", "Date"]].copy()
        test_keys["unique_id"] = (
            test_keys["Store"].astype(str) + "_" + test_keys["Dept"].astype(str)
        )
        test_keys["ds"] = test_keys["Date"]
        test_keys["Id"] = make_submission_id(test_keys)

        unique_ids = sorted(self.history_long_df["unique_id"].unique())
        contexts, hist_futr, last_ds = build_series_context(
            self.history_long_df, unique_ids, self.context_length, self.futr_cols
        )
        inputs = [contexts[uid] for uid in unique_ids]

        forecast_ds = {}
        dyn_num = {c: [] for c in self.futr_cols} if self.use_covariates else None
        static_num = {c: [] for c in self.static_num_cols} if self.use_covariates else None
        static_cat = {c: [] for c in self.static_cat_cols} if self.use_covariates else None

        if self.use_covariates:
            static_by_id = self.static_df.set_index("unique_id")
            store_dtype = self.store_date_features["Store"].dtype
            sdf_indexed = self.store_date_features.set_index(["Store", "ds"])

        for uid in unique_ids:
            future_dates = pd.date_range(
                last_ds[uid] + pd.Timedelta(weeks=1), periods=self.horizon, freq="W-FRI"
            )
            forecast_ds[uid] = future_dates

            if self.use_covariates:
                store_val = np.array([uid.split("_", 1)[0]]).astype(store_dtype)[0]
                future_rows = sdf_indexed.reindex(
                    pd.MultiIndex.from_product([[store_val], future_dates], names=["Store", "ds"])
                )

                for c in self.futr_cols:
                    dyn_num[c].append(
                        np.concatenate(
                            [hist_futr[uid][c], future_rows[c].to_numpy(dtype=np.float32)]
                        )
                    )

                static_row = static_by_id.loc[uid] if uid in static_by_id.index else None
                for c in self.static_num_cols:
                    static_num[c].append(float(static_row[c]) if static_row is not None else 0.0)
                for c in self.static_cat_cols:
                    static_cat[c].append(int(static_row[c]) if static_row is not None else 0)

        if self.use_covariates:
            outputs, _xreg_outputs = self.model.forecast_with_covariates(
                inputs=inputs,
                dynamic_numerical_covariates=dyn_num,
                static_numerical_covariates=static_num or None,
                static_categorical_covariates=static_cat or None,
                xreg_mode=self.xreg_mode,
                ridge=self.ridge,
            )
        else:
            point, _quantiles = self.model.forecast(horizon=self.horizon, inputs=inputs)
            outputs = list(point)

        forecast_rows = []
        for uid, series_out in zip(unique_ids, outputs):
            series_out = np.asarray(series_out).reshape(-1)[: self.horizon]
            for ds, val in zip(forecast_ds[uid], series_out):
                forecast_rows.append((uid, ds, float(val)))

        forecast_df = pd.DataFrame(forecast_rows, columns=["unique_id", "ds", self.model_col])

        preds = test_keys.merge(forecast_df, on=["unique_id", "ds"], how="left")
        preds = preds.rename(columns={self.model_col: "Weekly_Sales"})

        fallback_values = preds["unique_id"].map(self.fallback_by_id)
        preds["Weekly_Sales"] = (
            preds["Weekly_Sales"].fillna(fallback_values).fillna(self.global_fallback)
        )

        # Weekly_Sales cannot be negative; clip any negative forecasts to 0.
        preds["Weekly_Sales"] = preds["Weekly_Sales"].clip(lower=0.0)

        return preds[["Id", "Weekly_Sales"]]
