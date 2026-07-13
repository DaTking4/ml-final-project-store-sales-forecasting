from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.kalman_filter import MEMORY_CONSERVE
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.metrics import wmae_from_df


def to_arima_long(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["unique_id"] = df["Store"].astype(str) + "_" + df["Dept"].astype(str)
    df["ds"] = df["Date"]

    cols = ["unique_id", "ds", "Store", "Dept"]
    if include_target:
        df["y"] = df["Weekly_Sales"].astype(float)
        cols.append("y")

    if "IsHoliday" in df.columns:
        cols.append("IsHoliday")

    return df[cols].sort_values(["unique_id", "ds"]).reset_index(drop=True)


def arima_order(config: dict[str, Any]) -> tuple[int, int, int]:
    if "order" in config:
        return tuple(config["order"])
    return int(config["p"]), int(config["d"]), int(config["q"])


def fit_arima_result(y: pd.Series, config: dict[str, Any]):
    model = SARIMAX(
        y.astype(float),
        order=arima_order(config),
        seasonal_order=tuple(config.get("seasonal_order", (0, 0, 0, 0))),
        trend=config.get("trend", "n"),
        enforce_stationarity=bool(config.get("enforce_stationarity", False)),
        enforce_invertibility=bool(config.get("enforce_invertibility", False)),
        concentrate_scale=bool(config.get("concentrate_scale", False)),
    )
    # Drops per-timestep diagnostic history statsmodels keeps by default
    # (~275MB/series -> ~8MB across ~2,660 series), same forecast output.
    model.ssm.set_conserve_memory(MEMORY_CONSERVE)
    return model.fit(disp=False, maxiter=int(config["maxiter"]))


def is_valid_forecast(forecast, y_train) -> bool:
    """Reject non-finite or implausibly large forecasts.

    Every config in this project's sweeps sets enforce_stationarity=False and
    enforce_invertibility=False, so a badly-conditioned fit can converge to
    roots outside the unit circle and produce a forecast that diverges
    exponentially over a 26-39 step horizon -- still finite floats (e.g. ~1e17
    has been observed in practice), so a plain np.isnan() check misses them
    entirely and one such series can dominate an aggregate metric like WMAE,
    or leak a nonsense prediction into a real submission.
    """
    forecast = np.asarray(forecast, dtype=float)
    if forecast.size == 0 or not np.isfinite(forecast).all():
        return False
    y_train = np.asarray(y_train, dtype=float).ravel()
    scale = float(np.abs(y_train).max()) if y_train.size else 0.0
    bound = max(scale * 50.0, 1.0)
    return bool(np.abs(forecast).max() <= bound)


def forecast_one_series(
    y_train: pd.Series,
    steps: int,
    config: dict[str, Any],
    min_train_points: int,
    fallback_value: float,
) -> tuple[np.ndarray, bool]:
    if len(y_train) < min_train_points:
        return np.full(steps, fallback_value, dtype=float), True

    try:
        result = fit_arima_result(y_train, config=config)
        forecast = np.asarray(result.forecast(steps=steps), dtype=float)
        if len(forecast) != steps or not is_valid_forecast(forecast, y_train):
            return np.full(steps, fallback_value, dtype=float), True
        return forecast, False
    except Exception:
        return np.full(steps, fallback_value, dtype=float), True


def _evaluate_one_arima_series(
    unique_id: str,
    y_train: pd.Series,
    valid_group: pd.DataFrame,
    config: dict[str, Any],
    model_col: str,
    min_train_points: int,
):
    valid_group = valid_group.copy()
    fallback_value = float(y_train.iloc[-1])

    if len(y_train) < min_train_points:
        forecast = np.full(len(valid_group), fallback_value, dtype=float)
        failed = True
        fitted = pd.Series(dtype=float)
    else:
        try:
            result = fit_arima_result(y_train, config=config)
            forecast = np.asarray(result.forecast(steps=len(valid_group)), dtype=float)
            fitted = result.fittedvalues
            failed = len(forecast) != len(valid_group) or not is_valid_forecast(forecast, y_train)
            if failed:
                forecast = np.full(len(valid_group), fallback_value, dtype=float)
        except Exception:
            forecast = np.full(len(valid_group), fallback_value, dtype=float)
            failed = True
            fitted = pd.Series(dtype=float)

    valid_group[model_col] = forecast
    valid_group["used_fallback"] = failed

    train_row = None
    if not fitted.empty:
        train_row = pd.DataFrame({
            "unique_id": unique_id,
            "ds": fitted.index,
            "y": y_train.reindex(fitted.index).to_numpy(),
            model_col: fitted.to_numpy(),
        })

    return valid_group, train_row, failed


def evaluate_arima_config(
    config: dict[str, Any],
    arima_ids,
    train_by_id: dict[str, pd.Series],
    valid_by_id: dict[str, pd.DataFrame],
    holiday_lookup: pd.DataFrame,
    model_col: str,
    min_train_points: int,
    n_jobs: int = -2,
) -> tuple[pd.DataFrame, float, int, float]:
    from joblib import Parallel, delayed

    # Process-based (loky): the SARIMAX fit is CPU-bound, so threading would
    # stay GIL-bound (unlike Prophet's cmdstan-subprocess case).
    print(f"Evaluating {len(arima_ids):,} series for {config.get('label', '')} (n_jobs={n_jobs})")
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_evaluate_one_arima_series)(
            unique_id, train_by_id[unique_id], valid_by_id[unique_id], config, model_col, min_train_points,
        )
        for unique_id in arima_ids
    )

    rows = [r[0] for r in results]
    train_rows = [r[1] for r in results if r[1] is not None]
    failures = sum(r[2] for r in results)

    cv_df = pd.concat(rows, ignore_index=True)
    cv_df = cv_df.merge(holiday_lookup, on=["unique_id", "ds"], how="left")
    cv_df["IsHoliday"] = cv_df["IsHoliday"].fillna(False).astype(bool)

    val_wmae = wmae_from_df(
        cv_df,
        y_true_col="y",
        y_pred_col=model_col,
        holiday_col="IsHoliday",
    )

    if train_rows:
        train_cv_df = pd.concat(train_rows, ignore_index=True)
        train_cv_df = train_cv_df.merge(holiday_lookup, on=["unique_id", "ds"], how="left")
        train_cv_df["IsHoliday"] = train_cv_df["IsHoliday"].fillna(False).astype(bool)
        train_wmae = wmae_from_df(
            train_cv_df,
            y_true_col="y",
            y_pred_col=model_col,
            holiday_col="IsHoliday",
        )
    else:
        train_wmae = float("nan")

    return cv_df, float(val_wmae), failures, float(train_wmae)


def model_file_path(models_dir, unique_id: str) -> str:
    return os.path.join(str(models_dir), f"{unique_id.replace('/', '_')}.joblib")


def _fit_one_final_arima_series(unique_id: str, y: pd.Series, config: dict[str, Any], models_dir: str):
    import joblib

    y = y.copy()
    y.index = pd.DatetimeIndex(y.index)
    y = y.asfreq("W-FRI").interpolate(limit_direction="both")
    try:
        result = fit_arima_result(y, config=config)
        # Write straight to disk and return only a bool -- avoids returning
        # many large objects through the multiprocessing pipe at once, and
        # avoids one combined ~22GB artifact that breaks the upload to DagsHub.
        joblib.dump(result, model_file_path(models_dir, unique_id))
        return unique_id, True
    except Exception:
        return unique_id, False


def fit_arima_models(
    full_long_df: pd.DataFrame,
    ids,
    config: dict[str, Any],
    models_dir,
    n_jobs: int = -2,
) -> tuple[list[str], list[str]]:
    """Fit one SARIMAX/ARIMA per series, writing each directly to
    `models_dir` as `<unique_id>.joblib`. Returns (fitted_ids, failed_ids) --
    NOT the fitted objects themselves; load individual models back with
    `load_arima_model(models_dir, unique_id)` at prediction time instead of
    holding the whole fleet in memory at once.
    """
    from joblib import Parallel, delayed

    os.makedirs(str(models_dir), exist_ok=True)

    grouped = {
        unique_id: group.sort_values("ds").set_index("ds")["y"].astype(float)
        for unique_id, group in full_long_df[full_long_df["unique_id"].isin(ids)].groupby("unique_id")
    }

    print(f"Fitting {len(ids):,} final ARIMA models (n_jobs={n_jobs}) -> {models_dir}")
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_fit_one_final_arima_series)(unique_id, grouped[unique_id], config, models_dir)
        for unique_id in ids
    )

    fitted_ids = [unique_id for unique_id, ok in results if ok]
    failures = [unique_id for unique_id, ok in results if not ok]
    return fitted_ids, failures


def load_arima_model(models_dir, unique_id: str):
    import joblib

    path = model_file_path(models_dir, unique_id)
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None
