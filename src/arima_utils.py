from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
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
    return model.fit(disp=False, maxiter=int(config["maxiter"]))


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
        if len(forecast) != steps or np.isnan(forecast).any():
            return np.full(steps, fallback_value, dtype=float), True
        return forecast, False
    except Exception:
        return np.full(steps, fallback_value, dtype=float), True


def evaluate_arima_config(
    config: dict[str, Any],
    arima_ids,
    train_by_id: dict[str, pd.Series],
    valid_by_id: dict[str, pd.DataFrame],
    holiday_lookup: pd.DataFrame,
    model_col: str,
    min_train_points: int,
) -> tuple[pd.DataFrame, float, int, float]:
    rows = []
    train_rows = []
    failures = 0

    for idx, unique_id in enumerate(arima_ids, start=1):
        y_train = train_by_id[unique_id]
        valid_group = valid_by_id[unique_id].copy()
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
                failed = len(forecast) != len(valid_group) or np.isnan(forecast).any()
                if failed:
                    forecast = np.full(len(valid_group), fallback_value, dtype=float)
            except Exception:
                forecast = np.full(len(valid_group), fallback_value, dtype=float)
                failed = True
                fitted = pd.Series(dtype=float)

        failures += int(failed)
        valid_group[model_col] = forecast
        valid_group["used_fallback"] = failed
        rows.append(valid_group)

        if not fitted.empty:
            train_rows.append(pd.DataFrame({
                "unique_id": unique_id,
                "ds": fitted.index,
                "y": y_train.reindex(fitted.index).to_numpy(),
                model_col: fitted.to_numpy(),
            }))

        if idx % 250 == 0:
            print(f"Evaluated {idx:,}/{len(arima_ids):,} series for {config['label']}")

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


def fit_arima_models(
    full_long_df: pd.DataFrame,
    ids,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    models = {}
    failures = []

    for idx, unique_id in enumerate(ids, start=1):
        y = (
            full_long_df[full_long_df["unique_id"] == unique_id]
            .sort_values("ds")["y"]
            .astype(float)
        )
        y.index = pd.DatetimeIndex(y.index)
        y = y.asfreq("W-FRI")
        y = y.interpolate(limit_direction="both")
        try:
            models[unique_id] = fit_arima_result(y, config=config)
        except Exception:
            failures.append(unique_id)

        if idx % 250 == 0:
            print(f"Fit {idx:,}/{len(ids):,} final ARIMA models")

    return models, failures
