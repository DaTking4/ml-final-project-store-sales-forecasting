from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from src.arima_utils import to_arima_long
from src.metrics import wmae_from_df

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# Walmart competition holiday dates — passed to Prophet's holidays argument.
_HOLIDAY_ROWS = [
    ("Super_Bowl",   "2010-02-12"), ("Super_Bowl",   "2011-02-11"),
    ("Super_Bowl",   "2012-02-10"), ("Super_Bowl",   "2013-02-08"),
    ("Labor_Day",    "2010-09-10"), ("Labor_Day",    "2011-09-09"),
    ("Labor_Day",    "2012-09-07"), ("Labor_Day",    "2013-09-06"),
    ("Thanksgiving", "2010-11-26"), ("Thanksgiving", "2011-11-25"),
    ("Thanksgiving", "2012-11-23"), ("Thanksgiving", "2013-11-29"),
    ("Christmas",    "2010-12-31"), ("Christmas",    "2011-12-30"),
    ("Christmas",    "2012-12-28"), ("Christmas",    "2013-12-27"),
]


def make_holidays_df() -> pd.DataFrame:
    return pd.DataFrame(
        [{"holiday": name, "ds": pd.Timestamp(date)} for name, date in _HOLIDAY_ROWS]
    )


def fit_prophet_model(y: pd.Series, config: dict[str, Any]):
    from prophet import Prophet  # import locally to avoid slow top-level import

    params = {k: v for k, v in config.items() if k not in ("label", "regime")}
    model = Prophet(holidays=make_holidays_df(), **params)
    prophet_df = pd.DataFrame({"ds": y.index, "y": y.values})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prophet_df)
    return model


def evaluate_prophet_config(
    config: dict[str, Any],
    prophet_ids,
    train_by_id: dict[str, pd.Series],
    valid_by_id: dict[str, pd.DataFrame],
    holiday_lookup: pd.DataFrame,
    model_col: str,
    min_train_points: int,
) -> tuple[pd.DataFrame, float, int, float]:
    rows = []
    train_rows = []
    failures = 0

    for idx, unique_id in enumerate(prophet_ids, start=1):
        y_train = train_by_id[unique_id]
        valid_group = valid_by_id[unique_id].copy()
        fallback_value = float(y_train.iloc[-1])

        if len(y_train) < min_train_points:
            forecast = np.full(len(valid_group), fallback_value, dtype=float)
            failed = True
            fitted_vals = pd.Series(dtype=float)
        else:
            try:
                model = fit_prophet_model(y_train, config)

                future_df = pd.DataFrame({"ds": pd.DatetimeIndex(valid_group["ds"].values)})
                fc = model.predict(future_df)["yhat"].values.astype(float)

                if len(fc) != len(valid_group) or np.isnan(fc).any():
                    forecast = np.full(len(valid_group), fallback_value, dtype=float)
                    failed = True
                    fitted_vals = pd.Series(dtype=float)
                else:
                    forecast = fc
                    failed = False
                    insample = model.predict(pd.DataFrame({"ds": y_train.index}))
                    fitted_vals = pd.Series(insample["yhat"].values, index=y_train.index)
            except Exception:
                forecast = np.full(len(valid_group), fallback_value, dtype=float)
                failed = True
                fitted_vals = pd.Series(dtype=float)

        failures += int(failed)
        valid_group[model_col] = forecast
        valid_group["used_fallback"] = failed
        rows.append(valid_group)

        if not fitted_vals.empty:
            train_rows.append(pd.DataFrame({
                "unique_id": unique_id,
                "ds": fitted_vals.index,
                "y": y_train.reindex(fitted_vals.index).to_numpy(),
                model_col: fitted_vals.to_numpy(),
            }))

        if idx % 100 == 0:
            print(f"Evaluated {idx:,}/{len(prophet_ids):,} series for {config.get('label', '')}")

    cv_df = pd.concat(rows, ignore_index=True)
    cv_df = cv_df.merge(holiday_lookup, on=["unique_id", "ds"], how="left")
    cv_df["IsHoliday"] = cv_df["IsHoliday"].fillna(False).astype(bool)
    val_wmae = wmae_from_df(cv_df, y_true_col="y", y_pred_col=model_col, holiday_col="IsHoliday")

    if train_rows:
        train_cv_df = pd.concat(train_rows, ignore_index=True)
        train_cv_df = train_cv_df.merge(holiday_lookup, on=["unique_id", "ds"], how="left")
        train_cv_df["IsHoliday"] = train_cv_df["IsHoliday"].fillna(False).astype(bool)
        train_wmae = wmae_from_df(train_cv_df, y_true_col="y", y_pred_col=model_col, holiday_col="IsHoliday")
    else:
        train_wmae = float("nan")

    return cv_df, float(val_wmae), failures, float(train_wmae)


def fit_prophet_models(
    full_long_df: pd.DataFrame,
    ids,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    models: dict[str, Any] = {}
    failures: list[str] = []

    for idx, unique_id in enumerate(ids, start=1):
        y = (
            full_long_df[full_long_df["unique_id"] == unique_id]
            .sort_values("ds")["y"]
            .astype(float)
        )
        y.index = pd.DatetimeIndex(y.index)
        y = y.asfreq("W-FRI").interpolate(limit_direction="both")

        try:
            models[unique_id] = fit_prophet_model(y, config)
        except Exception:
            failures.append(unique_id)

        if idx % 100 == 0:
            print(f"Fit {idx:,}/{len(ids):,} final Prophet models")

    return models, failures
