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
    # We only ever read `yhat` from predict() output, never the uncertainty
    # interval columns, so skip posterior sampling entirely -- it's pure cost
    # for output nobody uses, and it runs on every predict() call.
    model = Prophet(holidays=make_holidays_df(), uncertainty_samples=0, **params)
    prophet_df = pd.DataFrame({"ds": y.index, "y": y.values})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prophet_df)
    return model


def _evaluate_one_series(
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

    valid_group[model_col] = forecast
    valid_group["used_fallback"] = failed

    train_row = None
    if not fitted_vals.empty:
        train_row = pd.DataFrame({
            "unique_id": unique_id,
            "ds": fitted_vals.index,
            "y": y_train.reindex(fitted_vals.index).to_numpy(),
            model_col: fitted_vals.to_numpy(),
        })

    return valid_group, train_row, failed


def evaluate_prophet_config(
    config: dict[str, Any],
    prophet_ids,
    train_by_id: dict[str, pd.Series],
    valid_by_id: dict[str, pd.DataFrame],
    holiday_lookup: pd.DataFrame,
    model_col: str,
    min_train_points: int,
    n_jobs: int = 8,
) -> tuple[pd.DataFrame, float, int, float]:
    from joblib import Parallel, delayed

    # threading, not the default process-pool (loky) backend: each series' real
    # work happens in an external cmdstan subprocess, and Python releases the
    # GIL while waiting on it, so threads already get true OS-level concurrency
    # here -- without loky's extra worker-process layer, which was crashing on
    # concurrent first-launches of the freshly-built stan binary.
    print(f"Evaluating {len(prophet_ids):,} series for {config.get('label', '')} (n_jobs={n_jobs}, threading)")
    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=1)(
        delayed(_evaluate_one_series)(
            unique_id, train_by_id[unique_id], valid_by_id[unique_id], config, model_col, min_train_points,
        )
        for unique_id in prophet_ids
    )

    rows = [r[0] for r in results]
    train_rows = [r[1] for r in results if r[1] is not None]
    failures = sum(r[2] for r in results)

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


def _fit_one_final_series(unique_id: str, y: pd.Series, config: dict[str, Any]):
    y = y.copy()
    y.index = pd.DatetimeIndex(y.index)
    y = y.asfreq("W-FRI").interpolate(limit_direction="both")
    try:
        return unique_id, fit_prophet_model(y, config), None
    except Exception:
        return unique_id, None, unique_id


def fit_prophet_models(
    full_long_df: pd.DataFrame,
    ids,
    config: dict[str, Any],
    n_jobs: int = 8,
) -> tuple[dict[str, Any], list[str]]:
    from joblib import Parallel, delayed

    grouped = {
        unique_id: group.sort_values("ds")["y"].astype(float)
        for unique_id, group in full_long_df[full_long_df["unique_id"].isin(ids)].groupby("unique_id")
    }

    print(f"Fitting {len(ids):,} final Prophet models (n_jobs={n_jobs}, threading)")
    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=1)(
        delayed(_fit_one_final_series)(unique_id, grouped[unique_id], config)
        for unique_id in ids
    )

    models: dict[str, Any] = {}
    failures: list[str] = []
    for unique_id, model, failure in results:
        if model is not None:
            models[unique_id] = model
        else:
            failures.append(failure)

    return models, failures
