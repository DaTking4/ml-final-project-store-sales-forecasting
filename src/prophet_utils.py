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


def _warm_up_prophet(holidays_df: pd.DataFrame) -> None:
    """Force Stan's one-time model-compile/cache check to happen once, here,
    synchronously, before any parallel worker touches it.

    cmdstanpy's "is the binary already compiled? if not, build it" check isn't
    lock-protected. If it hasn't run yet and several parallel workers (threads
    or processes) each hit that check at the same moment, they can all start
    compiling/writing the same output binary concurrently and corrupt it,
    crashing whichever worker tries to execute the half-written file. Fitting
    one throwaway series up front guarantees the binary exists on disk and is
    valid before the pool starts, so the race can't happen at all.
    """
    dummy_index = pd.date_range("2020-01-03", periods=20, freq="W-FRI")
    dummy_y = pd.Series(np.arange(20, dtype=float), index=dummy_index)
    fit_prophet_model(dummy_y, {"label": "_warmup", "regime": "_warmup"}, holidays_df=holidays_df)


def fit_prophet_model(y: pd.Series, config: dict[str, Any], holidays_df: pd.DataFrame | None = None):
    from prophet import Prophet  # import locally to avoid slow top-level import

    params = {k: v for k, v in config.items() if k not in ("label", "regime")}
    # We only ever read `yhat` from predict() output, never the uncertainty
    # interval columns, so skip posterior sampling entirely -- it's pure cost
    # for output nobody uses, and it runs on every predict() call.
    if holidays_df is None:
        holidays_df = make_holidays_df()
    model = Prophet(holidays=holidays_df, uncertainty_samples=0, **params)
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
    holidays_df: pd.DataFrame | None = None,
):
    valid_group = valid_group.copy()
    fallback_value = float(y_train.iloc[-1])

    if len(y_train) < min_train_points:
        forecast = np.full(len(valid_group), fallback_value, dtype=float)
        failed = True
        fitted_vals = pd.Series(dtype=float)
    else:
        try:
            model = fit_prophet_model(y_train, config, holidays_df=holidays_df)

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


def evaluate_prophet_configs(
    configs: list[dict[str, Any]],
    prophet_ids,
    train_by_id: dict[str, pd.Series],
    valid_by_id: dict[str, pd.DataFrame],
    holiday_lookup: pd.DataFrame,
    model_col: str,
    min_train_points: int,
    n_jobs: int = -1,
) -> dict[str, dict[str, Any]]:
    """Evaluate many Prophet configs across all series in one shared thread pool.

    Submitting the full (config x series) cross product as a single Parallel
    call -- instead of one Parallel call per config -- keeps every worker busy
    for the whole sweep: with a separate pool per config, the pool sits idle
    waiting for that config's slowest straggler series before the next config
    can even start, which stalls the sweep at every config boundary.
    """
    from joblib import Parallel, delayed

    holidays_df = make_holidays_df()  # built once and shared, not per-fit
    _warm_up_prophet(holidays_df)
    configs_by_label = {config["label"]: config for config in configs}
    jobs = [(label, unique_id) for label in configs_by_label for unique_id in prophet_ids]

    # threading, not the default process-pool (loky) backend: each series' real
    # work happens in an external cmdstan subprocess, and Python releases the
    # GIL while waiting on it, so threads already get true OS-level concurrency
    # here -- without loky's extra worker-process layer, which was crashing on
    # concurrent first-launches of the freshly-built stan binary.
    print(
        f"Evaluating {len(configs_by_label)} config(s) x {len(prophet_ids):,} series "
        f"({len(jobs):,} fits total, n_jobs={n_jobs}, threading)"
    )
    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=1)(
        delayed(_evaluate_one_series)(
            unique_id, train_by_id[unique_id], valid_by_id[unique_id],
            configs_by_label[label], model_col, min_train_points, holidays_df,
        )
        for label, unique_id in jobs
    )

    by_label: dict[str, dict[str, list]] = {
        label: {"rows": [], "train_rows": [], "failures": 0} for label in configs_by_label
    }
    for (label, _), (row, train_row, failed) in zip(jobs, results):
        bucket = by_label[label]
        bucket["rows"].append(row)
        if train_row is not None:
            bucket["train_rows"].append(train_row)
        bucket["failures"] += int(failed)

    summary: dict[str, dict[str, Any]] = {}
    for label, bucket in by_label.items():
        cv_df = pd.concat(bucket["rows"], ignore_index=True)
        cv_df = cv_df.merge(holiday_lookup, on=["unique_id", "ds"], how="left")
        cv_df["IsHoliday"] = cv_df["IsHoliday"].fillna(False).astype(bool)
        val_wmae = wmae_from_df(cv_df, y_true_col="y", y_pred_col=model_col, holiday_col="IsHoliday")

        if bucket["train_rows"]:
            train_cv_df = pd.concat(bucket["train_rows"], ignore_index=True)
            train_cv_df = train_cv_df.merge(holiday_lookup, on=["unique_id", "ds"], how="left")
            train_cv_df["IsHoliday"] = train_cv_df["IsHoliday"].fillna(False).astype(bool)
            train_wmae = wmae_from_df(train_cv_df, y_true_col="y", y_pred_col=model_col, holiday_col="IsHoliday")
        else:
            train_wmae = float("nan")

        summary[label] = {
            "cv_df": cv_df,
            "val_wmae": float(val_wmae),
            "failures": bucket["failures"],
            "train_wmae": float(train_wmae),
        }

    return summary


def evaluate_prophet_config(
    config: dict[str, Any],
    prophet_ids,
    train_by_id: dict[str, pd.Series],
    valid_by_id: dict[str, pd.DataFrame],
    holiday_lookup: pd.DataFrame,
    model_col: str,
    min_train_points: int,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, float, int, float]:
    """Single-config convenience wrapper around evaluate_prophet_configs."""
    summary = evaluate_prophet_configs(
        [config], prophet_ids, train_by_id, valid_by_id, holiday_lookup,
        model_col, min_train_points, n_jobs=n_jobs,
    )[config["label"]]
    return summary["cv_df"], summary["val_wmae"], summary["failures"], summary["train_wmae"]


def _fit_one_final_series(unique_id: str, y: pd.Series, config: dict[str, Any], holidays_df: pd.DataFrame):
    y = y.copy()
    y.index = pd.DatetimeIndex(y.index)
    y = y.asfreq("W-FRI").interpolate(limit_direction="both")
    try:
        return unique_id, fit_prophet_model(y, config, holidays_df=holidays_df), None
    except Exception:
        return unique_id, None, unique_id


def fit_prophet_models(
    full_long_df: pd.DataFrame,
    ids,
    config: dict[str, Any],
    n_jobs: int = -1,
) -> tuple[dict[str, Any], list[str]]:
    from joblib import Parallel, delayed

    grouped = {
        unique_id: group.sort_values("ds")["y"].astype(float)
        for unique_id, group in full_long_df[full_long_df["unique_id"].isin(ids)].groupby("unique_id")
    }
    holidays_df = make_holidays_df()
    _warm_up_prophet(holidays_df)

    print(f"Fitting {len(ids):,} final Prophet models (n_jobs={n_jobs}, threading)")
    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=1)(
        delayed(_fit_one_final_series)(unique_id, grouped[unique_id], config, holidays_df)
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
