from __future__ import annotations

import numpy as np
import pandas as pd

LAG_WEEKS = 52
ROLL_WINDOW = 26

CATEGORICAL_COLS = ["Store", "Dept"]

NUMERIC_COLS = [
    "IsHoliday",
    "Temperature",
    "Fuel_Price",
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
    "CPI",
    "Unemployment",
    "Size",
    "Type_A",
    "Type_B",
    "Type_C",
    "Year",
    "Month",
    "WeekOfYear",
    "DaysSinceLastHoliday",
    "DaysToNextHoliday",
    "lag_52",
    "roll_mean_26_lag52",
    "roll_std_26_lag52",
]

FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


def add_unique_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["unique_id"] = df["Store"].astype(str) + "_" + df["Dept"].astype(str)
    return df


def build_series_lookup(history_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Per Store-Dept table of Weekly_Sales history, indexed by Date, with a
    trailing rolling mean/std already attached.

    A prediction for date D only ever looks up this table at D - LAG_WEEKS,
    so as long as LAG_WEEKS (52) is greater than the forecast horizon (<=39
    weeks for this competition), the lookup always resolves to a date at or
    before the training cutoff -- never to a future value.
    """
    history_df = add_unique_id(history_df)
    history_df = history_df.assign(Date=pd.to_datetime(history_df["Date"]))

    lookup = {}
    for unique_id, group in history_df.groupby("unique_id"):
        series = (
            group.sort_values("Date")
            .set_index("Date")["Weekly_Sales"]
            .astype(float)
            .asfreq("W-FRI")
        )
        lookup[unique_id] = pd.DataFrame({
            "y": series,
            "roll_mean_26": series.rolling(ROLL_WINDOW, min_periods=1).mean(),
            "roll_std_26": series.rolling(ROLL_WINDOW, min_periods=2).std(),
        })

    return lookup


def attach_lag_features(
    df: pd.DataFrame,
    lookup: dict[str, pd.DataFrame],
    lag_weeks: int = LAG_WEEKS,
) -> pd.DataFrame:
    df = add_unique_id(df)
    df = df.assign(Date=pd.to_datetime(df["Date"]))

    lag_52 = np.full(len(df), np.nan)
    roll_mean = np.full(len(df), np.nan)
    roll_std = np.full(len(df), np.nan)

    ref_dates = df["Date"] - pd.Timedelta(weeks=lag_weeks)

    for unique_id, positions in df.groupby("unique_id").indices.items():
        table = lookup.get(unique_id)
        if table is None:
            continue
        aligned = table.reindex(ref_dates.iloc[positions])
        lag_52[positions] = aligned["y"].to_numpy()
        roll_mean[positions] = aligned["roll_mean_26"].to_numpy()
        roll_std[positions] = aligned["roll_std_26"].to_numpy()

    df["lag_52"] = lag_52
    df["roll_mean_26_lag52"] = roll_mean
    df["roll_std_26_lag52"] = roll_std

    return df


def prepare_model_frame(
    df: pd.DataFrame,
    lookup: dict[str, pd.DataFrame],
    feature_cols: list[str] = FEATURE_COLS,
    categorical_cols: list[str] = CATEGORICAL_COLS,
) -> pd.DataFrame:
    df = attach_lag_features(df, lookup)
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in categorical_cols:
        X[col] = X[col].astype("category")
    return X