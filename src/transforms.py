import pandas as pd
import numpy as np

_HOLIDAY_DATES = pd.to_datetime([
    # Super Bowl
    "2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08",
    # Labor Day
    "2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06",
    # Thanksgiving
    "2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29",
    # Christmas
    "2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27",
])


def encode_is_holiday(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    return df


def merge_features(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    features = encode_is_holiday(handle_missing(features.copy()))
    return df.merge(features.drop(columns="IsHoliday"), on=["Store", "Date"], how="left")


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    markdowns = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    df[markdowns] = df[markdowns].fillna(0)
    for col in ["CPI", "Unemployment"]:
        if col in df.columns:
            df[col] = df.groupby("Store")[col].transform(
                lambda s: s.ffill().bfill()
            )
    return df


def add_store_features(df: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    stores = stores[["Store", "Type", "Size"]].copy()
    type_dummies = pd.get_dummies(stores["Type"], prefix="Type")
    stores = pd.concat([stores.drop(columns="Type"), type_dummies], axis=1)
    return df.merge(stores, on="Store", how="left")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

    dates = df["Date"].values.astype("datetime64[D]")
    holiday_days = _HOLIDAY_DATES.values.astype("datetime64[D]")

    diffs = (dates[:, None] - holiday_days[None, :]).astype(int)

    days_since = np.where(diffs >= 0, diffs, np.inf).min(axis=1)
    days_until = np.where(diffs <= 0, -diffs, np.inf).min(axis=1)

    df["DaysSinceLastHoliday"] = days_since.astype(float)
    df["DaysToNextHoliday"] = days_until.astype(float)

    return df


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


def apply_shared_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    df = handle_missing(df)

    if "Type" in df.columns:
        type_dummies = pd.get_dummies(df["Type"], prefix="Type")
        df = pd.concat([df.drop(columns=["Type"]), type_dummies], axis=1)

    if "IsHoliday" in df.columns:
        df["IsHoliday"] = df["IsHoliday"].astype(int)

    df = add_time_features(df)

    sort_cols = [col for col in ["Store", "Dept", "Date"] if col in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    return df
