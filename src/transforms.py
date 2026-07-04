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
