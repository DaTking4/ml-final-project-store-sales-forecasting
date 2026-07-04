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
    df["DaysToNearestHoliday"] = np.minimum(days_since, days_until)

    return df
