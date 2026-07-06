import pandas as pd


def time_based_split(df: pd.DataFrame, valid_weeks: int):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    unique_dates = sorted(df["Date"].unique())

    if len(unique_dates) <= valid_weeks:
        raise ValueError("Not enough dates for the requested validation split.")

    split_date = unique_dates[-valid_weeks]

    train_part = df[df["Date"] < split_date].copy()
    valid_part = df[df["Date"] >= split_date].copy()

    return train_part, valid_part