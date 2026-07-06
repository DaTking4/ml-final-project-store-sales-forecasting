import numpy as np
import pandas as pd


def wmae_from_df(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    holiday_col: str = "IsHoliday",
) -> float:
    y_true = df[y_true_col].astype(float)
    y_pred = df[y_pred_col].astype(float)

    is_holiday = df[holiday_col].fillna(False).astype(bool)
    weights = np.where(is_holiday, 5.0, 1.0)

    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)