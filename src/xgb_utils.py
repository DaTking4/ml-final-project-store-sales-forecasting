from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.metrics import wmae_from_df

# Lags >= 26 are always available at test time for a 26-week horizon.
LAGS = [26, 52]
# Rolling windows computed from the lag-26-shifted series (no leakage).
ROLLING_WINDOWS = [4, 13, 26]

XGB_FEATURE_COLS: list[str] = (
    [f"lag_{l}" for l in LAGS]
    + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"rolling_std_{w}" for w in ROLLING_WINDOWS]
    + ["WeekOfYear", "Month", "Year"]
    + ["Type_A", "Type_B", "Type_C", "Size"]
    + ["IsHoliday", "DaysSinceLastHoliday", "DaysToNextHoliday"]
    + ["Fuel_Price", "Temperature", "CPI", "Unemployment"]
    + ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    + ["Store", "Dept"]
)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling features per Store-Dept. Requires a Weekly_Sales column."""
    df = df.copy().sort_values(["Store", "Dept", "Date"])

    def _group_feats(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        y = g["Weekly_Sales"]
        for lag in LAGS:
            g[f"lag_{lag}"] = y.shift(lag)
        # Compute rolling stats from the lag-26 position so they are always
        # available at inference time for a 26-week test horizon.
        shifted = y.shift(26)
        for w in ROLLING_WINDOWS:
            g[f"rolling_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()
            g[f"rolling_std_{w}"] = shifted.rolling(w, min_periods=1).std().fillna(0.0)
        return g

    return df.groupby(["Store", "Dept"], group_keys=False).apply(_group_feats)


def build_xgb_matrices(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "Weekly_Sales",
    holiday_col: str = "IsHoliday",
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, pd.DataFrame]:
    """Drop rows with NaN in any feature or target; return X, y, weights, clean_df."""
    clean = df.dropna(subset=[c for c in feature_cols if c in df.columns] + [target_col]).copy()
    available = [c for c in feature_cols if c in clean.columns]
    X = clean[available].astype(float)
    y = clean[target_col].astype(float)
    weights = np.where(clean[holiday_col].fillna(False).astype(bool), 5.0, 1.0)
    return X, y, weights, clean


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: np.ndarray,
    config: dict[str, Any],
) -> xgb.XGBRegressor:
    params = {k: v for k, v in config.items() if k not in ("label", "regime")}
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **params,
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    return model


def evaluate_xgb_config(
    config: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: np.ndarray,
    X_val: pd.DataFrame,
    val_meta: pd.DataFrame,
    model_col: str = "XGBoost",
    holiday_col: str = "IsHoliday",
) -> tuple[xgb.XGBRegressor, float, float]:
    """Train one config and return (model, train_wmae, val_wmae).

    val_meta must be aligned with X_val and contain Weekly_Sales + IsHoliday.
    """
    model = train_xgb(X_train, y_train, w_train, config)

    train_preds = model.predict(X_train)
    train_eval = pd.DataFrame({
        "y": y_train.values,
        model_col: train_preds,
        holiday_col: (w_train == 5.0),
    })
    train_wmae = wmae_from_df(train_eval, y_true_col="y", y_pred_col=model_col, holiday_col=holiday_col)

    val_preds = model.predict(X_val)
    val_result = val_meta[[holiday_col, "Weekly_Sales"]].copy()
    val_result[model_col] = val_preds
    val_wmae = wmae_from_df(val_result, y_true_col="Weekly_Sales", y_pred_col=model_col, holiday_col=holiday_col)

    return model, float(train_wmae), float(val_wmae)
