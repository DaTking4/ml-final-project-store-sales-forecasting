import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_train() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "train.csv", parse_dates=["Date"])


def load_test() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "test.csv", parse_dates=["Date"])


def load_features() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "features.csv", parse_dates=["Date"])


def load_stores() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "stores.csv")


def load_all():
    return load_train(), load_test(), load_features(), load_stores()
