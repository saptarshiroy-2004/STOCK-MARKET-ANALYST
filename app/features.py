from typing import Tuple
import pandas as pd
import numpy as np


def build_return_features(close: pd.Series, n_lags: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build lagged return features from a close-price series.

    X columns: r_lag_1..r_lag_n, roll_mean_5, roll_std_5
    y: next-day return
    """
    returns = close.pct_change()

    X = {}
    for i in range(1, n_lags + 1):
        X[f"r_lag_{i}"] = returns.shift(i)

    X["roll_mean_5"] = returns.rolling(5).mean()
    X["roll_std_5"] = returns.rolling(5).std()

    X_df = pd.DataFrame(X, index=close.index)
    y = returns.shift(-1)  # predict next-day return

    df = pd.concat([X_df, y.rename("target")], axis=1).dropna()
    X_final = df.drop(columns=["target"]).astype(np.float32)
    y_final = df["target"].astype(np.float32)
    return X_final, y_final
