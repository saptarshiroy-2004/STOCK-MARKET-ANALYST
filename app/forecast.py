from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
from .features import build_return_features
from .model import train_return_model, TrainResult


@dataclass
class ForecastOutput:
    ticker: str
    last_close: float
    horizon: int
    predictions: pd.DataFrame  # columns: [Date, PredictedClose]
    mae: float | float('nan')
    rmse: float | float('nan')


def train_and_forecast_single(
    df: pd.DataFrame,
    ticker: str,
    horizon: int = 5,
    n_lags: int = 10,
) -> ForecastOutput:
    if df.empty or "Close" not in df.columns:
        raise ValueError(f"No valid Close prices for {ticker}")

    df = df.sort_values("Date").reset_index(drop=True)
    close = df["Close"]
    # Ensure 'close' is a 1-D Series (some data sources may yield a 2-D single-column frame)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.Series(close.values.ravel(), index=df.index, name="Close").copy()

    # Build features/targets
    X, y = build_return_features(close, n_lags=n_lags)
    if X.empty:
        raise ValueError(f"Insufficient data after feature engineering for {ticker}")

    # Train model and get metrics
    res: TrainResult = train_return_model(X, y)

    # Prepare last feature row to roll forward
    last_idx = X.index[-1]

    # Build a rolling window of returns ending at the last available day
    returns = close.pct_change()
    window = list(returns.loc[:last_idx].dropna().values)[-max(n_lags, 5):]
    if len(window) < max(n_lags, 5):
        raise ValueError(f"Not enough history to forecast {ticker}")

    last_close = float(close.loc[last_idx])

    pred_rows = []
    current_date = pd.to_datetime(df.loc[df.index[-1], "Date"]).normalize()

    for step in range(1, horizon + 1):
        # Create a single feature row from the rolling window
        feats = {}
        for i in range(1, n_lags + 1):
            feats[f"r_lag_{i}"] = window[-i] if i <= len(window) else 0.0
        roll_last = window[-5:] if len(window) >= 5 else window
        feats["roll_mean_5"] = float(np.mean(roll_last))
        feats["roll_std_5"] = float(np.std(roll_last, ddof=1)) if len(roll_last) > 1 else 0.0

        X_one = pd.DataFrame([feats])
        pred_return = float(res.model.predict(X_one)[0])

        # Convert predicted return to price
        next_close = last_close * (1.0 + pred_return)
        current_date = current_date + pd.Timedelta(days=1)

        pred_rows.append({
            "Date": current_date,
            "PredictedClose": next_close,
            "PredictedReturn": pred_return,
        })

        # Update rolling window and last_close for next step
        window.append(pred_return)
        last_close = next_close

    pred_df = pd.DataFrame(pred_rows)
    return ForecastOutput(
        ticker=ticker,
        last_close=close.iloc[-1],
        horizon=horizon,
        predictions=pred_df,
        mae=res.mae,
        rmse=res.rmse,
    )
