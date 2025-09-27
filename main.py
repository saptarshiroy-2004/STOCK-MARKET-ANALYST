from __future__ import annotations
import argparse
import os
from datetime import datetime
from typing import List
import matplotlib.pyplot as plt
import pandas as pd

from app.data_loader import download_prices
from app.forecast import train_and_forecast_single


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stock price forecasting baseline")
    p.add_argument("--tickers", type=str, required=True, help="Comma-separated ticker symbols, e.g. AAPL,MSFT,GOOG")
    p.add_argument("--start", type=str, default="2018-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD or None for today")
    p.add_argument("--horizon", type=int, default=5, help="Forecast horizon in days")
    p.add_argument("--lags", type=int, default=10, help="Number of lagged return features")
    return p.parse_args()


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_outputs(ticker: str, preds: pd.DataFrame, metrics: dict, outdir: str) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(outdir, f"predictions_{ticker}_{ts}.csv")
    plot_path = os.path.join(outdir, f"plot_{ticker}_{ts}.png")

    preds.to_csv(csv_path, index=False)

    # Quick plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pd.to_datetime(preds["Date"]), preds["PredictedClose"], marker="o", label="Forecast")
    ax.set_title(f"Forecast for {ticker}\nMAE: {metrics.get('mae')}  RMSE: {metrics.get('rmse')}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    tickers: List[str] = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    print(f"Downloading data for: {tickers}")
    data = download_prices(tickers, start=args.start, end=args.end)

    outdir = os.path.join(os.path.dirname(__file__), "outputs")
    ensure_dir(outdir)

    for t in tickers:
        df = data.get(t)
        if df is None or df.empty:
            print(f"[WARN] No data for {t}, skipping")
            continue

        try:
            result = train_and_forecast_single(df, ticker=t, horizon=args.horizon, n_lags=args.lags)
        except Exception as e:
            print(f"[ERROR] {t}: {e}")
            continue

        print(f"\nTicker: {t}")
        print(f"Last observed close: {result.last_close:.2f}")
        print(f"Validation MAE: {result.mae}")
        print(f"Validation RMSE: {result.rmse}")
        print(result.predictions)

        save_outputs(
            t,
            result.predictions,
            metrics={"mae": result.mae, "rmse": result.rmse},
            outdir=outdir,
        )


if __name__ == "__main__":
    main()
