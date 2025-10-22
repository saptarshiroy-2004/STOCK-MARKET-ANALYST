from typing import Dict, List
import pandas as pd
import yfinance as yf


def download_prices(
    tickers: List[str],
    start: str = "2015-01-01",
    end: str | None = None,
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """
    Download historical price data for each ticker. Auto-adjusts for splits/dividends.

    Returns a dict mapping ticker -> DataFrame with columns: [Date, Open, High, Low, Close, Volume].
    """
    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        # progress=False to keep CLI output clean
        df = yf.download(
            t,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        df = df.copy()
        # If MultiIndex columns remain, flatten to the price field names
        if isinstance(df.columns, pd.MultiIndex):
            # yfinance may return MultiIndex with levels (Price, Ticker); keep the Price level
            df.columns = [str(col[0]) if isinstance(col, tuple) else str(col) for col in df.columns]
        df.index.name = "Date"
        df.reset_index(inplace=True)
        out[t] = df
    return out
