import io
import os
from datetime import date
from typing import List

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.stats import norm

from app.data_loader import download_prices
from app.forecast import train_and_forecast_single


st.set_page_config(page_title="Stock Forecast", layout="wide")
st.title("Stock Analysis and Forecast")
st.caption("Educational model: not investment advice. Predictions are probabilistic and may be wrong.")

# Popular markets with friendly company names
MARKETS = {
    "United States": [
        ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("GOOG", "Alphabet"), ("AMZN", "Amazon"), ("NVDA", "NVIDIA"), ("META", "Meta"), ("TSLA", "Tesla"), ("BRK-B", "Berkshire Hathaway"),
    ],
    "India": [
        ("RELIANCE.NS", "Reliance Industries"), ("TCS.NS", "Tata Consultancy Services"), ("INFY.NS", "Infosys"), ("HDFCBANK.NS", "HDFC Bank"), ("POWERGRID.NS", "Power Grid"), ("TATAMOTORS.NS", "Tata Motors"),
    ],
    "Japan": [
        ("7203.T", "Toyota"), ("6758.T", "Sony"), ("9984.T", "SoftBank"),
    ],
    "Hong Kong": [
        ("0700.HK", "Tencent"), ("0939.HK", "China Construction Bank"),
    ],
    "Germany": [
        ("SAP.DE", "SAP"), ("BMW.DE", "BMW"), ("SIE.DE", "Siemens"),
    ],
    "UK": [
        ("RIO.L", "Rio Tinto"), ("ULVR.L", "Unilever"), ("HSBA.L", "HSBC"),
    ],
    "Netherlands": [
        ("ASML", "ASML"),
    ],
    "Switzerland": [
        ("NESN.SW", "Nestlé"),
    ],
}

# Sidebar controls
with st.sidebar:
    st.header("Pick your stocks")
    market = st.selectbox("Market", options=list(MARKETS.keys()), index=1)
    choices = [f"{name} ({tick})" for tick, name in MARKETS[market]]
    default_choices = [choices[0]] if choices else []
    selected_choices = st.multiselect("Companies", options=choices, default=default_choices, help="Select one or more companies")

    tickers_text = st.text_input(
        "Add tickers manually (comma-separated)",
        value="",
        help="Use Yahoo Finance symbols (e.g., AAPL, MSFT, TCS.NS, POWERGRID.NS)",
    )
    start_date = st.date_input("Start date", value=date(2015, 1, 1))
    interval = st.selectbox("Data interval", options=["1d", "1wk", "1mo"], index=0)
    horizon = st.slider("Forecast horizon (days)", min_value=1, max_value=60, value=5)
    lags = st.slider("Lag features (days)", min_value=3, max_value=60, value=10)
    chart_range = st.selectbox("Chart range", options=["2Y", "5Y", "MAX"], index=0)
    run_btn = st.button("Run Analysis")


def compute_metrics(close: pd.Series) -> dict:
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    if len(close) < 2:
        return {}
    start_price = float(close.iloc[0])
    end_price = float(close.iloc[-1])
    days = (close.index[-1] - close.index[0]).days
    years = days / 365.25 if days > 0 else np.nan
    total_return = end_price / start_price - 1.0
    ret = close.pct_change().dropna()
    log_ret = np.log1p(ret)
    ann_vol = float(log_ret.std() * np.sqrt(252)) if len(log_ret) else np.nan
    mu_daily = float(log_ret.mean()) if len(log_ret) else np.nan
    mu_annual = mu_daily * 252 if not np.isnan(mu_daily) else np.nan
    cum = (1 + ret).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else np.nan
    cagr = (end_price / start_price) ** (1 / years) - 1 if years and years > 0 else np.nan
    return dict(
        start=close.index[0].date(),
        end=close.index[-1].date(),
        years=years,
        start_price=start_price,
        end_price=end_price,
        total_return=total_return,
        cagr=cagr,
        ann_vol=ann_vol,
        mu_annual=mu_annual,
        max_dd=max_dd,
    )


def expected_multi_step_return(preds: pd.DataFrame) -> float:
    if preds is None or preds.empty or "PredictedReturn" not in preds.columns:
        return float("nan")
    r = (1.0 + preds["PredictedReturn"]).prod() - 1.0
    return float(r)


def get_logo_and_name(ticker: str) -> tuple[str | None, str | None]:
    try:
        tk = yf.Ticker(ticker)
        info = None
        try:
            info = tk.get_info()
        except Exception:
            # fallback to .info if available
            info = getattr(tk, "info", None)
        logo = None
        name = None
        if isinstance(info, dict):
            logo = info.get("logo_url") or info.get("logo_url_png")
            name = info.get("shortName") or info.get("longName")
        return logo, name
    except Exception:
        return None, None


def recommendation_label(exp_ret: float, ann_vol: float) -> tuple[str, str]:
    """Return (label, explanation). Simple rule-based, educational only."""
    if np.isnan(exp_ret) or np.isnan(ann_vol):
        return "Hold", "Insufficient data for a confident view."
    # Heuristic thresholds
    if exp_ret > 0.02 and ann_vol < 0.45:
        return "Buy", f"Expected {exp_ret*100:.2f}% over horizon with annualized vol {ann_vol*100:.1f}%."
    if exp_ret < -0.01 and ann_vol > 0.50:
        return "Avoid", f"Negative expected return {exp_ret*100:.2f}% and high vol {ann_vol*100:.1f}%."
    return "Hold", f"Expected {exp_ret*100:.2f}% with vol {ann_vol*100:.1f}%."


def outlook_10y(mu_annual: float, sigma_annual: float) -> dict:
    """Probability stock is above today after 10 years using lognormal approx.
    Returns dict with prob_profit, exp_return_10y.
    """
    if np.isnan(mu_annual) or np.isnan(sigma_annual) or sigma_annual <= 0:
        return {"prob_profit": float("nan"), "exp_return_10y": float("nan")}
    # Log-return ~ N(mu_annual*10, sigma_annual*sqrt(10))
    mu10 = mu_annual * 10.0
    sig10 = sigma_annual * np.sqrt(10.0)
    prob = 1.0 - float(norm.cdf((0.0 - mu10) / (sig10 + 1e-12)))
    exp_ret = float(np.exp(mu10) - 1.0)
    return {"prob_profit": prob, "exp_return_10y": exp_ret}


def plot_history_and_forecast(df: pd.DataFrame, preds: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    df_plot = df.copy()
    df_plot["Date"] = pd.to_datetime(df_plot["Date"])  # ensure datetime
    ax.plot(df_plot["Date"], df_plot["Close"], label="Historical", linewidth=1.8, color="#1f77b4")

    if preds is not None and not preds.empty:
        preds_plot = preds.copy()
        preds_plot["Date"] = pd.to_datetime(preds_plot["Date"])  # ensure datetime
        ax.plot(preds_plot["Date"], preds_plot["PredictedClose"], marker="o", label="Forecast", color="#ff7f0e")

    ax.set_title(f"{ticker}: Historical and Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    return fig


def build_pdf_report(ticker: str, df_hist: pd.DataFrame, preds: pd.DataFrame, metrics: dict, label: str, expl: str, long_term_text: str | None = None) -> bytes:
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Page 1: Chart
        fig1, ax1 = plt.subplots(figsize=(11, 6))
        dfp = df_hist.copy()
        dfp["Date"] = pd.to_datetime(dfp["Date"])  # ensure datetime
        ax1.plot(dfp["Date"], dfp["Close"], label="Historical", linewidth=1.5)
        if preds is not None and not preds.empty:
            pr = preds.copy()
            pr["Date"] = pd.to_datetime(pr["Date"])  # ensure datetime
            ax1.plot(pr["Date"], pr["PredictedClose"], marker="o", label="Forecast")
        ax1.set_title(f"{ticker} — Historical and Forecast")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        ax1.legend()
        fig1.tight_layout()
        pdf.savefig(fig1)
        plt.close(fig1)

        # Page 2: Metrics + Recommendation
        fig2, ax2 = plt.subplots(figsize=(11, 6))
        ax2.axis('off')
        lines = [
            f"Ticker: {ticker}",
            f"Period: {metrics.get('start')} to {metrics.get('end')} ({metrics.get('years'):.2f} yrs)",
            f"Start price: {metrics.get('start_price'):.2f}",
            f"End price: {metrics.get('end_price'):.2f}",
            f"Total return: {metrics.get('total_return')*100:.2f}%",
            f"CAGR: {metrics.get('cagr')*100:.2f}%",
            f"Annualized volatility: {metrics.get('ann_vol')*100:.2f}%",
            f"Max drawdown: {metrics.get('max_dd')*100:.2f}%",
            "",
            f"Recommendation: {label}",
            f"Rationale: {expl}",
        ]
        if long_term_text:
            lines += ["", long_term_text]
        lines += [
            "",
            "Note: Educational template using a simple ML model on returns. Not investment advice.",
        ]
        y = 0.95
        for ln in lines:
            ax2.text(0.05, y, ln, fontsize=12, va='top')
            y -= 0.07
        fig2.tight_layout()
        pdf.savefig(fig2)
        plt.close(fig2)
    buf.seek(0)
    return buf.getvalue()


if run_btn:
    selected_market_tickers = []
    for ch in selected_choices:
        # ch like "Apple (AAPL)"
        sym = ch.split("(")[-1].rstrip(")").strip()
        if sym:
            selected_market_tickers.append(sym)
    tickers: List[str] = sorted(set([t.strip().upper() for t in tickers_text.split(",") if t.strip()] + selected_market_tickers))
    if not tickers:
        st.warning("Please enter at least one ticker.")
        st.stop()

    with st.spinner("Downloading data and running forecasts..."):
        data = download_prices(tickers, start=str(start_date), interval=interval)

    tabs = st.tabs(tickers)
    for idx, t in enumerate(tickers):
        with tabs[idx]:
            df = data.get(t)
            # Header with logo & name
            logo_url, name = get_logo_and_name(t)
            c0, c1 = st.columns([1, 6])
            with c0:
                if logo_url:
                    st.image(logo_url, width=64)
                else:
                    st.markdown(f"**{t}**")
            with c1:
                st.subheader(name if name else t)

            if df is None or df.empty or "Close" not in df.columns:
                st.error("No data available for this ticker.")
                continue

            # Metrics
            close = pd.Series(df["Close"]).copy()
            metrics = compute_metrics(pd.Series(close.values, index=pd.to_datetime(df["Date"])) )
            if metrics:
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total return", f"{metrics['total_return']*100:.2f}%")
                m2.metric("CAGR", f"{metrics['cagr']*100:.2f}%")
                m3.metric("Ann. vol", f"{metrics['ann_vol']*100:.2f}%")
                m4.metric("Max drawdown", f"{metrics['max_dd']*100:.2f}%")
                m5.metric("Period (yrs)", f"{metrics['years']:.2f}")
            else:
                st.info("Not enough data to compute metrics.")

            # Forecast
            try:
                result = train_and_forecast_single(df, ticker=t, horizon=horizon, n_lags=lags)
            except Exception as e:
                st.error(f"Forecast error: {e}")
                continue

            # Compute recommendation from expected multi-step return and risk
            exp_ret = expected_multi_step_return(result.predictions)
            label, expl = recommendation_label(exp_ret, metrics.get('ann_vol', float('nan')) if metrics else float('nan'))
            if label == "Buy":
                st.success(f"Suggestion: {label} — {expl}")
            elif label == "Avoid":
                st.error(f"Suggestion: {label} — {expl}")
            else:
                st.info(f"Suggestion: {label} — {expl}")

            # 10-year outlook
            ol = outlook_10y(metrics.get('mu_annual', float('nan')) if metrics else float('nan'), metrics.get('ann_vol', float('nan')) if metrics else float('nan'))
            if not np.isnan(ol.get('prob_profit', float('nan'))):
                prob_txt = f"Probability of profit in 10 years: {ol['prob_profit']*100:.1f}%"
                exp_txt = f"Expected 10-year return (drift-only): {ol['exp_return_10y']*100:.1f}%"
                if ol['prob_profit'] >= 0.65:
                    st.success(f"10-year outlook: Likely profitable. {prob_txt} — {exp_txt}")
                elif ol['prob_profit'] <= 0.45:
                    st.error(f"10-year outlook: Unlikely profitable. {prob_txt} — {exp_txt}")
                else:
                    st.warning(f"10-year outlook: Uncertain. {prob_txt} — {exp_txt}")
                long_term_text = f"10-year outlook: {prob_txt}. {exp_txt}."
            else:
                long_term_text = None

            # Show forecast table
            st.write("Validation (returns): MAE = ", result.mae, ", RMSE = ", result.rmse)
            st.dataframe(result.predictions)

            # Choose chart range
            if chart_range == "2Y":
                hist_df = df.tail(252*2).copy()
            elif chart_range == "5Y":
                hist_df = df.tail(252*5).copy()
            else:
                hist_df = df.copy()

            # Plot historical + forecast
            fig = plot_history_and_forecast(hist_df, result.predictions, t)

            # Downloads: CSV and PDF report
            cdl, cdr = st.columns(2)
            with cdl:
                csv_buf = io.StringIO()
                result.predictions.to_csv(csv_buf, index=False)
                st.download_button(
                    label="Download forecast CSV",
                    data=csv_buf.getvalue(),
                    file_name=f"predictions_{t}.csv",
                    mime="text/csv",
                )
            with cdr:
                pdf_bytes = build_pdf_report(t, hist_df, result.predictions, metrics if metrics else {}, label, expl, long_term_text)
                st.download_button(
                    label="Download PDF report",
                    data=pdf_bytes,
                    file_name=f"{t}_report.pdf",
                    mime="application/pdf",
                )
