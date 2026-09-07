"""Market data fetcher using yfinance."""
import logging
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

PAIRS = {
    "XAUUSD": "XAUUSD=X",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
}


def fetch_candles(ticker: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    """Fetch OHLCV candles from yfinance.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    and a UTC DatetimeIndex, or None on failure.
    """
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        if df is None or df.empty:
            logger.warning("No data returned for %s (%s/%s)", ticker, interval, period)
            return None
        # yfinance may return MultiIndex columns for single ticker; flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Ensure UTC index
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df
    except Exception:
        logger.exception("Failed to fetch data for %s", ticker)
        return None


def get_m5(ticker: str) -> Optional[pd.DataFrame]:
    return fetch_candles(ticker, interval="5m", period="1d")


def get_m15(ticker: str) -> Optional[pd.DataFrame]:
    return fetch_candles(ticker, interval="15m", period="5d")
