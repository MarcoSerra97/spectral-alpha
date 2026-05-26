import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def get_sp500_tickers() -> list[str]:
    """
    Pull current S&P 500 constituents from Wikipedia.

    Limitation: this is not point-in-time. Stocks that were removed
    from the index between 2005 and 2024 are excluded, introducing
    survivorship bias. This inflates backtest metrics by approximately
    1-2% annualised. The relative comparison between model variants
    is less affected since the bias is common to all models.
    """
    # Add a User-Agent to mimic a browser

    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers=headers,
    )
    table = pd.read_html(StringIO(response.text))[0]
    tickers = (
        table["Symbol"]
        .str.replace(".", "-", regex=False)
        .tolist()
    )
    return tickers


def download_prices(
    start: str = "2005-01-01",
    end: str = "2024-12-31",
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted weekly closing prices for S&P 500 constituents.

    Returns
    -------
    pd.DataFrame
        Rows = weekly Friday dates, columns = tickers.
        Values = adjusted close prices. NaN where unavailable.
    """
    cache_path = DATA_DIR / "weekly_prices.parquet"

    if cache and cache_path.exists():
        print("Loading prices from cache...")
        return pd.read_parquet(cache_path)

    tickers = get_sp500_tickers()
    print(f"Downloading prices for {len(tickers)} tickers...")
    
    # The procedure from YahooFinance is to download all daily prices from Monday to Friday 
    # for period from start to end, for all the tickers.
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
    )["Close"]  
    
    #raw is thus a table with rows=days between start and end, columns=tickers

    # The above daily prices are then resampled to keep only the one on weekly Friday close
    weekly_prices = raw.resample("W-FRI").last()
    # the raw table is thus transformed as
    '''
    raw (daily)                    weekly_prices (weekly)
     5,040 rows × 503 cols    →     1,043 rows × 503 cols
    one row per trading day        one row per Friday
    '''


    # drop tickers with more than 50% missing values
    weekly_prices = weekly_prices.loc[
        :, weekly_prices.isna().mean() < 0.5
    ]

    if cache:
        weekly_prices.to_parquet(cache_path)
        print(f"Saved to {cache_path}")

    return weekly_prices


def compute_weekly_returns(
    prices: pd.DataFrame,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Compute simple weekly returns from weekly prices.
    Applies cross-sectional winsorization at 1%/99% at each date
    to remove price adjustment errors and data artefacts.

    Returns
    -------
    pd.DataFrame
        Rows = weekly dates, columns = tickers.
        First row dropped (NaN from pct_change).
    """
    cache_path = DATA_DIR / "weekly_returns.parquet"

    if cache and cache_path.exists():
        print("Loading returns from cache...")
        return pd.read_parquet(cache_path)

    returns = prices.pct_change().iloc[1:]
   #pct_change() computes (price_t - price_{t-1}) / price_{t-1} for every cell. 
   # The first row is always NaN because there is no previous price — iloc[1:] drops it.
   #prices.pct_change() is a Panda method


    def winsorize_row(row: pd.Series) -> pd.Series:
        low = row.quantile(0.01)
        high = row.quantile(0.99)
        return row.clip(lower=low, upper=high)
      #Defines a function that takes one row 
      # — one week of returns across all tickers — and clips extreme values:
      # any return below the 1st percentile is set to the 1st percentile value;
      #  any above the 99th is set to the 99th. 
    returns = returns.apply(winsorize_row, axis=1)
      #This command applies this function to each row independently. 
      # The axis=1 is critical — it means row-by-row (cross-sectional),
      #  not column-by-column (time-series). Each week has its own winsorization thresholds.
    if cache:
        returns.to_parquet(cache_path)
        print(f"Saved to {cache_path}")

    return returns


def load_data(
    start: str = "2005-01-01",
    end: str = "2024-12-31",
    cache: bool = True,
) -> pd.DataFrame:
    """
    Main entry point. Downloads prices and computes returns.
    All downstream modules import this function.

    Returns
    -------
    pd.DataFrame
        Weekly returns, rows = dates, columns = tickers.
    """
    prices = download_prices(start=start, end=end, cache=cache)
    returns = compute_weekly_returns(prices, cache=cache)
    return returns
# It chains download_prices and compute_weekly_returns and returns the final returns DataFrame. 
# Downstream modules — features.py, backtest.py — only need to know about this one function.

if __name__ == "__main__":
    returns = load_data()
    print(f"Returns shape: {returns.shape}")
    print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"Tickers: {returns.shape[1]}")
    print(f"Sample tickers: {list(returns.columns[:5])}")
    print(f"\nMissing values: {returns.isna().mean().mean():.1%} on average per cell")