import numpy as np
import pandas as pd


def momentum_12_1(returns: pd.DataFrame) -> pd.DataFrame:
    """
    12-month minus 1-month momentum, Jegadeesh-Titman style.

    Cumulative return from 52 weeks ago to 4 weeks ago, excluding the most
    recent month to avoid short-term reversal contamination.

    Returns
    -------
    pd.DataFrame
        Rows = weekly dates, columns = tickers. Values = momentum signal.
        First ~52 rows are NaN due to insufficient history.
    """
    log_ret = np.log1p(returns) #create matrix of single-period log returns 
    momentum_log = log_ret.shift(4).rolling(window=48).sum() #builds \sum_{s=t-51}^{t-4}log(1+r_s); shift(4) shifts down by 4 rows implementing the K=4 skip, so that at row t we read \ell_{t-4}; .rolling(window=48).sum() sums the most recent 48 entries from t-4, i.e sums (t-4) back to (t-52).
    return np.expm1(momentum_log) #gives M_t^{(52,4)} (first 51 rows are NaN)


def short_term_reversal(returns: pd.DataFrame) -> pd.DataFrame:
    """
    1-week short-term reversal signal.
    The signal is the negative of last week's return — high last week implies
    likely reversal this week.
    """
    return -returns.shift(1) #shift(1) moves every value down one row, so at date t
                             #the entry is -r_{t-1}. (first row is NaN)


def realized_volatility(returns: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """
    Rolling realized volatility over the past `window` weeks.
    Used both as a feature and for risk normalization downstream.
    """
    return returns.rolling(window=window).std()


def amihud_illiquidity(
    returns: pd.DataFrame,
    window: int = 12,
) -> pd.DataFrame:
    """
    Amihud illiquidity proxy: mean of |return| over a rolling window.

    The true Amihud measure is |return| / dollar volume, but since we do not
    have dollar volume in this pipeline we use the absolute-return component
    only. This still captures the dispersion / instability aspect of
    illiquidity for the cross-sectional ranking.
    """
    return returns.abs().rolling(window=window).mean()


def hurst_exponent(
    returns: pd.DataFrame,
    window: int = 52,
) -> pd.DataFrame:
    """
    Rolling Hurst exponent estimated via rescaled range analysis.

    H ~ 0.5 → random walk (no memory)
    H > 0.5 → persistent / trending behavior
    H < 0.5 → mean-reverting behavior

    This is the physics-inspired feature: returns whose variance scales
    super-diffusively (H > 0.5) carry different information than purely
    diffusive ones.
    """
    def _hurst_single(series: np.ndarray) -> float:
        if np.any(np.isnan(series)) or len(series) < 20:
            return np.nan
        lags = range(2, min(20, len(series) // 2))
        tau = [np.std(series[lag:] - series[:-lag]) for lag in lags]
        tau = np.array(tau)  #it gives the array of std(\delta_\tau) for the various \tau
        if np.any(tau <= 0):
            return np.nan
        slope = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
        return slope

    return returns.rolling(window=window).apply(_hurst_single, raw=True)


def cross_sectional_demean(features: pd.DataFrame) -> pd.DataFrame:
    """
    At each date, subtract the cross-sectional mean from each feature.
    This neutralizes any market-wide level in the feature.
    """
    return features.sub(features.mean(axis=1), axis=0)


def cross_sectional_rank(features: pd.DataFrame) -> pd.DataFrame:
    """
    At each date, convert features to cross-sectional ranks in [-0.5, 0.5], such that mean is near 0 (demeaned features)
    This is the standard pre-processing for cross-sectional models —
    it makes the model scale-invariant and robust to outliers.
    """
    ranks = features.rank(axis=1, pct=True)
    return ranks - 0.5


def winsorize_cross_section(
    features: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """
    Cross-sectional winsorization at the given percentiles, applied per date.
    """
    def _winsorize_row(row: pd.Series) -> pd.Series:
        low = row.quantile(lower)
        high = row.quantile(upper)
        return row.clip(lower=low, upper=high)

    return features.apply(_winsorize_row, axis=1)


def build_feature_panel(returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build the full feature panel. Returns a dictionary mapping feature name
    to a DataFrame of (dates × tickers).

    Each feature is winsorized cross-sectionally to remove extreme outliers,
    then converted to cross-sectional ranks for use by the downstream model.
    """
    raw_features = {
        "momentum_12_1": momentum_12_1(returns),
        "short_term_reversal": short_term_reversal(returns),
        "realized_volatility": realized_volatility(returns),
        "amihud_illiquidity": amihud_illiquidity(returns),
        "hurst_exponent": hurst_exponent(returns),
    }

    processed = {}
    for name, feat in raw_features.items():
        feat_w = winsorize_cross_section(feat)
        feat_r = cross_sectional_rank(feat_w)
        processed[name] = feat_r

    return processed


def stack_features_for_modeling(
    feature_panel: dict[str, pd.DataFrame],
    target: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the feature panel and target into a long-format DataFrame
    suitable for cross-sectional modeling.

    Returns
    -------
    pd.DataFrame
        Multi-indexed by (date, ticker), columns are features plus 'target'.
        Rows with any NaN are dropped.
    """
    frames = []
    for name, feat in feature_panel.items():
        s = feat.stack()
        s.name = name
        frames.append(s)

    target_stacked = target.stack()
    target_stacked.name = "target"
    frames.append(target_stacked)

    df = pd.concat(frames, axis=1)
    df.index.names = ["date", "ticker"]
    df = df.dropna()
    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data import load_data

    print("Loading returns...")
    returns = load_data()
    print(f"Returns shape: {returns.shape}")

    # use a smaller subset for the demo — Hurst is slow
    subset = returns.iloc[:200, :50]
    print(f"\nUsing subset of shape {subset.shape} for feature demo.")

    print("\nBuilding feature panel...")
    panel = build_feature_panel(subset)

    for name, feat in panel.items():
        n_valid = feat.notna().sum().sum()
        total = feat.size
        print(f"  {name}: shape {feat.shape}, {n_valid}/{total} non-NaN values")

    # target: next-week return (shifted by -1)
    target = subset.shift(-1)

    print("\nStacking features and target into long format...")
    df = stack_features_for_modeling(panel, target)
    print(f"Final modeling DataFrame shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nFeature correlation with target:")
    for col in df.columns:
        if col == "target":
            continue
        corr = df[col].corr(df["target"])
        print(f"  {col}: {corr:+.4f}")