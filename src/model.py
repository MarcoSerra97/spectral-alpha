import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.rmt import MarchenkoPasturFilter


def estimate_weighting_matrix(
    returns_train: pd.DataFrame,
    method: str = "mp",
) -> tuple[np.ndarray, dict]:
    """
    Estimate the GLS weighting matrix W = Sigma^{-1} from training-window returns.

    For the "sample" and "mp" methods, the cleaning is performed on the
    correlation matrix (where Marchenko-Pastur applies directly), and the
    result is then rescaled by empirical asset volatilities to produce the
    covariance matrix whose inverse is used as the GLS weighting.

    Parameters
    ----------
    returns_train : pd.DataFrame
        Training-window returns, rows = dates, columns = tickers.
        Tickers with any NaN in the window are dropped.
    method : {"identity", "sample", "mp"}
        "identity" -> W = I (reduces GLS to OLS)
        "sample"   -> W = inverse raw sample covariance matrix
        "mp"       -> W = inverse MP-cleaned covariance matrix (headline method)

    Returns
    -------
    W : np.ndarray of shape (N, N)
        The weighting matrix.
    info : dict
        Diagnostic info: tickers kept, condition number, and (for mp)
        the MP filter summary.
    """
    R = returns_train.dropna(axis=1)
    tickers = R.columns.tolist()
    N = len(tickers)

    info = {"tickers": tickers, "N": N, "T": len(R), "method": method}

    if method == "identity":
        W = np.eye(N)
        info["condition_number"] = 1.0
        return W, info

    if method == "sample":
        sigmas = R.std(axis=0).values
        D = np.diag(sigmas)
        C = np.corrcoef(R.values.T)
        Sigma = D @ C @ D
        W = np.linalg.pinv(Sigma)
        info["condition_number"] = float(np.linalg.cond(Sigma))
        return W, info

    if method == "mp":
        sigmas = R.std(axis=0).values
        D = np.diag(sigmas)
        mp = MarchenkoPasturFilter().fit(R.values)
        C_cleaned = mp.correlation_matrix_
        Sigma_cleaned = D @ C_cleaned @ D
        W = np.linalg.pinv(Sigma_cleaned)
        info["condition_number"] = float(np.linalg.cond(Sigma_cleaned))
        info["mp_summary"] = mp.summary()
        return W, info

    raise ValueError(f"Unknown method: {method!r}")


def fit_gls_ridge(
    feature_panel: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    W: np.ndarray,
    tickers: list[str],
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Solve the pooled GLS-Ridge regression in closed form.

        beta_hat = (sum_t F_t^T W F_t + alpha I)^{-1} sum_t F_t^T W r_{t+1}

    where the sum runs over all dates in train_dates, and W is constant
    across dates within the training window.

    Parameters
    ----------
    feature_panel : dict of {name -> (dates x tickers) DataFrame}
        The rank-standardized features.
    target : pd.DataFrame
        Next-week returns, shape (dates x tickers).
    train_dates : pd.DatetimeIndex
        Dates over which to accumulate the regression.
    W : np.ndarray of shape (N, N)
        GLS weighting matrix.
    tickers : list[str]
        N tickers corresponding to the rows/columns of W.
    alpha : float
        Ridge penalty.

    Returns
    -------
    beta : np.ndarray of shape (p,)
        Coefficient vector for the p features, in the order of
        feature_panel.keys().
    """
    feature_names = list(feature_panel.keys())
    p = len(feature_names)

    A = np.zeros((p, p))
    b = np.zeros(p)

    for t in train_dates:
        rows = []
        for name in feature_names:
            if t not in feature_panel[name].index:
                rows = []
                break
            row = feature_panel[name].loc[t].reindex(tickers).values
            rows.append(row)

        if not rows:
            continue

        F_t = np.column_stack(rows)
        r_t = target.loc[t].reindex(tickers).values if t in target.index else None

        if r_t is None:
            continue

        mask = np.all(np.isfinite(F_t), axis=1) & np.isfinite(r_t)
        if mask.sum() < 10:
            continue

        F_t = F_t[mask]
        r_t = r_t[mask]
        valid_idx = np.where(mask)[0]
        W_t = W[np.ix_(valid_idx, valid_idx)]

        A += F_t.T @ W_t @ F_t
        b += F_t.T @ W_t @ r_t

    beta = np.linalg.solve(A + alpha * np.eye(p), b)
    return beta


def predict(
    feature_panel: dict[str, pd.DataFrame],
    test_dates: pd.DatetimeIndex,
    beta: np.ndarray,
    tickers: list[str],
) -> pd.DataFrame:
    """
    Apply fitted coefficients to each test date to produce predicted
    cross-sectional return scores.

    Returns
    -------
    pd.DataFrame of shape (n_test_dates, N)
        Predicted scores, rows = dates, columns = tickers.
    """
    feature_names = list(feature_panel.keys())
    predictions = {}

    for t in test_dates:
        rows = []
        for name in feature_names:
            if t not in feature_panel[name].index:
                rows = []
                break
            row = feature_panel[name].loc[t].reindex(tickers).values
            rows.append(row)

        if not rows:
            continue
        F_t = np.column_stack(rows)
        predictions[t] = F_t @ beta

    return pd.DataFrame(predictions, index=tickers).T


def select_ridge_alpha(
    feature_panel: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    W: np.ndarray,
    tickers: list[str],
    alphas: list[float] = None,
    n_splits: int = 5,
    gap: int = 4,
) -> float:
    """
    Select the ridge penalty alpha by time-series CV on the training dates.

    Uses TimeSeriesSplit with an embargo gap to prevent leakage. For each
    fold, fits GLS-Ridge with each candidate alpha and evaluates per-date
    Spearman IC on the validation fold. Chooses the alpha with the highest
    mean validation IC.
    """
    if alphas is None:
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    train_dates = pd.DatetimeIndex(train_dates).sort_values()
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    ic_scores = {a: [] for a in alphas}

    for fold_train_idx, fold_val_idx in tscv.split(train_dates):
        fold_train_dates = train_dates[fold_train_idx]
        fold_val_dates = train_dates[fold_val_idx]

        for a in alphas:
            beta = fit_gls_ridge(
                feature_panel, target, fold_train_dates, W, tickers, alpha=a
            )
            pred = predict(feature_panel, fold_val_dates, beta, tickers)
            ics = []
            for t in fold_val_dates:
                if t not in pred.index or t not in target.index:
                    continue
                p_t = pred.loc[t].reindex(tickers)
                r_t = target.loc[t].reindex(tickers)
                ic = p_t.corr(r_t, method="spearman")
                if np.isfinite(ic):
                    ics.append(ic)
            if ics:
                ic_scores[a].append(np.mean(ics))

    mean_ic = {a: np.mean(scores) if scores else -np.inf
               for a, scores in ic_scores.items()}
    best_alpha = max(mean_ic, key=mean_ic.get)
    return best_alpha


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data import load_data
    from src.features import build_feature_panel

    print("Loading returns...")
    returns = load_data()
    subset = returns.iloc[:300, :100]
    print(f"Using subset of shape {subset.shape} for model demo.")

    print("\nBuilding feature panel...")
    panel = build_feature_panel(subset)
    target = subset.shift(-1)

    all_dates = subset.index
    split_idx = int(len(all_dates) * 0.8)
    train_dates = all_dates[:split_idx]
    test_dates = all_dates[split_idx:]

    returns_train = subset.loc[train_dates]
    print(f"Training window for W estimation: {returns_train.shape}")

    results = {}
    for method in ["identity", "sample", "mp"]:
        print(f"\n=== variant: {method} ===")
        W, info = estimate_weighting_matrix(returns_train, method=method)
        tickers = info["tickers"]
        print(f"  N = {info['N']}, T = {info['T']}, "
              f"cond(Sigma) = {info['condition_number']:.2e}")
        if "mp_summary" in info:
            print(f"  signal eigenvalues: {info['mp_summary']['n_signal_eigenvalues']}")

        print("  selecting alpha via time-series CV...")
        best_alpha = select_ridge_alpha(panel, target, train_dates, W, tickers)
        print(f"  selected alpha = {best_alpha}")

        beta = fit_gls_ridge(panel, target, train_dates, W, tickers, alpha=best_alpha)
        print(f"  coefficients:")
        for name, c in zip(panel.keys(), beta):
            print(f"    {name}: {c:+.6f}")

        pred = predict(panel, test_dates, beta, tickers)
        ics = []
        for t in test_dates:
            if t not in pred.index or t not in target.index:
                continue
            p_t = pred.loc[t].reindex(tickers)
            r_t = target.loc[t].reindex(tickers)
            ic = p_t.corr(r_t, method="spearman")
            if np.isfinite(ic):
                ics.append(ic)

        ic_arr = np.array(ics)
        mean_ic = ic_arr.mean() if len(ic_arr) > 0 else np.nan
        std_ic = ic_arr.std() if len(ic_arr) > 0 else np.nan
        icir = mean_ic / std_ic if std_ic > 0 else np.nan
        print(f"  test mean IC: {mean_ic:+.4f}  std: {std_ic:.4f}  ICIR: {icir:+.4f}")
        results[method] = (mean_ic, std_ic, icir, best_alpha)

    print("\n\n===== summary =====")
    print(f"{'method':12s} {'alpha':>10s} {'IC mean':>10s} {'IC std':>10s} {'ICIR':>10s}")
    for method, (mic, sic, icir, a) in results.items():
        print(f"{method:12s} {a:>10.3f} {mic:>+10.4f} {sic:>10.4f} {icir:>+10.4f}")