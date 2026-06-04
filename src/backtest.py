import numpy as np
import pandas as pd

from src.model import (
    estimate_weighting_matrix,
    fit_gls_ridge,
    predict,
    select_ridge_alpha,
)


def walk_forward_backtest(
    feature_panel: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    returns: pd.DataFrame,
    method: str = "mp",
    initial_train_weeks: int = 260,
    retrain_every: int = 4,
    embargo_weeks: int = 4,
    alpha_grid: list[float] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward backtest for the GLS-Ridge model.

    At each refit step:
        1. Re-estimate the GLS weighting matrix W on the expanding
           training window of returns (from the start up to the current
           cutoff).
        2. Select the ridge penalty alpha via time-series CV on the
           training window, using the IC metric.
        3. Fit GLS-Ridge on the full training window with the selected alpha.
        4. Predict on the next `retrain_every` test weeks, separated from
           training by an `embargo_weeks` gap.

    Parameters
    ----------
    feature_panel : dict of {name -> (dates x tickers) DataFrame}
        Rank-standardized features.
    target : pd.DataFrame
        Next-week returns aligned with the same (dates, tickers) index.
    returns : pd.DataFrame
        Raw weekly returns used to estimate the GLS weighting matrix.
    method : {"identity", "sample", "mp"}
        Which weighting matrix to use.
    initial_train_weeks : int
        Number of weeks in the first training window.
    retrain_every : int
        Number of test weeks between successive refits.
    embargo_weeks : int
        Gap between the end of training and the start of testing.
    alpha_grid : list[float] or None
        Candidate ridge penalties for CV. Defaults to a standard grid.
    verbose : bool
        Print progress every refit step.

    Returns
    -------
    pd.DataFrame
        Per-test-date results: columns = ['date', 'ic', 'n_assets',
        'alpha', 'n_signal_eigenvalues']. One row per test week.
    """
    all_dates = returns.index.sort_values()
    T = len(all_dates)

    results = []
    refit_count = 0

    cursor = initial_train_weeks
    while cursor + embargo_weeks + retrain_every <= T:
        train_dates = all_dates[:cursor]
        test_start = cursor + embargo_weeks
        test_end = min(test_start + retrain_every, T)
        test_dates = all_dates[test_start:test_end]

        returns_train = returns.loc[train_dates]

        W, info = estimate_weighting_matrix(returns_train, method=method)
        tickers = info["tickers"]
        n_signal = info.get("mp_summary", {}).get(
            "n_signal_eigenvalues", np.nan
        )

        best_alpha = select_ridge_alpha(
            feature_panel, target, train_dates, W, tickers,
            alphas=alpha_grid,
        )

        beta = fit_gls_ridge(
            feature_panel, target, train_dates, W, tickers,
            alpha=best_alpha,
        )

        pred = predict(feature_panel, test_dates, beta, tickers)

        for t in test_dates:
            if t not in pred.index or t not in target.index:
                continue
            p_t = pred.loc[t].reindex(tickers)
            r_t = target.loc[t].reindex(tickers)
            ic = p_t.corr(r_t, method="spearman")
            if np.isfinite(ic):
                results.append({
                    "date": t,
                    "ic": ic,
                    "n_assets": len(tickers),
                    "alpha": best_alpha,
                    "n_signal_eigenvalues": n_signal,
                })

        refit_count += 1
        if verbose and refit_count % 10 == 0:
            n_done = len(results)
            print(f"  refit {refit_count:4d}  train_end={train_dates[-1].date()}  "
                  f"test_end={test_dates[-1].date()}  "
                  f"n_assets={len(tickers)}  "
                  f"n_test_weeks_collected={n_done}")

        cursor += retrain_every

    return pd.DataFrame(results)


def summarize_backtest(results: pd.DataFrame, label: str = "") -> dict:
    """
    Compute summary statistics for a completed walk-forward backtest.

    Returns IC mean, IC std, ICIR, hit rate, and number of test weeks.
    """
    ic = results["ic"].dropna().values
    if len(ic) == 0:
        return {"label": label, "n_weeks": 0}

    mean_ic = ic.mean()
    std_ic = ic.std()
    icir = mean_ic / std_ic * np.sqrt(52) if std_ic > 0 else np.nan
    hit_rate = (ic > 0).mean()

    return {
        "label": label,
        "n_weeks": len(ic),
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "icir_annualized": icir,
        "hit_rate": hit_rate,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.data import load_data
    from src.features import build_feature_panel

    print("Loading returns...")
    returns = load_data()
    print(f"Full returns shape: {returns.shape}")

    print("\nBuilding feature panel on full data — this is slow due to Hurst...")
    panel = build_feature_panel(returns)
    target = returns.shift(-1)
    print("Feature panel built.")

    summaries = {}
    for method in ["identity", "sample", "mp"]:
        print(f"\n========== walk-forward: {method} ==========")
        results = walk_forward_backtest(
            feature_panel=panel,
            target=target,
            returns=returns,
            method=method,
            initial_train_weeks=260,
            retrain_every=4,
            embargo_weeks=4,
        )
        summary = summarize_backtest(results, label=method)
        summaries[method] = summary

        results.to_parquet(
            Path(__file__).resolve().parent.parent / "data" /
            f"backtest_{method}.parquet"
        )
        print(f"\n  Saved to data/backtest_{method}.parquet")
        print(f"  N test weeks:   {summary['n_weeks']}")
        print(f"  Mean IC:        {summary['mean_ic']:+.4f}")
        print(f"  IC std:         {summary['std_ic']:.4f}")
        print(f"  ICIR (annual):  {summary['icir_annualized']:+.4f}")
        print(f"  Hit rate:       {summary['hit_rate']:.1%}")

    print("\n\n========== final comparison ==========")
    print(f"{'method':12s} {'n_weeks':>8s} {'IC mean':>10s} "
          f"{'IC std':>10s} {'ICIR (ann)':>12s} {'hit rate':>10s}")
    for method, s in summaries.items():
        print(f"{method:12s} {s['n_weeks']:>8d} {s['mean_ic']:>+10.4f} "
              f"{s['std_ic']:>10.4f} {s['icir_annualized']:>+12.4f} "
              f"{s['hit_rate']:>10.1%}")