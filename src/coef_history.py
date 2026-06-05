import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

from src.data import load_data
from src.features import build_feature_panel
from src.model import estimate_weighting_matrix, fit_gls_ridge

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def compute_beta_se(
    feature_panel, target, train_dates, W, tickers, beta, alpha,
):
    """Frequentist SE for GLS-Ridge coefficients."""
    feature_names = list(feature_panel.keys())
    p = len(feature_names)

    A = np.zeros((p, p))
    residual_sq_sum = 0.0
    n_obs = 0

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
        if t not in target.index:
            continue
        r_t = target.loc[t].reindex(tickers).values
        mask = np.all(np.isfinite(F_t), axis=1) & np.isfinite(r_t)
        if mask.sum() < 10:
            continue
        F_t = F_t[mask]
        r_t = r_t[mask]
        valid_idx = np.where(mask)[0]
        W_t = W[np.ix_(valid_idx, valid_idx)]

        A += F_t.T @ W_t @ F_t
        e = r_t - F_t @ beta
        residual_sq_sum += e @ W_t @ e
        n_obs += len(r_t)

    sigma2 = residual_sq_sum / max(n_obs - p, 1)
    A_reg = A + alpha * np.eye(p)
    A_reg_inv = np.linalg.inv(A_reg)
    cov_beta = sigma2 * A_reg_inv @ A @ A_reg_inv
    se = np.sqrt(np.abs(np.diag(cov_beta)))
    return se


def load_alpha_history(method: str) -> pd.Series:
    """Load the alpha-per-refit from the existing backtest parquet file."""
    path = DATA_DIR / f"backtest_{method}.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    # alpha is constant within each 4-week refit window; take one value per refit
    alpha_by_date = df.groupby("date")["alpha"].first().sort_index()
    return alpha_by_date


def run_coef_history(
    panel, target, returns,
    method: str,
    initial_train_weeks: int = 260,
    retrain_every: int = 4,
):
    """Iterate through refit schedule using pre-computed alpha values."""
    print(f"\n=== {method} ===")
    
    # Load alphas from existing backtest
    alpha_history = load_alpha_history(method)
    print(f"  loaded {len(alpha_history)} alpha values from backtest parquet")
    
    all_dates = returns.index.sort_values()
    T = len(all_dates)

    history = []
    feature_names = list(panel.keys())
    refit_count = 0

    cursor = initial_train_weeks
    embargo_weeks = 4
    while cursor + embargo_weeks + retrain_every <= T:
        train_dates = all_dates[:cursor]
        returns_train = returns.loc[train_dates]
        
        # Determine the test date that corresponds to this refit
        test_start_idx = cursor + embargo_weeks
        test_start_date = all_dates[test_start_idx]
        
        # Find the alpha for this refit window
        try:
            best_alpha = alpha_history.loc[test_start_date]
        except KeyError:
            # If the exact date isn't in the parquet, find the nearest one
            available_dates = alpha_history.index
            mask = available_dates >= test_start_date
            if mask.any():
                best_alpha = alpha_history.loc[available_dates[mask][0]]
            else:
                print(f"  no alpha found for refit at {test_start_date.date()}, skipping")
                cursor += retrain_every
                continue
        
        try:
            W, info = estimate_weighting_matrix(returns_train, method=method)
            tickers = info["tickers"]
            beta = fit_gls_ridge(
                panel, target, train_dates, W, tickers, alpha=best_alpha
            )
            se = compute_beta_se(
                panel, target, train_dates, W, tickers, beta, best_alpha
            )
        except Exception as e:
            print(f"  refit at {train_dates[-1].date()}: failed — {e}")
            cursor += retrain_every
            continue
        
        record = {
            "refit_date": train_dates[-1],
            "alpha": best_alpha,
            "n_tickers": info["N"],
        }
        for i, name in enumerate(feature_names):
            record[f"beta_{name}"] = beta[i]
            record[f"se_{name}"] = se[i]
            record[f"t_{name}"] = beta[i] / se[i] if se[i] > 0 else np.nan
        history.append(record)
        
        refit_count += 1
        if refit_count % 20 == 0:
            print(f"  {refit_count} refits done — current train_end="
                  f"{train_dates[-1].date()}, alpha={best_alpha}")
        
        cursor += retrain_every

    df = pd.DataFrame(history)
    out_path = DATA_DIR / f"coef_history_{method}.parquet"
    df.to_parquet(out_path)
    print(f"  saved {refit_count} refits to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["identity", "sample", "mp", "all"],
                        default="all")
    args = parser.parse_args()

    print("Loading returns and building features...")
    returns = load_data()
    panel = build_feature_panel(returns)
    target = returns.shift(-1)
    print("Done. Starting coefficient history runs.")

    methods = ["identity", "sample", "mp"] if args.method == "all" else [args.method]
    for m in methods:
        run_coef_history(panel, target, returns, method=m)