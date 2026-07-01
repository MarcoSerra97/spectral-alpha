"""
Controlled study of coefficient variance vs ridge alpha across the three
weighting variants.

This is a conditional-variance experiment: W is estimated once on the full
training window and held fixed across resamples. The bootstrap therefore
probes the variance of the estimators within a fixed training window, thus removing the across-refit dependence 
which is instead captured by the walking-forward analysis that gives the total variance.

This code is structured as follow:

  (1) Real Künsch moving-block bootstrap: blocks drawn with replacement,
      no deduplication, no sorting. The resample is a multiset of length
      exactly n_blocks * block_size.

  (3) Local pooled fit that
      handles multiplicity correctly — under MBB the same date can
      appear multiple times in the resample, and it must contribute
      with that multiplicity to F^T W F and F^T W r.

  (4) The diagnostic eigenvalue print is rescaled by the subsample-size
      ratio so the comparison with the alpha grid is the right one for
      the inner-loop regressions.

  (5) Monte Carlo SE of the std estimate is reported alongside the
      point estimates, so the reader knows the resolution of the chart.

  (6) Multiple seeds can be run to assess the seed-dependence of the
      conclusions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data import load_data
from src.features import build_feature_panel
from src.model import estimate_weighting_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# (1) Real Künsch moving-block bootstrap
# ---------------------------------------------------------------------------

def moving_block_bootstrap_indices(
    n_dates: int,
    n_subsamples: int,
    subsample_size: int,
    block_size: int = 12,
    seed: int = 42,
) -> list[np.ndarray]:
    """
    Generate moving-block bootstrap resamples of date indices.

    Each resample is a length-`subsample_size` array of integer indices into
    [0, n_dates), constructed by drawing ceil(subsample_size / block_size)
    block starts uniformly with replacement from [0, n_dates - block_size]
    and concatenating the resulting consecutive-index blocks. The output
    preserves multiplicities (no set(), no sort).

    Block starts can repeat; entire blocks then appear multiple times in the
    resample, which is the correct MBB behaviour.
    """
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(subsample_size / block_size))
    max_start = n_dates - block_size  # last valid start (inclusive)

    resamples = []
    for _ in range(n_subsamples):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
        # Truncate from the tail so the head still has clean block structure
        idx = idx[:subsample_size]
        resamples.append(idx)
    return resamples


# ---------------------------------------------------------------------------
# (3) Pooled GLS-Ridge fit that respects multiplicities
# ---------------------------------------------------------------------------

def fit_pooled_with_multiplicities(
    feature_panel: dict,
    target: pd.DataFrame,
    date_indices: np.ndarray,
    all_dates: pd.DatetimeIndex,
    W: np.ndarray,
    tickers: list[str],
    alpha: float,
) -> np.ndarray:
    """
    Solve the pooled GLS-Ridge normal equations over a multiset of date
    indices. A date appearing k times in `date_indices` contributes k copies
    of F_t^T W F_t and F_t^T W r_t to the normal equations — equivalent to
    pooling over the resample (with duplicates) directly.

    Implemented by counting multiplicities once with np.unique, then weighting
    each unique date's contribution. Avoids the O(k) overhead of looping over
    duplicates.
    """
    feature_names = list(feature_panel.keys())
    p = len(feature_names)
    A = np.zeros((p, p))
    b = np.zeros(p)

    unique_idx, counts = np.unique(date_indices, return_counts=True)

    for ix, k in zip(unique_idx, counts):
        t = all_dates[ix]

        rows = []
        skip = False
        for name in feature_names:
            if t not in feature_panel[name].index:
                skip = True
                break
            rows.append(feature_panel[name].loc[t].reindex(tickers).values)
        if skip:
            continue
        if t not in target.index:
            continue

        F_t = np.column_stack(rows)
        r_t = target.loc[t].reindex(tickers).values

        mask = np.all(np.isfinite(F_t), axis=1) & np.isfinite(r_t)
        if mask.sum() < 10:
            continue

        F_t = F_t[mask]
        r_t = r_t[mask]
        valid_idx = np.where(mask)[0]
        W_t = W[np.ix_(valid_idx, valid_idx)]

        # Multiplicity k enters linearly
        A += k * (F_t.T @ W_t @ F_t)
        b += k * (F_t.T @ W_t @ r_t)

    return np.linalg.solve(A + alpha * np.eye(p), b)


# ---------------------------------------------------------------------------
# (4) Eigenvalue diagnostic, rescaled to the subsample regime
# ---------------------------------------------------------------------------

def compute_normal_matrix_eigs(
    panel: dict,
    target: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    W: np.ndarray,
    tickers: list[str],
) -> np.ndarray:
    """Compute eigenvalues of A = sum_t F_t^T W F_t over train_dates."""
    feature_names = list(panel.keys())
    p = len(feature_names)
    A = np.zeros((p, p))

    for t in train_dates:
        rows = []
        skip = False
        for name in feature_names:
            if t not in panel[name].index:
                skip = True
                break
            rows.append(panel[name].loc[t].reindex(tickers).values)
        if skip or t not in target.index:
            continue

        F_t = np.column_stack(rows)
        r_t = target.loc[t].reindex(tickers).values
        mask = np.all(np.isfinite(F_t), axis=1) & np.isfinite(r_t)
        if mask.sum() < 10:
            continue
        F_t = F_t[mask]
        valid_idx = np.where(mask)[0]
        W_t = W[np.ix_(valid_idx, valid_idx)]
        A += F_t.T @ W_t @ F_t

    return np.linalg.eigvalsh(A)


# ---------------------------------------------------------------------------
# Main study
# ---------------------------------------------------------------------------

def run_alpha_study(
    n_bootstrap: int = 200,
    subsample_size: int = 200,
    block_size: int = 12,
    alphas: np.ndarray = None,
    n_seeds: int = 4,
):
    """
    Run the corrected alpha-variance study.

    Parameters
    ----------
    n_bootstrap : MBB resamples per seed (200 is enough for ~5% SE on the std).
    subsample_size : exact resample length (preserved by the corrected MBB).
    block_size : MBB block length (12 weeks ≈ one quarter, captures persistence).
    alphas : log-spaced ridge penalties.
    n_seeds : number of independent MBB realisations to assess seed-dependence.
    """
    if alphas is None:
        alphas = np.logspace(-1, 3.5, 25)

    print("Loading data and building features...")
    returns = load_data()
    panel = build_feature_panel(returns)
    target = returns.shift(-1)
    all_dates = returns.index.sort_values()

    train_dates = all_dates[:260]
    returns_train = returns.loc[train_dates]
    print(f"Training window: {train_dates[0].date()} → {train_dates[-1].date()}, "
          f"{len(train_dates)} weeks")

    # (3) - W estimated once on the full window, held fixed across resamples
    print("\nPre-estimating weighting matrices on full training window "
          "(fixed across MBB resamples — this is a conditional-variance "
          "experiment)...")
    weighting_info = {}
    for method in ["identity", "sample", "mp"]:
        W, info = estimate_weighting_matrix(returns_train, method=method)
        weighting_info[method] = (W, info)
        print(f"  {method}: N = {info['N']}, "
              f"cond(Sigma) = {info['condition_number']:.2e}")

    # (4) - eigenvalue diagnostic rescaled to subsample size
    print("\nDiagnostic: eigenvalues of (subsample_size/T_full) · X^T W X "
          "on full training window (rescaled to inner-loop regime).")
    print("  Ridge is ineffective when these eigenvalues dominate the alpha grid.")
    rescale = subsample_size / len(train_dates)
    feature_names_list = list(panel.keys())

    eig_table = {}
    for method in ["identity", "sample", "mp"]:
        W, info = weighting_info[method]
        tickers = info["tickers"]
        eigs = compute_normal_matrix_eigs(panel, target, train_dates, W,
                                          tickers) * rescale
        eig_table[method] = eigs
        eigs_sorted = sorted(eigs, reverse=True)
        print(f"  {method}: [" +
              ", ".join(f"{e:.2e}" for e in eigs_sorted) + "]")
    print(f"\n  alpha grid: [{alphas[0]:.2f}, {alphas[-1]:.2f}]   "
          f"(min normal-matrix eigenvalue across methods: "
          f"{min(e.min() for e in eig_table.values()):.2e})")

    # ----- (1)+(2) Generate Künsch MBB resamples for each seed -----
    seeds = list(range(42, 42 + n_seeds))
    print(f"\nGenerating MBB resamples for {n_seeds} seeds × {n_bootstrap} "
          f"resamples × subsample_size {subsample_size} × block_size "
          f"{block_size} ...")

    all_resamples = {
        seed: moving_block_bootstrap_indices(
            n_dates=len(train_dates),
            n_subsamples=n_bootstrap,
            subsample_size=subsample_size,
            block_size=block_size,
            seed=seed,
        )
        for seed in seeds
    }

    sizes = [len(r) for r in all_resamples[seeds[0]]]
    print(f"  Resample lengths: min={min(sizes)}, max={max(sizes)} "
          f"(should all equal {subsample_size}).")

    # ----- Fit regressions and collect coefficients -----
    p = len(feature_names_list)
    # results[seed][method][alpha] -> (n_bootstrap, p) array
    results = {
        seed: {m: {a: np.zeros((n_bootstrap, p)) for a in alphas}
               for m in ["identity", "sample", "mp"]}
        for seed in seeds
    }

    total_per_seed = n_bootstrap * len(alphas) * 3
    print(f"\nFitting {total_per_seed * n_seeds} regressions "
          f"({n_seeds} seeds × {n_bootstrap} resamples × {len(alphas)} "
          f"alphas × 3 methods)...")

    for s_i, seed in enumerate(seeds):
        for b, idx in enumerate(all_resamples[seed]):
            for method in ["identity", "sample", "mp"]:
                W, info = weighting_info[method]
                tickers = info["tickers"]
                for j, alpha in enumerate(alphas):
                    beta = fit_pooled_with_multiplicities(
                        panel, target, idx, train_dates, W, tickers, alpha
                    )
                    results[seed][method][alpha][b] = beta
            if (b + 1) % 50 == 0:
                print(f"  seed {seed} ({s_i+1}/{n_seeds}): "
                      f"resample {b+1}/{n_bootstrap}")

    # ----- Aggregate std and its MC SE across seeds -----
    print("\nAggregating std estimates and Monte Carlo SE across seeds...")

    raw_rows = []
    for method in ["identity", "sample", "mp"]:
        for alpha in alphas:
            for j, fname in enumerate(feature_names_list):
                # Per-seed std estimates
                stds_per_seed = np.array([
                    results[seed][method][alpha][:, j].std()
                    for seed in seeds
                ])
                row = {
                    "method": method,
                    "alpha": alpha,
                    "feature": fname,
                    "std_mean": stds_per_seed.mean(),
                    "std_seed_se": (stds_per_seed.std(ddof=1)
                                    if n_seeds > 1 else 0.0),
                    "std_chi_se": (
                        stds_per_seed.mean()
                        / np.sqrt(2 * (n_bootstrap - 1) * n_seeds)
                    ),
                }
                raw_rows.append(row)

    df_results = pd.DataFrame(raw_rows)
    out_csv = DATA_DIR / "alpha_variance_study_fixed.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"  Saved to {out_csv}")

    # ----- Plot with error bands -----
    print("\nGenerating plot with Monte Carlo error bands...")
    feature_labels = {
        "momentum_12_1": "12-1 momentum",
        "short_term_reversal": "Short-term reversal",
        "realized_volatility": "Realized volatility",
        "amihud_illiquidity": "Amihud illiquidity",
        "hurst_exponent": "Hurst exponent",
    }
    method_colors = {"identity": "tab:gray", "sample": "tab:orange",
                     "mp": "tab:blue"}
    method_labels = {
        "identity": "Identity (OLS-Ridge)",
        "sample": r"GLS-Ridge (sample $\Sigma^{-1}$)",
        "mp": r"GLS-Ridge (MP-cleaned $\Sigma^{-1}$)",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for j, fname in enumerate(feature_names_list):
        ax = axes[j]
        sub = df_results[df_results["feature"] == fname]
        for method in ["identity", "sample", "mp"]:
            sm = sub[sub["method"] == method].sort_values("alpha")
            ax.plot(sm["alpha"], sm["std_mean"],
                    label=method_labels[method],
                    color=method_colors[method],
                    linewidth=1.8, marker="o", markersize=3)
            # Combined SE = sqrt(seed_se^2 + chi_se^2) at each point
            se = np.sqrt(sm["std_seed_se"]**2 + sm["std_chi_se"]**2)
            ax.fill_between(sm["alpha"], sm["std_mean"] - se,
                            sm["std_mean"] + se,
                            color=method_colors[method], alpha=0.2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$\alpha$ (log scale)")
        ax.set_ylabel(r"Std of $\hat\beta$ across resamples (log scale)")
        ax.set_title(feature_labels[fname], fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    axes[5].axis("off")
    plt.suptitle(
        f"Coefficient variance vs ridge α (Künsch MBB, "
        f"{n_seeds}×{n_bootstrap} resamples, n={subsample_size}, "
        f"block={block_size})\n"
        f"shaded band = Monte Carlo SE of the std estimate",
        y=1.00, fontsize=12,
    )
    plt.tight_layout()
    out_png = FIGURES_DIR / "coefficient_variance_vs_alpha_fixed.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot to {out_png}")

    # ----- Summary printout at the regime where ridge is ineffective -----
    print("\nCoefficient std (mean ± SE across seeds + Monte Carlo) at α≈0.1 "
          "[ridge-ineffective regime for MP & Sample]:")
    a_low = alphas[0]
    print(f"  α = {a_low:.3f}")
    print(f"  {'feature':<25s} {'identity':>20s} {'sample':>20s} {'mp':>20s}")
    for fname in feature_names_list:
        row_strs = [f"{fname:<25s}"]
        for method in ["identity", "sample", "mp"]:
            r = df_results[
                (df_results["method"] == method) &
                (np.isclose(df_results["alpha"], a_low)) &
                (df_results["feature"] == fname)
            ].iloc[0]
            se = np.sqrt(r["std_seed_se"]**2 + r["std_chi_se"]**2)
            row_strs.append(f"{r['std_mean']:.4g} ± {se:.2g}".rjust(20))
        print("  " + " ".join(row_strs))

    print("\nDone.")
    return df_results


if __name__ == "__main__":
    df = run_alpha_study(
        n_bootstrap=200,
        subsample_size=200,
        block_size=12,
        n_seeds=4,
    )