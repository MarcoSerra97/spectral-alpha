import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.data import load_data
from src.rmt import MarchenkoPasturFilter


# ---------- paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# ---------- load results ----------
def load_backtest_results() -> dict[str, pd.DataFrame]:
    """Load the three backtest result parquet files."""
    results = {}
    for method in ["identity", "sample", "mp"]:
        path = DATA_DIR / f"backtest_{method}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing backtest file: {path}")
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        results[method] = df
    return results


# ---------- plot 1: cumulative IC ----------
def plot_cumulative_ic(results: dict[str, pd.DataFrame]) -> None:
    """
    Plot cumulative sum of weekly IC across time, one line per variant.

    A consistently upward-trending line indicates persistent positive
    predictive power; a flat or downward line indicates no signal.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    colors = {"identity": "tab:gray", "sample": "tab:orange", "mp": "tab:blue"}
    labels = {
        "identity": "Identity (OLS-Ridge)",
        "sample":   "GLS-Ridge (sample $\\Sigma^{-1}$)",
        "mp":       "GLS-Ridge (MP-cleaned $\\Sigma^{-1}$)",
    }

    for method, df in results.items():
        cum_ic = df["ic"].cumsum()
        ax.plot(df["date"], cum_ic, label=labels[method],
                color=colors[method], linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative weekly IC")
    ax.set_title("Cumulative information coefficient — three GLS-Ridge variants")
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out = FIGURES_DIR / "cumulative_ic.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved {out}")


# ---------- plot 2: IC distribution ----------
def plot_ic_distribution(results: dict[str, pd.DataFrame]) -> None:
    """
    Histogram of weekly IC values for each variant, with mean marked.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    labels = {
        "identity": "Identity (OLS-Ridge)",
        "sample":   "GLS-Ridge (sample)",
        "mp":       "GLS-Ridge (MP-cleaned)",
    }
    colors = {"identity": "tab:gray", "sample": "tab:orange", "mp": "tab:blue"}

    for ax, (method, df) in zip(axes, results.items()):
        ic = df["ic"].dropna().values
        ax.hist(ic, bins=40, color=colors[method], alpha=0.7, edgecolor="black", linewidth=0.4)
        ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6, label="zero")
        ax.axvline(ic.mean(), color="red", linewidth=1.5,
                   label=f"mean = {ic.mean():+.4f}")
        ax.set_title(labels[method])
        ax.set_xlabel("Weekly IC")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Count")

    plt.suptitle("Distribution of weekly information coefficients", y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "ic_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ---------- plot 3: eigenvalue spectrum ----------
def plot_eigenvalue_spectrum() -> None:
    """
    Compare the eigenvalue spectrum of the raw sample correlation matrix
    to the MP-cleaned correlation matrix, plus the theoretical MP curve.
    """
    from scipy.stats import gaussian_kde
    
    returns = load_data()
    R = returns.iloc[:260].dropna(axis=1)
    print(f"  spectrum computed on first 260-week window, "
          f"{R.shape[1]} tickers")

    raw_C = np.corrcoef(R.values.T)
    raw_eigs = np.sort(np.linalg.eigvalsh(raw_C))[::-1]
    

    mp = MarchenkoPasturFilter().fit(R.values)
    cleaned_C = mp.correlation_matrix_
    cleaned_eigs = np.sort(np.linalg.eigvalsh(cleaned_C))[::-1]
     
    raw_eigs = np.clip(raw_eigs, 1e-14, None)
    cleaned_eigs = np.clip(cleaned_eigs, 1e-14, None)


    N, T = R.shape[1], R.shape[0]
    q = N / T
    lam_plus = (1 + np.sqrt(q)) ** 2
    lam_minus = (1 - np.sqrt(q)) ** 2

    # theoretical MP density:  p(lam) = sqrt((lam_+ - lam)(lam - lam_-)) / (2*pi*q*lam)
    def mp_pdf(lam, q, lam_minus, lam_plus):
        out = np.zeros_like(lam)
        mask = (lam >= lam_minus) & (lam <= lam_plus)
        out[mask] = np.sqrt(
            np.maximum(0, (lam_plus - lam[mask]) * (lam[mask] - lam_minus))
        ) / (2 * np.pi * q * lam[mask])
        return out

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left: sorted eigenvalue curve, lam_k vs k
    ax = axes[0]
    # restrict to positive eigenvalues for a clean curve
    ax.plot(np.arange(len(raw_eigs)), raw_eigs,
            color="tab:orange", linewidth=1.5, label="Raw sample $C$")
    ax.plot(np.arange(len(cleaned_eigs)), cleaned_eigs,
            color="tab:blue", linewidth=1.5, label="MP-cleaned $\\tilde C$")
    ax.axhline(lam_plus, color="black", linestyle=":", linewidth=1,
               alpha=0.7, label=f"$\\lambda_+ = {lam_plus:.2f}$")
    ax.axhline(lam_minus, color="black", linestyle="--", linewidth=1,
               alpha=0.7, label=f"$\\lambda_- = {lam_minus:.2f}$")
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue index $k$ (sorted)")
    ax.set_ylabel("$\\lambda_k$ (log scale)")
    ax.set_title("Sorted eigenvalue spectrum")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
# right: smooth density distributions, log x-axis to show all eigenvalues
    ax = axes[1]
    raw_positive_for_density = raw_eigs[raw_eigs > 0.01]
    cleaned_positive_for_density = cleaned_eigs[cleaned_eigs > 0.01]

    lam_max = max(raw_eigs.max(), cleaned_eigs.max()) * 1.3
    lam_grid = np.logspace(np.log10(0.01), np.log10(lam_max), 500)

    # kernel density estimate of raw eigenvalues — in log space
    log_raw = np.log10(raw_positive_for_density)
    kde_raw = gaussian_kde(log_raw, bw_method=0.1)
    raw_density = kde_raw(np.log10(lam_grid)) / (lam_grid * np.log(10))
    ax.fill_between(lam_grid, raw_density, alpha=0.4,
                    color="tab:orange", label="Raw sample (KDE)")
    ax.plot(lam_grid, raw_density, color="tab:orange", linewidth=1)

    # cleaned eigenvalues: histogram with log bins
    log_bins = np.logspace(np.log10(0.01), np.log10(lam_max), 80)
    ax.hist(cleaned_positive_for_density, bins=log_bins, density=True,
            alpha=0.4, color="tab:blue", label="MP-cleaned (histogram)")

    # theoretical MP density (defined only on the bulk range)
    mp_density = mp_pdf(lam_grid, q, lam_minus, lam_plus)
    ax.plot(lam_grid, mp_density, color="black", linewidth=2,
            label="Theoretical MP density")

    ax.axvline(lam_minus, color="black", linestyle="--", linewidth=1,
               alpha=0.5)
    ax.axvline(lam_plus, color="black", linestyle=":", linewidth=1,
               alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, None)
    ax.set_xlabel("$\\lambda$ (log scale)")
    ax.set_ylabel("Density (log scale)")
    ax.set_title(f"Eigenvalue density (q = N/T = {q:.2f})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle("Eigenvalue spectrum: raw vs MP-cleaned correlation matrix",
                 y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "eigenvalue_spectrum.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ---------- plot 4: IC std comparison ----------
def plot_ic_std_comparison(results: dict[str, pd.DataFrame]) -> None:
    """
    Bar chart of the IC standard deviation across the three variants.
    This is the project's headline finding: MP cleaning reduces
    estimator variance.
    """
    methods = list(results.keys())
    ic_stds = [results[m]["ic"].std() for m in methods]
    labels = {
        "identity": "Identity\n(OLS-Ridge)",
        "sample":   "GLS-Ridge\n(sample)",
        "mp":       "GLS-Ridge\n(MP-cleaned)",
    }
    colors = {"identity": "tab:gray", "sample": "tab:orange", "mp": "tab:blue"}

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar([labels[m] for m in methods], ic_stds,
                  color=[colors[m] for m in methods], edgecolor="black")
    for bar, val in zip(bars, ic_stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("IC standard deviation (weekly)")
    ax.set_title("Cross-sectional IC volatility across variants")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, max(ic_stds) * 1.15)

    plt.tight_layout()
    out = FIGURES_DIR / "ic_std_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved {out}")

def plot_coefficient_t_statistics() -> None:
    """
    Plot the t-statistics of regression coefficients over time,
    one panel per variant, with a line per feature.
    """
    methods = ["identity", "sample", "mp"]
    method_labels = {
        "identity": "Identity (OLS-Ridge)",
        "sample":   "GLS-Ridge (sample $\\Sigma^{-1}$)",
        "mp":       "GLS-Ridge (MP-cleaned $\\Sigma^{-1}$)",
    }
    
    # Feature names and colors
    feature_names = ["momentum_12_1", "short_term_reversal",
                     "realized_volatility", "amihud_illiquidity",
                     "hurst_exponent"]
    feature_labels = {
        "momentum_12_1": "12-1 momentum",
        "short_term_reversal": "Short-term reversal",
        "realized_volatility": "Realized volatility",
        "amihud_illiquidity": "Amihud illiquidity",
        "hurst_exponent": "Hurst exponent",
    }
    feature_colors = {
        "momentum_12_1": "tab:blue",
        "short_term_reversal": "tab:orange",
        "realized_volatility": "tab:green",
        "amihud_illiquidity": "tab:red",
        "hurst_exponent": "tab:purple",
    }
    
    # Load coefficient histories
    histories = {}
    for method in methods:
        path = DATA_DIR / f"coef_history_{method}.parquet"
        if not path.exists():
            print(f"  skipping {method}: no coefficient history file")
            continue
        df = pd.read_parquet(path)
        df["refit_date"] = pd.to_datetime(df["refit_date"])
        df = df.sort_values("refit_date").reset_index(drop=True)
        histories[method] = df
    
    if not histories:
        print("  no coefficient history files found")
        return
    
    n_panels = len(histories)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.5 * n_panels),
                              sharex=True)
    if n_panels == 1:
        axes = [axes]
    
    for ax, (method, df) in zip(axes, histories.items()):
        for fname in feature_names:
            t_col = f"t_{fname}"
            if t_col not in df.columns:
                continue
            ax.plot(df["refit_date"], df[t_col],
                    linewidth=1.2, label=feature_labels[fname],
                    color=feature_colors[fname])
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.axhline(2, color="black", linewidth=0.5, alpha=0.5,
                   linestyle="--", label="$|t| = 2$ threshold")
        ax.axhline(-2, color="black", linewidth=0.5, alpha=0.5,
                   linestyle="--")
        ax.set_ylabel("Coefficient t-statistic")
        ax.set_title(method_labels[method], loc="left", fontsize=11)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
    
    axes[-1].set_xlabel("Refit date")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    
    plt.suptitle("Coefficient t-statistics across refits — by variant", y=1.00)
    plt.tight_layout()
    out = FIGURES_DIR / "coefficient_t_statistics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ---------- subperiod analysis ----------
def subperiod_analysis(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute mean IC and t-statistic per subperiod for each variant.

    Subperiods are chosen to include the major regime episodes:
    pre-crisis recovery (2010-2013), bull market (2014-2017),
    late cycle / pandemic (2018-2020), post-pandemic (2021-2024).
    """
    periods = [
        ("2010-2013", "2010-01-01", "2013-12-31"),
        ("2014-2017", "2014-01-01", "2017-12-31"),
        ("2018-2020", "2018-01-01", "2020-12-31"),
        ("2021-2024", "2021-01-01", "2024-12-31"),
    ]

    rows = []
    for method, df in results.items():
        for label, start, end in periods:
            mask = (df["date"] >= start) & (df["date"] <= end)
            ic = df.loc[mask, "ic"].dropna().values
            n = len(ic)
            if n == 0:
                continue
            mean_ic = ic.mean()
            std_ic = ic.std()
            se = std_ic / np.sqrt(n) if std_ic > 0 else np.nan
            t = mean_ic / se if se > 0 else np.nan
            rows.append({
                "method": method,
                "period": label,
                "n_weeks": n,
                "mean_ic": mean_ic,
                "ic_std": std_ic,
                "t_stat": t,
            })
    return pd.DataFrame(rows)


def plot_subperiod_ic(subperiod_df: pd.DataFrame) -> None:
    """Grouped bar chart of mean IC per subperiod and variant."""
    pivot = subperiod_df.pivot(index="period", columns="method",
                                values="mean_ic")
    pivot = pivot[["identity", "sample", "mp"]]

    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", ax=ax,
               color=["tab:gray", "tab:orange", "tab:blue"],
               edgecolor="black", width=0.75)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Mean weekly IC")
    ax.set_xlabel("Subperiod")
    ax.set_title("Subperiod mean IC by variant")
    ax.legend(title="Method")
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=0)
    plt.tight_layout()
    out = FIGURES_DIR / "subperiod_ic.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved {out}")


def plot_alpha_over_time(results: dict[str, pd.DataFrame]) -> None:
    """
    Plot the ridge alpha selected by time-series CV at each refit,
    for the three GLS-Ridge variants in separate panels.
    """
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, sharey=True)

    colors = {"identity": "tab:gray", "sample": "tab:orange", "mp": "tab:blue"}
    labels = {
        "identity": "Identity (OLS-Ridge)",
        "sample":   "GLS-Ridge (sample $\\Sigma^{-1}$)",
        "mp":       "GLS-Ridge (MP-cleaned $\\Sigma^{-1}$)",
    }

    for ax, (method, df) in zip(axes, results.items()):
        alpha_by_date = df.groupby("date")["alpha"].first().sort_index()
        alpha_per_refit = alpha_by_date.iloc[::4]
        ax.plot(alpha_per_refit.index, alpha_per_refit.values,
                marker="o", markersize=3, linewidth=1.0,
                color=colors[method])
        # also overlay a smoothed median for trend visibility
        smoothed = alpha_by_date.rolling(window=26, min_periods=5).median()
        ax.plot(smoothed.index, smoothed.values,
                linewidth=2, color=colors[method], alpha=0.6,
                linestyle="--", label="6-month rolling median")
        ax.set_yscale("log")
        ax.set_ylabel("$\\alpha$ (log)")
        ax.set_title(labels[method], loc="left", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)
        print(f"  {method}: {len(alpha_per_refit)} alpha values plotted")

    axes[-1].set_xlabel("Refit date")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.suptitle("Ridge penalty $\\alpha$ selected by time-series CV across refits",
                 y=1.00)
    plt.tight_layout()
    out = FIGURES_DIR / "alpha_over_time.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")

# ---------- summary table ----------
def print_summary_table(results: dict[str, pd.DataFrame]) -> None:
    """Print the final comparison table."""
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"{'method':12s} {'n':>6s} {'mean IC':>10s} {'IC std':>10s} "
          f"{'t-stat':>8s} {'ICIR(ann)':>11s} {'hit rate':>10s}")
    print("-" * 70)
    for method, df in results.items():
        ic = df["ic"].dropna().values
        n = len(ic)
        mean_ic = ic.mean()
        std_ic = ic.std()
        se = std_ic / np.sqrt(n) if std_ic > 0 else np.nan
        t = mean_ic / se if se > 0 else np.nan
        icir = mean_ic / std_ic * np.sqrt(52) if std_ic > 0 else np.nan
        hit = (ic > 0).mean()
        print(f"{method:12s} {n:>6d} {mean_ic:>+10.4f} {std_ic:>10.4f} "
              f"{t:>+8.2f} {icir:>+11.4f} {hit:>9.1%}")
    print("=" * 70)


if __name__ == "__main__":
    print("Loading backtest results...")
    results = load_backtest_results()
    for method, df in results.items():
        print(f"  {method}: {len(df)} test weeks "
              f"from {df['date'].min().date()} to {df['date'].max().date()}")

    print("\nGenerating plots...")
    plot_cumulative_ic(results)
    plot_ic_distribution(results)
    plot_ic_std_comparison(results)
    plot_alpha_over_time(results) 
    plot_coefficient_t_statistics()
    plot_eigenvalue_spectrum()

    print("\nSubperiod analysis...")
    subperiod_df = subperiod_analysis(results)
    print(subperiod_df.to_string(index=False))
    plot_subperiod_ic(subperiod_df)
    subperiod_df.to_csv(DATA_DIR / "subperiod_analysis.csv", index=False)
    print(f"  saved subperiod table to {DATA_DIR / 'subperiod_analysis.csv'}")

    print_summary_table(results)

    print(f"\nAll figures saved to {FIGURES_DIR}")