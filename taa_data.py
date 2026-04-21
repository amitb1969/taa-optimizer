#!/usr/bin/env python3
"""
TAA Optimizer — Data Feed (v2)
================================
Fetches daily prices from Yahoo Finance, computes:
  - Pairwise correlation matrix (1yr trailing)
  - Ledoit-Wolf shrinkage for stable out-of-sample covariance
  - Positive-definiteness check + Higham nearest-PD correction
  - Annualized volatility per ETF
  - Latest closing price per ETF
  - Last update timestamp

Outputs: taa_data.json (same directory as this script)

Schedule: cron daily at 7pm ET (after market close)
  0 19 * * 1-5 cd /path/to/dir && python3 taa_data.py

The HTML tool auto-loads this file on startup.
"""

import json
import os
import sys
from datetime import datetime, timedelta

try:
    import yfinance as yf
    import numpy as np
    import pandas as pd
except ImportError:
    print("Install required packages:")
    print("  pip install yfinance numpy pandas")
    sys.exit(1)


# ─── CONFIG ───
TICKERS = [
    "SPY", "XLI", "SMH",                    # US equity
    "EWJ",                                    # Japan
    "FEZ", "EWP", "EWI", "GREK",            # Europe / Greece
    "EPOL", "EZA", "ECH", "EIS", "EPU",     # EM single-country
    "SHY",                                    # Cash proxy
    "GLD", "USO", "COPX",                    # Commodities
]

LOOKBACK_DAYS = 365
MIN_OBS = 200

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taa_data.json")


# ═══════════════════════════════════════════════════════════
#  LEDOIT-WOLF SHRINKAGE
# ═══════════════════════════════════════════════════════════
# Ledoit & Wolf (2004) — shrink sample covariance toward a
# constant-correlation target to reduce estimation noise.
#
#   Σ_shrunk = δ·F + (1-δ)·S
#
# F = constant correlation target (avg corr × vol structure)
# S = sample covariance
# δ = optimal shrinkage intensity (analytic formula)
#
# This pulls extreme correlations toward the mean, stabilizing
# the optimizer against noisy sample estimates.
# ═══════════════════════════════════════════════════════════

def ledoit_wolf_shrinkage(returns):
    """
    Ledoit-Wolf shrunk covariance with constant-correlation target.
    Returns: (shrunk_cov, shrunk_corr, delta, rbar)
    """
    T, N = returns.shape
    X = returns.values
    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / T

    var = np.diag(S).copy()
    std = np.sqrt(np.maximum(var, 1e-20))

    sample_corr = S / np.outer(std, std)
    np.fill_diagonal(sample_corr, 0)
    rbar = sample_corr.sum() / (N * (N - 1))

    F = rbar * np.outer(std, std)
    np.fill_diagonal(F, var)

    # Optimal shrinkage intensity
    Y = Xc ** 2
    pi = ((Y.T @ Y) / T - S ** 2).sum()
    gamma = np.sum((F - S) ** 2)

    delta = min(1.0, max(0.0, pi / (T * gamma))) if gamma > 0 else 1.0

    Sigma = delta * F + (1 - delta) * S

    shrunk_std = np.sqrt(np.maximum(np.diag(Sigma), 1e-20))
    shrunk_corr = Sigma / np.outer(shrunk_std, shrunk_std)
    np.fill_diagonal(shrunk_corr, 1.0)
    shrunk_corr = np.clip(shrunk_corr, -1.0, 1.0)

    return Sigma, shrunk_corr, delta, rbar


# ═══════════════════════════════════════════════════════════
#  POSITIVE DEFINITENESS CHECK + HIGHAM CORRECTION
# ═══════════════════════════════════════════════════════════
# The optimizer needs Σ to be PSD (all eigenvalues ≥ 0).
# If not PD, portfolio variance can go negative (impossible)
# and the optimizer finds spurious "free lunch" portfolios.
#
# Higham (2002) alternating projection finds the nearest PD
# correlation matrix in Frobenius norm.
# ═══════════════════════════════════════════════════════════

def nearest_pd(A, max_iter=100, tol=1e-10):
    """Higham (2002) nearest positive-definite correlation matrix."""
    n = A.shape[0]
    eigvals = np.linalg.eigvalsh(A)
    if eigvals.min() > tol:
        return A, True, eigvals.min()

    Y = A.copy()
    dS = np.zeros_like(A)

    for _ in range(max_iter):
        R = Y - dS
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals = np.maximum(eigvals, tol)
        X = eigvecs @ np.diag(eigvals) @ eigvecs.T
        dS = X - R
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)
        Y = (Y + Y.T) / 2.0
        if np.linalg.norm(Y - X, 'fro') < tol:
            break

    eigvals, eigvecs = np.linalg.eigh(Y)
    eigvals = np.maximum(eigvals, tol)
    result = eigvecs @ np.diag(eigvals) @ eigvecs.T
    np.fill_diagonal(result, 1.0)
    result = np.clip((result + result.T) / 2.0, -1.0, 1.0)

    return result, False, np.linalg.eigvalsh(result).min()


def check_and_fix_pd(corr_matrix, label="correlation"):
    """Check PD, apply Higham correction if needed."""
    eigvals = np.linalg.eigvalsh(corr_matrix)
    min_eig = eigvals.min()
    cond = eigvals.max() / max(min_eig, 1e-15)

    print(f"\n── PD Check ({label}) ──")
    print(f"  Min eigenvalue:   {min_eig:.6f}")
    print(f"  Condition number: {cond:.1f}")

    if min_eig > 1e-10:
        print(f"  ✓ Positive definite")
        return corr_matrix, True
    else:
        print(f"  ✗ NOT positive definite — applying Higham correction")
        fixed, _, new_min = nearest_pd(corr_matrix)
        print(f"  → Fixed min eigenvalue: {new_min:.6f}")
        print(f"  → Max element change:   {np.abs(fixed - corr_matrix).max():.6f}")
        return fixed, False


# ═══════════════════════════════════════════════════════════
#  DATA FETCHING & PROCESSING
# ═══════════════════════════════════════════════════════════

def fetch_prices(tickers, days):
    """Fetch adjusted close prices from Yahoo Finance."""
    end = datetime.now()
    start = end - timedelta(days=days + 30)
    print(f"Fetching {len(tickers)} tickers, {start.date()} → {end.date()}...")

    data = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data
    return prices.tail(days)


def compute_stats(prices, use_shrinkage=True):
    """Compute correlation, vol, prices. Optionally with shrinkage + PD fix."""
    returns = np.log(prices / prices.shift(1)).dropna()

    if len(returns) < MIN_OBS:
        print(f"WARNING: Only {len(returns)} observations (need {MIN_OBS})")

    print(f"Computing from {len(returns)} daily returns "
          f"({returns.index[0].date()} → {returns.index[-1].date()})")

    if use_shrinkage:
        shrunk_cov, shrunk_corr, delta, rbar = ledoit_wolf_shrinkage(returns)

        print(f"\n── Ledoit-Wolf Shrinkage ──")
        print(f"  δ = {delta:.4f} ({delta*100:.1f}% target + {(1-delta)*100:.1f}% sample)")
        print(f"  ρ̄ = {rbar:.4f} (average pairwise correlation)")
        if delta > 0.5:
            print(f"  ⚠ High shrinkage — sample correlations are noisy")

        ann_vol = pd.Series(
            np.sqrt(np.diag(shrunk_cov) * 252) * 100,
            index=returns.columns
        )
        corr = pd.DataFrame(shrunk_corr, index=returns.columns, columns=returns.columns)
    else:
        corr = returns.corr()
        ann_vol = returns.std() * np.sqrt(252) * 100

    # PD check
    corr_vals = corr.values.copy()
    corr_fixed, was_pd = check_and_fix_pd(corr_vals, "shrunk" if use_shrinkage else "sample")
    if not was_pd:
        corr = pd.DataFrame(corr_fixed, index=corr.index, columns=corr.columns)

    latest_px = prices.iloc[-1]
    return corr, ann_vol, latest_px, returns


def build_output(tickers, corr, ann_vol, latest_px, metadata=None):
    """Build JSON for the HTML tool."""
    pairs = {}
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if j <= i:
                continue
            val = corr.loc[t1, t2]
            if not np.isnan(val):
                pairs[f"{t1}-{t2}"] = round(float(val), 4)

    stats = {}
    for t in tickers:
        stats[t] = {
            "vol": round(float(ann_vol.get(t, 0)), 1),
            "px": round(float(latest_px.get(t, 0)), 2),
        }

    output = {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "lookback_days": int(len(corr)),
        "tickers": tickers,
        "correlations": pairs,
        "stats": stats,
    }
    if metadata:
        output["metadata"] = metadata
    return output


def print_summary(tickers, corr, ann_vol):
    """Terminal summary."""
    print("\n── Annualized Volatility ──")
    for t in sorted(tickers, key=lambda x: -ann_vol.get(x, 0)):
        v = ann_vol.get(t, 0)
        print(f"  {t:6s}  {v:5.1f}%  {'█' * int(v / 2)}")

    print("\n── Top 10 Correlations ──")
    all_pairs = []
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if j <= i:
                continue
            val = corr.loc[t1, t2]
            if not np.isnan(val):
                all_pairs.append((t1, t2, val))
    all_pairs.sort(key=lambda x: -abs(x[2]))
    for t1, t2, v in all_pairs[:10]:
        print(f"  {t1:5s} — {t2:5s}  {v:+.3f}")

    print("\n── Lowest 5 Correlations ──")
    for t1, t2, v in all_pairs[-5:]:
        print(f"  {t1:5s} — {t2:5s}  {v:+.3f}")

    print("\n── Flags ──")
    for t1, t2, v in all_pairs:
        if abs(v) > 0.92:
            print(f"  ⚠ {t1}-{t2}: {v:+.3f} — near-duplicate?")
    for t1, t2, v in all_pairs:
        if abs(v) < 0.05 and t1 != "SHY" and t2 != "SHY":
            print(f"  💡 {t1}-{t2}: {v:+.3f} — strong diversifier")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TAA Data Feed v2")
    parser.add_argument("--no-shrinkage", action="store_true",
                        help="Use raw sample covariance (skip Ledoit-Wolf)")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_DAYS,
                        help=f"Lookback days (default: {LOOKBACK_DAYS})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print diagnostics only, don't write file")
    args = parser.parse_args()

    prices = fetch_prices(TICKERS, args.lookback)

    available = [t for t in TICKERS if t in prices.columns]
    missing = [t for t in TICKERS if t not in prices.columns]
    if missing:
        print(f"WARNING: Missing tickers: {missing}")
    if len(available) < 2:
        print("ERROR: Not enough tickers")
        sys.exit(1)

    corr, ann_vol, latest_px, returns = compute_stats(
        prices[available], use_shrinkage=not args.no_shrinkage
    )
    print_summary(available, corr, ann_vol)

    metadata = {
        "shrinkage": not args.no_shrinkage,
        "observations": len(returns),
        "assets": len(available),
    }
    if not args.no_shrinkage:
        _, _, delta, rbar = ledoit_wolf_shrinkage(returns[available])
        metadata["shrinkage_intensity"] = round(float(delta), 4)
        metadata["avg_correlation"] = round(float(rbar), 4)

    output = build_output(available, corr, ann_vol, latest_px, metadata)

    if args.dry_run:
        print("\n── Dry run ──")
        print(json.dumps(output, indent=2)[:500] + "...")
    else:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Written to {OUTPUT_FILE}")
        print(f"  {len(output['correlations'])} pairs · {len(output['stats'])} tickers")
        si = metadata.get('shrinkage_intensity', '?')
        print(f"  Shrinkage: {'ON (δ=' + str(si) + ')' if metadata['shrinkage'] else 'OFF'}")


if __name__ == "__main__":
    main()
