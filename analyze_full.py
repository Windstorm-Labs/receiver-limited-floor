#!/usr/bin/env python3
"""
Phase 3: Statistical analysis of full-scale BPB experiment.
Generates charts and summary statistics from full_bpb_results.csv.
"""

import csv
import json
import math
import os
import sys
import numpy as np
from datetime import datetime

BASE_DIR = os.path.expanduser("~/throughput_experiment")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHARTS_DIR = os.path.join(BASE_DIR, "charts_full")
BPB_PATH = os.path.join(RESULTS_DIR, "full_bpb_results.csv")
PILOT_BPB_PATH = os.path.join(RESULTS_DIR, "bpb_results.csv")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "full_statistical_summary.json")

os.makedirs(CHARTS_DIR, exist_ok=True)


def load_results(path):
    """Load successful results from CSV."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] == "success" and row["BPB"]:
                try:
                    row["BPB"] = float(row["BPB"])
                    row["vocab_size"] = int(row["vocab_size"])
                    row["log2_vocab"] = float(row.get("log2_vocab", 0)) or math.log2(row["vocab_size"])
                    row["bits_per_token"] = float(row.get("bits_per_token", 0))
                    row["avg_bytes_per_token"] = float(row.get("avg_bytes_per_token", 0))
                    row["perplexity"] = float(row.get("perplexity", 0))
                    rows.append(row)
                except (ValueError, KeyError):
                    continue
    return rows


def linear_regression(x, y):
    """Simple OLS regression returning slope, intercept, r_squared, p_value."""
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return {
        "slope": round(slope, 6),
        "intercept": round(intercept, 4),
        "r_squared": round(r_value**2, 6),
        "p_value": round(p_value, 6),
        "std_err": round(std_err, 6),
    }


def spearman_corr(x, y):
    """Spearman rank correlation."""
    from scipy import stats
    rho, p_value = stats.spearmanr(x, y)
    return {"rho": round(rho, 4), "p_value": round(p_value, 6)}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    # Load full results
    rows = load_results(BPB_PATH)
    print(f"Full experiment: {len(rows)} successful evaluations")

    if len(rows) < 10:
        print("Too few results for analysis. Exiting.")
        sys.exit(1)

    # Also load pilot results for comparison
    pilot_rows = []
    if os.path.exists(PILOT_BPB_PATH):
        pilot_rows = load_results(PILOT_BPB_PATH)
        print(f"Pilot experiment: {len(pilot_rows)} successful evaluations")

    # Extract arrays
    vocab_sizes = np.array([r["vocab_size"] for r in rows])
    log2_vocabs = np.array([r["log2_vocab"] for r in rows])
    bpbs = np.array([r["BPB"] for r in rows])
    bits_per_token = np.array([r["bits_per_token"] for r in rows])
    bytes_per_token = np.array([r["avg_bytes_per_token"] for r in rows])

    # Filter outliers for cleaner analysis (BPB > 20 is likely broken)
    mask = bpbs < 20
    vocab_clean = log2_vocabs[mask]
    bpb_clean = bpbs[mask]
    print(f"After outlier filter (BPB<20): {mask.sum()} models")

    # ============================================================
    # STATISTICAL TESTS
    # ============================================================
    summary = {
        "experiment": "full_scale_throughput",
        "date": datetime.now().isoformat(),
        "n_total": len(rows),
        "n_clean": int(mask.sum()),
        "n_pilot": len(pilot_rows),
    }

    # 1. Linear regression: BPB vs log2(vocab)
    reg = linear_regression(vocab_clean, bpb_clean)
    summary["linear_fit"] = reg
    print(f"\nLinear fit: slope={reg['slope']}, R²={reg['r_squared']}, p={reg['p_value']}")

    # 2. Spearman rank correlation
    sp = spearman_corr(vocab_clean, bpb_clean)
    summary["spearman"] = sp
    print(f"Spearman: ρ={sp['rho']}, p={sp['p_value']}")

    # 3. Descriptive stats
    summary["bpb_stats"] = {
        "mean": round(float(bpb_clean.mean()), 4),
        "median": round(float(np.median(bpb_clean)), 4),
        "std": round(float(bpb_clean.std()), 4),
        "min": round(float(bpb_clean.min()), 4),
        "max": round(float(bpb_clean.max()), 4),
        "q25": round(float(np.percentile(bpb_clean, 25)), 4),
        "q75": round(float(np.percentile(bpb_clean, 75)), 4),
    }

    # 4. Vocab size distribution
    summary["vocab_stats"] = {
        "mean": round(float(vocab_sizes.mean()), 0),
        "median": round(float(np.median(vocab_sizes)), 0),
        "min": int(vocab_sizes.min()),
        "max": int(vocab_sizes.max()),
        "unique_sizes": int(len(np.unique(vocab_sizes))),
    }

    # 5. Bits per event (biological channel comparison)
    bits_per_event = bits_per_token[mask]
    in_bio_band = np.sum((bits_per_event >= 3) & (bits_per_event <= 5.5))
    summary["bits_per_event"] = {
        "mean": round(float(bits_per_event.mean()), 4),
        "median": round(float(np.median(bits_per_event)), 4),
        "in_biological_band_3_5_5": int(in_bio_band),
        "percent_in_band": round(100 * in_bio_band / len(bits_per_event), 1),
    }

    # 6. Rate-distortion bound comparison
    R_M = np.log2(vocab_sizes[mask]) / bytes_per_token[mask]
    residuals = bpb_clean - R_M
    summary["rate_distortion"] = {
        "mean_residual": round(float(residuals.mean()), 4),
        "median_residual": round(float(np.median(residuals)), 4),
        "pct_below_bound": round(100 * np.sum(residuals < 0) / len(residuals), 1),
    }

    # 7. Vocab size bins analysis
    bins = [(0, 35000), (35000, 50000), (50000, 65000), (65000, 100000), (100000, 300000)]
    bin_stats = []
    for lo, hi in bins:
        bmask = (vocab_sizes >= lo) & (vocab_sizes < hi) & mask
        if bmask.sum() > 0:
            bin_stats.append({
                "range": f"{lo}-{hi}",
                "count": int(bmask.sum()),
                "bpb_mean": round(float(bpbs[bmask].mean()), 4),
                "bpb_median": round(float(np.median(bpbs[bmask])), 4),
                "bpb_std": round(float(bpbs[bmask].std()), 4),
            })
    summary["vocab_bins"] = bin_stats

    # 8. ANOVA across vocab bins
    bin_groups = []
    for lo, hi in bins:
        bmask = (vocab_sizes >= lo) & (vocab_sizes < hi) & mask
        if bmask.sum() >= 5:
            bin_groups.append(bpbs[bmask])
    if len(bin_groups) >= 2:
        f_stat, anova_p = stats.f_oneway(*bin_groups)
        summary["anova"] = {
            "f_statistic": round(float(f_stat), 4),
            "p_value": round(float(anova_p), 6),
            "n_groups": len(bin_groups),
        }
        print(f"ANOVA: F={f_stat:.4f}, p={anova_p:.6f}")

    # 9. By quality label
    quality_groups = {}
    for r in rows:
        ql = r.get("quality_label", "unknown")
        if ql not in quality_groups:
            quality_groups[ql] = []
        quality_groups[ql].append(r["BPB"])
    summary["by_quality"] = {
        k: {"count": len(v), "mean_bpb": round(np.mean(v), 4)}
        for k, v in quality_groups.items() if len(v) >= 3
    }

    # Save summary
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {SUMMARY_PATH}")

    # ============================================================
    # CHARTS
    # ============================================================
    fig_dpi = 150

    # Chart 1: BPB vs log2(vocab) — scatter with regression line
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(vocab_clean, bpb_clean, alpha=0.3, s=15, c="steelblue", label=f"Full (n={len(vocab_clean)})")
    if pilot_rows:
        pilot_v = [math.log2(r["vocab_size"]) for r in pilot_rows if r["BPB"] < 20]
        pilot_b = [r["BPB"] for r in pilot_rows if r["BPB"] < 20]
        ax.scatter(pilot_v, pilot_b, alpha=0.8, s=40, c="crimson", marker="D", label=f"Pilot (n={len(pilot_v)})")
    # Regression line
    x_fit = np.linspace(vocab_clean.min(), vocab_clean.max(), 100)
    y_fit = reg["slope"] * x_fit + reg["intercept"]
    ax.plot(x_fit, y_fit, "r--", linewidth=2, alpha=0.7,
            label=f"OLS: slope={reg['slope']:.4f}, R²={reg['r_squared']:.4f}, p={reg['p_value']:.3f}")
    ax.set_xlabel("log₂(Vocabulary Size)", fontsize=14)
    ax.set_ylabel("Bits per Byte (BPB)", fontsize=14)
    ax.set_title(f"BPB vs Vocabulary Size — Full Scale (n={len(vocab_clean)})", fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(CHARTS_DIR, "full_bpb_vs_vocab.png"), dpi=fig_dpi, bbox_inches="tight")
    plt.close()
    print("Chart 1: BPB vs vocab")

    # Chart 2: BPB distribution histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(bpb_clean, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(np.median(bpb_clean), color="red", linestyle="--", linewidth=2,
               label=f"Median={np.median(bpb_clean):.2f}")
    ax.axvline(np.mean(bpb_clean), color="orange", linestyle="--", linewidth=2,
               label=f"Mean={np.mean(bpb_clean):.2f}")
    ax.set_xlabel("Bits per Byte (BPB)", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title(f"BPB Distribution (n={len(bpb_clean)})", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(CHARTS_DIR, "full_bpb_distribution.png"), dpi=fig_dpi, bbox_inches="tight")
    plt.close()
    print("Chart 2: BPB distribution")

    # Chart 3: Vocab size bins box plot
    fig, ax = plt.subplots(figsize=(10, 6))
    box_data = []
    box_labels = []
    for lo, hi in bins:
        bmask = (vocab_sizes >= lo) & (vocab_sizes < hi) & mask
        if bmask.sum() > 0:
            box_data.append(bpbs[bmask])
            box_labels.append(f"{lo//1000}K-{hi//1000}K\n(n={bmask.sum()})")
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
    colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xlabel("Vocabulary Size Range", fontsize=14)
    ax.set_ylabel("Bits per Byte (BPB)", fontsize=14)
    ax.set_title("BPB by Vocabulary Size Bin", fontsize=16)
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(os.path.join(CHARTS_DIR, "full_vocab_bins_boxplot.png"), dpi=fig_dpi, bbox_inches="tight")
    plt.close()
    print("Chart 3: Vocab bins boxplot")

    # Chart 4: BPB vs Windy Pro quality stars
    fig, ax = plt.subplots(figsize=(10, 6))
    stars_list = []
    bpb_by_stars = []
    for r in rows:
        try:
            s = float(r.get("stars", 0))
            if s > 0 and r["BPB"] < 20:
                stars_list.append(s)
                bpb_by_stars.append(r["BPB"])
        except (ValueError, TypeError):
            pass
    if stars_list:
        ax.scatter(stars_list, bpb_by_stars, alpha=0.3, s=15, c="steelblue")
        sp_stars = spearman_corr(stars_list, bpb_by_stars)
        ax.set_xlabel("Windy Pro Quality Stars", fontsize=14)
        ax.set_ylabel("Bits per Byte (BPB)", fontsize=14)
        ax.set_title(f"BPB vs Quality Stars (ρ={sp_stars['rho']:.3f}, p={sp_stars['p_value']:.4f})", fontsize=16)
        ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(CHARTS_DIR, "full_bpb_vs_stars.png"), dpi=fig_dpi, bbox_inches="tight")
    plt.close()
    print("Chart 4: BPB vs stars")

    # Chart 5: Cumulative BPB — what % of models fall below each BPB threshold
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_bpb = np.sort(bpb_clean)
    cdf = np.arange(1, len(sorted_bpb) + 1) / len(sorted_bpb)
    ax.plot(sorted_bpb, cdf, linewidth=2, color="steelblue")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(np.median(bpb_clean), color="red", linestyle="--", alpha=0.5,
               label=f"Median={np.median(bpb_clean):.2f}")
    ax.set_xlabel("Bits per Byte (BPB)", fontsize=14)
    ax.set_ylabel("Cumulative Fraction", fontsize=14)
    ax.set_title("CDF of BPB Across All Models", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(CHARTS_DIR, "full_bpb_cdf.png"), dpi=fig_dpi, bbox_inches="tight")
    plt.close()
    print("Chart 5: BPB CDF")

    # Chart 6: Language pair heatmap (top languages)
    from collections import Counter
    lang_bpb = {}
    for r in rows:
        if r["BPB"] < 20:
            sl = r.get("source_lang", "?")
            tl = r.get("target_lang", "?")
            pair = f"{sl}→{tl}"
            lang_bpb[pair] = r["BPB"]

    # Group by source language
    src_stats = {}
    for r in rows:
        if r["BPB"] < 20:
            sl = r.get("source_lang", "?")
            if sl not in src_stats:
                src_stats[sl] = []
            src_stats[sl].append(r["BPB"])

    # Top 20 source languages by count
    top_src = sorted(src_stats.items(), key=lambda x: -len(x[1]))[:20]
    fig, ax = plt.subplots(figsize=(12, 8))
    labels = [f"{lang} (n={len(vals)})" for lang, vals in top_src]
    means = [np.mean(vals) for _, vals in top_src]
    stds = [np.std(vals) for _, vals in top_src]
    y_pos = range(len(labels))
    ax.barh(y_pos, means, xerr=stds, color="steelblue", alpha=0.7, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean BPB", fontsize=14)
    ax.set_title("Mean BPB by Source Language (Top 20)", fontsize=16)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    fig.savefig(os.path.join(CHARTS_DIR, "full_bpb_by_language.png"), dpi=fig_dpi, bbox_inches="tight")
    plt.close()
    print("Chart 6: BPB by language")

    print(f"\nAll charts saved to {CHARTS_DIR}")
    print(f"Summary saved to {SUMMARY_PATH}")
    print("\n" + "="*60)
    print("KEY RESULTS")
    print("="*60)
    print(f"Models evaluated: {len(rows)} ({len(vocab_clean)} after outlier filter)")
    print(f"Vocab range: {int(vocab_sizes.min())} - {int(vocab_sizes.max())}")
    print(f"BPB median: {np.median(bpb_clean):.4f}")
    print(f"Linear fit: slope={reg['slope']}, R²={reg['r_squared']}, p={reg['p_value']}")
    print(f"Spearman: ρ={sp['rho']}, p={sp['p_value']}")
    sig = "SIGNIFICANT" if reg['p_value'] < 0.05 else "NOT significant"
    print(f"Verdict: Vocab size effect is {sig} (α=0.05)")


if __name__ == "__main__":
    main()
