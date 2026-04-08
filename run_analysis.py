#!/usr/bin/env python3
"""
Steps 4-5: Statistical analysis and visualization.
"""

import csv
import math
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import ruptures
import plotly.graph_objects as go
import statsmodels.api as sm

BASE_DIR = os.path.expanduser("~/throughput_experiment")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")

##############################################################################
# LOAD DATA
##############################################################################
df = pd.read_csv(os.path.join(RESULTS_DIR, "bpb_results.csv"))
df_ok = df[df["status"] == "success"].copy()
for col in ["BPB", "log2_vocab", "bits_per_token", "avg_bytes_per_token", "perplexity", "vocab_size"]:
    df_ok[col] = pd.to_numeric(df_ok[col], errors="coerce")
df_ok = df_ok.dropna(subset=["BPB", "log2_vocab"])

# Filter clearly nonsensical (BPB > 10 not meaningful for English)
df_ok = df_ok[df_ok["BPB"] < 10].copy()

print(f"Models: {len(df_ok)}")
print(f"Vocab range: {df_ok['vocab_size'].min():.0f} - {df_ok['vocab_size'].max():.0f}")
print(f"log₂(V) range: {df_ok['log2_vocab'].min():.1f} - {df_ok['log2_vocab'].max():.1f}")
print(f"BPB range: {df_ok['BPB'].min():.4f} - {df_ok['BPB'].max():.4f}")
print()

# Print full table
print("MODEL RESULTS (sorted by vocab size):")
print("-" * 100)
for _, row in df_ok.sort_values("vocab_size").iterrows():
    print(f"  {row['model_name']:45s} V={row['vocab_size']:>8.0f}  BPB={row['BPB']:.4f}  "
          f"bits/tok={row['bits_per_token']:.2f}  bytes/tok={row['avg_bytes_per_token']:.2f}  "
          f"PPL={row['perplexity']:.2f}")
print()

##############################################################################
# STEP 4: STATISTICAL ANALYSIS
##############################################################################
print("=" * 70)
print("STATISTICAL ANALYSIS")
print("=" * 70)

# A. LINEAR FIT TEST
print("\n--- A. LINEAR FIT: BPB ~ β₀ + β₁·log₂(V) ---")
slope, intercept, r_value, p_value, std_err = stats.linregress(df_ok["log2_vocab"], df_ok["BPB"])
r_squared = r_value ** 2
print(f"  β₀ (intercept) = {intercept:.4f}")
print(f"  β₁ (slope)     = {slope:.6f}")
print(f"  R²             = {r_squared:.4f}")
print(f"  p(β₁)          = {p_value:.6f}")
print(f"  std_err(β₁)    = {std_err:.6f}")

if p_value < 0.05 and slope > 0:
    print(f"  → SIGNIFICANT positive slope: larger vocabs = higher BPB")
    print(f"    This supports linear scaling (falsifies plateau)")
elif p_value < 0.05 and slope < 0:
    print(f"  → SIGNIFICANT negative slope: larger vocabs = lower BPB")
elif p_value >= 0.05:
    print(f"  → NOT significant: BPB does NOT depend on vocab size")
    print(f"    This supports the plateau hypothesis")

# Also test within just causal LMs (apples to apples)
df_causal = df_ok[df_ok["model_type"] == "LLM"].copy()
if len(df_causal) >= 5:
    s2, i2, r2, p2, se2 = stats.linregress(df_causal["log2_vocab"], df_causal["BPB"])
    print(f"\n  Causal LMs only ({len(df_causal)} models):")
    print(f"  β₁ = {s2:.6f}, R² = {r2**2:.4f}, p = {p2:.6f}")

# B. SPEARMAN RANK CORRELATION (more robust to outliers)
print("\n--- B. SPEARMAN RANK CORRELATION ---")
rho, sp_pval = stats.spearmanr(df_ok["log2_vocab"], df_ok["BPB"])
print(f"  ρ = {rho:.4f}, p = {sp_pval:.6f}")

# C. PLATEAU DETECTION
print("\n--- C. PLATEAU / CHANGEPOINT DETECTION ---")
v_plateau = "N/A"
try:
    df_sorted = df_ok.sort_values("log2_vocab").reset_index(drop=True)
    signal = df_sorted["BPB"].values
    
    if len(signal) >= 5:
        algo = ruptures.Pelt(model="rbf", min_size=3).fit(signal.reshape(-1, 1))
        breakpoints = algo.predict(pen=1)
        
        valid_bp = [b for b in breakpoints if b < len(signal)]
        if valid_bp:
            bp_idx = valid_bp[0]
            v_plateau = df_sorted.iloc[bp_idx]["vocab_size"]
            log2_p = df_sorted.iloc[bp_idx]["log2_vocab"]
            
            var_before = np.var(signal[:bp_idx]) if bp_idx > 1 else float('nan')
            var_after = np.var(signal[bp_idx:]) if bp_idx < len(signal) - 1 else float('nan')
            mean_before = np.mean(signal[:bp_idx]) if bp_idx > 1 else float('nan')
            mean_after = np.mean(signal[bp_idx:]) if bp_idx < len(signal) - 1 else float('nan')
            
            print(f"  Changepoint at index {bp_idx}")
            print(f"  V_plateau ≈ {v_plateau:.0f} (log₂ ≈ {log2_p:.1f})")
            print(f"  Mean BPB before: {mean_before:.4f} (var={var_before:.6f})")
            print(f"  Mean BPB after:  {mean_after:.4f} (var={var_after:.6f})")
        else:
            print("  No changepoint detected")
except Exception as e:
    print(f"  Detection failed: {e}")

# D. BITS/EVENT
print("\n--- D. BITS PER EVENT ---")
df_ok["bits_per_event"] = df_ok["BPB"] * df_ok["avg_bytes_per_token"]
bpe_mean = df_ok["bits_per_event"].mean()
bpe_median = df_ok["bits_per_event"].median()
bpe_q25 = df_ok["bits_per_event"].quantile(0.25)
bpe_q75 = df_ok["bits_per_event"].quantile(0.75)
bpe_std = df_ok["bits_per_event"].std()
in_band = df_ok[(df_ok["bits_per_event"] >= 3) & (df_ok["bits_per_event"] <= 5.5)]
print(f"  Mean:    {bpe_mean:.4f}")
print(f"  Median:  {bpe_median:.4f}")
print(f"  Std:     {bpe_std:.4f}")
print(f"  IQR:     [{bpe_q25:.4f}, {bpe_q75:.4f}]")
print(f"  In [3, 5.5] band: {len(in_band)}/{len(df_ok)} ({100*len(in_band)/len(df_ok):.1f}%)")

# E. RATE-DISTORTION
print("\n--- E. RATE-DISTORTION BOUND ---")
def h_binary(eps):
    if eps <= 0 or eps >= 1:
        return 0
    return -eps * math.log2(eps) - (1 - eps) * math.log2(1 - eps)

df_ok["epsilon"] = (1 - (1 / df_ok["perplexity"])).clip(0.001, 0.999)
residuals = []
for idx, row in df_ok.iterrows():
    eps = row["epsilon"]
    V = row["vocab_size"]
    H_b = h_binary(eps)
    R_M = math.log2(V) - H_b - eps * math.log2(max(V - 1, 1))
    R_M = max(R_M, 0)
    df_ok.at[idx, "R_M"] = R_M
    df_ok.at[idx, "rd_residual"] = row["BPB"] - R_M
    residuals.append(row["BPB"] - R_M)

print(f"  Mean residual (BPB - R_M): {np.mean(residuals):.4f}")
print(f"  Std residual:              {np.std(residuals):.4f}")
print(f"  Median residual:           {np.median(residuals):.4f}")

# F. SCALING WITH MODEL SIZE (within same tokenizer)
print("\n--- F. SCALING WITH MODEL SIZE (fixed tokenizer families) ---")
families = {
    "GPT-2 (50257)": df_ok[df_ok["model_name"].str.contains("gpt2|distilgpt2", case=False)],
    "Pythia (50254)": df_ok[df_ok["model_name"].str.contains("pythia", case=False)],
    "OPT (50265)": df_ok[df_ok["model_name"].str.contains("opt-", case=False)],
    "Qwen (151643)": df_ok[df_ok["model_name"].str.contains("Qwen", case=False)],
    "BLOOM (250680)": df_ok[df_ok["model_name"].str.contains("bloom", case=False)],
}
for name, fam in families.items():
    if len(fam) >= 2:
        bpbs = fam["BPB"].values
        print(f"  {name}: n={len(fam)}, BPB range [{min(bpbs):.4f}, {max(bpbs):.4f}], "
              f"mean={np.mean(bpbs):.4f}")

# Save summary
summary_path = os.path.join(RESULTS_DIR, "statistical_summary.txt")
mean_bpb = df_ok["BPB"].mean()
with open(summary_path, "w") as f:
    f.write("THROUGHPUT CONSTRAINT EXPERIMENT — STATISTICAL SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Date: 2025-03-28\n")
    f.write(f"Corpus: WikiText-2 test set\n")
    f.write(f"Models analyzed: {len(df_ok)}\n")
    f.write(f"Vocab range: {df_ok['vocab_size'].min():.0f} - {df_ok['vocab_size'].max():.0f}\n")
    f.write(f"log₂(V) range: {df_ok['log2_vocab'].min():.1f} - {df_ok['log2_vocab'].max():.1f}\n\n")
    
    f.write("A. LINEAR FIT: BPB ~ β₀ + β₁·log₂(V)\n")
    f.write(f"   β₀ = {intercept:.4f}, β₁ = {slope:.6f}\n")
    f.write(f"   R² = {r_squared:.4f}, p(β₁) = {p_value:.6f}\n")
    if len(df_causal) >= 5:
        f.write(f"   Causal LMs only: β₁={s2:.6f}, R²={r2**2:.4f}, p={p2:.6f}\n")
    f.write(f"\nB. SPEARMAN RANK\n")
    f.write(f"   ρ = {rho:.4f}, p = {sp_pval:.6f}\n")
    f.write(f"\nC. PLATEAU\n")
    f.write(f"   V_plateau ≈ {v_plateau}\n")
    f.write(f"\nD. BITS/EVENT\n")
    f.write(f"   Mean={bpe_mean:.4f}, Median={bpe_median:.4f}, Std={bpe_std:.4f}\n")
    f.write(f"   IQR=[{bpe_q25:.4f}, {bpe_q75:.4f}]\n")
    f.write(f"   In [3, 5.5]: {len(in_band)}/{len(df_ok)} ({100*len(in_band)/len(df_ok):.1f}%)\n")
    f.write(f"\nE. RATE-DISTORTION\n")
    f.write(f"   Mean residual: {np.mean(residuals):.4f}\n")
    f.write(f"   Std residual: {np.std(residuals):.4f}\n")

df_ok.to_csv(os.path.join(RESULTS_DIR, "bpb_results_augmented.csv"), index=False)
print(f"\nSaved: {summary_path}")
print(f"Saved: bpb_results_augmented.csv")

##############################################################################
# STEP 5: VISUALIZATIONS
##############################################################################
print("\n" + "=" * 70)
print("GENERATING CHARTS")
print("=" * 70)

# Tick labels for x-axis
all_log2 = df_ok["log2_vocab"].values
tick_min = int(np.floor(all_log2.min()))
tick_max = int(np.ceil(all_log2.max()))
tick_vals = list(range(tick_min, tick_max + 1))
tick_texts = [f"{2**v:,.0f}" for v in tick_vals]

# ── CHART 1: BPB vs log₂(Vocab Size) ──
print("  Chart 1: BPB vs log₂(Vocab Size)...")
fig1 = go.Figure()

colors_map = {"LLM": "#1f77b4", "translation": "#ff7f0e"}
for mt in df_ok["model_type"].unique():
    sub = df_ok[df_ok["model_type"] == mt]
    fig1.add_trace(go.Scatter(
        x=sub["log2_vocab"], y=sub["BPB"], mode="markers", name=mt,
        marker=dict(size=10, color=colors_map.get(mt, "#888"), opacity=0.8),
        text=sub["model_name"],
        hovertemplate="%{text}<br>log₂(V)=%{x:.1f}<br>BPB=%{y:.4f}<extra></extra>"
    ))

# Linear fit
x_range = np.linspace(all_log2.min() - 0.5, all_log2.max() + 0.5, 100)
fig1.add_trace(go.Scatter(
    x=x_range, y=intercept + slope * x_range, mode="lines",
    name=f"OLS: BPB={intercept:.2f}+{slope:.4f}·log₂(V) (R²={r_squared:.3f}, p={p_value:.4f})",
    line=dict(color="red", dash="dash", width=2)
))

# LOWESS
try:
    lowess = sm.nonparametric.lowess(df_ok["BPB"].values, df_ok["log2_vocab"].values, frac=0.4)
    fig1.add_trace(go.Scatter(
        x=lowess[:, 0], y=lowess[:, 1], mode="lines", name="LOWESS (frac=0.4)",
        line=dict(color="green", width=2)
    ))
except: pass

fig1.add_hline(y=mean_bpb, line_dash="dot", line_color="gray",
               annotation_text=f"Mean BPB = {mean_bpb:.3f}")

fig1.update_layout(
    title=dict(text="BPB vs Vocabulary Size: Testing the Throughput Constraint", font=dict(size=18)),
    xaxis_title="Vocabulary Size", yaxis_title="Bits Per Byte (BPB)",
    width=1400, height=800, template="plotly_white", font=dict(size=14),
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)")
)
fig1.update_xaxes(tickvals=tick_vals, ticktext=tick_texts)
fig1.write_image(os.path.join(CHARTS_DIR, "chart1_bpb_vs_vocab.png"), scale=2)
print("    Done")

# ── CHART 2: Bits per Event Distribution ──
print("  Chart 2: Bits per Event Distribution...")
fig2 = go.Figure()
fig2.add_trace(go.Histogram(
    x=df_ok["bits_per_event"], nbinsx=30,
    marker_color="#1f77b4", opacity=0.7, name="Observed"
))
fig2.add_vrect(x0=3, x1=5.5, fillcolor="green", opacity=0.1,
               annotation_text="Theoretical band [3, 5.5]", annotation_position="top left")
fig2.add_vline(x=4.39, line_dash="dash", line_color="red",
               annotation_text="Ribosome: 4.39 bits/event")
fig2.add_vline(x=bpe_median, line_dash="dot", line_color="purple",
               annotation_text=f"Median: {bpe_median:.2f}")
fig2.update_layout(
    title=dict(text="Bits per Event Distribution: Convergence to Biological Band?", font=dict(size=18)),
    xaxis_title="Bits per Event (BPB × bytes/token)",
    yaxis_title="Count", width=1100, height=650, template="plotly_white", font=dict(size=14)
)
fig2.write_image(os.path.join(CHARTS_DIR, "chart2_bits_per_event.png"), scale=2)
print("    Done")

# ── CHART 3: Bytes/Token vs Vocab Size ──
print("  Chart 3: Bytes/Token vs Vocab Size...")
fig3 = go.Figure()

for mt in df_ok["model_type"].unique():
    sub = df_ok[df_ok["model_type"] == mt]
    fig3.add_trace(go.Scatter(
        x=sub["log2_vocab"], y=sub["avg_bytes_per_token"],
        mode="markers", name=mt,
        marker=dict(size=10, color=colors_map.get(mt, "#888"), opacity=0.8),
        text=sub["model_name"],
        hovertemplate="%{text}<br>log₂(V)=%{x:.1f}<br>bytes/tok=%{y:.2f}<extra></extra>"
    ))

try:
    s_bt, i_bt, _, p_bt, _ = stats.linregress(df_ok["log2_vocab"], df_ok["avg_bytes_per_token"])
    fig3.add_trace(go.Scatter(
        x=x_range, y=i_bt + s_bt * x_range, mode="lines",
        name=f"OLS: slope={s_bt:.3f}, p={p_bt:.4f}",
        line=dict(color="red", dash="dash", width=2)
    ))
except: pass

fig3.update_layout(
    title=dict(text="Tokenization Efficiency: Bytes per Token vs Vocabulary Size", font=dict(size=18)),
    xaxis_title="Vocabulary Size", yaxis_title="Average Bytes per Token",
    width=1100, height=650, template="plotly_white", font=dict(size=14),
    legend=dict(x=0.02, y=0.98)
)
fig3.update_xaxes(tickvals=tick_vals, ticktext=tick_texts)
fig3.write_image(os.path.join(CHARTS_DIR, "chart3_bytes_per_token.png"), scale=2)
print("    Done")

# ── CHART 4: Observed BPB vs Rate-Distortion ──
print("  Chart 4: BPB vs R_M(ε)...")
fig4 = go.Figure()
df_rd = df_ok.dropna(subset=["R_M"])
fig4.add_trace(go.Scatter(
    x=df_rd["R_M"], y=df_rd["BPB"], mode="markers",
    marker=dict(size=10, color=df_rd["log2_vocab"], colorscale="Viridis",
                colorbar=dict(title="log₂(V)"), showscale=True),
    text=df_rd["model_name"],
    hovertemplate="%{text}<br>R_M(ε)=%{x:.3f}<br>BPB=%{y:.4f}<extra></extra>",
    name="Models"
))
rd_max = max(df_rd["R_M"].max(), df_rd["BPB"].max()) * 1.1
fig4.add_trace(go.Scatter(
    x=[0, rd_max], y=[0, rd_max], mode="lines", name="Perfect prediction (y=x)",
    line=dict(color="red", dash="dash", width=2)
))
fig4.update_layout(
    title=dict(text="Observed BPB vs Rate-Distortion Bound R_M(ε)", font=dict(size=18)),
    xaxis_title="R_M(ε) — Rate-Distortion Bound",
    yaxis_title="Observed BPB", width=1100, height=750,
    template="plotly_white", font=dict(size=14)
)
fig4.write_image(os.path.join(CHARTS_DIR, "chart4_rd_vs_observed.png"), scale=2)
print("    Done")

# ── CHART 5 (BONUS): Model family scaling ──
print("  Chart 5: BPB by Model Family...")
fig5 = go.Figure()
family_colors = {"GPT-2": "#e41a1c", "Pythia": "#377eb8", "OPT": "#4daf4a",
                 "Qwen": "#984ea3", "BLOOM": "#ff7f00", "Other": "#999999"}

for _, row in df_ok.iterrows():
    name = row["model_name"]
    if "gpt2" in name.lower() or "distilgpt2" in name.lower():
        row["family"] = "GPT-2"
    elif "pythia" in name.lower():
        row["family"] = "Pythia"
    elif "opt-" in name.lower():
        row["family"] = "OPT"
    elif "qwen" in name.lower():
        row["family"] = "Qwen"
    elif "bloom" in name.lower():
        row["family"] = "BLOOM"
    else:
        row["family"] = "Other"
    df_ok.at[row.name, "family"] = row["family"]

for fam in ["GPT-2", "Pythia", "OPT", "Qwen", "BLOOM", "Other"]:
    sub = df_ok[df_ok["family"] == fam]
    if len(sub) > 0:
        fig5.add_trace(go.Scatter(
            x=sub["log2_vocab"], y=sub["BPB"], mode="markers+text" if fam != "Other" else "markers",
            name=fam,
            marker=dict(size=12, color=family_colors.get(fam, "#999"), opacity=0.8),
            text=[n.split("/")[-1] for n in sub["model_name"]] if fam != "Other" else None,
            textposition="top center", textfont=dict(size=9),
        ))

fig5.update_layout(
    title=dict(text="BPB by Model Family: Fixed Tokenizer, Varying Scale", font=dict(size=18)),
    xaxis_title="Vocabulary Size", yaxis_title="BPB",
    width=1400, height=800, template="plotly_white", font=dict(size=14)
)
fig5.update_xaxes(tickvals=tick_vals, ticktext=tick_texts)
fig5.write_image(os.path.join(CHARTS_DIR, "chart5_model_families.png"), scale=2)
print("    Done")

print(f"\nAll charts saved to {CHARTS_DIR}/")
print("\n" + "=" * 70)
print("EXPERIMENT PIPELINE COMPLETE")
print("=" * 70)
