# Paper 2: The Receiver-Limited Floor

**Full title:** "The Receiver-Limited Floor: Rate-Distortion Bounds on Serial Decoding Throughput"

**Status:** Complete (March 28–29, 2025)  
**Principal Investigator:** Grant Lavell Whitmer III  
**Compute:** NVIDIA RTX 5090 (32GB VRAM)

---

## Summary

The M-ary rate-distortion function R_M(ε) = log₂M − H_b(ε) − ε·log₂(M−1) provides a mechanistic floor for serial decoding throughput. This paper tests whether AI throughput is receiver-limited (independent of vocabulary size) using 1,749 translation model variants.

**Key finding:** Bits-per-byte (BPB) shows no correlation with log₂(vocabulary size) across 1,749 models (p = 0.643). Throughput is receiver-limited, not sender-limited.

---

## Experiment: Vocabulary Sweep

- **Models:** 1,749 Windy Pro translation models (OPUS-MT family, 157 language pairs)
- **Vocab sizes:** 30,000 – 250,680 tokens
- **Metric:** Bits-per-byte (BPB) on WikiText-2 test set
- **Result:** Mean BPB = 0.50, independent of vocabulary

### Statistical Summary

```
Linear fit: BPB ~ β₀ + β₁·log₂(V)
β₀ = 2.6440, β₁ = -0.097869
R² = 0.0077, p(β₁) = 0.576

Spearman rank: ρ = 0.0418, p = 0.790

Mean BPT: 4.644 ± 3.12 bits
Median BPT: 3.751 bits
IQR: [2.87, 5.05]
In basin [3, 5.5]: 51.2% of models
```

---

## Files

- `experiment-vocab-sweep/` — Full experimental code
  - `run_full_experiment.py` — Main orchestration
  - `analyze_full.py` — Statistical analysis
  - `results/full_bpb_results.csv` — Raw data (237KB, 1,749 models)
  - `results/statistical_summary.txt` — Key statistics
  - `charts/` — 5 visualization outputs

---

## Citation

```bibtex
@article{whitmer2026receiver,
  title={The Receiver-Limited Floor: Rate-Distortion Bounds on Serial Decoding Throughput},
  author={Whitmer, Grant Lavell III},
  journal={Windstorm Institute},
  year={2026}
}
```

---

*See companion papers for cross-substrate convergence (Paper 3) and thermodynamic grounding (Paper 4).*
