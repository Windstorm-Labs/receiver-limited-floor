# Paper 2: The Receiver-Limited Floor — Experiments & Code

**Rate-Distortion Bounds on Serial Decoding Throughput**

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19322973-blue)](https://doi.org/10.5281/zenodo.19322973)
[![License: MIT](https://img.shields.io/badge/Code-MIT-green)](https://opensource.org/licenses/MIT)
[![Track: Throughput Basin](https://img.shields.io/badge/Track-1_·_Throughput_Basin-3b82f6)](https://windstorminstitute.org/#track1)

---

## Published Paper

**[Windstorm-Institute/receiver-limited-floor](https://github.com/Windstorm-Institute/receiver-limited-floor)** — paper PDF, article HTML, Zenodo DOI

**Website article:** [windstorminstitute.org/articles/receiver-limited-floor.html](https://windstorminstitute.org/articles/receiver-limited-floor.html)

## Quick Start

```bash
git clone https://github.com/Windstorm-Labs/receiver-limited-floor.git
cd receiver-limited-floor
# Install deps (no requirements.txt yet — see Windstorm-Labs/throughput-experiments/requirements.txt
# for the closest-known dependency pin: torch, transformers, datasets, accelerate, bitsandbytes,
# numpy, scipy, pandas, scikit-learn, tqdm, tokenizers).

# Canonical entry point: run_full_experiment.py
#   - Subprocess isolation per model (CUDA recovery on crash)
#   - Disk-managed model cache cleanup
#   - Resumable across interruptions
python run_full_experiment.py
```

**Three `run_*.py` scripts are present in this repo** (legacy from iterative development):
- `run_full_experiment.py` — **canonical**: subprocess isolation, disk management, resumable
- `run_experiment_v2.py` — earlier robust version with CUDA recovery
- `run_experiment.py` — original 5-step pipeline

Use `run_full_experiment.py` unless you have a specific reason to use one of the older scripts.

## Hardware

- **GPU:** Current-generation Nvidia GPU (32 GB VRAM, CUDA)
- **OS:** Ubuntu 24.04
- **Python:** 3.11+

See individual experiment scripts for runtime estimates and specific dependencies.

---

## Discuss this code

- **Bug, reproduction failure, or unexpected output?** → [Open an Issue](../../issues)
- **Q&A — version compatibility, hardware, generalization to other inputs?** → [Start a Discussion](../../discussions)
- **Discuss the paper itself** → [Comments on the website article](https://windstorminstitute.org/articles/receiver-limited-floor.html#comments) or [Issues on the Institute repo](https://github.com/Windstorm-Institute/receiver-limited-floor/issues)

---

---

## The Windstorm Institute — Two Research Tracks

### Track 1 — The Throughput Basin · 9 papers (Papers 1–9 globally; 1st through 9th in this track; arc complete)

| # | Paper | DOI |
|---|-------|-----|
| 1 | [The Fons Constraint](https://github.com/Windstorm-Institute/fons-constraint) | [10.5281/zenodo.19274048](https://doi.org/10.5281/zenodo.19274048) |
| 2 | [The Receiver-Limited Floor](https://github.com/Windstorm-Institute/receiver-limited-floor) *(this paper)* | [10.5281/zenodo.19322973](https://doi.org/10.5281/zenodo.19322973) |
| 3 | [The Throughput Basin](https://github.com/Windstorm-Institute/throughput-basin) | [10.5281/zenodo.19323194](https://doi.org/10.5281/zenodo.19323194) |
| 4 | [The Serial Decoding Basin τ](https://github.com/Windstorm-Institute/serial-decoding-basin) | [10.5281/zenodo.19323423](https://doi.org/10.5281/zenodo.19323423) |
| 5 | [The Dissipative Decoder](https://github.com/Windstorm-Institute/dissipative-decoder) | [10.5281/zenodo.19433048](https://doi.org/10.5281/zenodo.19433048) |
| 6 | [The Inherited Constraint](https://github.com/Windstorm-Institute/inherited-constraint) | [10.5281/zenodo.19432911](https://doi.org/10.5281/zenodo.19432911) |
| 7 | [The Throughput Basin Origin](https://github.com/Windstorm-Institute/throughput-basin-origin) | [10.5281/zenodo.19498582](https://doi.org/10.5281/zenodo.19498582) |
| 8 | [The Vision Basin](https://github.com/Windstorm-Institute/vision-basin) | [10.5281/zenodo.19672827](https://doi.org/10.5281/zenodo.19672827) |
| 9 | [The Hardware Basin](https://github.com/Windstorm-Institute/hardware-basin) | [10.5281/zenodo.19672921](https://doi.org/10.5281/zenodo.19672921) |

### Track 2 — Entropic Bounds in Analog Systems · 7 papers (Papers 10–16 globally; 1st through 4th in this track; line of inquiry active)

| # | Paper | DOI |
|---|-------|-----|
| 10 | [Phonon Extraction Bound (BEC Analog Gravity)](https://github.com/Windstorm-Institute/phonon-extraction-bound) | [10.5281/zenodo.20014391](https://doi.org/10.5281/zenodo.20014391) |
| 11 | [Gravitational Entropy Escrow](https://github.com/Windstorm-Institute/gravitational-entropy-escrow) | [10.5281/zenodo.20032023](https://doi.org/10.5281/zenodo.20032023) |
| 12 | [C8 Clarification Note](https://github.com/Windstorm-Institute/c8-clarification-note) | [10.5281/zenodo.20041992](https://doi.org/10.5281/zenodo.20041992) |

| 13 | [Lattice QFT Test of the Static Escrow Postulate](https://github.com/Windstorm-Institute/lattice-qft-test) *(4th in track; supplement to Paper 11)* | [10.5281/zenodo.20057538](https://doi.org/10.5281/zenodo.20057538) |
| 14 | [Spacetime as Escrow Bookkeeping](https://github.com/Windstorm-Institute/escrow-spacetime) *(5th in track; translation of standard GR results into the escrow vocabulary; companion to Paper 11)* | [10.5281/zenodo.20126091](https://doi.org/10.5281/zenodo.20126091) |
| 15 | [The 𝒩<sub>esc</sub> Recipe](https://github.com/Windstorm-Institute/nesc-recipe) *(6th in track; formalizes 𝒩<sub>esc</sub> as a cross-regime function; continuation of Paper 14)* | [10.5281/zenodo.20145106](https://doi.org/10.5281/zenodo.20145106) |
**Website:** [windstorminstitute.org](https://windstorminstitute.org)

---

*Code: MIT License · Data: CC BY 4.0*
