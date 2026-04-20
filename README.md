# Paper 2: The Receiver-Limited Floor — Experiments & Code

**Rate-Distortion Bounds on Serial Decoding Throughput**

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19322973-blue)](https://doi.org/10.5281/zenodo.19322973)
[![License: MIT](https://img.shields.io/badge/Code-MIT-green)](https://opensource.org/licenses/MIT)

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

- **GPU:** NVIDIA RTX 5090 (32 GB VRAM)
- **OS:** Ubuntu 24.04
- **Python:** 3.11+

See individual experiment scripts for runtime estimates and specific dependencies.

---

## The Windstorm Series

| # | Paper | DOI |
|---|-------|-----|
| 1 | [The Fons Constraint](https://github.com/Windstorm-Institute/fons-constraint) | [10.5281/zenodo.19274048](https://doi.org/10.5281/zenodo.19274048) |
| 2 | [The Receiver-Limited Floor](https://github.com/Windstorm-Institute/receiver-limited-floor) | [10.5281/zenodo.19322973](https://doi.org/10.5281/zenodo.19322973) |
| 3 | [The Throughput Basin](https://github.com/Windstorm-Institute/throughput-basin) | [10.5281/zenodo.19323194](https://doi.org/10.5281/zenodo.19323194) |
| 4 | [The Serial Decoding Basin τ](https://github.com/Windstorm-Institute/serial-decoding-basin) | [10.5281/zenodo.19323423](https://doi.org/10.5281/zenodo.19323423) |
| 5 | [The Dissipative Decoder](https://github.com/Windstorm-Institute/dissipative-decoder) | [10.5281/zenodo.19433048](https://doi.org/10.5281/zenodo.19433048) |
| 6 | [The Inherited Constraint](https://github.com/Windstorm-Institute/inherited-constraint) | [10.5281/zenodo.19432911](https://doi.org/10.5281/zenodo.19432911) |
| 7 | [The Throughput Basin Origin](https://github.com/Windstorm-Institute/throughput-basin-origin) | [10.5281/zenodo.19498582](https://doi.org/10.5281/zenodo.19498582) |
| 8 | [The Vision Basin](https://github.com/Windstorm-Institute/vision-basin) | [10.5281/zenodo.19672827](https://doi.org/10.5281/zenodo.19672827) |
| 9 | [The Hardware Basin](https://github.com/Windstorm-Institute/hardware-basin) | Preprint (DOI pending) |

**Website:** [windstorminstitute.org](https://windstorminstitute.org)

---

*Code: MIT License · Data: CC BY 4.0*
