#!/bin/bash
# Full-scale throughput experiment launcher
# Run with: nohup bash run_full.sh &> results/full_experiment.log &
#
# Monitor with:
#   tail -f ~/throughput_experiment/results/full_experiment.log
#   cat ~/throughput_experiment/results/progress.json | python3 -m json.tool

set -euo pipefail
cd ~/throughput_experiment
source /home/hermes-oc1/.hermes/hermes-agent/venv/bin/activate

echo "============================================="
echo "FULL-SCALE THROUGHPUT EXPERIMENT"
echo "Started: $(date)"
echo "============================================="
echo ""

# Phase 1 + 2: Catalog + Evaluation
python3 run_full_experiment.py --min-disk-gb 5

echo ""
echo "============================================="
echo "PHASE 3: Analysis"
echo "============================================="

# Phase 3: Analysis
python3 analyze_full.py

echo ""
echo "============================================="
echo "COMPLETE: $(date)"
echo "============================================="
