#!/usr/bin/env python3
"""
Full-scale throughput experiment: evaluate 1,826 models from Windy Pro catalog.

Features:
- Resumable: skips already-completed models
- Disk management: cleans up model cache after each eval to stay within disk budget
- Subprocess isolation: each model runs in its own process
- Progress logging: writes status to a log file for monitoring

Usage:
    python run_full_experiment.py [--skip-catalog] [--max-models N] [--cleanup-aggressive]
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta

BASE_DIR = os.path.expanduser("~/throughput_experiment")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CATALOG_PATH = os.path.join(RESULTS_DIR, "full_catalog.csv")
BPB_RESULTS_PATH = os.path.join(RESULTS_DIR, "full_bpb_results.csv")
LOG_PATH = os.path.join(RESULTS_DIR, "experiment_log.txt")
PROGRESS_PATH = os.path.join(RESULTS_DIR, "progress.json")

BPB_FIELDS = [
    "model_name", "model_id", "vocab_size", "log2_vocab", "BPB",
    "bits_per_token", "avg_bytes_per_token", "perplexity",
    "model_type", "source_lang", "target_lang", "stars",
    "quality_label", "status", "error_msg", "eval_time_sec"
]

HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub/")


def log(msg: str):
    """Write to both stdout and log file."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def get_disk_free_gb() -> float:
    """Get free disk space in GB."""
    stat = os.statvfs("/home")
    return (stat.f_bavail * stat.f_frsize) / (1024**3)


def cleanup_model_cache(repo_id: str):
    """Remove cached model weights to free disk. Keep tokenizer files."""
    # HF cache uses -- as separator: models--Helsinki-NLP--opus-mt-en-de
    cache_name = f"models--{repo_id.replace('/', '--')}"
    cache_path = os.path.join(HF_CACHE, cache_name)
    if os.path.exists(cache_path):
        size_mb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, dn, fnames in os.walk(cache_path)
            for f in fnames
        ) / (1024 * 1024)
        shutil.rmtree(cache_path, ignore_errors=True)
        return size_mb
    return 0


def load_existing_results() -> dict:
    """Load already-completed evaluations."""
    existing = {}
    if os.path.exists(BPB_RESULTS_PATH):
        with open(BPB_RESULTS_PATH) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row["model_id"]] = row
    return existing


def save_results(results: list):
    """Save all results to CSV."""
    with open(BPB_RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BPB_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: int(r.get("vocab_size", 0))))


def save_progress(done: int, total: int, successes: int, failures: int,
                  start_time: float, estimated_remaining: str):
    """Save progress JSON for external monitoring."""
    with open(PROGRESS_PATH, "w") as f:
        json.dump({
            "done": done,
            "total": total,
            "successes": successes,
            "failures": failures,
            "percent": round(100 * done / max(total, 1), 1),
            "elapsed_minutes": round((time.time() - start_time) / 60, 1),
            "estimated_remaining": estimated_remaining,
            "last_update": datetime.now().isoformat(),
        }, f, indent=2)


def eval_model(repo_id: str, vocab_size: int, timeout: int = 300) -> dict:
    """Run evaluation in subprocess. Returns result dict."""
    python = sys.executable
    script = os.path.join(BASE_DIR, "eval_marian.py")

    try:
        proc = subprocess.run(
            [python, script, repo_id, str(vocab_size)],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
                 "TOKENIZERS_PARALLELISM": "false"}
        )

        if proc.returncode != 0:
            stderr = proc.stderr.strip()[-300:]
            return {"status": "error", "error_msg": f"exit={proc.returncode}: {stderr}"}

        # Parse JSON from stdout (last line)
        for line in reversed(proc.stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)

        return {"status": "error", "error_msg": "No JSON output"}

    except subprocess.TimeoutExpired:
        return {"status": "error", "error_msg": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"status": "error", "error_msg": str(e)[:300]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-catalog", action="store_true",
                        help="Skip catalog building, use existing catalog")
    parser.add_argument("--max-models", type=int, default=0,
                        help="Limit number of models to evaluate (0 = all)")
    parser.add_argument("--cleanup-aggressive", action="store_true",
                        help="Clean up ALL model caches after eval, not just new ones")
    parser.add_argument("--min-disk-gb", type=float, default=5.0,
                        help="Minimum free disk GB before aggressive cleanup")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Phase 1: Build catalog if needed
    if not args.skip_catalog or not os.path.exists(CATALOG_PATH):
        log("=== PHASE 1: Building model catalog ===")
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "build_catalog.py")],
                       check=True)
    else:
        log("=== PHASE 1: Skipping catalog (--skip-catalog) ===")

    # Load catalog
    catalog = []
    with open(CATALOG_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row.get("vocab_size", 0)) > 0:
                catalog.append(row)
    log(f"Catalog: {len(catalog)} models with valid vocab sizes")

    if args.max_models > 0:
        catalog = catalog[:args.max_models]
        log(f"Limited to {args.max_models} models")

    # Load existing results
    existing = load_existing_results()
    log(f"Already evaluated: {len(existing)} models")

    # Merge existing into results list
    all_results = list(existing.values())

    # Determine what needs evaluation
    to_eval = [(row["model_id"], row) for row in catalog
               if row["model_id"] not in existing]
    log(f"Need to evaluate: {len(to_eval)} models")

    if not to_eval:
        log("Nothing to do!")
        return

    # Phase 2: Evaluate
    log(f"\n=== PHASE 2: Evaluating {len(to_eval)} models ===")
    log(f"Free disk: {get_disk_free_gb():.1f} GB")

    start_time = time.time()
    successes = sum(1 for r in all_results if r.get("status") == "success")
    failures = sum(1 for r in all_results if r.get("status") != "success")
    disk_freed_total = 0

    for i, (model_id, cat_row) in enumerate(to_eval):
        repo = cat_row["source_repo"]
        vocab_size = int(cat_row["vocab_size"])
        log2_vocab = round(math.log2(vocab_size), 3) if vocab_size > 0 else 0

        # Progress estimate
        elapsed = time.time() - start_time
        if i > 0:
            avg_per_model = elapsed / i
            remaining = avg_per_model * (len(to_eval) - i)
            eta = str(timedelta(seconds=int(remaining)))
        else:
            eta = "calculating..."

        log(f"[{i+1}/{len(to_eval)}] {repo} (V={vocab_size}) — ETA: {eta}")

        # Check disk space
        free_gb = get_disk_free_gb()
        if free_gb < args.min_disk_gb:
            log(f"  ⚠ Low disk: {free_gb:.1f}GB — running aggressive cleanup")
            # Clean up models not in our current batch
            cleaned = 0
            if os.path.exists(HF_CACHE):
                for d in os.listdir(HF_CACHE):
                    if d.startswith("models--") and os.path.isdir(os.path.join(HF_CACHE, d)):
                        dp = os.path.join(HF_CACHE, d)
                        try:
                            shutil.rmtree(dp)
                            cleaned += 1
                        except Exception:
                            pass
            log(f"  Cleaned {cleaned} cached models")

        # Evaluate
        t0 = time.time()
        result = eval_model(repo, vocab_size, timeout=300)
        eval_time = round(time.time() - t0, 1)

        # Build result row
        row = {
            "model_name": repo,
            "model_id": model_id,
            "vocab_size": result.get("vocab_size", vocab_size),
            "log2_vocab": log2_vocab,
            "BPB": result.get("BPB", ""),
            "bits_per_token": result.get("bits_per_token", ""),
            "avg_bytes_per_token": result.get("avg_bytes_per_token", ""),
            "perplexity": result.get("perplexity", ""),
            "model_type": cat_row.get("model_type", "translation"),
            "source_lang": cat_row.get("source_lang", ""),
            "target_lang": cat_row.get("target_lang", ""),
            "stars": cat_row.get("stars", ""),
            "quality_label": cat_row.get("quality_label", ""),
            "status": result.get("status", "error"),
            "error_msg": result.get("error_msg", ""),
            "eval_time_sec": eval_time,
        }

        all_results.append(row)

        if result["status"] == "success":
            successes += 1
            log(f"  ✓ BPB={result['BPB']:.4f}, time={eval_time}s")
        else:
            failures += 1
            log(f"  ✗ {result.get('error_msg', 'unknown')[:100]}, time={eval_time}s")

        # Cleanup model cache to manage disk
        freed = cleanup_model_cache(repo)
        if freed > 0:
            disk_freed_total += freed

        # Save every 10 models
        if (i + 1) % 10 == 0:
            save_results(all_results)
            save_progress(i + 1, len(to_eval), successes, failures,
                         start_time, eta)

        # Brief pause to let GPU memory clear
        time.sleep(1)

    # Final save
    save_results(all_results)

    elapsed_total = time.time() - start_time
    log(f"\n=== EXPERIMENT COMPLETE ===")
    log(f"Total evaluated: {len(to_eval)}")
    log(f"Successes: {successes}")
    log(f"Failures: {failures}")
    log(f"Total time: {timedelta(seconds=int(elapsed_total))}")
    log(f"Avg per model: {elapsed_total/max(len(to_eval),1):.1f}s")
    log(f"Disk freed during run: {disk_freed_total/1024:.1f}GB")
    log(f"Results: {BPB_RESULTS_PATH}")

    save_progress(len(to_eval), len(to_eval), successes, failures,
                 start_time, "complete")


if __name__ == "__main__":
    main()
