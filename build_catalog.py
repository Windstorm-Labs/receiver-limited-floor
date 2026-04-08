#!/usr/bin/env python3
"""
Phase 1: Build full model catalog with vocab sizes.
Pulls tokenizer config from HuggingFace API for each model in the Windy Pro roster.
Falls back to downloading just the tokenizer if API doesn't have vocab_size.
"""

import csv
import json
import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.expanduser("~/throughput_experiment")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ROSTER_PATH = "/srv/repos/windy-pro/THE_CLINIC/MASTER_ROSTER.json"
CATALOG_PATH = os.path.join(RESULTS_DIR, "full_catalog.csv")

FIELDNAMES = [
    "model_id", "source_repo", "source_lang", "target_lang",
    "vocab_size", "model_type", "eval_method", "variants",
    "stars", "quality_label", "status"
]


def get_vocab_size_api(repo_id: str) -> int | None:
    """Try to get vocab_size from HF API without downloading anything."""
    # Try config.json first
    url = f"https://huggingface.co/{repo_id}/raw/main/config.json"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            config = r.json()
            vs = config.get("vocab_size") or config.get("encoder", {}).get("vocab_size")
            if vs:
                return int(vs)
    except Exception:
        pass

    # Try tokenizer_config.json
    url = f"https://huggingface.co/{repo_id}/raw/main/tokenizer_config.json"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            config = r.json()
            vs = config.get("vocab_size")
            if vs:
                return int(vs)
    except Exception:
        pass

    return None


def get_vocab_size_tokenizer(repo_id: str) -> int | None:
    """Download just the tokenizer and check vocab size."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        return tok.vocab_size
    except Exception as e:
        print(f"    tokenizer fallback failed for {repo_id}: {e}")
        return None


def classify_model(repo_id: str, info: dict) -> tuple[str, str]:
    """Determine model_type and eval_method."""
    repo_lower = repo_id.lower()
    if "helsinki-nlp/opus-mt" in repo_lower:
        if "-tc-big-" in repo_lower:
            return "translation", "seq2seq"
        return "translation", "seq2seq"
    elif "hplt" in repo_lower:
        return "translation", "seq2seq"
    elif "windyprolabs" in repo_lower:
        return "translation", "seq2seq"
    else:
        return "translation", "seq2seq"  # Default for this catalog


def process_model(model_id: str, info: dict) -> dict:
    """Process a single model entry."""
    repo = info.get("source_repo", "")
    if not repo:
        return None

    model_type, eval_method = classify_model(repo, info)

    # Get vocab size
    vocab_size = get_vocab_size_api(repo)
    status = "api_success" if vocab_size else "needs_tokenizer"

    return {
        "model_id": model_id,
        "source_repo": repo,
        "source_lang": info.get("source_lang", ""),
        "target_lang": info.get("target_lang", ""),
        "vocab_size": vocab_size or 0,
        "model_type": model_type,
        "eval_method": eval_method,
        "variants": ",".join(info.get("variants", [])),
        "stars": info.get("stars", ""),
        "quality_label": info.get("quality_label", ""),
        "status": status,
    }


def main():
    # Load roster
    with open(ROSTER_PATH) as f:
        roster = json.load(f)
    patients = roster["patients"]
    print(f"Roster has {len(patients)} models")

    # Load existing catalog for resume capability
    existing = {}
    if os.path.exists(CATALOG_PATH):
        with open(CATALOG_PATH) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("vocab_size") and int(row["vocab_size"]) > 0:
                    existing[row["model_id"]] = row
        print(f"Already cataloged: {len(existing)} models with vocab sizes")

    results = list(existing.values())
    to_process = [(mid, info) for mid, info in patients.items() if mid not in existing]
    print(f"Need to process: {len(to_process)} models")

    # Phase 1: API-based vocab size lookup (fast, parallel)
    print("\n=== Phase 1: API lookup ===")
    needs_tokenizer = []

    def api_worker(args):
        model_id, info = args
        return process_model(model_id, info)

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(api_worker, (mid, info)): mid
                   for mid, info in to_process}
        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result is None:
                continue
            if result["status"] == "api_success":
                results.append(result)
            else:
                needs_tokenizer.append((futures[future],
                                       patients[futures[future]]))
            if done % 100 == 0:
                print(f"  API checked: {done}/{len(to_process)}, "
                      f"got vocab: {len(results) - len(existing)}, "
                      f"need tokenizer: {len(needs_tokenizer)}")

    print(f"\nAPI phase complete. Got vocab for {len(results) - len(existing)} new models.")
    print(f"Need tokenizer download for {len(needs_tokenizer)} models.")

    # Save intermediate
    save_catalog(results)

    # Phase 2: Tokenizer-based fallback (slower, sequential to manage memory)
    if needs_tokenizer:
        print(f"\n=== Phase 2: Tokenizer fallback ({len(needs_tokenizer)} models) ===")
        for i, (model_id, info) in enumerate(needs_tokenizer):
            repo = info.get("source_repo", "")
            print(f"  [{i+1}/{len(needs_tokenizer)}] {repo}", end="", flush=True)
            vocab_size = get_vocab_size_tokenizer(repo)
            entry = process_model(model_id, info)
            if vocab_size:
                entry["vocab_size"] = vocab_size
                entry["status"] = "tokenizer_success"
            else:
                entry["status"] = "failed"
            results.append(entry)
            print(f" → vocab={vocab_size}")

            # Save every 50
            if (i + 1) % 50 == 0:
                save_catalog(results)

    save_catalog(results)
    print(f"\n=== CATALOG COMPLETE ===")
    print(f"Total: {len(results)}")
    success = sum(1 for r in results if int(r.get('vocab_size', 0)) > 0)
    print(f"With vocab size: {success}")
    print(f"Failed: {len(results) - success}")


def save_catalog(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CATALOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: int(r.get("vocab_size", 0))))
    print(f"  Saved {len(results)} entries to {CATALOG_PATH}")


if __name__ == "__main__":
    main()
