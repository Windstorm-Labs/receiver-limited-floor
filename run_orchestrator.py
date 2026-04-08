#!/usr/bin/env python3
"""
Orchestrator: runs each model in a separate subprocess for CUDA isolation.
If one model crashes, the others continue.
"""

import csv
import json
import math
import os
import subprocess
import sys
import time

BASE_DIR = os.path.expanduser("~/throughput_experiment")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CORPUS_PATH = os.path.join(BASE_DIR, "data/reference_corpus.txt")
results_path = os.path.join(RESULTS_DIR, "bpb_results.csv")
EVAL_SCRIPT = os.path.join(BASE_DIR, "eval_single_model.py")

FIELDNAMES = ["model_name", "vocab_size", "log2_vocab", "BPB", "bits_per_token",
              "avg_bytes_per_token", "perplexity", "model_type", "status", "error_msg"]

# Load existing successes
existing = set()
all_results = []
if os.path.exists(results_path):
    with open(results_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] == "success":
                existing.add(row["model_name"])
                all_results.append(row)
print(f"Already completed: {len(existing)} models")

MODELS = [
    ("google/flan-t5-small", "seq2seq", 32100, "translation"),
    ("Helsinki-NLP/opus-mt-tc-big-en-ko", "seq2seq", 32001, "translation"),
    ("albert/albert-base-v2", "masked", 30000, "LLM"),
    ("distilbert-base-uncased", "masked", 30522, "LLM"),
    ("bert-base-uncased", "masked", 30522, "LLM"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "causal", 32000, "LLM"),
    ("openlm-research/open_llama_3b", "causal", 32000, "LLM"),
    ("mistralai/Mistral-7B-v0.1", "causal", 32000, "LLM"),
    ("NousResearch/Llama-2-7b-hf", "causal", 32000, "LLM"),
    ("Helsinki-NLP/opus-mt-en-tvl", "seq2seq", 38380, "translation"),
    ("Helsinki-NLP/opus-mt-en-ho", "seq2seq", 42463, "translation"),
    ("Helsinki-NLP/opus-mt-en-iso", "seq2seq", 45029, "translation"),
    ("HuggingFaceTB/SmolLM-135M", "causal", 49152, "LLM"),
    ("sshleifer/tiny-gpt2", "causal", 50257, "LLM"),
    ("distilgpt2", "causal", 50257, "LLM"),
    ("gpt2", "causal", 50257, "LLM"),
    ("gpt2-medium", "causal", 50257, "LLM"),
    ("gpt2-large", "causal", 50257, "LLM"),
    ("microsoft/phi-1", "causal", 50257, "LLM"),
    ("microsoft/phi-1_5", "causal", 50257, "LLM"),
    ("microsoft/phi-2", "causal", 50257, "LLM"),
    ("EleutherAI/pythia-14m", "causal", 50254, "LLM"),
    ("EleutherAI/pythia-70m", "causal", 50254, "LLM"),
    ("EleutherAI/pythia-160m", "causal", 50254, "LLM"),
    ("EleutherAI/pythia-410m", "causal", 50254, "LLM"),
    ("EleutherAI/pythia-1b", "causal", 50254, "LLM"),
    ("EleutherAI/pythia-1.4b", "causal", 50254, "LLM"),
    ("facebook/opt-125m", "causal", 50265, "LLM"),
    ("facebook/opt-350m", "causal", 50265, "LLM"),
    ("facebook/opt-1.3b", "causal", 50265, "LLM"),
    ("cerebras/Cerebras-GPT-111M", "causal", 50257, "LLM"),
    ("cerebras/Cerebras-GPT-256M", "causal", 50257, "LLM"),
    ("allenai/OLMo-1B-hf", "causal", 50280, "LLM"),
    ("roberta-base", "masked", 50265, "LLM"),
    ("Helsinki-NLP/opus-mt-tc-big-en-pt", "seq2seq", 54776, "translation"),
    ("Helsinki-NLP/opus-mt-en-de", "seq2seq", 58101, "translation"),
    ("Helsinki-NLP/opus-mt-en-fr", "seq2seq", 59514, "translation"),
    ("Helsinki-NLP/opus-mt-en-ru", "seq2seq", 62518, "translation"),
    ("Helsinki-NLP/opus-mt-en-es", "seq2seq", 65001, "translation"),
    ("baichuan-inc/Baichuan-7B", "causal", 64000, "LLM"),
    ("01-ai/Yi-6B", "causal", 63992, "LLM"),
    ("Helsinki-NLP/opus-mt-en-it", "seq2seq", 80035, "translation"),
    ("google/pegasus-xsum", "seq2seq", 96103, "LLM"),
    ("stabilityai/stablelm-2-zephyr-1_6b", "causal", 100289, "LLM"),
    ("Qwen/Qwen2.5-0.5B", "causal", 151643, "LLM"),
    ("Qwen/Qwen2.5-1.5B", "causal", 151643, "LLM"),
    ("bigscience/bloom-560m", "causal", 250680, "LLM"),
    ("bigscience/bloom-1b1", "causal", 250680, "LLM"),
]

def save_results():
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_results)

python_exe = sys.executable

for i, (model_id, method, vocab_size, model_type) in enumerate(MODELS):
    if model_id in existing:
        print(f"[{i+1}/{len(MODELS)}] {model_id} — done, skipping")
        continue
    
    print(f"\n[{i+1}/{len(MODELS)}] {model_id} (V={vocab_size}, {method})", flush=True)
    log2_vocab = math.log2(vocab_size)
    t_start = time.time()
    
    try:
        proc = subprocess.run(
            [python_exe, EVAL_SCRIPT, model_id, method, str(vocab_size), model_type, CORPUS_PATH],
            capture_output=True, text=True, timeout=600  # 10 min per model
        )
        
        # Find the RESULT: line in stdout
        result_line = None
        for line in proc.stdout.split("\n"):
            if line.startswith("RESULT:"):
                result_line = line[7:]
                break
        
        if result_line:
            result = json.loads(result_line)
        else:
            # Check stderr for clues
            err = proc.stderr[-500:] if proc.stderr else "No output"
            result = {"status": "error", "error": f"No RESULT line. stderr: {err[:200]}"}
        
    except subprocess.TimeoutExpired:
        result = {"status": "error", "error": "Timeout (600s)"}
    except Exception as e:
        result = {"status": "error", "error": str(e)[:200]}
    
    elapsed = time.time() - t_start
    
    if result.get("status") == "success":
        row = {
            "model_name": model_id,
            "vocab_size": str(vocab_size),
            "log2_vocab": str(round(log2_vocab, 3)),
            "BPB": str(round(result["BPB"], 6)),
            "bits_per_token": str(round(result["bits_per_token"], 4)),
            "avg_bytes_per_token": str(round(result["avg_bytes_per_token"], 4)),
            "perplexity": str(round(result["perplexity"], 4)),
            "model_type": model_type,
            "status": "success",
            "error_msg": "",
        }
        existing.add(model_id)
        print(f"  OK: BPB={result['BPB']:.4f}, bits/tok={result['bits_per_token']:.2f}, "
              f"bytes/tok={result['avg_bytes_per_token']:.2f}, PPL={result['perplexity']:.2f} "
              f"({elapsed:.1f}s)")
    else:
        status = result.get("status", "error")
        err_msg = result.get("error", "unknown")[:200]
        row = {
            "model_name": model_id, "vocab_size": str(vocab_size),
            "log2_vocab": str(round(log2_vocab, 3)),
            "BPB": "", "bits_per_token": "", "avg_bytes_per_token": "",
            "perplexity": "", "model_type": model_type,
            "status": status, "error_msg": err_msg,
        }
        print(f"  {status.upper()}: {err_msg} ({elapsed:.1f}s)")
    
    all_results.append(row)
    save_results()
    sys.stdout.flush()

successes = sum(1 for r in all_results if r["status"] == "success")
errors = sum(1 for r in all_results if r["status"] in ("error", "oom"))
print(f"\n{'='*70}")
print(f"ALL MODELS EVALUATED: {successes} success, {errors} error/oom")
print(f"{'='*70}")
