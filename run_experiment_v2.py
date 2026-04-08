#!/usr/bin/env python3
"""
Throughput Constraint Tokenizer Sweep Experiment v2
====================================================
Robust version with proper error handling, CUDA recovery, and 
domain-appropriate model selection.
"""

import csv
import json
import math
import os
import sys
import time
import traceback
import warnings
import gc
warnings.filterwarnings("ignore")

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Async for speed, we catch errors properly
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch

# Directories
BASE_DIR = os.path.expanduser("~/throughput_experiment")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
for d in [DATA_DIR, RESULTS_DIR, CHARTS_DIR]:
    os.makedirs(d, exist_ok=True)

##############################################################################
# STEP 1 — MODEL CATALOG (curated for English-language BPB evaluation)
##############################################################################
print("=" * 70)
print("STEP 1: Building model catalog")
print("=" * 70)

# Only models that can meaningfully process English text.
# Format: (model_id, model_type, approx_params, load_method, expected_vocab)
MODELS = [
    # 32K vocab - Seq2Seq 
    ("google/flan-t5-small", "translation", "80M", "seq2seq", 32100),
    ("Helsinki-NLP/opus-mt-tc-big-en-ko", "translation", "230M", "seq2seq", 32001),

    # 30K vocab - Masked LM
    ("albert/albert-base-v2", "LLM", "11M", "masked", 30000),
    ("distilbert-base-uncased", "LLM", "66M", "masked", 30522),
    ("bert-base-uncased", "LLM", "110M", "masked", 30522),

    # 32K vocab - Causal LM
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "LLM", "1.1B", "causal", 32000),
    ("openlm-research/open_llama_3b", "LLM", "3B", "causal", 32000),
    ("mistralai/Mistral-7B-v0.1", "LLM", "7B", "causal", 32000),
    ("NousResearch/Llama-2-7b-hf", "LLM", "7B", "causal", 32000),

    # 38-45K vocab - Translation
    ("Helsinki-NLP/opus-mt-en-tvl", "translation", "77M", "seq2seq", 38380),
    ("Helsinki-NLP/opus-mt-en-ho", "translation", "77M", "seq2seq", 42463),
    ("Helsinki-NLP/opus-mt-en-iso", "translation", "77M", "seq2seq", 45029),

    # 49K vocab
    ("HuggingFaceTB/SmolLM-135M", "LLM", "135M", "causal", 49152),

    # ~50K vocab - GPT-2 family (different sizes, same tokenizer = nice control)
    ("sshleifer/tiny-gpt2", "LLM", "2M", "causal", 50257),
    ("distilgpt2", "LLM", "82M", "causal", 50257),
    ("gpt2", "LLM", "117M", "causal", 50257),
    ("gpt2-medium", "LLM", "345M", "causal", 50257),
    ("gpt2-large", "LLM", "774M", "causal", 50257),
    ("microsoft/phi-1", "LLM", "1.3B", "causal", 50257),
    ("microsoft/phi-1_5", "LLM", "1.3B", "causal", 50257),
    ("microsoft/phi-2", "LLM", "2.7B", "causal", 50257),

    # ~50K vocab - Pythia family (different sizes, same tokenizer)
    ("EleutherAI/pythia-14m", "LLM", "14M", "causal", 50254),
    ("EleutherAI/pythia-70m", "LLM", "70M", "causal", 50254),
    ("EleutherAI/pythia-160m", "LLM", "160M", "causal", 50254),
    ("EleutherAI/pythia-410m", "LLM", "410M", "causal", 50254),
    ("EleutherAI/pythia-1b", "LLM", "1B", "causal", 50254),
    ("EleutherAI/pythia-1.4b", "LLM", "1.4B", "causal", 50254),

    # ~50K vocab - OPT family
    ("facebook/opt-125m", "LLM", "125M", "causal", 50265),
    ("facebook/opt-350m", "LLM", "350M", "causal", 50265),
    ("facebook/opt-1.3b", "LLM", "1.3B", "causal", 50265),

    # ~50K vocab - Cerebras
    ("cerebras/Cerebras-GPT-111M", "LLM", "111M", "causal", 50257),
    ("cerebras/Cerebras-GPT-256M", "LLM", "256M", "causal", 50257),

    # ~50K vocab - other
    ("allenai/OLMo-1B-hf", "LLM", "1B", "causal", 50280),
    ("roberta-base", "LLM", "125M", "masked", 50265),

    # 54-65K vocab - Translation
    ("Helsinki-NLP/opus-mt-tc-big-en-pt", "translation", "230M", "seq2seq", 54776),
    ("Helsinki-NLP/opus-mt-en-de", "translation", "77M", "seq2seq", 58101),
    ("Helsinki-NLP/opus-mt-en-fr", "translation", "77M", "seq2seq", 59514),
    ("Helsinki-NLP/opus-mt-en-ru", "translation", "77M", "seq2seq", 62518),
    ("Helsinki-NLP/opus-mt-en-es", "translation", "77M", "seq2seq", 65001),

    # 64K
    ("baichuan-inc/Baichuan-7B", "LLM", "7B", "causal", 64000),
    ("01-ai/Yi-6B", "LLM", "6B", "causal", 63992),

    # 80K
    ("Helsinki-NLP/opus-mt-en-it", "translation", "77M", "seq2seq", 80035),

    # 96K
    ("google/pegasus-xsum", "LLM", "570M", "seq2seq", 96103),

    # 100K
    ("stabilityai/stablelm-2-zephyr-1_6b", "LLM", "1.6B", "causal", 100289),

    # 151K
    ("Qwen/Qwen2.5-0.5B", "LLM", "500M", "causal", 151643),
    ("Qwen/Qwen2.5-1.5B", "LLM", "1.5B", "causal", 151643),

    # 250K
    ("bigscience/bloom-560m", "LLM", "560M", "causal", 250680),
    ("bigscience/bloom-1b1", "LLM", "1.1B", "causal", 250680),
]

# Save catalog
catalog_path = os.path.join(RESULTS_DIR, "model_catalog.csv")
with open(catalog_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model_name", "vocab_size", "model_type", "quantization", "param_count", "path"])
    for model_id, mtype, params, method, vocab in MODELS:
        writer.writerow([model_id, vocab, mtype, "fp16", params, model_id])

print(f"Catalog: {len(MODELS)} models")
print(f"Vocab range: {min(m[4] for m in MODELS)} - {max(m[4] for m in MODELS)}")
print(f"Saved to {catalog_path}")

##############################################################################
# STEP 2 — REFERENCE CORPUS
##############################################################################
print("\n" + "=" * 70)
print("STEP 2: Preparing reference corpus (WikiText-2 test)")
print("=" * 70)

corpus_path = os.path.join(DATA_DIR, "reference_corpus.txt")
if os.path.exists(corpus_path):
    with open(corpus_path, "r") as f:
        corpus_text = f.read()
    print(f"Corpus already exists: {len(corpus_text)} chars")
else:
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    corpus_text = "\n".join(dataset["text"])
    with open(corpus_path, "w") as f:
        f.write(corpus_text)
    print(f"Downloaded WikiText-2 test: {len(corpus_text)} chars")

total_bytes = len(corpus_text.encode("utf-8"))
print(f"Total bytes: {total_bytes:,}")

##############################################################################
# STEP 3 — COMPUTE BPB FOR EACH MODEL
##############################################################################
print("\n" + "=" * 70)
print("STEP 3: Computing BPB for each model")
print("=" * 70)

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM
)

results = []
results_path = os.path.join(RESULTS_DIR, "bpb_results.csv")
FIELDNAMES = ["model_name", "vocab_size", "log2_vocab", "BPB", "bits_per_token",
              "avg_bytes_per_token", "perplexity", "model_type", "status", "error_msg"]


def save_results():
    """Save current results to CSV."""
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)


def reset_cuda():
    """Best-effort CUDA cleanup after errors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def compute_bpb_causal(model_id, corpus, total_bytes, max_length=2048):
    """Compute BPB for causal LM models."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    model.eval()
    
    # Tokenize full corpus to get token count
    full_encodings = tokenizer(corpus, return_tensors="pt", truncation=False)
    full_num_tokens = full_encodings.input_ids.shape[1]
    avg_bytes_per_token = total_bytes / full_num_tokens
    
    # Compute loss using sliding window
    stride = max_length
    total_loss = 0.0
    total_counted = 0
    
    input_ids = full_encodings.input_ids
    
    for begin_loc in range(0, min(full_num_tokens, max_length * 10), stride):
        end_loc = min(begin_loc + max_length, full_num_tokens)
        chunk_ids = input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = chunk_ids.clone()
        
        # Don't count overlap tokens (except first window)
        if begin_loc > 0:
            overlap = max_length - stride
            if overlap > 0:
                target_ids[:, :overlap] = -100
        
        with torch.no_grad():
            outputs = model(chunk_ids, labels=target_ids)
        
        counted = (target_ids != -100).sum().item()
        total_loss += outputs.loss.item() * counted
        total_counted += counted
        
        if end_loc >= full_num_tokens or end_loc >= max_length * 10:
            break
    
    avg_loss_nats = total_loss / total_counted if total_counted > 0 else float('inf')
    bits_per_token = avg_loss_nats / math.log(2)
    bpb = bits_per_token / avg_bytes_per_token
    perplexity = math.exp(min(avg_loss_nats, 100))
    
    del model, full_encodings, input_ids
    reset_cuda()
    
    return {
        "BPB": bpb,
        "bits_per_token": bits_per_token,
        "avg_bytes_per_token": avg_bytes_per_token,
        "perplexity": perplexity,
        "num_tokens": full_num_tokens,
    }


def compute_bpb_seq2seq(model_id, corpus, total_bytes, max_length=512):
    """Compute BPB for encoder-decoder models."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "</s>"
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    model.eval()
    
    # Split corpus into sentences for teacher-forced decoding
    sentences = [s.strip() for s in corpus.split("\n") if len(s.strip()) > 20][:500]
    
    total_loss = 0.0
    total_counted = 0
    
    batch_size = 8
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                           max_length=max_length, padding=True)
        labels = tokenizer(text_target=batch, return_tensors="pt", truncation=True,
                          max_length=max_length, padding=True)
        
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        label_ids = labels.input_ids.to(model.device)
        label_ids[label_ids == tokenizer.pad_token_id] = -100
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)
        
        counted = (label_ids != -100).sum().item()
        total_loss += outputs.loss.item() * counted
        total_counted += counted
    
    avg_loss_nats = total_loss / total_counted if total_counted > 0 else float('inf')
    
    # Full tokenization for bytes/token
    full_enc = tokenizer(corpus, truncation=False)
    num_tokens = len(full_enc.input_ids)
    
    bits_per_token = avg_loss_nats / math.log(2)
    avg_bytes_per_token = total_bytes / num_tokens
    bpb = bits_per_token / avg_bytes_per_token
    perplexity = math.exp(min(avg_loss_nats, 100))
    
    del model
    reset_cuda()
    
    return {
        "BPB": bpb,
        "bits_per_token": bits_per_token,
        "avg_bytes_per_token": avg_bytes_per_token,
        "perplexity": perplexity,
        "num_tokens": num_tokens,
    }


def compute_bpb_masked(model_id, corpus, total_bytes, max_length=512):
    """Compute pseudo-BPB for masked LM models."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    model.eval()
    
    # Take a sample of the corpus
    sentences = [s.strip() for s in corpus.split("\n") if len(s.strip()) > 20][:200]
    sample_text = " ".join(sentences)
    
    encodings = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids.to(model.device)
    num_tokens = input_ids.shape[1]
    
    mask_token_id = tokenizer.mask_token_id
    
    if mask_token_id is None:
        # Fallback: use forward pass with labels
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            avg_loss_nats = outputs.loss.item()
    else:
        # Pseudo-log-likelihood: mask one token at a time, sample positions
        n_samples = min(300, num_tokens - 2)  # exclude [CLS] and [SEP]
        positions = np.random.choice(range(1, num_tokens - 1), n_samples, replace=False)
        
        total_log_prob = 0.0
        for pos in positions:
            masked_input = input_ids.clone()
            masked_input[0, pos] = mask_token_id
            
            with torch.no_grad():
                outputs = model(masked_input)
                logits = outputs.logits[0, pos].float()
                log_probs = torch.log_softmax(logits, dim=-1)
                true_token = input_ids[0, pos]
                total_log_prob += log_probs[true_token].item()
        
        avg_loss_nats = -total_log_prob / n_samples
    
    # Full tokenization for bytes/token
    full_enc = tokenizer(corpus, truncation=False)
    full_num_tokens = len(full_enc.input_ids)
    
    bits_per_token = avg_loss_nats / math.log(2)
    avg_bytes_per_token = total_bytes / full_num_tokens
    bpb = bits_per_token / avg_bytes_per_token
    perplexity = math.exp(min(avg_loss_nats, 100))
    
    del model
    reset_cuda()
    
    return {
        "BPB": bpb,
        "bits_per_token": bits_per_token,
        "avg_bytes_per_token": avg_bytes_per_token,
        "perplexity": perplexity,
        "num_tokens": full_num_tokens,
    }


# Main evaluation loop
for i, (model_id, mtype, params, method, vocab_size) in enumerate(MODELS):
    print(f"\n[{i+1}/{len(MODELS)}] {model_id} (vocab={vocab_size}, method={method})")
    
    log2_vocab = math.log2(vocab_size) if vocab_size > 0 else 0
    
    try:
        t_start = time.time()
        
        if method == "causal":
            result = compute_bpb_causal(model_id, corpus_text, total_bytes)
        elif method == "seq2seq":
            result = compute_bpb_seq2seq(model_id, corpus_text, total_bytes)
        elif method == "masked":
            result = compute_bpb_masked(model_id, corpus_text, total_bytes)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        elapsed = time.time() - t_start
        
        row = {
            "model_name": model_id,
            "vocab_size": vocab_size,
            "log2_vocab": round(log2_vocab, 3),
            "BPB": round(result["BPB"], 6),
            "bits_per_token": round(result["bits_per_token"], 4),
            "avg_bytes_per_token": round(result["avg_bytes_per_token"], 4),
            "perplexity": round(result["perplexity"], 4),
            "model_type": mtype,
            "status": "success",
            "error_msg": "",
        }
        results.append(row)
        
        print(f"  OK: BPB={result['BPB']:.4f}, bits/tok={result['bits_per_token']:.2f}, "
              f"bytes/tok={result['avg_bytes_per_token']:.2f}, PPL={result['perplexity']:.2f} "
              f"({elapsed:.1f}s)")
        
    except torch.cuda.OutOfMemoryError:
        reset_cuda()
        row = {
            "model_name": model_id, "vocab_size": vocab_size, "log2_vocab": round(log2_vocab, 3),
            "BPB": "", "bits_per_token": "", "avg_bytes_per_token": "", "perplexity": "",
            "model_type": mtype, "status": "oom", "error_msg": "CUDA OOM",
        }
        results.append(row)
        print(f"  OOM: Skipping")
        
    except Exception as e:
        reset_cuda()
        err_msg = str(e)[:200]
        row = {
            "model_name": model_id, "vocab_size": vocab_size, "log2_vocab": round(log2_vocab, 3),
            "BPB": "", "bits_per_token": "", "avg_bytes_per_token": "", "perplexity": "",
            "model_type": mtype, "status": "error", "error_msg": err_msg,
        }
        results.append(row)
        print(f"  ERROR: {err_msg}")
        traceback.print_exc()
    
    # Save incrementally
    save_results()
    sys.stdout.flush()

print(f"\n{'=' * 70}")
print(f"STEP 3 COMPLETE")
successes = sum(1 for r in results if r["status"] == "success")
failures = sum(1 for r in results if r["status"] == "error")
ooms = sum(1 for r in results if r["status"] == "oom")
print(f"Success: {successes}, Error: {failures}, OOM: {ooms}")
print(f"Results saved to: {results_path}")

##############################################################################
# STEP 4 — STATISTICAL ANALYSIS
##############################################################################
print("\n" + "=" * 70)
print("STEP 4: Statistical Analysis")
print("=" * 70)

import pandas as pd
from scipy import stats
import ruptures

df = pd.read_csv(results_path)
df_ok = df[df["status"] == "success"].copy()
for col in ["BPB", "log2_vocab", "bits_per_token", "avg_bytes_per_token", "perplexity", "vocab_size"]:
    df_ok[col] = pd.to_numeric(df_ok[col], errors="coerce")
df_ok = df_ok.dropna(subset=["BPB", "log2_vocab"])

# Filter out clearly broken results (BPB > 10 is nonsensical for real language)
df_ok = df_ok[df_ok["BPB"] < 10].copy()

print(f"Analyzing {len(df_ok)} successful models (after filtering)")
print(f"Vocab range: {df_ok['vocab_size'].min():.0f} - {df_ok['vocab_size'].max():.0f}")
print(f"log2(vocab) range: {df_ok['log2_vocab'].min():.1f} - {df_ok['log2_vocab'].max():.1f}")

# A. LINEAR FIT TEST
print("\n--- A. LINEAR FIT TEST ---")
slope, intercept, r_value, p_value, std_err = stats.linregress(df_ok["log2_vocab"], df_ok["BPB"])
r_squared = r_value ** 2
print(f"  β₀ = {intercept:.4f}")
print(f"  β₁ = {slope:.6f}")
print(f"  R² = {r_squared:.4f}")
print(f"  p(β₁) = {p_value:.6f}")
if p_value < 0.05:
    print(f"  → β₁ significant (p < 0.05) — supports linear scaling")
else:
    print(f"  → β₁ NOT significant (p ≥ 0.05) — supports plateau")

# B. PLATEAU DETECTION
print("\n--- B. PLATEAU DETECTION ---")
v_plateau = "N/A"
try:
    df_sorted = df_ok.sort_values("log2_vocab").reset_index(drop=True)
    signal = df_sorted["BPB"].values
    
    if len(signal) >= 5:
        algo = ruptures.Pelt(model="rbf", min_size=3).fit(signal.reshape(-1, 1))
        breakpoints = algo.predict(pen=1)
        
        if breakpoints and breakpoints[0] < len(signal):
            bp_idx = min(breakpoints[0], len(signal) - 1)
            v_plateau = df_sorted.iloc[bp_idx]["vocab_size"]
            log2_plateau = df_sorted.iloc[bp_idx]["log2_vocab"]
            
            var_before = np.var(signal[:bp_idx]) if bp_idx > 1 else float('nan')
            var_after = np.var(signal[bp_idx:]) if bp_idx < len(signal) - 1 else float('nan')
            
            print(f"  Changepoint at index {bp_idx}")
            print(f"  V_plateau ≈ {v_plateau:.0f} (log₂ ≈ {log2_plateau:.1f})")
            print(f"  Variance before: {var_before:.6f}")
            print(f"  Variance after:  {var_after:.6f}")
        else:
            print("  No changepoint detected")
except Exception as e:
    print(f"  Detection failed: {e}")

# C. BITS/EVENT
print("\n--- C. BITS/EVENT ---")
df_ok["bits_per_event"] = df_ok["BPB"] * df_ok["avg_bytes_per_token"]
bpe_mean = df_ok["bits_per_event"].mean()
bpe_median = df_ok["bits_per_event"].median()
bpe_q25 = df_ok["bits_per_event"].quantile(0.25)
bpe_q75 = df_ok["bits_per_event"].quantile(0.75)
in_band = df_ok[(df_ok["bits_per_event"] >= 3) & (df_ok["bits_per_event"] <= 5.5)]
print(f"  Mean:   {bpe_mean:.4f}")
print(f"  Median: {bpe_median:.4f}")
print(f"  IQR:    [{bpe_q25:.4f}, {bpe_q75:.4f}]")
print(f"  In [3, 5.5]: {len(in_band)}/{len(df_ok)} ({100*len(in_band)/len(df_ok):.1f}%)")

# D. RATE-DISTORTION
print("\n--- D. RATE-DISTORTION ---")
def h_binary(eps):
    if eps <= 0 or eps >= 1:
        return 0
    return -eps * math.log2(eps) - (1 - eps) * math.log2(1 - eps)

df_ok["epsilon"] = (1 - (1 / df_ok["perplexity"])).clip(0.001, 0.999)
R_M_vals = []
residuals = []
for idx, row in df_ok.iterrows():
    eps = row["epsilon"]
    V = row["vocab_size"]
    H_b = h_binary(eps)
    R_M = math.log2(V) - H_b - eps * math.log2(max(V - 1, 1))
    R_M = max(R_M, 0)
    df_ok.at[idx, "R_M"] = R_M
    df_ok.at[idx, "rd_residual"] = row["BPB"] - R_M
    R_M_vals.append(R_M)
    residuals.append(row["BPB"] - R_M)

print(f"  Mean residual: {np.mean(residuals):.4f}")
print(f"  Std residual:  {np.std(residuals):.4f}")

# Save summary
summary_path = os.path.join(RESULTS_DIR, "statistical_summary.txt")
with open(summary_path, "w") as f:
    f.write("THROUGHPUT CONSTRAINT — STATISTICAL SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Models analyzed: {len(df_ok)}\n")
    f.write(f"Vocab range: {df_ok['vocab_size'].min():.0f} - {df_ok['vocab_size'].max():.0f}\n")
    f.write(f"log₂(V) range: {df_ok['log2_vocab'].min():.1f} - {df_ok['log2_vocab'].max():.1f}\n\n")
    
    f.write("A. LINEAR FIT\n")
    f.write(f"   BPB ~ {intercept:.4f} + {slope:.6f} · log₂(V)\n")
    f.write(f"   R² = {r_squared:.4f}, p(β₁) = {p_value:.6f}\n\n")
    
    f.write("B. PLATEAU\n")
    f.write(f"   V_plateau ≈ {v_plateau}\n\n")
    
    f.write("C. BITS/EVENT\n")
    f.write(f"   Mean={bpe_mean:.4f}, Median={bpe_median:.4f}\n")
    f.write(f"   IQR=[{bpe_q25:.4f}, {bpe_q75:.4f}]\n")
    f.write(f"   In [3, 5.5]: {len(in_band)}/{len(df_ok)}\n\n")
    
    f.write("D. RATE-DISTORTION\n")
    f.write(f"   Mean residual: {np.mean(residuals):.4f}\n")
    f.write(f"   Std residual: {np.std(residuals):.4f}\n")

df_ok.to_csv(os.path.join(RESULTS_DIR, "bpb_results_augmented.csv"), index=False)
print(f"\nSaved: {summary_path}")

##############################################################################
# STEP 5 — VISUALIZATIONS
##############################################################################
print("\n" + "=" * 70)
print("STEP 5: Generating Visualizations")
print("=" * 70)

import plotly.graph_objects as go
import statsmodels.api as sm

# CHART 1
print("  Chart 1: BPB vs log₂(Vocab Size)...")
fig1 = go.Figure()
colors = {"LLM": "#1f77b4", "translation": "#ff7f0e"}
for mt in df_ok["model_type"].unique():
    sub = df_ok[df_ok["model_type"] == mt]
    fig1.add_trace(go.Scatter(
        x=sub["log2_vocab"], y=sub["BPB"], mode="markers", name=mt,
        marker=dict(size=10, color=colors.get(mt, "#888")),
        text=sub["model_name"],
        hovertemplate="%{text}<br>log₂(V)=%{x:.1f}<br>BPB=%{y:.4f}<extra></extra>"
    ))

x_range = np.linspace(df_ok["log2_vocab"].min() - 0.5, df_ok["log2_vocab"].max() + 0.5, 100)
fig1.add_trace(go.Scatter(
    x=x_range, y=intercept + slope * x_range, mode="lines",
    name=f"Linear fit (β₁={slope:.4f}, p={p_value:.4f})",
    line=dict(color="red", dash="dash")
))

try:
    lowess = sm.nonparametric.lowess(df_ok["BPB"].values, df_ok["log2_vocab"].values, frac=0.4)
    fig1.add_trace(go.Scatter(
        x=lowess[:, 0], y=lowess[:, 1], mode="lines", name="LOWESS",
        line=dict(color="green", width=2)
    ))
except: pass

mean_bpb = df_ok["BPB"].mean()
fig1.add_hline(y=mean_bpb, line_dash="dot", line_color="gray",
               annotation_text=f"Mean BPB = {mean_bpb:.3f}")

tick_vals = list(range(int(df_ok["log2_vocab"].min()), int(df_ok["log2_vocab"].max()) + 2))
tick_texts = [f"{2**v:,.0f}" for v in tick_vals]

fig1.update_layout(
    title="BPB vs log₂(Vocab Size): Plateau or Linear Scaling?",
    xaxis_title="log₂(Vocab Size)", yaxis_title="Bits Per Byte (BPB)",
    width=1200, height=700, template="plotly_white", font=dict(size=14)
)
fig1.update_xaxes(tickvals=tick_vals, ticktext=tick_texts)
fig1.write_image(os.path.join(CHARTS_DIR, "chart1_bpb_vs_vocab.png"), scale=2)
print("    Done")

# CHART 2
print("  Chart 2: Bits per Event Distribution...")
fig2 = go.Figure()
fig2.add_trace(go.Histogram(
    x=df_ok["bits_per_event"], nbinsx=30,
    marker_color="#1f77b4", opacity=0.7, name="bits/event"
))
fig2.add_vrect(x0=3, x1=5.5, fillcolor="green", opacity=0.15,
               annotation_text="Theoretical [3, 5.5]")
fig2.add_vline(x=4.39, line_dash="dash", line_color="red",
               annotation_text="Ribosome: 4.39 bits")
fig2.update_layout(
    title="Effective bits/event: Do LLMs Converge to the Biological Band?",
    xaxis_title="bits per event", yaxis_title="Count",
    width=1000, height=600, template="plotly_white", font=dict(size=14)
)
fig2.write_image(os.path.join(CHARTS_DIR, "chart2_bits_per_event.png"), scale=2)
print("    Done")

# CHART 3
print("  Chart 3: Bytes/Token vs Vocab Size...")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=df_ok["log2_vocab"], y=df_ok["avg_bytes_per_token"],
    mode="markers", marker=dict(size=10, color="#1f77b4"),
    text=df_ok["model_name"],
    hovertemplate="%{text}<br>log₂(V)=%{x:.1f}<br>bytes/tok=%{y:.2f}<extra></extra>"
))
try:
    s_bt, i_bt, _, _, _ = stats.linregress(df_ok["log2_vocab"], df_ok["avg_bytes_per_token"])
    fig3.add_trace(go.Scatter(
        x=x_range, y=i_bt + s_bt * x_range, mode="lines",
        name=f"Fit (slope={s_bt:.3f})", line=dict(color="red", dash="dash")
    ))
except: pass
fig3.update_layout(
    title="Tokenization Efficiency: Bytes/Token vs Vocab Size",
    xaxis_title="log₂(Vocab Size)", yaxis_title="Average Bytes per Token",
    width=1000, height=600, template="plotly_white", font=dict(size=14)
)
fig3.update_xaxes(tickvals=tick_vals, ticktext=tick_texts)
fig3.write_image(os.path.join(CHARTS_DIR, "chart3_bytes_per_token.png"), scale=2)
print("    Done")

# CHART 4
print("  Chart 4: BPB vs R_M(ε)...")
fig4 = go.Figure()
df_rd = df_ok.dropna(subset=["R_M"])
fig4.add_trace(go.Scatter(
    x=df_rd["R_M"], y=df_rd["BPB"], mode="markers",
    marker=dict(size=10, color=df_rd["log2_vocab"], colorscale="Viridis",
                colorbar=dict(title="log₂(V)"), showscale=True),
    text=df_rd["model_name"],
    hovertemplate="%{text}<br>R_M(ε)=%{x:.3f}<br>BPB=%{y:.4f}<extra></extra>"
))
rd_max = max(df_rd["R_M"].max(), df_rd["BPB"].max()) * 1.1
fig4.add_trace(go.Scatter(
    x=[0, rd_max], y=[0, rd_max], mode="lines", name="y = x",
    line=dict(color="red", dash="dash")
))
fig4.update_layout(
    title="Observed BPB vs Rate-Distortion Prediction R_M(ε)",
    xaxis_title="R_M(ε)", yaxis_title="Observed BPB",
    width=1000, height=700, template="plotly_white", font=dict(size=14)
)
fig4.write_image(os.path.join(CHARTS_DIR, "chart4_rd_vs_observed.png"), scale=2)
print("    Done")

##############################################################################
# FINAL
##############################################################################
print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
print(f"Models: {successes} success, {failures} error, {ooms} OOM")
print(f"Linear fit: β₁ = {slope:.6f}, p = {p_value:.6f}, R² = {r_squared:.4f}")
print(f"Mean BPB: {mean_bpb:.4f}")
print(f"Bits/event: mean={bpe_mean:.4f}, IQR=[{bpe_q25:.4f}, {bpe_q75:.4f}]")
print(f"In [3, 5.5] band: {len(in_band)}/{len(df_ok)} ({100*len(in_band)/len(df_ok):.1f}%)")
