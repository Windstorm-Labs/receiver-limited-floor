#!/usr/bin/env python3
"""
Throughput Constraint Tokenizer Sweep Experiment
=================================================
Tests whether BPB plateaus across vocabulary sizes, or scales linearly with log2(V).

Steps 1-5 in a single pipeline:
  1. Build model catalog
  2. Prepare reference corpus (WikiText-2 test)
  3. Compute BPB for each model
  4. Statistical analysis
  5. Generate visualizations
"""

import csv
import json
import math
import os
import sys
import time
import traceback
import warnings
warnings.filterwarnings("ignore")

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
# STEP 1 — MODEL CATALOG
##############################################################################
print("=" * 70)
print("STEP 1: Building model catalog")
print("=" * 70)

# Curated models spanning the widest vocab range possible.
# We focus on models that support causal LM (AutoModelForCausalLM) or 
# encoder models where we can compute masked LM loss.
# Format: (model_id, model_type, approx_params, load_method)
# load_method: "causal" = AutoModelForCausalLM, "seq2seq" = encoder-decoder,
#              "masked" = AutoModelForMaskedLM, "protein" = ESM, "dna" = DNA model

MODELS = [
    # Byte-level (256 vocab)
    ("google/byt5-small", "translation", "300M", "seq2seq", 256),

    # Protein (33 vocab)
    ("facebook/esm2_t6_8M_UR50D", "protein", "8M", "protein", 33),

    # DNA models (~4K vocab)
    ("zhihan1996/DNABERT-2-117M", "LLM", "117M", "masked", 4096),
    ("armheb/DNA_bert_6", "LLM", "110M", "masked", 4101),
    ("InstaDeepAI/nucleotide-transformer-500m-human-ref", "LLM", "500M", "masked", 4107),

    # Chinese GPT-2 (21K vocab)
    ("uer/gpt2-chinese-cluecorpussmall", "LLM", "110M", "causal", 21128),

    # ALBERT/BERT-family (30K vocab)
    ("albert/albert-base-v2", "LLM", "11M", "masked", 30000),
    ("distilbert-base-uncased", "LLM", "66M", "masked", 30522),
    ("bert-base-uncased", "LLM", "110M", "masked", 30522),

    # 32K vocab models
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "LLM", "1.1B", "causal", 32000),
    ("mistralai/Mistral-7B-v0.1", "LLM", "7B", "causal", 32000),
    ("NousResearch/Llama-2-7b-hf", "LLM", "7B", "causal", 32000),
    ("openlm-research/open_llama_3b", "LLM", "3B", "causal", 32000),
    ("google/flan-t5-small", "translation", "80M", "seq2seq", 32100),
    ("Helsinki-NLP/opus-mt-tc-big-en-ko", "translation", "230M", "seq2seq", 32001),

    # Tuvaluan MT (38K) 
    ("Helsinki-NLP/opus-mt-en-tvl", "translation", "77M", "seq2seq", 38380),
    
    # Hindi BERT (39K)
    ("monsoon-nlp/hindi-bert", "LLM", "110M", "masked", 39628),

    # Hiri Motu MT (42K)
    ("Helsinki-NLP/opus-mt-en-ho", "translation", "77M", "seq2seq", 42463),

    # Isoko MT (45K)
    ("Helsinki-NLP/opus-mt-en-iso", "translation", "77M", "seq2seq", 45029),

    # SmolLM (49K)
    ("HuggingFaceTB/SmolLM-135M", "LLM", "135M", "causal", 49152),

    # GPT-2 family (~50K vocab)
    ("sshleifer/tiny-gpt2", "LLM", "2M", "causal", 50257),
    ("distilgpt2", "LLM", "82M", "causal", 50257),
    ("gpt2", "LLM", "117M", "causal", 50257),
    ("gpt2-medium", "LLM", "345M", "causal", 50257),
    ("gpt2-large", "LLM", "774M", "causal", 50257),

    # Pythia family (~50K vocab, different sizes)
    ("EleutherAI/pythia-14m", "LLM", "14M", "causal", 50254),
    ("EleutherAI/pythia-70m", "LLM", "70M", "causal", 50254),
    ("EleutherAI/pythia-160m", "LLM", "160M", "causal", 50254),
    ("EleutherAI/pythia-410m", "LLM", "410M", "causal", 50254),
    ("EleutherAI/pythia-1b", "LLM", "1B", "causal", 50254),

    # OPT family (~50K vocab)
    ("facebook/opt-125m", "LLM", "125M", "causal", 50265),
    ("facebook/opt-350m", "LLM", "350M", "causal", 50265),

    # Opus MT with varied vocabs (54K-80K)
    ("Helsinki-NLP/opus-mt-en-de", "translation", "77M", "seq2seq", 58101),
    ("Helsinki-NLP/opus-mt-en-fr", "translation", "77M", "seq2seq", 59514),
    ("Helsinki-NLP/opus-mt-en-ru", "translation", "77M", "seq2seq", 62518),
    ("Helsinki-NLP/opus-mt-en-it", "translation", "77M", "seq2seq", 80035),

    # Baichuan (64K)
    ("baichuan-inc/Baichuan-7B", "LLM", "7B", "causal", 64000),

    # Pegasus (96K)
    ("google/pegasus-xsum", "LLM", "570M", "seq2seq", 96103),

    # StableLM (100K)
    ("stabilityai/stablelm-2-zephyr-1_6b", "LLM", "1.6B", "causal", 100289),

    # Bangla BERT (102K)
    ("sagorsarker/bangla-bert-base", "LLM", "110M", "masked", 101975),

    # Qwen (151K vocab)
    ("Qwen/Qwen2.5-0.5B", "LLM", "500M", "causal", 151643),

    # BLOOM (250K vocab)
    ("bigscience/bloom-560m", "LLM", "560M", "causal", 250680),

    # LEALLA (501K vocab)
    ("setu4993/LEALLA-small", "LLM", "85M", "masked", 501153),

    # CANINE (1.1M vocab - Unicode)
    ("google/canine-s", "LLM", "110M", "masked", 1114112),
]

# Save catalog
catalog_path = os.path.join(RESULTS_DIR, "model_catalog.csv")
with open(catalog_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model_name", "vocab_size", "model_type", "quantization", "param_count", "path"])
    for model_id, mtype, params, method, vocab in MODELS:
        writer.writerow([model_id, vocab, mtype, "fp16", params, model_id])

print(f"Catalog: {len(MODELS)} models, vocab range {min(m[4] for m in MODELS)} - {max(m[4] for m in MODELS)}")
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
    AutoModelForMaskedLM, AutoModel, EsmForMaskedLM
)

results = []
results_path = os.path.join(RESULTS_DIR, "bpb_results.csv")

# Write header
with open(results_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model_name", "vocab_size", "log2_vocab", "BPB", "bits_per_token",
                      "avg_bytes_per_token", "perplexity", "model_type", "status", "error_msg"])

def compute_bpb_causal(model_id, corpus, total_bytes, max_length=2048):
    """Compute BPB for causal LM models."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    model.eval()
    
    # Tokenize the corpus
    encodings = tokenizer(corpus, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids.to(model.device)
    
    num_tokens = input_ids.shape[1]
    
    # Compute loss in chunks to handle long sequences
    # Use sliding window approach
    stride = max_length // 2
    total_loss = 0.0
    total_counted = 0
    
    for begin_loc in range(0, num_tokens, stride):
        end_loc = min(begin_loc + max_length, num_tokens)
        trg_len = end_loc - begin_loc
        
        chunk_ids = input_ids[:, begin_loc:end_loc]
        target_ids = chunk_ids.clone()
        
        # Only count loss for the non-overlapping part (except first chunk)
        if begin_loc > 0:
            target_ids[:, :stride] = -100
        
        with torch.no_grad():
            outputs = model(chunk_ids, labels=target_ids)
            # Loss is averaged over non-ignored tokens
            neg_log_likelihood = outputs.loss
        
        # Count how many tokens contributed
        counted = (target_ids != -100).sum().item()
        total_loss += neg_log_likelihood.item() * counted
        total_counted += counted
        
        if end_loc >= num_tokens:
            break
    
    avg_loss_nats = total_loss / total_counted if total_counted > 0 else float('inf')
    
    # Compute metrics
    bits_per_token = avg_loss_nats / math.log(2)
    avg_bytes_per_token = total_bytes / num_tokens
    bpb = bits_per_token / avg_bytes_per_token
    perplexity = math.exp(min(avg_loss_nats, 100))  # cap to avoid overflow
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return {
        "BPB": bpb,
        "bits_per_token": bits_per_token,
        "avg_bytes_per_token": avg_bytes_per_token,
        "perplexity": perplexity,
        "num_tokens": num_tokens,
    }


def compute_bpb_seq2seq(model_id, corpus, total_bytes, max_length=512):
    """Compute BPB for encoder-decoder (seq2seq) models using decoder loss."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "</s>"
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    model.eval()
    
    # For seq2seq, we use teacher-forced decoding:
    # Split corpus into chunks, use each chunk as both input and target
    sentences = [s.strip() for s in corpus.split("\n") if len(s.strip()) > 20][:500]
    
    total_loss = 0.0
    total_counted = 0
    total_tokens = 0
    
    batch_size = 8
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, 
                           max_length=max_length, padding=True)
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(batch, return_tensors="pt", truncation=True,
                              max_length=max_length, padding=True)
        
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        label_ids = labels.input_ids.to(model.device)
        
        # Replace padding token id with -100 so it's not counted in loss
        label_ids[label_ids == tokenizer.pad_token_id] = -100
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)
        
        counted = (label_ids != -100).sum().item()
        total_loss += outputs.loss.item() * counted
        total_counted += counted
        total_tokens += inputs.input_ids.numel()  # for bytes/token estimate
    
    avg_loss_nats = total_loss / total_counted if total_counted > 0 else float('inf')
    
    # For bytes/token we use the encoder tokenization of the full corpus
    full_enc = tokenizer(corpus, truncation=True, max_length=100000)
    num_tokens = len(full_enc.input_ids)
    
    bits_per_token = avg_loss_nats / math.log(2)
    avg_bytes_per_token = total_bytes / num_tokens
    bpb = bits_per_token / avg_bytes_per_token
    perplexity = math.exp(min(avg_loss_nats, 100))
    
    del model
    torch.cuda.empty_cache()
    
    return {
        "BPB": bpb,
        "bits_per_token": bits_per_token,
        "avg_bytes_per_token": avg_bytes_per_token,
        "perplexity": perplexity,
        "num_tokens": num_tokens,
    }


def compute_bpb_masked(model_id, corpus, total_bytes, max_length=512):
    """Compute pseudo-BPB for masked LM models using pseudo-perplexity."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Check if it's an ESM model
    is_esm = "esm" in model_id.lower()
    if is_esm:
        model = EsmForMaskedLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16
        )
    else:
        model = AutoModelForMaskedLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
        )
    model.eval()
    
    # For masked LMs, compute pseudo-log-likelihood:
    # Mask each token one at a time, compute loss
    # We sample a subset for efficiency
    sentences = [s.strip() for s in corpus.split("\n") if len(s.strip()) > 20][:200]
    sample_text = "\n".join(sentences)
    
    encodings = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids.to(model.device)
    
    num_tokens = input_ids.shape[1]
    mask_token_id = tokenizer.mask_token_id
    
    if mask_token_id is None:
        # Fallback: use regular forward pass with labels
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            avg_loss_nats = outputs.loss.item()
    else:
        # True pseudo-log-likelihood: mask one token at a time
        # Sample 200 random positions for efficiency
        positions = np.random.choice(num_tokens, min(200, num_tokens), replace=False)
        total_log_prob = 0.0
        
        for pos in positions:
            masked_input = input_ids.clone()
            masked_input[0, pos] = mask_token_id
            
            with torch.no_grad():
                outputs = model(masked_input)
                logits = outputs.logits[0, pos]
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                true_token = input_ids[0, pos]
                total_log_prob += log_probs[true_token].item()
        
        avg_loss_nats = -total_log_prob / len(positions)
    
    # Compute full tokenization for bytes/token
    full_enc = tokenizer(corpus, truncation=True, max_length=100000)
    full_num_tokens = len(full_enc.input_ids)
    
    bits_per_token = avg_loss_nats / math.log(2)
    avg_bytes_per_token = total_bytes / full_num_tokens
    bpb = bits_per_token / avg_bytes_per_token
    perplexity = math.exp(min(avg_loss_nats, 100))
    
    del model
    torch.cuda.empty_cache()
    
    return {
        "BPB": bpb,
        "bits_per_token": bits_per_token,
        "avg_bytes_per_token": avg_bytes_per_token,
        "perplexity": perplexity,
        "num_tokens": full_num_tokens,
    }


# Run evaluations
for i, (model_id, mtype, params, method, vocab_size) in enumerate(MODELS):
    print(f"\n[{i+1}/{len(MODELS)}] {model_id} (vocab={vocab_size}, method={method})")
    
    log2_vocab = math.log2(vocab_size) if vocab_size > 0 else 0
    
    try:
        t_start = time.time()
        
        if method == "causal":
            result = compute_bpb_causal(model_id, corpus_text, total_bytes)
        elif method == "seq2seq":
            result = compute_bpb_seq2seq(model_id, corpus_text, total_bytes)
        elif method in ("masked", "protein", "dna"):
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
        torch.cuda.empty_cache()
        row = {
            "model_name": model_id, "vocab_size": vocab_size, "log2_vocab": round(log2_vocab, 3),
            "BPB": "", "bits_per_token": "", "avg_bytes_per_token": "", "perplexity": "",
            "model_type": mtype, "status": "oom", "error_msg": "CUDA OOM",
        }
        results.append(row)
        print(f"  OOM: Skipping (insufficient VRAM)")
        
    except Exception as e:
        torch.cuda.empty_cache()
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
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "vocab_size", "log2_vocab", "BPB",
                                                "bits_per_token", "avg_bytes_per_token", "perplexity",
                                                "model_type", "status", "error_msg"])
        writer.writeheader()
        writer.writerows(results)

print(f"\n{'=' * 70}")
print(f"STEP 3 COMPLETE")
successes = sum(1 for r in results if r["status"] == "success")
failures = sum(1 for r in results if r["status"] == "error")
ooms = sum(1 for r in results if r["status"] == "oom")
print(f"Success: {successes}, Error: {failures}, OOM: {ooms}")
print(f"Results saved to: {results_path}")
print(f"{'=' * 70}")

##############################################################################
# STEP 4 — STATISTICAL ANALYSIS
##############################################################################
print("\n" + "=" * 70)
print("STEP 4: Statistical Analysis")
print("=" * 70)

import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
import ruptures

# Load successful results
df = pd.read_csv(results_path)
df_ok = df[df["status"] == "success"].copy()
df_ok["BPB"] = pd.to_numeric(df_ok["BPB"], errors="coerce")
df_ok["log2_vocab"] = pd.to_numeric(df_ok["log2_vocab"], errors="coerce")
df_ok["bits_per_token"] = pd.to_numeric(df_ok["bits_per_token"], errors="coerce")
df_ok["avg_bytes_per_token"] = pd.to_numeric(df_ok["avg_bytes_per_token"], errors="coerce")
df_ok["perplexity"] = pd.to_numeric(df_ok["perplexity"], errors="coerce")
df_ok["vocab_size"] = pd.to_numeric(df_ok["vocab_size"], errors="coerce")
df_ok = df_ok.dropna(subset=["BPB", "log2_vocab"])

print(f"Analyzing {len(df_ok)} successful models")
print(f"Vocab range: {df_ok['vocab_size'].min():.0f} - {df_ok['vocab_size'].max():.0f}")
print(f"log2(vocab) range: {df_ok['log2_vocab'].min():.1f} - {df_ok['log2_vocab'].max():.1f}")

# A. LINEAR FIT TEST
print("\n--- A. LINEAR FIT TEST (Null Hypothesis: BPB ~ β₀ + β₁·log₂(V)) ---")
slope, intercept, r_value, p_value, std_err = stats.linregress(df_ok["log2_vocab"], df_ok["BPB"])
r_squared = r_value ** 2
print(f"  β₀ (intercept) = {intercept:.4f}")
print(f"  β₁ (slope)     = {slope:.6f}")
print(f"  R²             = {r_squared:.4f}")
print(f"  p-value(β₁)    = {p_value:.6f}")
print(f"  std_err(β₁)    = {std_err:.6f}")
if p_value < 0.05:
    print(f"  → β₁ IS significantly different from zero (p < 0.05)")
    print(f"    NULL HYPOTHESIS (linear scaling) receives support")
else:
    print(f"  → β₁ is NOT significantly different from zero (p ≥ 0.05)")
    print(f"    PLATEAU HYPOTHESIS receives support")

# B. PLATEAU DETECTION
print("\n--- B. PLATEAU DETECTION ---")
try:
    # Sort by log2_vocab for changepoint detection
    df_sorted = df_ok.sort_values("log2_vocab").reset_index(drop=True)
    signal = df_sorted["BPB"].values
    
    if len(signal) >= 5:
        # Use ruptures for changepoint detection
        algo = ruptures.Pelt(model="rbf", min_size=3).fit(signal.reshape(-1, 1))
        breakpoints = algo.predict(pen=1)
        
        if breakpoints and breakpoints[0] < len(signal):
            bp_idx = breakpoints[0] if breakpoints[0] < len(signal) else len(signal) - 1
            v_plateau = df_sorted.iloc[bp_idx]["vocab_size"] if bp_idx < len(df_sorted) else "N/A"
            log2_plateau = df_sorted.iloc[bp_idx]["log2_vocab"] if bp_idx < len(df_sorted) else "N/A"
            
            bpb_before = signal[:bp_idx]
            bpb_after = signal[bp_idx:]
            var_before = np.var(bpb_before) if len(bpb_before) > 1 else float('nan')
            var_after = np.var(bpb_after) if len(bpb_after) > 1 else float('nan')
            
            print(f"  Changepoint detected at index {bp_idx}")
            print(f"  V_plateau ≈ {v_plateau} (log₂ ≈ {log2_plateau})")
            print(f"  BPB variance before plateau: {var_before:.6f}")
            print(f"  BPB variance after plateau:  {var_after:.6f}")
        else:
            print(f"  No clear changepoint detected")
            v_plateau = "N/A"
    else:
        print(f"  Too few data points for changepoint detection")
        v_plateau = "N/A"
except Exception as e:
    print(f"  Changepoint detection failed: {e}")
    v_plateau = "N/A"

# C. BITS/EVENT ESTIMATE
print("\n--- C. BITS/EVENT ESTIMATE ---")
df_ok["bits_per_event"] = df_ok["BPB"] * df_ok["avg_bytes_per_token"]
bpe_mean = df_ok["bits_per_event"].mean()
bpe_q25 = df_ok["bits_per_event"].quantile(0.25)
bpe_q75 = df_ok["bits_per_event"].quantile(0.75)
bpe_median = df_ok["bits_per_event"].median()
print(f"  Mean bits/event:   {bpe_mean:.4f}")
print(f"  Median bits/event: {bpe_median:.4f}")
print(f"  IQR: [{bpe_q25:.4f}, {bpe_q75:.4f}]")
in_band = df_ok[(df_ok["bits_per_event"] >= 3) & (df_ok["bits_per_event"] <= 5.5)]
print(f"  Models in [3, 5.5] band: {len(in_band)}/{len(df_ok)} ({100*len(in_band)/len(df_ok):.1f}%)")

# D. RATE-DISTORTION BOUND
print("\n--- D. RATE-DISTORTION BOUND COMPARISON ---")
def h_binary(eps):
    """Binary entropy function."""
    if eps <= 0 or eps >= 1:
        return 0
    return -eps * math.log2(eps) - (1 - eps) * math.log2(1 - eps)

df_ok["epsilon"] = 1 - (1 / df_ok["perplexity"])
df_ok["epsilon"] = df_ok["epsilon"].clip(0.001, 0.999)

rd_residuals = []
for idx, row in df_ok.iterrows():
    eps = row["epsilon"]
    V = row["vocab_size"]
    H_b = h_binary(eps)
    R_M = math.log2(V) - H_b - eps * math.log2(max(V - 1, 1))
    R_M = max(R_M, 0)  # R(ε) can't be negative
    residual = row["BPB"] - R_M
    rd_residuals.append({"model_name": row["model_name"], "R_M": R_M, "residual": residual})
    df_ok.at[idx, "R_M"] = R_M
    df_ok.at[idx, "rd_residual"] = residual

print(f"  Mean residual (BPB - R_M(ε)): {np.mean([r['residual'] for r in rd_residuals]):.4f}")
print(f"  Std residual:                  {np.std([r['residual'] for r in rd_residuals]):.4f}")

# Save statistical summary
summary_path = os.path.join(RESULTS_DIR, "statistical_summary.txt")
with open(summary_path, "w") as f:
    f.write("THROUGHPUT CONSTRAINT — STATISTICAL SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Models analyzed: {len(df_ok)}\n")
    f.write(f"Vocab size range: {df_ok['vocab_size'].min():.0f} - {df_ok['vocab_size'].max():.0f}\n")
    f.write(f"log₂(vocab) range: {df_ok['log2_vocab'].min():.1f} - {df_ok['log2_vocab'].max():.1f}\n\n")
    
    f.write("A. LINEAR FIT TEST\n")
    f.write(f"   BPB ~ {intercept:.4f} + {slope:.6f} · log₂(V)\n")
    f.write(f"   R² = {r_squared:.4f}\n")
    f.write(f"   p-value(β₁) = {p_value:.6f}\n")
    f.write(f"   std_err(β₁) = {std_err:.6f}\n\n")
    
    f.write("B. PLATEAU DETECTION\n")
    f.write(f"   V_plateau ≈ {v_plateau}\n\n")
    
    f.write("C. BITS/EVENT\n")
    f.write(f"   Mean: {bpe_mean:.4f}\n")
    f.write(f"   Median: {bpe_median:.4f}\n")
    f.write(f"   IQR: [{bpe_q25:.4f}, {bpe_q75:.4f}]\n")
    f.write(f"   In [3, 5.5] band: {len(in_band)}/{len(df_ok)}\n\n")
    
    f.write("D. RATE-DISTORTION\n")
    f.write(f"   Mean residual: {np.mean([r['residual'] for r in rd_residuals]):.4f}\n")
    f.write(f"   Std residual: {np.std([r['residual'] for r in rd_residuals]):.4f}\n")

print(f"\nStatistical summary saved to: {summary_path}")

# Save augmented results
df_ok.to_csv(os.path.join(RESULTS_DIR, "bpb_results_augmented.csv"), index=False)

##############################################################################
# STEP 5 — VISUALIZATIONS
##############################################################################
print("\n" + "=" * 70)
print("STEP 5: Generating Visualizations")
print("=" * 70)

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import statsmodels.api as sm

# CHART 1: BPB vs log2(vocab_size)
print("  Generating Chart 1: BPB vs log₂(Vocab Size)...")

fig1 = go.Figure()

# Color by model type
colors = {"LLM": "#1f77b4", "translation": "#ff7f0e", "protein": "#2ca02c"}
for mtype in df_ok["model_type"].unique():
    subset = df_ok[df_ok["model_type"] == mtype]
    fig1.add_trace(go.Scatter(
        x=subset["log2_vocab"], y=subset["BPB"],
        mode="markers", name=mtype,
        marker=dict(size=10, color=colors.get(mtype, "#888")),
        text=subset["model_name"],
        hovertemplate="%{text}<br>log₂(V)=%{x:.1f}<br>BPB=%{y:.4f}<extra></extra>"
    ))

# Linear regression line
x_range = np.linspace(df_ok["log2_vocab"].min() - 0.5, df_ok["log2_vocab"].max() + 0.5, 100)
y_fit = intercept + slope * x_range
fig1.add_trace(go.Scatter(
    x=x_range, y=y_fit, mode="lines", name=f"Linear fit (β₁={slope:.4f}, p={p_value:.4f})",
    line=dict(color="red", dash="dash")
))

# LOWESS smoothed curve
try:
    lowess = sm.nonparametric.lowess(df_ok["BPB"].values, df_ok["log2_vocab"].values, frac=0.4)
    fig1.add_trace(go.Scatter(
        x=lowess[:, 0], y=lowess[:, 1], mode="lines", name="LOWESS trend",
        line=dict(color="green", width=2)
    ))
except:
    pass

# Mean BPB line
mean_bpb = df_ok["BPB"].mean()
fig1.add_hline(y=mean_bpb, line_dash="dot", line_color="gray",
               annotation_text=f"Mean BPB = {mean_bpb:.3f}")

fig1.update_layout(
    title="BPB vs log₂(Vocab Size): Plateau or Linear Scaling?",
    xaxis_title="log₂(Vocab Size)",
    yaxis_title="Bits Per Byte (BPB)",
    width=1200, height=700,
    template="plotly_white",
    font=dict(size=14)
)

# Custom x-axis tick labels showing actual vocab sizes
tick_vals = [8, 10, 12, 14, 15, 16, 17, 18, 20]
tick_texts = [f"{2**v:,.0f}" for v in tick_vals]
fig1.update_xaxes(tickvals=tick_vals, ticktext=tick_texts)

fig1.write_image(os.path.join(CHARTS_DIR, "chart1_bpb_vs_vocab.png"), scale=2)
print("    Saved chart1_bpb_vs_vocab.png")

# CHART 2: bits_per_event distribution
print("  Generating Chart 2: Bits per Event Distribution...")

fig2 = go.Figure()

fig2.add_trace(go.Histogram(
    x=df_ok["bits_per_event"], nbinsx=30,
    marker_color="#1f77b4", opacity=0.7,
    name="bits/event"
))

# Theoretical convergence band [3, 5.5]
fig2.add_vrect(x0=3, x1=5.5, fillcolor="green", opacity=0.15,
               annotation_text="Theoretical band [3, 5.5]", annotation_position="top")

# Ribosome prediction
fig2.add_vline(x=4.39, line_dash="dash", line_color="red",
               annotation_text="Ribosome: 4.39 bits")

fig2.update_layout(
    title="Effective bits/event: Do LLMs Converge to the Biological Band?",
    xaxis_title="bits per event",
    yaxis_title="Count",
    width=1000, height=600,
    template="plotly_white",
    font=dict(size=14)
)

fig2.write_image(os.path.join(CHARTS_DIR, "chart2_bits_per_event.png"), scale=2)
print("    Saved chart2_bits_per_event.png")

# CHART 3: avg_bytes_per_token vs log2(vocab_size)
print("  Generating Chart 3: Bytes/Token vs Vocab Size...")

fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=df_ok["log2_vocab"], y=df_ok["avg_bytes_per_token"],
    mode="markers", name="Models",
    marker=dict(size=10, color="#1f77b4"),
    text=df_ok["model_name"],
    hovertemplate="%{text}<br>log₂(V)=%{x:.1f}<br>bytes/tok=%{y:.2f}<extra></extra>"
))

# Log fit: avg_bytes ~ a + b * log2(V)
try:
    slope_bt, intercept_bt, r_bt, p_bt, se_bt = stats.linregress(
        df_ok["log2_vocab"], df_ok["avg_bytes_per_token"])
    x_fit = np.linspace(df_ok["log2_vocab"].min() - 0.5, df_ok["log2_vocab"].max() + 0.5, 100)
    y_fit_bt = intercept_bt + slope_bt * x_fit
    fig3.add_trace(go.Scatter(
        x=x_fit, y=y_fit_bt, mode="lines",
        name=f"Linear fit (slope={slope_bt:.3f})",
        line=dict(color="red", dash="dash")
    ))
except:
    pass

fig3.update_layout(
    title="Tokenization Efficiency: Bytes/Token vs Vocab Size",
    xaxis_title="log₂(Vocab Size)",
    yaxis_title="Average Bytes per Token",
    width=1000, height=600,
    template="plotly_white",
    font=dict(size=14)
)
fig3.update_xaxes(tickvals=tick_vals, ticktext=tick_texts)

fig3.write_image(os.path.join(CHARTS_DIR, "chart3_bytes_per_token.png"), scale=2)
print("    Saved chart3_bytes_per_token.png")

# CHART 4: Observed BPB vs Rate-Distortion Bound
print("  Generating Chart 4: BPB vs R_M(ε)...")

fig4 = go.Figure()

df_ok_rd = df_ok.dropna(subset=["R_M"])

fig4.add_trace(go.Scatter(
    x=df_ok_rd["R_M"], y=df_ok_rd["BPB"],
    mode="markers",
    marker=dict(size=10, color=df_ok_rd["log2_vocab"], colorscale="Viridis",
                colorbar=dict(title="log₂(V)"), showscale=True),
    text=df_ok_rd["model_name"],
    hovertemplate="%{text}<br>R_M(ε)=%{x:.3f}<br>BPB=%{y:.4f}<extra></extra>"
))

# Identity line y=x
rd_range = [0, max(df_ok_rd["R_M"].max(), df_ok_rd["BPB"].max()) * 1.1]
fig4.add_trace(go.Scatter(
    x=rd_range, y=rd_range, mode="lines", name="y = x (theory = observation)",
    line=dict(color="red", dash="dash")
))

fig4.update_layout(
    title="Observed BPB vs Rate-Distortion Prediction R_M(ε)",
    xaxis_title="R_M(ε) — Rate-Distortion Bound",
    yaxis_title="Observed BPB",
    width=1000, height=700,
    template="plotly_white",
    font=dict(size=14)
)

fig4.write_image(os.path.join(CHARTS_DIR, "chart4_rd_vs_observed.png"), scale=2)
print("    Saved chart4_rd_vs_observed.png")

print("\nAll charts generated!")

##############################################################################
# FINAL SUMMARY
##############################################################################
print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
print(f"Models evaluated: {successes} success, {failures} error, {ooms} OOM")
print(f"Linear fit: β₁ = {slope:.6f}, p = {p_value:.6f}")
print(f"Mean BPB: {mean_bpb:.4f}")
print(f"Bits/event IQR: [{bpe_q25:.4f}, {bpe_q75:.4f}]")
print(f"\nOutput files:")
print(f"  data/reference_corpus.txt")
print(f"  results/model_catalog.csv")
print(f"  results/bpb_results.csv")
print(f"  results/bpb_results_augmented.csv")
print(f"  results/statistical_summary.txt")
print(f"  charts/chart1_bpb_vs_vocab.png")
print(f"  charts/chart2_bits_per_event.png")
print(f"  charts/chart3_bytes_per_token.png")
print(f"  charts/chart4_rd_vs_observed.png")
