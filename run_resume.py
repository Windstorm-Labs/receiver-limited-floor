#!/usr/bin/env python3
"""
Throughput Constraint Tokenizer Sweep — Resumable runner.
Skips already-completed models, avoids 7B+ models to prevent OOM.
"""

import csv
import math
import os
import sys
import time
import traceback
import warnings
import gc
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch

BASE_DIR = os.path.expanduser("~/throughput_experiment")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
results_path = os.path.join(RESULTS_DIR, "bpb_results.csv")

# Load corpus
with open(os.path.join(BASE_DIR, "data/reference_corpus.txt")) as f:
    corpus_text = f.read()
total_bytes = len(corpus_text.encode("utf-8"))
print(f"Corpus: {total_bytes:,} bytes")

# Load existing results
existing = {}
if os.path.exists(results_path):
    with open(results_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] == "success":
                existing[row["model_name"]] = row
print(f"Already completed: {len(existing)} models")

FIELDNAMES = ["model_name", "vocab_size", "log2_vocab", "BPB", "bits_per_token",
              "avg_bytes_per_token", "perplexity", "model_type", "status", "error_msg"]

# Models — skip 7B+ to avoid RAM OOM. Focus on models <= 3B.
MODELS = [
    ("google/flan-t5-small", "translation", "80M", "seq2seq", 32100),
    ("Helsinki-NLP/opus-mt-tc-big-en-ko", "translation", "230M", "seq2seq", 32001),
    ("albert/albert-base-v2", "LLM", "11M", "masked", 30000),
    ("distilbert-base-uncased", "LLM", "66M", "masked", 30522),
    ("bert-base-uncased", "LLM", "110M", "masked", 30522),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "LLM", "1.1B", "causal", 32000),
    ("openlm-research/open_llama_3b", "LLM", "3B", "causal", 32000),
    # SKIP: ("mistralai/Mistral-7B-v0.1", "LLM", "7B", "causal", 32000),
    # SKIP: ("NousResearch/Llama-2-7b-hf", "LLM", "7B", "causal", 32000),
    ("Helsinki-NLP/opus-mt-en-tvl", "translation", "77M", "seq2seq", 38380),
    ("Helsinki-NLP/opus-mt-en-ho", "translation", "77M", "seq2seq", 42463),
    ("Helsinki-NLP/opus-mt-en-iso", "translation", "77M", "seq2seq", 45029),
    ("HuggingFaceTB/SmolLM-135M", "LLM", "135M", "causal", 49152),
    ("sshleifer/tiny-gpt2", "LLM", "2M", "causal", 50257),
    ("distilgpt2", "LLM", "82M", "causal", 50257),
    ("gpt2", "LLM", "117M", "causal", 50257),
    ("gpt2-medium", "LLM", "345M", "causal", 50257),
    ("gpt2-large", "LLM", "774M", "causal", 50257),
    ("microsoft/phi-1", "LLM", "1.3B", "causal", 50257),
    ("microsoft/phi-1_5", "LLM", "1.3B", "causal", 50257),
    ("microsoft/phi-2", "LLM", "2.7B", "causal", 50257),
    ("EleutherAI/pythia-14m", "LLM", "14M", "causal", 50254),
    ("EleutherAI/pythia-70m", "LLM", "70M", "causal", 50254),
    ("EleutherAI/pythia-160m", "LLM", "160M", "causal", 50254),
    ("EleutherAI/pythia-410m", "LLM", "410M", "causal", 50254),
    ("EleutherAI/pythia-1b", "LLM", "1B", "causal", 50254),
    ("EleutherAI/pythia-1.4b", "LLM", "1.4B", "causal", 50254),
    ("facebook/opt-125m", "LLM", "125M", "causal", 50265),
    ("facebook/opt-350m", "LLM", "350M", "causal", 50265),
    ("facebook/opt-1.3b", "LLM", "1.3B", "causal", 50265),
    ("cerebras/Cerebras-GPT-111M", "LLM", "111M", "causal", 50257),
    ("cerebras/Cerebras-GPT-256M", "LLM", "256M", "causal", 50257),
    ("allenai/OLMo-1B-hf", "LLM", "1B", "causal", 50280),
    ("roberta-base", "LLM", "125M", "masked", 50265),
    ("Helsinki-NLP/opus-mt-tc-big-en-pt", "translation", "230M", "seq2seq", 54776),
    ("Helsinki-NLP/opus-mt-en-de", "translation", "77M", "seq2seq", 58101),
    ("Helsinki-NLP/opus-mt-en-fr", "translation", "77M", "seq2seq", 59514),
    ("Helsinki-NLP/opus-mt-en-ru", "translation", "77M", "seq2seq", 62518),
    ("Helsinki-NLP/opus-mt-en-es", "translation", "77M", "seq2seq", 65001),
    # SKIP 7B: ("baichuan-inc/Baichuan-7B", ...),
    # SKIP 6B: ("01-ai/Yi-6B", ...),
    ("Helsinki-NLP/opus-mt-en-it", "translation", "77M", "seq2seq", 80035),
    ("google/pegasus-xsum", "LLM", "570M", "seq2seq", 96103),
    ("stabilityai/stablelm-2-zephyr-1_6b", "LLM", "1.6B", "causal", 100289),
    ("Qwen/Qwen2.5-0.5B", "LLM", "500M", "causal", 151643),
    ("Qwen/Qwen2.5-1.5B", "LLM", "1.5B", "causal", 151643),
    ("bigscience/bloom-560m", "LLM", "560M", "causal", 250680),
    ("bigscience/bloom-1b1", "LLM", "1.1B", "causal", 250680),
]

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM
)

def reset_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def compute_bpb_causal(model_id, corpus, total_bytes, max_length=2048):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    model.eval()
    
    full_encodings = tokenizer(corpus, return_tensors="pt", truncation=False)
    full_num_tokens = full_encodings.input_ids.shape[1]
    avg_bytes_per_token = total_bytes / full_num_tokens
    
    # Evaluate up to 20K tokens (enough for stable BPB estimate)
    max_eval_tokens = min(full_num_tokens, 20480)
    input_ids = full_encodings.input_ids[:, :max_eval_tokens]
    
    stride = max_length
    total_loss = 0.0
    total_counted = 0
    
    for begin_loc in range(0, max_eval_tokens, stride):
        end_loc = min(begin_loc + max_length, max_eval_tokens)
        chunk_ids = input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = chunk_ids.clone()
        
        with torch.no_grad():
            outputs = model(chunk_ids, labels=target_ids)
        
        counted = (target_ids != -100).sum().item()
        total_loss += outputs.loss.item() * counted
        total_counted += counted
        
        if end_loc >= max_eval_tokens:
            break
    
    avg_loss_nats = total_loss / total_counted if total_counted > 0 else float('inf')
    bits_per_token = avg_loss_nats / math.log(2)
    bpb = bits_per_token / avg_bytes_per_token
    perplexity = math.exp(min(avg_loss_nats, 100))
    
    del model, full_encodings, input_ids
    reset_cuda()
    
    return {"BPB": bpb, "bits_per_token": bits_per_token,
            "avg_bytes_per_token": avg_bytes_per_token,
            "perplexity": perplexity, "num_tokens": full_num_tokens}


def compute_bpb_seq2seq(model_id, corpus, total_bytes, max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "</s>"
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    model.eval()
    
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
    
    full_enc = tokenizer(corpus, truncation=False)
    num_tokens = len(full_enc.input_ids)
    
    bits_per_token = avg_loss_nats / math.log(2)
    avg_bytes_per_token = total_bytes / num_tokens
    bpb = bits_per_token / avg_bytes_per_token
    perplexity = math.exp(min(avg_loss_nats, 100))
    
    del model
    reset_cuda()
    
    return {"BPB": bpb, "bits_per_token": bits_per_token,
            "avg_bytes_per_token": avg_bytes_per_token,
            "perplexity": perplexity, "num_tokens": num_tokens}


def compute_bpb_masked(model_id, corpus, total_bytes, max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    model.eval()
    
    sentences = [s.strip() for s in corpus.split("\n") if len(s.strip()) > 20][:200]
    sample_text = " ".join(sentences)
    
    encodings = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids.to(model.device)
    num_tokens = input_ids.shape[1]
    
    mask_token_id = tokenizer.mask_token_id
    
    if mask_token_id is None:
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            avg_loss_nats = outputs.loss.item()
    else:
        n_samples = min(300, num_tokens - 2)
        positions = np.random.choice(range(1, num_tokens - 1), n_samples, replace=False)
        total_log_prob = 0.0
        for pos in positions:
            masked_input = input_ids.clone()
            masked_input[0, pos] = mask_token_id
            with torch.no_grad():
                outputs = model(masked_input)
                logits = outputs.logits[0, pos].float()
                log_probs = torch.log_softmax(logits, dim=-1)
                total_log_prob += log_probs[input_ids[0, pos]].item()
        avg_loss_nats = -total_log_prob / n_samples
    
    full_enc = tokenizer(corpus, truncation=False)
    full_num_tokens = len(full_enc.input_ids)
    
    bits_per_token = avg_loss_nats / math.log(2)
    avg_bytes_per_token = total_bytes / full_num_tokens
    bpb = bits_per_token / avg_bytes_per_token
    perplexity = math.exp(min(avg_loss_nats, 100))
    
    del model
    reset_cuda()
    
    return {"BPB": bpb, "bits_per_token": bits_per_token,
            "avg_bytes_per_token": avg_bytes_per_token,
            "perplexity": perplexity, "num_tokens": full_num_tokens}


# Collect all results (existing + new)
all_results = list(existing.values())

for i, (model_id, mtype, params, method, vocab_size) in enumerate(MODELS):
    if model_id in existing:
        print(f"[{i+1}/{len(MODELS)}] {model_id} — already done, skipping")
        continue
    
    print(f"\n[{i+1}/{len(MODELS)}] {model_id} (vocab={vocab_size}, method={method})")
    log2_vocab = math.log2(vocab_size)
    
    try:
        t_start = time.time()
        
        if method == "causal":
            result = compute_bpb_causal(model_id, corpus_text, total_bytes)
        elif method == "seq2seq":
            result = compute_bpb_seq2seq(model_id, corpus_text, total_bytes)
        elif method == "masked":
            result = compute_bpb_masked(model_id, corpus_text, total_bytes)
        
        elapsed = time.time() - t_start
        
        row = {
            "model_name": model_id, "vocab_size": str(vocab_size),
            "log2_vocab": str(round(log2_vocab, 3)),
            "BPB": str(round(result["BPB"], 6)),
            "bits_per_token": str(round(result["bits_per_token"], 4)),
            "avg_bytes_per_token": str(round(result["avg_bytes_per_token"], 4)),
            "perplexity": str(round(result["perplexity"], 4)),
            "model_type": mtype, "status": "success", "error_msg": "",
        }
        all_results.append(row)
        existing[model_id] = row
        
        print(f"  OK: BPB={result['BPB']:.4f}, bits/tok={result['bits_per_token']:.2f}, "
              f"bytes/tok={result['avg_bytes_per_token']:.2f}, PPL={result['perplexity']:.2f} "
              f"({elapsed:.1f}s)")
        
    except torch.cuda.OutOfMemoryError:
        reset_cuda()
        row = {"model_name": model_id, "vocab_size": str(vocab_size),
               "log2_vocab": str(round(log2_vocab, 3)),
               "BPB": "", "bits_per_token": "", "avg_bytes_per_token": "",
               "perplexity": "", "model_type": mtype, "status": "oom", "error_msg": "OOM"}
        all_results.append(row)
        print(f"  OOM")
        
    except Exception as e:
        reset_cuda()
        err = str(e)[:200]
        row = {"model_name": model_id, "vocab_size": str(vocab_size),
               "log2_vocab": str(round(log2_vocab, 3)),
               "BPB": "", "bits_per_token": "", "avg_bytes_per_token": "",
               "perplexity": "", "model_type": mtype, "status": "error", "error_msg": err}
        all_results.append(row)
        print(f"  ERROR: {err}")
        traceback.print_exc()
    
    # Save after each model
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_results)
    
    sys.stdout.flush()

successes = sum(1 for r in all_results if r["status"] == "success")
failures = sum(1 for r in all_results if r["status"] == "error")
ooms = sum(1 for r in all_results if r["status"] == "oom")
print(f"\n{'='*70}")
print(f"DONE: {successes} success, {failures} error, {ooms} OOM out of {len(all_results)} total")
print(f"{'='*70}")
