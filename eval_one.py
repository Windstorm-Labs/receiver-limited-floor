#!/usr/bin/env python3
"""Evaluate a single model and append result to CSV. Run as a subprocess."""
import csv, math, os, sys, time, warnings, gc, json
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import torch

# Reset CUDA device at start to clear any lingering error state
if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.empty_cache()
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForMaskedLM
)

BASE_DIR = os.path.expanduser("~/throughput_experiment")
results_path = os.path.join(BASE_DIR, "results/bpb_results.csv")

model_id = sys.argv[1]
mtype = sys.argv[2]
method = sys.argv[3]
vocab_size = int(sys.argv[4])

with open(os.path.join(BASE_DIR, "data/reference_corpus.txt")) as f:
    corpus_text = f.read()
total_bytes = len(corpus_text.encode("utf-8"))
log2_vocab = math.log2(vocab_size)

FIELDNAMES = ["model_name","vocab_size","log2_vocab","BPB","bits_per_token",
              "avg_bytes_per_token","perplexity","model_type","status","error_msg"]

def compute_bpb_causal(model_id, corpus, total_bytes, max_length=2048):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()
    
    # Check if model is actually usable (some tiny test models have d_model=2)
    emb_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else \
              model.config.d_model if hasattr(model.config, 'd_model') else 0
    if emb_dim < 32:
        raise ValueError(f"Model embedding dim too small ({emb_dim}), not a real LM")
    
    full_enc = tokenizer(corpus, return_tensors="pt", truncation=False)
    full_num = full_enc.input_ids.shape[1]
    abpt = total_bytes / full_num
    max_eval = min(full_num, 20480)
    ids = full_enc.input_ids[:, :max_eval]
    
    # Clamp token IDs to valid range
    vocab_sz = model.config.vocab_size
    ids = ids.clamp(0, vocab_sz - 1)
    total_loss, total_counted = 0.0, 0
    for b in range(0, max_eval, max_length):
        e = min(b + max_length, max_eval)
        chunk = ids[:, b:e].to(model.device)
        tgt = chunk.clone()
        with torch.no_grad():
            out = model(chunk, labels=tgt)
        c = (tgt != -100).sum().item()
        total_loss += out.loss.item() * c
        total_counted += c
        if e >= max_eval: break
    loss = total_loss / total_counted
    bpt = loss / math.log(2)
    return {"BPB": bpt/abpt, "bits_per_token": bpt, "avg_bytes_per_token": abpt,
            "perplexity": math.exp(min(loss, 100)), "num_tokens": full_num}

def compute_bpb_seq2seq(model_id, corpus, total_bytes, max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "</s>"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()
    sents = [s.strip() for s in corpus.split("\n") if len(s.strip()) > 20][:500]
    total_loss, total_counted = 0.0, 0
    bs = 8
    for i in range(0, len(sents), bs):
        batch = sents[i:i+bs]
        inp = tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
        lab = tokenizer(text_target=batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
        iid = inp.input_ids.to(model.device)
        am = inp.attention_mask.to(model.device)
        lid = lab.input_ids.to(model.device)
        lid[lid == tokenizer.pad_token_id] = -100
        with torch.no_grad():
            out = model(input_ids=iid, attention_mask=am, labels=lid)
        c = (lid != -100).sum().item()
        total_loss += out.loss.item() * c
        total_counted += c
    loss = total_loss / total_counted
    fe = tokenizer(corpus, truncation=False)
    nt = len(fe.input_ids)
    bpt = loss / math.log(2)
    abpt = total_bytes / nt
    return {"BPB": bpt/abpt, "bits_per_token": bpt, "avg_bytes_per_token": abpt,
            "perplexity": math.exp(min(loss, 100)), "num_tokens": nt}

def compute_bpb_masked(model_id, corpus, total_bytes, max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()
    sents = [s.strip() for s in corpus.split("\n") if len(s.strip()) > 20][:200]
    sample = " ".join(sents)
    enc = tokenizer(sample, return_tensors="pt", truncation=True, max_length=max_length)
    iid = enc.input_ids.to(model.device)
    nt = iid.shape[1]
    mid = tokenizer.mask_token_id
    if mid is None:
        with torch.no_grad():
            out = model(iid, labels=iid)
            loss = out.loss.item()
    else:
        ns = min(300, nt - 2)
        pos = np.random.choice(range(1, nt - 1), ns, replace=False)
        tlp = 0.0
        for p in pos:
            mi = iid.clone()
            mi[0, p] = mid
            with torch.no_grad():
                out = model(mi)
                logits = out.logits[0, p].float()
                lp = torch.log_softmax(logits, dim=-1)
                tlp += lp[iid[0, p]].item()
        loss = -tlp / ns
    fe = tokenizer(corpus, truncation=False)
    fnt = len(fe.input_ids)
    bpt = loss / math.log(2)
    abpt = total_bytes / fnt
    return {"BPB": bpt/abpt, "bits_per_token": bpt, "avg_bytes_per_token": abpt,
            "perplexity": math.exp(min(loss, 100)), "num_tokens": fnt}

try:
    t0 = time.time()
    if method == "causal":
        r = compute_bpb_causal(model_id, corpus_text, total_bytes)
    elif method == "seq2seq":
        r = compute_bpb_seq2seq(model_id, corpus_text, total_bytes)
    elif method == "masked":
        r = compute_bpb_masked(model_id, corpus_text, total_bytes)
    elapsed = time.time() - t0
    
    row = {"model_name": model_id, "vocab_size": str(vocab_size),
           "log2_vocab": str(round(log2_vocab, 3)),
           "BPB": str(round(r["BPB"], 6)), "bits_per_token": str(round(r["bits_per_token"], 4)),
           "avg_bytes_per_token": str(round(r["avg_bytes_per_token"], 4)),
           "perplexity": str(round(r["perplexity"], 4)),
           "model_type": mtype, "status": "success", "error_msg": ""}
    
    print(f"OK|BPB={r['BPB']:.4f}|bpt={r['bits_per_token']:.2f}|abpt={r['avg_bytes_per_token']:.2f}|ppl={r['perplexity']:.2f}|{elapsed:.1f}s")

except torch.cuda.OutOfMemoryError:
    row = {"model_name": model_id, "vocab_size": str(vocab_size),
           "log2_vocab": str(round(log2_vocab, 3)),
           "BPB": "", "bits_per_token": "", "avg_bytes_per_token": "",
           "perplexity": "", "model_type": mtype, "status": "oom", "error_msg": "OOM"}
    print("OOM")

except Exception as e:
    row = {"model_name": model_id, "vocab_size": str(vocab_size),
           "log2_vocab": str(round(log2_vocab, 3)),
           "BPB": "", "bits_per_token": "", "avg_bytes_per_token": "",
           "perplexity": "", "model_type": mtype, "status": "error", "error_msg": str(e)[:200]}
    print(f"ERROR|{str(e)[:200]}")

# Append to CSV
needs_header = not os.path.exists(results_path) or os.path.getsize(results_path) == 0
with open(results_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if needs_header:
        writer.writeheader()
    writer.writerow(row)
