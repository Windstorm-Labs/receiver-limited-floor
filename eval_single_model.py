#!/usr/bin/env python3
"""
Evaluate a single model's BPB. Runs in its own process to isolate CUDA errors.
Usage: python eval_single_model.py <model_id> <method> <vocab_size> <model_type> <corpus_path>
Outputs JSON to stdout on success, or error JSON on failure.
"""

import json
import math
import os
import sys
import warnings
import gc
import numpy as np
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

def reset_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def compute_bpb_causal(model_id, corpus, total_bytes, max_length=2048):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get actual vocab size from tokenizer
    actual_vocab = tokenizer.vocab_size
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    model.eval()
    
    # Check embedding size matches vocab
    if hasattr(model, 'get_input_embeddings'):
        emb = model.get_input_embeddings()
        emb_size = emb.weight.shape[0]
    else:
        emb_size = actual_vocab
    
    # Tokenize full corpus
    full_encodings = tokenizer(corpus, return_tensors="pt", truncation=False)
    input_ids = full_encodings.input_ids
    
    # Clamp token IDs to valid embedding range
    if input_ids.max().item() >= emb_size:
        input_ids = input_ids.clamp(max=emb_size - 1)
    
    full_num_tokens = input_ids.shape[1]
    avg_bytes_per_token = total_bytes / full_num_tokens
    
    # Evaluate up to 20K tokens
    max_eval_tokens = min(full_num_tokens, 20480)
    eval_ids = input_ids[:, :max_eval_tokens]
    
    stride = max_length
    total_loss = 0.0
    total_counted = 0
    
    for begin_loc in range(0, max_eval_tokens, stride):
        end_loc = min(begin_loc + max_length, max_eval_tokens)
        chunk_ids = eval_ids[:, begin_loc:end_loc].to(model.device)
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
    
    del model, full_encodings, eval_ids
    reset_cuda()
    
    return {"BPB": bpb, "bits_per_token": bits_per_token,
            "avg_bytes_per_token": avg_bytes_per_token,
            "perplexity": perplexity, "actual_vocab": actual_vocab}


def compute_bpb_seq2seq(model_id, corpus, total_bytes, max_length=512):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
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
    actual_vocab = tokenizer.vocab_size
    
    bits_per_token = avg_loss_nats / math.log(2)
    avg_bytes_per_token = total_bytes / num_tokens
    bpb = bits_per_token / avg_bytes_per_token
    perplexity = math.exp(min(avg_loss_nats, 100))
    
    del model
    reset_cuda()
    
    return {"BPB": bpb, "bits_per_token": bits_per_token,
            "avg_bytes_per_token": avg_bytes_per_token,
            "perplexity": perplexity, "actual_vocab": actual_vocab}


def compute_bpb_masked(model_id, corpus, total_bytes, max_length=512):
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    
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
    actual_vocab = tokenizer.vocab_size
    
    bits_per_token = avg_loss_nats / math.log(2)
    avg_bytes_per_token = total_bytes / full_num_tokens
    bpb = bits_per_token / avg_bytes_per_token
    perplexity = math.exp(min(avg_loss_nats, 100))
    
    del model
    reset_cuda()
    
    return {"BPB": bpb, "bits_per_token": bits_per_token,
            "avg_bytes_per_token": avg_bytes_per_token,
            "perplexity": perplexity, "actual_vocab": actual_vocab}


if __name__ == "__main__":
    model_id = sys.argv[1]
    method = sys.argv[2]
    vocab_size = int(sys.argv[3])
    model_type = sys.argv[4]
    corpus_path = sys.argv[5]
    
    with open(corpus_path) as f:
        corpus = f.read()
    total_bytes = len(corpus.encode("utf-8"))
    
    try:
        if method == "causal":
            result = compute_bpb_causal(model_id, corpus, total_bytes)
        elif method == "seq2seq":
            result = compute_bpb_seq2seq(model_id, corpus, total_bytes)
        elif method == "masked":
            result = compute_bpb_masked(model_id, corpus, total_bytes)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        result["status"] = "success"
        result["model_name"] = model_id
        result["vocab_size"] = vocab_size
        result["model_type"] = model_type
        print("RESULT:" + json.dumps(result))
        
    except torch.cuda.OutOfMemoryError:
        print("RESULT:" + json.dumps({"status": "oom", "model_name": model_id, 
              "vocab_size": vocab_size, "model_type": model_type, "error": "CUDA OOM"}))
    except Exception as e:
        print("RESULT:" + json.dumps({"status": "error", "model_name": model_id,
              "vocab_size": vocab_size, "model_type": model_type, "error": str(e)[:300]}))
