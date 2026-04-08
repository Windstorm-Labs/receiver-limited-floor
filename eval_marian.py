#!/usr/bin/env python3
"""
Evaluate a single MarianMT/seq2seq model for BPB measurement.
Designed to be called as subprocess for CUDA isolation.
Outputs JSON result to stdout.

Usage: python eval_marian.py <repo_id> <vocab_size>
"""

import gc
import json
import math
import os
import sys
import time

import torch
import numpy as np


def eval_seq2seq_model(repo_id: str, vocab_size: int, corpus_path: str) -> dict:
    """Evaluate a seq2seq model using teacher-forcing cross-entropy."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        repo_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    actual_vocab = tokenizer.vocab_size
    if actual_vocab != vocab_size and vocab_size > 0:
        vocab_size = actual_vocab  # Trust the loaded tokenizer

    # Load corpus
    with open(corpus_path, "r") as f:
        text = f.read()

    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=False)
    total_bytes = len(text.encode("utf-8"))
    total_tokens = len(tokens)

    if total_tokens == 0:
        return {"status": "error", "error_msg": "No tokens produced"}

    # For seq2seq: use decoder-only teacher forcing
    # Feed tokens as both encoder input and decoder target in chunks
    chunk_size = 256  # Smaller chunks for seq2seq
    stride = chunk_size // 2

    total_nll = 0.0
    total_count = 0

    with torch.no_grad():
        for start in range(0, len(tokens) - 1, stride):
            end = min(start + chunk_size, len(tokens))
            chunk_ids = tokens[start:end]

            input_ids = torch.tensor([chunk_ids], device=device)
            # Use same sequence as both encoder and decoder input (teacher forcing)
            decoder_input_ids = torch.tensor([chunk_ids], device=device)

            try:
                outputs = model(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    labels=input_ids,
                )
                loss = outputs.loss
                n_tokens_in_chunk = input_ids.shape[1]
                total_nll += loss.item() * n_tokens_in_chunk
                total_count += n_tokens_in_chunk
            except Exception as e:
                # Some models may have issues with certain chunk sizes
                continue

            if end >= len(tokens):
                break

    if total_count == 0:
        return {"status": "error", "error_msg": "No valid chunks evaluated"}

    avg_nll = total_nll / total_count  # nats per token
    bits_per_token = avg_nll / math.log(2)
    avg_bytes_per_token = total_bytes / total_tokens
    bpb = bits_per_token / avg_bytes_per_token
    perplexity = math.exp(min(avg_nll, 100))  # Cap to avoid overflow

    # Cleanup
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "status": "success",
        "vocab_size": vocab_size,
        "BPB": round(bpb, 6),
        "bits_per_token": round(bits_per_token, 4),
        "avg_bytes_per_token": round(avg_bytes_per_token, 4),
        "perplexity": round(perplexity, 4),
        "total_tokens": total_tokens,
        "total_bytes": total_bytes,
        "chunks_evaluated": total_count,
    }


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"status": "error", "error_msg": "Usage: eval_marian.py <repo_id> <vocab_size>"}))
        sys.exit(1)

    repo_id = sys.argv[1]
    vocab_size = int(sys.argv[2])
    corpus_path = os.path.expanduser("~/throughput_experiment/data/reference_corpus.txt")

    if not os.path.exists(corpus_path):
        print(json.dumps({"status": "error", "error_msg": f"Corpus not found: {corpus_path}"}))
        sys.exit(1)

    try:
        result = eval_seq2seq_model(repo_id, vocab_size, corpus_path)
        result["model_name"] = repo_id
        print(json.dumps(result))
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({
            "status": "error",
            "model_name": repo_id,
            "error_msg": "CUDA OOM"
        }))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "model_name": repo_id,
            "error_msg": str(e)[:500]
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
