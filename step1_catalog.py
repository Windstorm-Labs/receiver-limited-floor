#!/usr/bin/env python3
"""
Step 1: Build model catalog spanning diverse vocabulary sizes.
We select publicly available HuggingFace models that cover vocab sizes
from ~256 to ~256K, prioritizing small models that fit in 32GB VRAM.
"""

import csv
import os
import json
import math
import sys
from transformers import AutoTokenizer

# Models curated for maximum vocab size diversity.
# Format: (model_id, model_type, approx_params, notes)
# We include LLMs and translation models with diverse tokenizers.
CANDIDATE_MODELS = [
    # Very small vocab (character-level and byte-level models)
    ("google/canine-s", "LLM", "110M", "character-level, 256 codepoints"),
    ("google/byt5-small", "translation", "300M", "byte-level, 256 vocab"),
    
    # Small vocab (~1K-4K range)
    ("benjamin/gerpt2", "LLM", "117M", "German GPT-2, custom tokenizer"),
    ("Helsinki-NLP/opus-mt-en-de", "translation", "77M", "Marian MT en-de"),
    ("Helsinki-NLP/opus-mt-en-fr", "translation", "77M", "Marian MT en-fr"),
    ("Helsinki-NLP/opus-mt-en-es", "translation", "77M", "Marian MT en-es"),
    ("Helsinki-NLP/opus-mt-en-zh", "translation", "77M", "Marian MT en-zh"),
    ("Helsinki-NLP/opus-mt-en-ru", "translation", "77M", "Marian MT en-ru"),
    ("Helsinki-NLP/opus-mt-en-ar", "translation", "77M", "Marian MT en-ar"),
    ("Helsinki-NLP/opus-mt-en-ja", "translation", "77M", "Marian MT en-ja"),
    ("Helsinki-NLP/opus-mt-en-fi", "translation", "77M", "Marian MT en-fi"),
    ("Helsinki-NLP/opus-mt-en-it", "translation", "77M", "Marian MT en-it"),
    ("Helsinki-NLP/opus-mt-tc-big-en-pt", "translation", "230M", "Marian MT en-pt big"),
    
    # Medium vocab (~8K-32K range)
    ("sshleifer/tiny-gpt2", "LLM", "2M", "tiny GPT-2"),
    ("distilgpt2", "LLM", "82M", "distilled GPT-2"),
    ("gpt2", "LLM", "117M", "GPT-2 small"),
    ("gpt2-medium", "LLM", "345M", "GPT-2 medium"),
    ("gpt2-large", "LLM", "774M", "GPT-2 large"),
    ("EleutherAI/pythia-70m", "LLM", "70M", "Pythia 70M"),
    ("EleutherAI/pythia-160m", "LLM", "160M", "Pythia 160M"),
    ("EleutherAI/pythia-410m", "LLM", "410M", "Pythia 410M"),
    ("EleutherAI/pythia-1b", "LLM", "1B", "Pythia 1B"),
    ("EleutherAI/pythia-1.4b", "LLM", "1.4B", "Pythia 1.4B"),
    ("cerebras/Cerebras-GPT-111M", "LLM", "111M", "Cerebras GPT 111M"),
    ("cerebras/Cerebras-GPT-256M", "LLM", "256M", "Cerebras GPT 256M"),
    ("facebook/opt-125m", "LLM", "125M", "OPT 125M"),
    ("facebook/opt-350m", "LLM", "350M", "OPT 350M"),
    ("facebook/opt-1.3b", "LLM", "1.3B", "OPT 1.3B"),
    ("bigscience/bloom-560m", "LLM", "560M", "BLOOM 560M"),
    ("bigscience/bloom-1b1", "LLM", "1.1B", "BLOOM 1.1B"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "LLM", "1.1B", "TinyLlama 1.1B"),
    ("microsoft/phi-1", "LLM", "1.3B", "Phi-1"),
    ("microsoft/phi-1_5", "LLM", "1.3B", "Phi-1.5"),
    ("microsoft/phi-2", "LLM", "2.7B", "Phi-2"),
    ("Qwen/Qwen2.5-0.5B", "LLM", "500M", "Qwen2.5 0.5B"),
    ("Qwen/Qwen2.5-1.5B", "LLM", "1.5B", "Qwen2.5 1.5B"),
    ("google/gemma-2b", "LLM", "2B", "Gemma 2B"),
    ("stabilityai/stablelm-2-zephyr-1_6b", "LLM", "1.6B", "StableLM 2 1.6B"),
    ("01-ai/Yi-6B", "LLM", "6B", "Yi 6B"),
    
    # Large vocab (50K-64K range) 
    ("albert/albert-base-v2", "LLM", "11M", "ALBERT base"),
    ("google/t5-small", "translation", "60M", "T5 small"),
    ("google/flan-t5-small", "translation", "80M", "Flan-T5 small"),
    ("xlnet/xlnet-base-cased", "LLM", "110M", "XLNet base"),
    
    # Very large vocab (100K-256K range)
    ("cl100k_base_models/gpt-j-6b", "LLM", "6B", "GPT-J 6B, 100K vocab"),  # might not exist
    ("meta-llama/Llama-2-7b-hf", "LLM", "7B", "Llama 2 7B"),
    ("mistralai/Mistral-7B-v0.1", "LLM", "7B", "Mistral 7B"),
    ("allenai/OLMo-1B-hf", "LLM", "1B", "OLMo 1B"),
]

RESULTS_DIR = os.path.expanduser("~/throughput_experiment/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

catalog = []
errors = []

for model_id, model_type, param_count, notes in CANDIDATE_MODELS:
    print(f"  Loading tokenizer: {model_id}...", end=" ", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)
        # Some tokenizers report vocab_size differently
        actual_len = len(tokenizer)
        print(f"vocab={vocab_size} (len={actual_len})")
        
        catalog.append({
            "model_name": model_id,
            "vocab_size": vocab_size,
            "tokenizer_len": actual_len,
            "model_type": model_type,
            "quantization": "fp16",
            "param_count": param_count,
            "path": model_id,
            "notes": notes,
        })
    except Exception as e:
        err_msg = str(e)[:100]
        print(f"FAILED: {err_msg}")
        errors.append((model_id, err_msg))

# Sort by vocab size
catalog.sort(key=lambda x: x["vocab_size"])

# Save catalog
csv_path = os.path.join(RESULTS_DIR, "model_catalog.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model_name", "vocab_size", "tokenizer_len", 
                                            "model_type", "quantization", "param_count", "path", "notes"])
    writer.writeheader()
    writer.writerows(catalog)

print(f"\n=== CATALOG SUMMARY ===")
print(f"Successfully loaded: {len(catalog)} models")
print(f"Failed: {len(errors)}")
print(f"\nVocab size range: {catalog[0]['vocab_size']} - {catalog[-1]['vocab_size']}")
print(f"\nVocab size distribution:")
for m in catalog:
    print(f"  {m['vocab_size']:>8d}  (log2={math.log2(m['vocab_size']):.1f})  {m['model_name']}")

if errors:
    print(f"\nFailed models:")
    for model_id, err in errors:
        print(f"  {model_id}: {err}")

print(f"\nCatalog saved to: {csv_path}")
