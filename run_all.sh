#!/bin/bash
# Run each model in its own process to avoid memory accumulation.
# Checks CSV for already-completed models and skips them.

cd ~/throughput_experiment
RESULTS="results/bpb_results.csv"

eval_model() {
    local model="$1" mtype="$2" method="$3" vocab="$4"
    
    # Check if already done
    if grep -q "^${model}," "$RESULTS" 2>/dev/null; then
        if grep "^${model}," "$RESULTS" | grep -q ",success,"; then
            echo "SKIP: $model (already done)"
            return 0
        fi
    fi
    
    echo ""
    echo ">>> $model (vocab=$vocab, method=$method)"
    python3 eval_one.py "$model" "$mtype" "$method" "$vocab"
    echo "    exit code: $?"
    
    # Brief pause to let GPU memory fully clear
    sleep 2
}

# Models ordered by vocab size for the sweep
eval_model "albert/albert-base-v2" "LLM" "masked" "30000"
eval_model "distilbert-base-uncased" "LLM" "masked" "30522"
eval_model "bert-base-uncased" "LLM" "masked" "30522"
eval_model "Helsinki-NLP/opus-mt-tc-big-en-ko" "translation" "seq2seq" "32001"
eval_model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" "LLM" "causal" "32000"
eval_model "openlm-research/open_llama_3b" "LLM" "causal" "32000"
eval_model "google/flan-t5-small" "translation" "seq2seq" "32100"
eval_model "Helsinki-NLP/opus-mt-en-tvl" "translation" "seq2seq" "38380"
eval_model "Helsinki-NLP/opus-mt-en-ho" "translation" "seq2seq" "42463"
eval_model "Helsinki-NLP/opus-mt-en-iso" "translation" "seq2seq" "45029"
eval_model "HuggingFaceTB/SmolLM-135M" "LLM" "causal" "49152"
# sshleifer/tiny-gpt2 REMOVED — d_model=2, not a real LM
eval_model "distilgpt2" "LLM" "causal" "50257"
eval_model "gpt2" "LLM" "causal" "50257"
eval_model "gpt2-medium" "LLM" "causal" "50257"
eval_model "gpt2-large" "LLM" "causal" "50257"
eval_model "microsoft/phi-1" "LLM" "causal" "50257"
eval_model "microsoft/phi-1_5" "LLM" "causal" "50257"
eval_model "microsoft/phi-2" "LLM" "causal" "50257"
eval_model "EleutherAI/pythia-14m" "LLM" "causal" "50254"
eval_model "EleutherAI/pythia-70m" "LLM" "causal" "50254"
eval_model "EleutherAI/pythia-160m" "LLM" "causal" "50254"
eval_model "EleutherAI/pythia-410m" "LLM" "causal" "50254"
eval_model "EleutherAI/pythia-1b" "LLM" "causal" "50254"
eval_model "EleutherAI/pythia-1.4b" "LLM" "causal" "50254"
eval_model "facebook/opt-125m" "LLM" "causal" "50265"
eval_model "facebook/opt-350m" "LLM" "causal" "50265"
eval_model "facebook/opt-1.3b" "LLM" "causal" "50265"
eval_model "cerebras/Cerebras-GPT-111M" "LLM" "causal" "50257"
eval_model "cerebras/Cerebras-GPT-256M" "LLM" "causal" "50257"
eval_model "allenai/OLMo-1B-hf" "LLM" "causal" "50280"
eval_model "roberta-base" "LLM" "masked" "50265"
eval_model "Helsinki-NLP/opus-mt-tc-big-en-pt" "translation" "seq2seq" "54776"
eval_model "Helsinki-NLP/opus-mt-en-de" "translation" "seq2seq" "58101"
eval_model "Helsinki-NLP/opus-mt-en-fr" "translation" "seq2seq" "59514"
eval_model "Helsinki-NLP/opus-mt-en-ru" "translation" "seq2seq" "62518"
eval_model "Helsinki-NLP/opus-mt-en-es" "translation" "seq2seq" "65001"
eval_model "Helsinki-NLP/opus-mt-en-it" "translation" "seq2seq" "80035"
eval_model "google/pegasus-xsum" "LLM" "seq2seq" "96103"
eval_model "stabilityai/stablelm-2-zephyr-1_6b" "LLM" "causal" "100289"
eval_model "Qwen/Qwen2.5-0.5B" "LLM" "causal" "151643"
eval_model "Qwen/Qwen2.5-1.5B" "LLM" "causal" "151643"
eval_model "bigscience/bloom-560m" "LLM" "causal" "250680"
eval_model "bigscience/bloom-1b1" "LLM" "causal" "250680"

echo ""
echo "========================================="
echo "ALL MODELS COMPLETE"
echo "========================================="
grep ",success," "$RESULTS" | wc -l
echo "successful evaluations"
