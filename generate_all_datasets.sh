#!/usr/bin/env bash
set -euo pipefail

# Batch generation of TokenSim datasets for multiple models
# Includes slicing for BookCorpus (first 5k) and Wikipedia (first 1%)

SCRIPT=gendataset.py
mkdir -p dataset_masuqur/longbench
mkdir -p dataset_masuqur/needle
mkdir -p dataset_masuqur/sharegpt
mkdir -p dataset_masuqur/bookcorpus
mkdir -p dataset_masuqur/wikipedia

declare -A models=(
  [llama2-7b]=meta-llama/Llama-2-7b
  [mistral7b]=mistralai/Mistral-7B-v0.3
  [internlm2-7b]=internlm/internlm2-7b
)

for label in "${!models[@]}"; do
  model="${models[$label]}"
  echo
  echo "=== Generating all datasets for $label ($model) ==="

  # 1) LongBench-v2
  python3 "$SCRIPT" \
    --model   "$model" \
    --dataset THUDM/LongBench-v2 \
    --split   train \
    --field   context \
    --out     dataset_masuqur/longbench/${label}_longbench_v2.json

  # 2) Needle-in-a-Haystack: texts, ATC needles, multi-needle reasoning
  python3 "$SCRIPT" \
    --model   "$model" \
    --dataset opencompass/NeedleBench \
    --config  en_haystack_texts \
    --split   test \
    --field   text \
    --out     dataset_masuqur/needle/${label}_needle_haystack.json

  python3 "$SCRIPT" \
    --model   "$model" \
    --dataset opencompass/NeedleBench \
    --config  atc_needles \
    --split   test \
    --field   English \
    --out     dataset_masuqur/needle/${label}_atc_needles.json

  python3 "$SCRIPT" \
    --model   "$model" \
    --dataset opencompass/NeedleBench \
    --config  multi_needle_reasoning_needle \
    --split   test \
    --field   question_translated \
    --out     dataset_masuqur/needle/${label}_multi_reasoning.json

  # 3) ShareGPT
  python3 "$SCRIPT" \
    --model   "$model" \
    --dataset RyokoAI/ShareGPT52K \
    --field   conversations \
    --out     dataset_masuqur/sharegpt/${label}_sharegpt.json

  # 4) BookCorpus (first 5000 examples)
  python3 "$SCRIPT" \
    --model   "$model" \
    --dataset bookcorpus \
    --split   train \
    --field   text \
    --out     dataset_masuqur/bookcorpus/${label}_bookcorpus.json

  # 5) Wikipedia Structured English (first 1%)
  python3 "$SCRIPT" \
    --model   "$model" \
    --dataset wikipedia \
    --config  20220301.en \
    --split   "train[:1%]" \
    --field   text \
    --out     dataset_masuqur/wikipedia/${label}_wikipedia_1pct.json

  echo "→ Done with $label"
done

echo "\nAll datasets generated under ./dataset/"

