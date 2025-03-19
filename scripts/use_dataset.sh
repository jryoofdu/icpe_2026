#!/usr/bin/env bash

./benchmark.py \
    --batching paged-attn \
    --block_size 16 \
    --swap_policy eager \
    --prompt_count 100 \
    --cluster ./data/clusters/1_a100/h1.json \
    --dataset_json_path ./data/llmb_ver.json \
    --qps 50 \
    --max_parallem_sum 100 \
    --verbose none \
    --psla ./data/psla/llama-7b.json