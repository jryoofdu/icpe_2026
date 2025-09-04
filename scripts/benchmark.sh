#!/usr/bin/env bash

./benchmark.py \
    --batching paged-attn \
    --block_size 64\
    --swap_policy eager \
    --prompt_count 25 \
    --prompt_lens_mean 100 \
    --generation_lens_mean 14 \
    --cluster ./data/clusters/8_a100/p2g6.json \
    --results_path ./results\
    --qps 10 \
    --eviction_policy lru \
    --psla ./data/psla/llama-7b-shared.json \
    --dataset_json_path ./dataset/sharegpt/converted/llama-2-7b.json \
  
