#!/usr/bin/env bash
set -e

OUT=cache_ablation.csv
echo "dataset,model,cache_size_GB,policy,shared,sessions,latency_ms,throughput_toks_per_s,hit_rate_pct,avg_token_ms" > $OUT

for MODEL in internlm2-7b mistral7b llama2-7b; do
  for DATASET in longbench needle sharegpt bookcorpus wikipedia; do
    DJSON="dataset_masuqur/${DATASET}/${MODEL}_${DATASET}.json"
    PJSON="data/psla/masuqur/${MODEL}_${DATASET}.json"

    for SIZE in 1 2 4; do
      for POLICY in lru lfu; do
        for SESS in 1 2 4; do
          for SH in "" "--shared_cache"; do
            SHARED_FLAG=$([ -z "$SH" ] && echo "isolated" || echo "shared")
            # pick cluster config with correct #sessions
            CLUSTER="data/clusters/1_a100/h${SESS}.json"

            # run one experiment
            RESULT=$(./scripts/benchmark.py \
              --cluster            $CLUSTER \
              --psla               $PJSON \
              --dataset_json_path  $DJSON \
              --cache_size         $SIZE \
              --eviction_policy    $POLICY \
              $SH \
              --sim_time           0 \
              --qps                50 \
              --batching           paged-attn \
              --distribution       uniform \
              --swap_policy        eager \
              --prompt_count       1000 \
              --prompt_lens_mean   55 \
              --prompt_lens_range  -1 \
              --generation_lens_mean 7 \
              --generation_lens_range -1 \
              --generation_lens_distribution uniform \
              --block_size         32 \
              --verbose            tqdm \
              --random_seed        42 \
              --program_id         0 \
              --results_path       /dev/null \
              --llm_compass        LLMCompass)

            # RESULT prints: latency_ms,throughput,hit_rate_pct,avg_token_ms
            echo "${DATASET},${MODEL},${SIZE},${POLICY},${SHARED_FLAG},${SESS},${RESULT}" \
              >> $OUT
          done
        done
      done
    done
  done
done

echo "🗄️  All runs complete — results in $OUT"

