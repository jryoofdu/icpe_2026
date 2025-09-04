# Group 25
## a. System Requriement
- OS: Ubuntu 22.04 (tested), 
- CPU: 13th Gen Intel(R) Core(TM) i7-1355U (1.70 GHz) x86_64 processor or Intel® Xeon® @ 2.30 GHz, x86\_64 (8 vCPUs, HT enabled)
- RAM: >= 8 GB
- GPU: None (CUDA unavailable)
- Python: 3.11.13
- Conda: 24.9.2
- Git: 2.43.0
- Dependencies: Listed in requirements.txt

## b. Setup Instructions
- Estimate time to finish all 4 steps (10 Minutes)
 ```shell
 # Step 1: Clone the repository
$ git clone https://github.com/JRYOO-FDU-CAPSTONE/Group-25-TokenSim.git
# Step 2: Update the submodule
$ git submodule update --init --recursive
# Step 3: Create and activate the virtual environment
$ conda create -n tokensim python=3.11
$ conda activate tokensim
# Step 4: Install the depencies
$ pip install -r requirements.txt
```
## c. How to Reproduce Results
```shell
# To generate Figure 1 (Cache Size vs Latency) , Figure 2 (Cache Size vs Throughput(tokens/sec)) and Table 1 (Latency and throughput summary by cache size) (Estimated Time to run = 5 min)
python ./scripts/figure1and2.py --config ./scripts/figure1and2.yaml --outdir ./results/benchmark
```
```shell
# To generate Figure 3 (Eviction Policy vs Cache Hit Rate(%)) , Figure 4 (Eviction Policy vs Average Token Generation Time(ms)),  Table 2 (Cache hit rates across eviction polies) and Table 3(Token genrateion item comparison by eviction policy) (Estimated Time to run = 5 min)
python ./scripts/figure3and4.py --config ./scripts/figure3and4.yaml --outdir ./results/benchmark
```
```shell
# To generate Figure 5 (Cache Architecture vs Latency(ms)) , Figure 6 (Cache Architecture vs Throughput (tokens/sec)),  Table 4 (Comparision of isolated vs shared cache in concurrent environment) and Table 5(Model-specific performance under different cache configurations) (Estimated Time to run = 5 min)
python ./scripts/figure5and6.py --config ./scripts/figure5and6.yaml --outdir ./results/benchmark
```
```shell
# To generate Figure 7a(Request Latency across different models atvarying session counts (QPS).) (Estimated Time to run = 5 min)
python ./scripts/figure7a.py --config ./scripts/figure7a.yaml --outdir ./results/benchmark
```
```shell
# To generate Figure 7b(Request Latency(ms) across different datasetsat varying session counts (QPS).) and Table 6 (Latency data per session count and dataset) (Estimated Time to run = 5 min)
python ./scripts/figure7b.py --config ./scripts/figure7b.yaml --outdir ./results/benchmark
```
```shell
# To generate Figure 8a(Throughput (in tokens/sec) across different models at varying session counts (QPS).) (Estimated Time to run = 5 min)
python ./scripts/figure8a.py --config ./scripts/figure8a.yaml --outdir ./results/benchmark
```
```shell
# To generate Figure 8b(Throughput (in tokens/sec) across different datasets at varying session counts (QPS).) (Estimated Time to run = 5 min)
python ./scripts/figure8b.py --config ./scripts/figure8b.yaml --outdir ./results/benchmark
```
```shell
# To generate Figure 9(Token per Session vs Latency(ms)), Figure 10(Token per Session vs Throughtput(tokens/sec)), Figure 11(Token per Session vs Cache Hit Rate(%) and Table 7, Table 8 and Table 9) (Estimated Time to run = 5 min)
python ./scripts/figure910and11.py --config ./scripts/figure910and11.yaml --outdir ./results/benchmark
```
```shell
# To generate Figure 12(Model Scalability Comparison: Stacked area chart of latency for all models)) (Estimated Time to run = 5 min)
python ./scripts/figure12.py --config ./scripts/figure12.yaml --outdir ./results/benchmark
```
```shell
# To generate Figure 13 to Figure 16 and Table 11 to Table 15 (Estimated Time to run = 1 min)
python ./scripts/plot_latency.py --results-root ./results/benchmark/dataset_masuqur --output-dir ./results/stage_latency_plots
  ```

## d. Validataion Checklist
```shell
# Test 1: Confirm that Figure 1, Figure 2 and Table 1 are generated successfully. 
python ./scripts/figure1and2.py --config ./scripts/figure1and2.yaml --outdir ./results/benchmark
# Expected Outout:
[✓] Saved: results/benchmark/figure/combined/figure_combined_latency.pdf
[✓] Saved: results/benchmark/figure/combined/figure_combined_throughput.pdf
[✓] Combined CSV written: results/benchmark/csv/benchmark_summary.csv
```
```shell
# Test 2: Confirm that Figure 3, Figure 4, Table 2 and Table 3are generated successfully. 
python ./scripts/figure3and4.py --config ./scripts/figure3and4.yaml --outdir ./results/benchmark
# Expected Outout:
[✓] Saved: results/benchmark/figure/figure3_combined_hit_rate.pdf
[✓] Saved: results/benchmark/figure/figure4_combined_token_gen.pdf
[✓] Saved: results/benchmark/csv/cache_hit_rates_by_eviction.csv
[✓] Saved: results/benchmark/csv/token_generation_by_eviction.csv
```
```shell
# Test 3: Confirm that Figure 5, Figure 6, Table 4 and Table 5 are generated successfully. 
python ./scripts/figure5and6.py --config ./scripts/figure5and6.yaml --outdir ./results/benchmark
# Expected Outout:
[✓] Saved: results/benchmark/figure/figure5_latency_shared_vs_isolated.pdf
[✓] Saved: results/benchmark/figure/figure6_throughput_shared_vs_isolated.pdf
[✓] Saved: results/benchmark/csv/shared_vs_isolated_summary.csv
[✓] Finished: Plots and CSV generated for shared vs. isolated analysis.
```
```shell
# Test 4: Confirm that Figure 7b and Table 6 are generated successfully. 
python ./scripts/figure7b.py --config ./scripts/figure5and6.yaml --outdir ./results/benchmark

# Expected Outout:
✅ Saved bar chart to: results/benchmark/figure/figure7b.pdf
✅ Saved latency data CSV to: results/benchmark/figure/Table6latency_data.csv
```
```shell
# Test 5: Confirm that Figure 13 to Figure 16 and Table 11 to Table 15  are generated successfully. 
python ./scripts/plot_latency.py --results-root ./results/benchmark/dataset_masuqur --output-dir ./results/stage_latency_plots
# Expected Outout:
Table 11: Median (p50) latency per stage (ms)

|                                |   preprocessing |   inference |   cache_access |   postprocessing |
|:-------------------------------|----------------:|------------:|---------------:|-----------------:|
| ('bookcorpus', 'internlm2-7b') |               0 |        0.81 |           0.01 |             0.09 |
| ('bookcorpus', 'llama2-7b')    |               0 |        0.72 |           0.01 |             0.09 |
| ('bookcorpus', 'mistral7b')    |               0 |        0.84 |           0.01 |             0.09 |
| ('longbench', 'internlm2-7b')  |               0 |        1    |           0.56 |             0.42 |
| ('longbench', 'llama2-7b')     |               0 |        0.88 |           0.65 |             0.42 |
| ('longbench', 'mistral7b')     |               0 |        0.97 |           0.66 |             0.38 |
| ('needle', 'internlm2-7b')     |               0 |        0.86 |           0.01 |             0.08 |
| ('needle', 'llama2-7b')        |               0 |        0.81 |           0.01 |             0.1  |
| ('needle', 'mistral7b')        |               0 |        0.9  |           0.01 |             0.11 |
| ('sharegpt', 'internlm2-7b')   |               0 |        0.84 |           0.02 |             0.12 |
| ('sharegpt', 'llama2-7b')      |               0 |        0.94 |           0.02 |             0.12 |
| ('sharegpt', 'mistral7b')      |               0 |        0.81 |           0.02 |             0.11 |
| ('wikipedia', 'internlm2-7b')  |               0 |        0.92 |           0.04 |             0.1  |
| ('wikipedia', 'llama2-7b')     |               0 |        0.92 |           0.05 |             0.1  |
| ('wikipedia', 'mistral7b')     |               0 |        0.86 |           0.05 |             0.09 |

Table 12: Percentage contribution per stage (%)

|                                |   preprocessing |   inference |   cache_access |   postprocessing |
|:-------------------------------|----------------:|------------:|---------------:|-----------------:|
| ('bookcorpus', 'internlm2-7b') |               0 |     89.011  |       1.0989   |          9.89011 |
| ('bookcorpus', 'llama2-7b')    |               0 |     87.8049 |       1.21951  |         10.9756  |
| ('bookcorpus', 'mistral7b')    |               0 |     89.3617 |       1.06383  |          9.57447 |
| ('longbench', 'internlm2-7b')  |               0 |     50.5051 |      28.2828   |         21.2121  |
| ('longbench', 'llama2-7b')     |               0 |     45.1282 |      33.3333   |         21.5385  |
| ('longbench', 'mistral7b')     |               0 |     48.2587 |      32.8358   |         18.9055  |
| ('needle', 'internlm2-7b')     |               0 |     90.5263 |       1.05263  |          8.42105 |
| ('needle', 'llama2-7b')        |               0 |     88.0435 |       1.08696  |         10.8696  |
| ('needle', 'mistral7b')        |               0 |     88.2353 |       0.980392 |         10.7843  |
| ('sharegpt', 'internlm2-7b')   |               0 |     85.7143 |       2.04082  |         12.2449  |
| ('sharegpt', 'llama2-7b')      |               0 |     87.037  |       1.85185  |         11.1111  |
| ('sharegpt', 'mistral7b')      |               0 |     86.1702 |       2.12766  |         11.7021  |
| ('wikipedia', 'internlm2-7b')  |               0 |     86.7925 |       3.77358  |          9.43396 |
| ('wikipedia', 'llama2-7b')     |               0 |     85.9813 |       4.6729   |          9.34579 |
| ('wikipedia', 'mistral7b')     |               0 |     86      |       5        |          9       |

Table 13: Latency (p50 & max) per stage for llama2-7b across all datasets

|                                  |   p50_ms |   max_ms |
|:---------------------------------|---------:|---------:|
| ('bookcorpus', 'cache_access')   |     0.01 |     0.1  |
| ('bookcorpus', 'inference')      |     0.72 |     1.58 |
| ('bookcorpus', 'postprocessing') |     0.09 |     0.45 |
| ('bookcorpus', 'preprocessing')  |     0    |     0.03 |
| ('longbench', 'cache_access')    |     0.65 |     7.5  |
| ('longbench', 'inference')       |     0.88 |     1.68 |
| ('longbench', 'postprocessing')  |     0.42 |     0.54 |
| ('longbench', 'preprocessing')   |     0    |     0.04 |
| ('needle', 'cache_access')       |     0.01 |     0.09 |
| ('needle', 'inference')          |     0.81 |     1.62 |
| ('needle', 'postprocessing')     |     0.1  |     0.41 |
| ('needle', 'preprocessing')      |     0    |     0.03 |
| ('sharegpt', 'cache_access')     |     0.02 |     0.36 |
| ('sharegpt', 'inference')        |     0.94 |     1.65 |
| ('sharegpt', 'postprocessing')   |     0.12 |     0.54 |
| ('sharegpt', 'preprocessing')    |     0    |     0.04 |
| ('wikipedia', 'cache_access')    |     0.05 |     0.17 |
| ('wikipedia', 'inference')       |     0.92 |     1.65 |
| ('wikipedia', 'postprocessing')  |     0.1  |     0.5  |
| ('wikipedia', 'preprocessing')   |     0    |     0.03 |

Table 14: Std. dev. of p50 latency per stage (rows) × dataset (cols)

|                |   bookcorpus |   longbench |    needle |   sharegpt |   wikipedia |
|:---------------|-------------:|------------:|----------:|-----------:|------------:|
| preprocessing  |      0       |   0         | 0         |  0         |   0         |
| inference      |      0.06245 |   0.06245   | 0.0450925 |  0.0680686 |   0.034641  |
| cache_access   |      0       |   0.0550757 | 0         |  0         |   0.0057735 |
| postprocessing |      0       |   0.023094  | 0.0152753 |  0.0057735 |   0.0057735 |

Table 15: Cache access latency (p50 & max)

| dataset    | model        |   cache_p50 |   cache_max |
|:-----------|:-------------|------------:|------------:|
| bookcorpus | internlm2-7b |        0.01 |        0.03 |
| bookcorpus | llama2-7b    |        0.01 |        0.1  |
| bookcorpus | mistral7b    |        0.01 |        0.03 |
| longbench  | internlm2-7b |        0.56 |        6.36 |
| longbench  | llama2-7b    |        0.65 |        7.5  |
| longbench  | mistral7b    |        0.66 |        6.83 |
| needle     | internlm2-7b |        0.01 |        0.1  |
| needle     | llama2-7b    |        0.01 |        0.09 |
| needle     | mistral7b    |        0.01 |        0.08 |
| sharegpt   | internlm2-7b |        0.02 |        0.42 |
| sharegpt   | llama2-7b    |        0.02 |        0.36 |
| sharegpt   | mistral7b    |        0.02 |        0.41 |
| wikipedia  | internlm2-7b |        0.04 |        0.14 |
| wikipedia  | llama2-7b    |        0.05 |        0.17 |
| wikipedia  | mistral7b    |        0.05 |        0.22 |

✅ All tables & figures saved under `results/stage_latency_plots`
```

## e. Limitations
- No support for Windows envrionment 
- Script 'convert_all_datasets.py' need api key for huggiface portal
- For Longbench, trancate promt length to 2048
- All benchmarks were conducted on Ubuntu 25.04; results may vary on Windows/macOS.
- GPU-accelerated inference was not utilized in this setup.
