import subprocess, itertools

# 1) Define your parameter grid:
cache_sizes = ["1GB","2GB","4GB"]
evictions   = ["lru","lfu"]
architects  = ["shared","isolated"]
datasets    = ["wiki", "squad", "openwebtext"]    # whatever you have
models      = ["llama-7b","gpt2"]                 # etc.

# 2) For each combination, invoke benchmark.py:
for ds, mdl, size, pol, arch in itertools.product(datasets, models,
                                                    cache_sizes,
                                                    evictions,
                                                    architects):
    out = f"out/{mdl}_{ds}_{size}_{pol}_{arch}.json"
    cmd = [
      "python", "benchmark.py",
      "--dataset", ds,
      "--model", mdl,
      "--cache-size", size,
      "--eviction-policy", pol,
      "--cache-arch", arch,
      "--swap-policy", "eager",          # pick one swap policy per run
      "--output-metrics", out
    ]
    subprocess.run(cmd, check=True)

