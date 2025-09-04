#!/usr/bin/env python3
import os, sys, time, csv
from pathlib import Path

# ─── 0) Bootstrap PYTHONPATH ────────────────────────────────────────────────
ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
LLMCOMPASS = os.path.join(ROOT, "LLMCompass")
if LLMCOMPASS not in sys.path:
    sys.path.insert(0, LLMCOMPASS)

# ─── 1) Imports ─────────────────────────────────────────────────────────────
import simpy
from TokenSim.llm.llm_request import Request, g_time

# pointer to the currently–running Request
CURRENT_REQ = None

# ─── 2) Monkey–patch Request.__init__ to add our timing dicts ───────────────
_orig_init = Request.__init__
def _patched_init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    self.stage_start = {}
    self.stage_end   = {}
    self.stage_lat   = {}
Request.__init__ = _patched_init

# ─── 3) Wall–clock mark_start / mark_end ────────────────────────────────────
def mark_start(self, stage: str):
    self.stage_start[stage] = time.time()
def mark_end(self, stage: str):
    t = time.time()
    self.stage_end[stage] = t
    self.stage_lat[stage] = t - self.stage_start[stage]

Request.mark_start = mark_start
Request.mark_end   = mark_end

# ─── 4) Instrument “preprocessing” in get_requests() ───────────────────────
from util.request import get_requests as _orig_get_requests
def get_requests(*args, **kwargs):
    reqs, plens, glens = _orig_get_requests(*args, **kwargs)
    for r in reqs:
        r.mark_start("preprocessing")
        r.mark_end("preprocessing")
    return reqs, plens, glens

import util.request as _ur
_ur.get_requests = get_requests

# ─── 5) Always subclass LLMEngine to time “inference” & “postprocessing” ──
from TokenSim.llm.llm_engine import LLMEngine as _OrigEngine
class InstrumentedEngine(_OrigEngine):
    def _run_batch(self, req, *a, **kw):
        global CURRENT_REQ
        CURRENT_REQ = req
        req.mark_start("inference")
        super()._run_batch(req, *a, **kw)
        req.mark_end("inference")

    def _finish_request(self, req, *a, **kw):
        req.mark_start("postprocessing")
        super()._finish_request(req, *a, **kw)
        req.mark_end("postprocessing")
        global CURRENT_REQ
        CURRENT_REQ = None

import TokenSim.llm.llm_engine as _le
_le.LLMEngine = InstrumentedEngine

# ─── 6) Hook SharedMemoryCache for “cache_access” ───────────────────────────
from TokenSim.block.shared_cache import SharedMemoryCache as _OrigCache
class InstrumentedCache(_OrigCache):
    def get(self, token_sequence):
        req = globals().get("CURRENT_REQ")
        if req: req.mark_start("cache_access")
        out = super().get(token_sequence)
        if req: req.mark_end("cache_access")
        return out

    def put(self, token_sequence, block_ids):
        req = globals().get("CURRENT_REQ")
        if req: req.mark_start("cache_access")
        ev = super().put(token_sequence, block_ids)
        if req: req.mark_end("cache_access")
        return ev

import TokenSim.block.shared_cache as _sc
_sc.SharedMemoryCache = InstrumentedCache

# ─── 7) Capture the final request list in check_results ─────────────────────
import benchmark
_CAPTURE = {}
_orig_check = benchmark.check_results

def _patched_check(args, requests, engine, psla, cluster,
                   duration, prompt_count, prompt_lens, generation_lens):
    _CAPTURE['requests']    = requests
    _CAPTURE['prompt_lens'] = prompt_lens
    _CAPTURE['generation_lens'] = generation_lens
    return _orig_check(args, requests, engine, psla, cluster,
                       duration, prompt_count, prompt_lens, generation_lens)

benchmark.check_results = _patched_check

# ─── 8) Delegate to benchmark.main() and then dump CSV ──────────────────────
from benchmark import main

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sim_time", type=float, default=None)
    p.add_argument("--qps",      type=float, required=True)
    p.add_argument("--batching",
                   choices=["static","dynamic","paged-attn"],
                   default="dynamic")
    p.add_argument("--distribution",
                   choices=["burst","uniform","poisson"],
                   default="uniform")
    p.add_argument("--swap_policy", type=str, required=True)
    p.add_argument("--prompt_count", type=int, default=100)
    p.add_argument("--prompt_lens_mean", type=int, default=None)
    p.add_argument("--prompt_lens_range", type=int, default=None)
    p.add_argument("--generation_lens_mean", type=int, default=None)
    p.add_argument("--generation_lens_range", type=int, default=None)
    p.add_argument("--generation_lens_distribution",
                   choices=["uniform","exponential","capped_exponential","burst"],
                   default="uniform")
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--psla",       type=str, required=True)
    p.add_argument("--cluster",    type=str, required=True)
    p.add_argument("--pworker_pool_type",
                   choices=["Depool","Cepool"], default="Depool")
    p.add_argument("--gworker_pool_type",
                   choices=["Depool","Cepool"], default="Cepool")
    p.add_argument("--max_parallem_sum", type=int, default=99999)
    p.add_argument("--max_occupy_ratio", type=float, default=1.0)
    p.add_argument("--pp_dim",            type=int, default=1)
    p.add_argument("--verbose",
                   choices=["none","simple","tqdm"], default="tqdm")
    p.add_argument("--program_id", type=int,   default=0)
    p.add_argument("--results_path", type=str, default="")
    p.add_argument("--dataset_json_path",
                   type=str, required=True)
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("--llm_compass", type=str, default=None)
    p.add_argument("--eviction_policy", type=str, default="lru")
    args = p.parse_args()
    if args.distribution == "burst":
        args.qps = float("inf")

    # run the sim (our patched check_results will stash the requests)
    main(args)

    # now write out the per‐stage CSV
    reqs   = _CAPTURE['requests']
    outdir = Path(args.results_path or ".")
    outdir.mkdir(exist_ok=True, parents=True)
    csv_path = outdir / "latency_breakdown.csv"

    stages = list(reqs[0].stage_lat.keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["request_id","dataset","model"] + stages)
        for r in reqs:
            w.writerow([
                r.id,
                Path(args.dataset_json_path).stem,
                Path(args.psla).stem,
                *(r.stage_lat[s] for s in stages)
            ])

    print(f"Wrote per-stage latencies → {csv_path}")

