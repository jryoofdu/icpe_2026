import json
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Any

from TokenSim.config.psla_config import PSLAConfig, LLMResult, MetricData
from TokenSim.config.config import ClusterConfig
from TokenSim.llm.llm_request import g_time, LLMTime, Request
from TokenSim.llm.llm_engine import LLMEngine
#from TokenSim.config.hardware_model import model_map

def print_all_stats(
    g_time: LLMTime,
    requests: list[Request],
    engine: LLMEngine,
    duration: float,
    prompt_count: int,
    prompt_lens: list[int],
):
    request_time = MetricData.from_list(g_time.request_time)
    prompt_time = MetricData.from_list(g_time.prompt_time)
    decode_time = MetricData.from_list(g_time.decode_time)
    decode_max_time = MetricData.from_list(g_time.decode_max_time)
    prompt_idle_time = MetricData.from_list(g_time.prompt_idle)
    decode_idle_time = MetricData.from_list(g_time.decode_idle)

    # 打印性能指标
    print_metrics(
        request_time,
        prompt_time,
        decode_time,
        decode_max_time,
        prompt_idle_time,
        decode_idle_time,
    )
    prompt_latency = [req.time[0] for req in requests]
     
    
    generation_latency = [sum(req.time[1:]) / (len(req.time) - 1) for req in requests]
    # 打印延迟统计
    print_latency_stats(prompt_latency, generation_latency)
    # 打印吞吐量统计
    print_throughput_stats(duration, prompt_count, prompt_lens)
    print(f"Total preemptions: {sum([wkr.preempted_cnt for wkr in engine.workers])}")
    
    # Print cache statistics if shared cache is enabled
    for worker in engine.workers:
        if hasattr(worker.scheduler, 'block_manager') and worker.scheduler.block_manager.shared_cache:
            cache_stats = worker.scheduler.block_manager.get_cache_stats()
            print("\nCache Statistics:")
            print(f"Cache Hits: {cache_stats['hits']}")
            print(f"Cache Misses: {cache_stats['misses']}")
            print(f"Cache Hit Rate: {cache_stats['hit_rate']:.2f}%")
            print(f"Total Allocations: {cache_stats['total']}")
            if 'evictions' in cache_stats:
                print(f"Cache Evictions: {cache_stats['evictions']}")
                
    # ─── stage-wise summary metrics ─────────────────────────────
    # collect each stage into its own list
    pre_ls  = [r.stage_lat.get("preprocessing",0)   for r in requests]
    inf_ls  = [r.stage_lat.get("inference",0)       for r in requests]
    cache_ls= [r.stage_lat.get("cache_access",0)    for r in requests]
    post_ls = [r.stage_lat.get("postprocessing",0)  for r in requests]

    # wrap them in MetricData (min, max, mean, stddev, percentiles)
    pre_md   = MetricData.from_list(pre_ls)
    inf_md   = MetricData.from_list(inf_ls)
    cache_md = MetricData.from_list(cache_ls)
    post_md  = MetricData.from_list(post_ls)

#    print("\nStage-wise latency summary:")
#    print(f" Preprocessing: {pre_md}")
#    print(f" Inference    : {inf_md}")
#    print(f" Cache Access : {cache_md}")
#    print(f" Postprocess  : {post_md}")
    # ─── stage‐wise latency summary in ms ────────────────────
    def _fmt_ms(md):
        # takes a MetricData and returns "(p50, p99, max)" in ms
        return (
            f"p50={md.p50*1e3:.2f} ms, "
            f"p99={md.p99*1e3:.2f} ms, "
            f"max={md.max*1e3:.2f} ms"            
        )

    print("\nStage-wise latency (ms):")
    print(f" Preprocessing: {_fmt_ms(pre_md)}")
    print(f" Inference    : {_fmt_ms(inf_md)}")
    print(f" Cache Access : {_fmt_ms(cache_md)}")
    print(f" Postprocess  : {_fmt_ms(post_md)}")
    # ───────────────────────────────────────────────────────────
                
    
    # 打印SLO统计
    print_slo_stats(duration)


def print_metrics(
    request_time: MetricData,
    prompt_time: MetricData,
    decode_time: MetricData,
    decode_max_time: MetricData,
    prompt_idle_time: MetricData,
    decode_idle_time: MetricData,
):
    """打印性能指标

    Args:
        request_time: 请求时间指标
        prompt_time: 提示词处理时间指标
        decode_time: 解码时间指标
        decode_max_time: 最大解码时间指标
        prompt_idle_time: 提示词空闲时间指标
        decode_idle_time: 解码空闲时间指标
    """
    print(f"{request_time=}")
    print(f"{prompt_time=}")
    print(f"{decode_time=}")
    print(f"{decode_max_time=}")
    print(f"{prompt_idle_time=}")
    print(f"{decode_idle_time=}")


def print_latency_stats(prompt_latency: List[float], generation_latency: List[float]):
    print(f"Average prompt latency: {sum(prompt_latency) / len(prompt_latency)}")
    print(f"prompt latency: {sum(prompt_latency) }")
    print(f"generation latency: {sum(generation_latency)}")
    print(f"total latency: {sum(prompt_latency) + sum(generation_latency)}")
    print(f"Max prompt latency: {max(prompt_latency)}")
    print(f"Min prompt latency: {min(prompt_latency)}")
    print(
        f"Average generation latency: {sum(generation_latency) / len(generation_latency)}"
    )
    print(f"Max generation latency: {max(generation_latency)}")
    print(f"Min generation latency: {min(generation_latency)}")


def print_throughput_stats(dur: float, prompt_count: int, prompt_lens: List[int]):
    print(f"total time: {dur}")
    print(f"Thoughput: {prompt_count / dur} r/s, {sum(prompt_lens) / dur} token/s")


def print_slo_stats(dur: float, prompt_slo: float = 15, decode_slo: float = 0.15):
    prompt_slo_good_cnt = sum([1 if t <= prompt_slo else 0 for t in g_time.prompt_time])
    decode_slo_good_cnt = sum([1 if t <= decode_slo else 0 for t in g_time.decode_time])
    req_slo_good_cnt = sum(
        [
            1 if (p <= prompt_slo and d <= decode_slo) else 0
            for p, d in zip(g_time.prompt_time, g_time.decode_time)
        ]
    )

    decode_max_slo_good_cnt = sum(
        [1 if t <= decode_slo else 0 for t in g_time.decode_max_time]
    )
    req_max_slo_good_cnt = sum(
        [
            1 if (p <= prompt_slo and d <= decode_slo) else 0
            for p, d in zip(g_time.prompt_time, g_time.decode_max_time)
        ]
    )

    print(
        f"Prompt SLO Good Throughput: {prompt_slo_good_cnt} r, {prompt_slo_good_cnt / dur} r/s"
    )
    print(
        f"Max-Decode SLO Good Throughput: {decode_max_slo_good_cnt} r, {decode_max_slo_good_cnt / dur} r/s"
    )
    print(
        f"Prompt and Max-Decode SLO Good Throughput: {req_max_slo_good_cnt} r, {req_max_slo_good_cnt / dur} r/s"
    )
    print(
        f"Decode SLO Good Throughput: {decode_slo_good_cnt} r, {decode_slo_good_cnt / dur} r/s"
    )
    print(
        f"Prompt and Decode SLO Good Throughput: {req_slo_good_cnt} r, {req_slo_good_cnt / dur} r/s"
    )


def export_resultold(
    args: Any,
    g_time: LLMTime,
    psla: PSLAConfig,
    cluster: ClusterConfig,
    prompt_count: int,
    prompt_lens: list[int],
    generation_lens: list[int],
    notdone: list[int],
    duration: float,
    engine: LLMEngine = None,  # Add engine parameter
):
    request_time = MetricData.from_list(g_time.request_time)
    prompt_time = MetricData.from_list(g_time.prompt_time)
    decode_time = MetricData.from_list(g_time.decode_time)

    # Get cache statistics if available
    cache_stats = None
    if engine:
        for worker in engine.workers:
            if hasattr(worker.scheduler, 'block_manager') and worker.scheduler.block_manager.shared_cache:
                cache_stats = worker.scheduler.block_manager.get_cache_stats()
                break

    result = LLMResult(
        qps=args.qps,
        cluster=cluster,
        request_time=request_time,
        prompt_time=prompt_time,
        decode_time=decode_time,
        prompt_count=prompt_count,
        batching=args.batching,
        duration=duration,
        output_qps=prompt_count / duration,
        output_token_ps=(sum(prompt_lens) + sum(generation_lens)) / duration,
        notdone=notdone,
        cache_stats=cache_stats,  # Add cache stats to result
    )

    if args.results_path == "":
        results_path = (
            Path.cwd()
            / "results"
            / psla.name
            / Path(args.cluster).parent.name
            / Path(args.cluster).stem
        )
    else:
        results_path = Path(args.results_path)

    results_path.mkdir(parents=True, exist_ok=True)

    result_dict = asdict(result)
    result_dict["cluster"] = asdict(result.cluster)
    result_dict["request_time"] = asdict(result.request_time)
    result_dict["prompt_time"] = asdict(result.prompt_time)
    result_dict["decode_time"] = asdict(result.decode_time)

    result_file = results_path / f"result_{args.qps}.json"
    with open(result_file, "w") as f:
        json.dump(result_dict, f, indent=4)


def export_result(
    args: Any,
    g_time: LLMTime,
    requests: list[Request],
    psla: PSLAConfig,
    cluster: ClusterConfig,
    prompt_count: int,
    prompt_lens: list[int],
    generation_lens: list[int],
    notdone: list[int],
    duration: float,
    engine: LLMEngine = None,
):
    # Metrics
    request_time = MetricData.from_list(g_time.request_time)
    prompt_time = MetricData.from_list(g_time.prompt_time)
    decode_time = MetricData.from_list(g_time.decode_time)

    # Cache stats
    cache_stats = None
    if engine:
      
      
        for worker in engine.workers:
            print(f"Scheduler: {worker.scheduler.block_manager.shared_cache}")
            shared_cache = bool(worker.scheduler.block_manager.shared_cache)
            if hasattr(worker.scheduler, 'block_manager') :#and worker.scheduler.block_manager.shared_cache
               
                cache_stats = worker.scheduler.block_manager.get_cache_stats()
                break

    # ─── override cache_access with “real” cache‐stats quantiles ─────────────────
    #if cache_stats is not None and hasattr(cache_stats, "quantiles"):
    #    # quantiles() should return something like { "p50":…, "p99":…, "max":… }
    #    cache_quota = cache_stats.quantiles()
    #else:
    #    cache_quota = None
    

    # Extract metadata
    dataset_name = Path(args.dataset_json_path).parent.parent.name
    psla_model = Path(args.psla).stem.replace("-shared", "").replace("-isolated", "")
    #shared_cache = "shared" in Path(args.psla).stem
    block_size = args.block_size
    num_blocks = args.num_blocks if hasattr(args, "num_blocks") else 4096
    cache_policy = args.eviction_policy

    # Load hardware model config
    try:
        with open("TransformerRoofline/hardware_models.json", "r") as f:
            hardware_models = json.load(f)

        gpu_list = hardware_models["hardware"] # Assuming this is a list of dicts with "Name" and "Dmodel" keys
        matched_model = next(
            (m for m in gpu_list if m["Name"].lower() in psla_model.lower()),
            None
        )
    except Exception as e:
            print(f"[Warning] Failed to load hardware model info: {e}")
            matched_model = None

    # Compute cache size
    d_model = matched_model["Dmodel"] if matched_model else 4096
    bytes_per_token = 2 * d_model  # fp16 = 2 bytes
    total_cache_bytes = num_blocks * block_size * bytes_per_token
    cache_size_gb = round(total_cache_bytes / (1024 ** 3), 2)

    # Create result object
    result = LLMResult(
        qps=args.qps,
        cluster=cluster,
        request_time=request_time,
        prompt_time=prompt_time,
        decode_time=decode_time,
        prompt_count=prompt_count,
        batching=args.batching,
        duration=duration,
        output_qps=prompt_count / duration,
        output_token_ps=(sum(prompt_lens) + sum(generation_lens)) / duration,
        notdone=notdone,
        cache_stats=cache_stats,
    )

    # Output path: results/benchmark/{dataset}/{model}/{policy}/qps_{qps}/
    results_path = (
        Path.cwd()
        / "results"
        / "benchmark"
        / dataset_name
        / psla_model
        / cache_policy
        / f"qps_{int(args.qps)}"
    )
    results_path.mkdir(parents=True, exist_ok=True)

    # Serialize and add additional metadata
    result_dict = asdict(result)
    result_dict["cluster"] = asdict(result.cluster)
    result_dict["request_time"] = asdict(result.request_time)
    result_dict["prompt_time"] = asdict(result.prompt_time)
    result_dict["decode_time"] = asdict(result.decode_time)
    result_dict.update({
        "psla_model": psla_model,
        "dataset_name": dataset_name,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "cache_size_gb": cache_size_gb,
        "cache_policy": cache_policy,
        "shared_cache": shared_cache
    })


    # ─── compute & inject per-stage summary ──────────────────────
    # build lists of per-request stage latencies directly from requests
    pre_ls   = [r.stage_lat.get("preprocessing",   0) for r in requests]
    inf_ls   = [r.stage_lat.get("inference",       0) for r in requests]
    cache_ls = [r.stage_lat.get("cache_access",    0) for r in requests]
    post_ls  = [r.stage_lat.get("postprocessing",  0) for r in requests]

    # summarize as MetricData
#    stage_summary = {
#        "preprocessing":  MetricData.from_list(pre_ls).__dict__,
#        "inference":      MetricData.from_list(inf_ls).__dict__,
#        "cache_access":   MetricData.from_list(cache_ls).__dict__,
#        "postprocessing": MetricData.from_list(post_ls).__dict__,
#    }

    # helper: convert MetricData dict (secs) to ms with 2 decimals
    def _to_ms(d: dict[str, float]) -> dict[str, float]:
        return {k: round(v * 1000, 2) for k, v in d.items()}

    # build stage summary in ms
    stage_summary = {
        "preprocessing":  _to_ms(vars(MetricData.from_list(pre_ls))),
        "inference":      _to_ms(vars(MetricData.from_list(inf_ls))),
        "cache_access":   _to_ms(vars(MetricData.from_list(cache_ls))),
        "postprocessing": _to_ms(vars(MetricData.from_list(post_ls))),
    }

    result_dict["stage_latency_summary"] = stage_summary
    # ──────────────────────────────────────────────────────────────


    # Short filename (no duplication of info already in folders)
    filename = f"{'shared' if shared_cache else 'isolated'}_{cache_size_gb}GB.json"
    result_file = results_path / filename

    # Write to JSON
    with open(result_file, "w") as f:
        json.dump(result_dict, f, indent=4)

    print(f"[✓] Exported result to: {result_file}")


def export_result(
    args: Any,
    g_time: LLMTime,
    psla: PSLAConfig,
    cluster: ClusterConfig,
    prompt_count: int,
    prompt_lens: list[int],
    generation_lens: list[int],
    notdone: list[int],
    duration: float,
    engine: LLMEngine = None,
    requests: List[Request] = None,  # Add requests parameter for latency stats
):
    # Metrics
    request_time = MetricData.from_list(g_time.request_time)
    prompt_time = MetricData.from_list(g_time.prompt_time)
    decode_time = MetricData.from_list(g_time.decode_time)

    # Cache stats
    cache_stats = None
    if engine:
      
      
        for worker in engine.workers:
            print(f"Scheduler: {worker.scheduler.block_manager.shared_cache}")
            shared_cache = bool(worker.scheduler.block_manager.shared_cache)
            if hasattr(worker.scheduler, 'block_manager') :#and worker.scheduler.block_manager.shared_cache
               
                cache_stats = worker.scheduler.block_manager.get_cache_stats()
                break

    # Extract metadata
    if hasattr(args, "dataset_json_path") and args.dataset_json_path:
        dataset_name = Path(args.dataset_json_path).parent.parent.name
    else:
        dataset_name = "Custom"

 
    psla_model = Path(args.psla).stem.replace("-shared", "").replace("-isolated", "")
    #shared_cache = "shared" in Path(args.psla).stem
    block_size = args.block_size
    num_blocks = args.num_blocks if hasattr(args, "num_blocks") else 4096
    cache_policy = args.eviction_policy

   # Load cluster to compute cache capacity
    cluster_file = Path(args.cluster)
    with open(cluster_file) as f:
        cluster_json = json.load(f)

# Compute total hardware capacity from new format
    try:
        with open("TransformerRoofline/hardware_models.json", "r") as f:
            hardware_models = json.load(f)

        gpu_list = hardware_models["hardware"] # Assuming this is a list of dicts with "Name" and "Dmodel" keys
        matched_model = next(
            (m for m in gpu_list if m["Name"].lower() in psla_model.lower()),
            None
        )
        worker_groups = cluster_json.get("worker_groups", [])
        total_capacity_blocks = 0

        for wg in worker_groups:
            hw_name = wg["hardware"]
            num = wg["num_workers"]
            matched_hw = next((h for h in gpu_list if h["Name"] == hw_name), None)
            if matched_hw:
                cap = matched_hw.get("Capacity", 0)
                total_capacity_blocks += cap * num

        d_model = matched_model["Dmodel"] if matched_model else 4096
        bytes_per_token = 2 * d_model
        total_cache_bytes = total_capacity_blocks * block_size * bytes_per_token
        cache_size_gb = total_capacity_blocks#round(total_cache_bytes / (1024 ** 3), 2)

    except Exception as e:
        print(f"[!] Error computing cache size from cluster file: {e}")
        cache_size_gb = 0


    # Create result object
    result = LLMResult(
        qps=args.qps,
        cluster=cluster,
        request_time=request_time,
        prompt_time=prompt_time,
        decode_time=decode_time,
        prompt_count=prompt_count,
        batching=args.batching,
        duration=duration,
        output_qps=prompt_count / duration,
        output_token_ps=(sum(prompt_lens) + sum(generation_lens)) / duration,
        notdone=notdone,
        cache_stats=cache_stats,
    )

    # Output path: results/benchmark/{dataset}/{model}/{policy}/qps_{qps}/
    results_path = (
        Path.cwd()
        / "results"
        / "benchmark"
        / dataset_name
        / psla_model
        / cache_policy
        / f"qps_{int(args.qps)}"
    )
    tps_value = getattr(args, "tps", None)
    if tps_value:
        results_path = results_path / f"tps_{int(tps_value)}"

    prompt_latency = [req.time[0] for req in requests]
    generation_latency = [sum(req.time[1:]) / (len(req.time) - 1) for req in requests]
    results_path.mkdir(parents=True, exist_ok=True)
    total_prompt_latency = sum(prompt_latency)
    total_generation_latency = sum(generation_latency)
    total_latency = total_prompt_latency + total_generation_latency
    # Serialize and add additional metadata
    result_dict = asdict(result)
    result_dict["cluster"] = asdict(result.cluster)
    result_dict["request_time"] = asdict(result.request_time)
    result_dict["prompt_time"] = asdict(result.prompt_time)
    result_dict["decode_time"] = asdict(result.decode_time)
    result_dict.update({
        "psla_model": psla_model,
        "dataset_name": dataset_name,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "cache_size_gb": cache_size_gb,
        "cache_policy": cache_policy,
        "shared_cache": shared_cache,
        "total_prompt_latency": total_prompt_latency,
        "total_generation_latency": total_generation_latency,
        "total_latency": total_latency,
    })

    # Short filename (no duplication of info already in folders)
    filename = f"{'shared' if shared_cache else 'isolated'}_{cache_size_gb}GB.json"
    result_file = results_path / filename

    # Write to JSON
    with open(result_file, "w") as f:
        json.dump(result_dict, f, indent=4)

    print(f"[✓] Exported result to: {result_file}")