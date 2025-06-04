import json
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Any

from TokenSim.config.psla_config import PSLAConfig, LLMResult, MetricData
from TokenSim.config.config import ClusterConfig
from TokenSim.llm.llm_request import g_time, LLMTime, Request
from TokenSim.llm.llm_engine import LLMEngine


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
