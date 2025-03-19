#!/usr/bin/env python3

import argparse
import simpy
from pathlib import Path
import sys
import os

# 添加 LLMCompass 目录到 Python 路径
llm_compass_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LLMCompass')
sys.path.append(llm_compass_path)

from util.results import export_result, print_all_stats
from util.request import get_requests, LLMSource
from util.tqdm import TqdmManager
from util.compass import get_compass_vars

from TokenSim.llm.llm_engine import LLMEngine, SwapPolicy
from TokenSim.llm.llm_request import g_time, Request
from TokenSim.config.config import ClusterConfig
from TokenSim.config.psla_config import PSLAConfig
from TransformerRoofline import TransformerRoofline


def check_results(
    args: argparse.Namespace,
    requests: list[Request],
    engine: LLMEngine,
    psla: PSLAConfig,
    cluster: ClusterConfig,
    duration: float,
    prompt_count: int,
    prompt_lens: list[int],
    generation_lens: list[int],
):
    notdone = [r.id for r in requests if not r.is_done]
    if notdone:
        failed_path = Path.cwd() / "results"
        failed_path.mkdir(parents=True, exist_ok=True)

        failed = "Failed " + args.cluster + "_" + str(args.qps) + ": " + str(notdone)
        with open(failed_path / "failed.txt", "a", newline="\n") as file:
            file.write(failed + "\n")

        return

    print_all_stats(g_time, requests, engine, duration, prompt_count, prompt_lens)
    export_result(
        args=args,
        g_time=g_time,
        psla=psla,
        cluster=cluster,
        prompt_count=prompt_count,
        prompt_lens=prompt_lens,
        generation_lens=generation_lens,
        notdone=notdone,
        duration=duration,
    )


def main(args: argparse.Namespace):
    wrapped_llmcompass_vars = get_compass_vars(args)
    roofline = TransformerRoofline(
        "./TransformerRoofline/hardware_models.json",
        "./TransformerRoofline/allreduce_v100.xlsx",
        "./TransformerRoofline/hardware_elements.json",
    )

    cluster = ClusterConfig.from_file(args.cluster)
    psla = PSLAConfig.from_file(args.psla).from_args(args)

    tqdm_manager = TqdmManager(verbose=args.verbose, program_id=args.program_id)
    env = simpy.Environment()
    requests, prompt_lens, generation_lens = get_requests(
        args=args,
        psla=psla,
        block_size=args.block_size,
        tqdm_submit_func=lambda req_num: tqdm_manager.update(req_num),
    )

    prompt_count = len(requests)
    tqdm_manager.set_total(prompt_count)

    engine = LLMEngine(
        env=env,
        block_size=args.block_size,
        batching=args.batching,
        swap_policy=args.swap_policy,
        psla_config=psla,
        cluster_config=cluster,
        roofline=roofline,
        pworker_pool_type=args.pworker_pool_type,
        gworker_pool_type=args.gworker_pool_type,
        max_parallem_sum=args.max_parallem_sum,
        max_occupy_ratio=args.max_occupy_ratio,
        pp_dim=args.pp_dim,
        wrapped_llmcompass_vars=wrapped_llmcompass_vars,
    )

    source = LLMSource(
        env=env,
        engine=engine,
        requests=requests,
        qps=args.qps,
        distribution=psla.distribution,
    )

    env.process(source)

    if args.sim_time is not None:
        env.run(args.sim_time)
    else:
        env.run()

    duration = env.now

    check_results(
        args,
        requests,
        engine,
        psla,
        cluster,
        duration,
        prompt_count,
        prompt_lens,
        generation_lens,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sim_time", type=float, default=None)

    parser.add_argument("--qps", type=float, required=True)
    parser.add_argument(
        "--batching", choices=["static", "dynamic", "paged-attn"], default="dynamic"
    )
    parser.add_argument(
        "--distribution", choices=["burst", "uniform", "poisson"], default="uniform"
    )
    parser.add_argument("--swap_policy", type=SwapPolicy, choices=list(SwapPolicy), required=True)

    parser.add_argument("--prompt_count", type=int, default=100)
    parser.add_argument("--prompt_lens_mean", type=int)
    parser.add_argument("--prompt_lens_range", type=int)
    parser.add_argument("--generation_lens_mean", type=int)
    parser.add_argument("--generation_lens_range", type=int)
    parser.add_argument(
        "--generation_lens_distribution",
        choices=["uniform", "exponential", "capped_exponential", "burst"],
        default="uniform",
    )

    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--psla", type=str, default="psla/test.json")
    parser.add_argument("--cluster", type=str, default="clusters/8_a100/p4g4.json")
    parser.add_argument("--pworker_pool_type", choices=["Depool", "Cepool"], default="Depool")
    parser.add_argument("--gworker_pool_type", choices=["Depool", "Cepool"], default="Cepool")
    parser.add_argument("--max_parallem_sum", type=int, default=99999)
    parser.add_argument("--max_occupy_ratio", type=float, default=1.0)
    parser.add_argument("--pp_dim", type=int, default=1)

    parser.add_argument("--verbose", type=str, choices=["none", "simple", "tqdm"], default="tqdm")
    parser.add_argument("--program_id", type=int, default=0)
    parser.add_argument("--results_path", type=str, default="")
    parser.add_argument("--dataset_json_path", type=str, default=None)
    parser.add_argument("--random_seed", type=int, default=0)

    parser.add_argument("--llm_compass", type=str, default=None)

    args = parser.parse_args()

    if args.distribution == "burst":
        args.qps = float("inf")
    main(args)
