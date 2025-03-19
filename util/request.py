import simpy
import argparse
import random

from TokenSim.llm.llm_request import Request
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.utils import get_wait_time, get_prompt_lens, get_generation_lens, get_lens_from_file
from TokenSim.config.psla_config import PSLAConfig


def get_requests(
    args: argparse.Namespace,
    psla: PSLAConfig,
    block_size: int,
    tqdm_submit_func=None,
) -> list[Request]:
    if args.dataset_json_path is None:
        random.seed(args.random_seed)
        prompt_lens = get_prompt_lens(
            len_mean=psla.prompt_lens_mean,
            len_range=psla.prompt_lens_range,
            num_prompt=args.prompt_count,
        )
        random.seed(args.random_seed)
        generation_lens = get_generation_lens(
            distribution=psla.generation_lens_distribution,
            len_mean=psla.generation_lens_mean,
            len_range=psla.generation_lens_range,
            num_prompt=args.prompt_count,
        )
    else:
        prompt_lens, generation_lens = get_lens_from_file(
            dataset_path=args.dataset_json_path, prompt_count=args.prompt_count
        )

    requests = [
        Request(id, prompt_len, generation_len, block_size)
        for id, (prompt_len, generation_len) in enumerate(
            zip(prompt_lens, generation_lens)
        )
    ]
    for req in requests:
        req.tqdm_submit_func = tqdm_submit_func
    return requests, prompt_lens, generation_lens

def LLMSource(
    env: simpy.Environment,
    engine: LLMEngine,
    requests: list[Request],
    qps: float,
    distribution: str,
):
    for req in requests:
        req.arrive(env)
        if distribution == "burst":
            engine.add_requests_burst([req])
        else:
            engine.add_requests([req])
            yield env.timeout(get_wait_time(1.0 / qps, distribution))
