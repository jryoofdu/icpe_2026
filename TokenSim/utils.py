from __future__ import annotations
import random
import numpy as np
import os
import json


def get_prompt_lens(len_mean: int, len_range: int, num_prompt: int) -> list[int]:
    if len_range < 0:
        return [len_mean] * num_prompt
    low = len_mean - (len_range // 2)
    high = len_mean + (len_range // 2)
    prompt_lens = list(map(lambda _: random.randint(low, high), range(num_prompt)))
    return prompt_lens


def get_generation_lens(
    distribution: str, len_mean: int, len_range: int, num_prompt: int
) -> list[int]:
    if len_range <= 0:
        return [len_mean] * num_prompt
    if distribution == "uniform":
        low = len_mean - (len_range // 2)
        high = len_mean + (len_range // 2)
        generation_lens = list(map(lambda _: random.randint(low, high), range(num_prompt)))
        return generation_lens
    elif distribution == "exponential":
        np.random.seed(0)
        return [
            min(round(s), len_range) for s in np.random.exponential(scale=len_mean, size=num_prompt)
        ]
    elif distribution == "capped_exponential":
        np.random.seed(0)
        response_lens = []
        while len(response_lens) < num_prompt:
            sample = round(np.random.exponential(scale=len_mean))
            if sample >= 2 and sample <= len_range:
                response_lens.append(sample)
        return response_lens
    elif distribution == "burst":
        return [len_mean] * num_prompt
    else:
        raise ValueError(f"unknown distribution {distribution=}")


def get_wait_time(mean_time_between_requests: float, distribution: str) -> float:
    if distribution == "uniform":
        return mean_time_between_requests
    else:
        return np.random.exponential(mean_time_between_requests)


def get_lens_from_file(dataset_path, prompt_count=None):
    assert os.path.exists(dataset_path)
    assert os.path.isfile(dataset_path)
    with open(dataset_path, "r") as f:
        data = json.load(f)
    if prompt_count is not None and prompt_count > len(data):
        data = data * (prompt_count // len(data) + 1)
    data = data[6789:]
    data = data[:prompt_count]
    prompt_lens = [x[0] - 2 if x[0] > 2 else 1 for x in data]
    gen_lens = [x[1] for x in data]
    print(
        f'[get_lens_from_file()] Read {len(data)} reqs from dataset "{dataset_path}",'
        + f"prompt average {sum(prompt_lens)/len(prompt_lens)}, gen average {sum(gen_lens)/len(gen_lens)}"
    )
    return prompt_lens, gen_lens
