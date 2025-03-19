from pydantic.dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from TokenSim.config.config import ClusterConfig


@dataclass
class MetricData:
    p50: float
    p99: float
    max: float

    @classmethod
    def from_list(cls, latencies):
        return cls(np.median(latencies), np.percentile(latencies, 99), max(latencies))

    @classmethod
    def inf(cls):
        return cls(999999, 999999, 999999)


@dataclass
class LLMResult:
    cluster: ClusterConfig
    qps: float
    prompt_count: int
    batching: str
    request_time: MetricData
    prompt_time: MetricData
    decode_time: MetricData
    duration: float
    output_qps: float
    output_token_ps: float
    notdone: list[int]

    @classmethod
    def from_file(cls, filename):
        return cls(**json.loads(Path(filename).read_text()))


@dataclass
class PSLAConfig:
    name: str
    model: str
    distribution: str
    prompt_lens_mean: int
    prompt_lens_range: int
    generation_lens_mean: int
    generation_lens_range: int
    generation_lens_distribution: str
    first_token_latency: MetricData
    decode_token_latency: MetricData
    qps: float

    @classmethod
    def from_file(cls, filename):
        return cls(**json.loads(Path(filename).read_text()))

    def from_args(self, args):
        self.distribution = args.distribution or self.distribution
        self.prompt_lens_mean = args.prompt_lens_mean or self.prompt_lens_mean
        self.prompt_lens_range = args.prompt_lens_range or self.prompt_lens_range
        self.generation_lens_mean = args.generation_lens_mean or self.generation_lens_mean
        self.generation_lens_range = args.generation_lens_range or self.generation_lens_range
        self.generation_lens_distribution = (
            args.generation_lens_distribution or self.generation_lens_distribution
        )
        return self
