import json
from pathlib import Path
from dataclasses import asdict
from pydantic.dataclasses import dataclass, Field
from TransformerRoofline import TransformerRoofline

_GB = 1 << 30


class CacheConfig:
    def __init__(self, block_size: int, hardware: str, model: str, roofline: TransformerRoofline, debug: bool = False):
        # block size
        self.block_size: int = block_size
        self.debug: bool = debug
        hardware_conf = roofline.hardwares[hardware]
        model_conf = roofline.models[model]
        self.size_per_token = model_conf.Dmodel * 2 * 2 * model_conf.Nlayer

        # size = 12 * l * d * d, then we assume vocab size is 5e4
        self.model_param_size = (
            12 * model_conf.Nlayer * model_conf.Dmodel * model_conf.Dmodel
            + 50000 * model_conf.Dmodel
        ) * 2  # sizeof fp16 is 2 bytes
        # num blocks
        # FIXME: actual host memory size can reach terrabytes, much larger than accelerator memory
        # however in our application, host memory serves as swap space, and leveraging large swap space is not desirable for performance
        #  so the number is set small so out simulation exits early
        assert hardware_conf.MM_Card_Num * hardware_conf.Capacity * _GB > self.model_param_size
        self.num_cpu_blocks: int = 32768 * _GB / self.size_per_token // self.block_size
        self.num_gpu_blocks: int = (
            (hardware_conf.MM_Card_Num * hardware_conf.Capacity * _GB - self.model_param_size)
            / self.size_per_token
            // self.block_size
        )


@dataclass
class WorkerConfig:
    role: str
    hardware: str
    network: str
    nettype: str


@dataclass
class WorkerGroupConfig:
    role: str
    hardware: str
    num_workers: int
    network: str = "net1"

    def workers(self, networks):
        return [
            WorkerConfig(self.role, self.hardware, self.network, networks[self.network])
            for _ in range(self.num_workers)
        ]

    def __str__(self):
        return f"{self.role[0]}{self.num_workers:g}"


@dataclass
class ClusterConfig:
    num_workers: int
    networks: dict[str, str]
    worker_groups: list[WorkerGroupConfig] = Field(default_factory=list)

    @classmethod
    def from_file(cls, filename):
        return cls(**json.loads(Path(filename).read_text()))

    def workers(self):
        return [
            worker
            for worker_group in self.worker_groups
            for worker in worker_group.workers(self.networks)
        ]

    def __str__(self):
        return "".join(str(worker_group) for worker_group in self.worker_groups)
