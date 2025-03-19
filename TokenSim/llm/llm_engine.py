from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any, Tuple
import logging
import math
from enum import Enum
from abc import ABC, abstractmethod
import simpy


from TokenSim.llm.llm_comm import SwapMetadata, SwapInMetadata
from TokenSim.llm.llm_request import Request
from TokenSim.config.config import ClusterConfig, WorkerConfig, CacheConfig, _GB
from TokenSim.config.psla_config import PSLAConfig
from TokenSim.llm.llm_scheduler import (
    LLMStaticScheduler,
    LLMDynamicScheduler,
    LLMPagedAttnScheduler,
    LLMPromptScheduler,
)

from TransformerRoofline import TransformerRoofline

from LLMCompass.software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from LLMCompass.hardware_model.system import System as LLMCompassSystem
from LLMCompass.software_model.utils import Tensor, data_type_dict

g_trace = []

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Task(Enum):
    ADD = 10  # 添加新请求
    STEP = 20  # 执行一步计算
    STOP = 30  # 停止执行
    SWAP = 0  # 交换内存
    SWAP_LOCAL = 1  # 本地内存交换
    SWAP_IN_REMOTE = 2  # 从远程加载内存
    SWAP_OUT_REMOTE = 3  # 写出到远程内存
    SWAP_IN_REMOTE_DONE = -1
    SWAP_OUT_REMOTE_DONE = -1


@dataclass
class Message:
    task: Task
    sender_id: int = 0
    requests: list[Request] | None = None
    metadata: Any | None = None

    def __lt__(self, other: Message):
        return self.task.value < other.task.value


class WorkerStatus(Enum):
    WAITING = 0
    RUNNING = 1
    SWAPPING = 2
    STOPPED = 3


class SwapPolicy(Enum):
    lazy = "lazy"
    eager = "eager"

    def __str__(self):
        return self.value


class Worker:
    def __init__(self, env: simpy.Environment, id: int) -> None:
        self.env = env
        self.id = id
        self.msg_queue = simpy.PriorityStore(self.env)
        self.memory = []
        self.cpu_mem = []

    def send_task(
        self, worker: Worker, task: Task, requests: Optional[list[Request]] = None, metadata=None
    ):
        worker.msg_queue.put(Message(task, self.id, requests, metadata))

    def trace(self, name, dur, requests: list[Request]):
        trace = {}
        trace["name"] = name + str(len(requests))
        trace["pid"] = self.id
        trace["tid"] = 1
        trace["ph"] = "X"
        trace["ts"] = self.env.now * 1e6
        trace["dur"] = dur * 1e6
        trace["args"] = {"requests": [(req.id, req.generation_idx) for req in requests]}
        g_trace.append(trace)

    def check(self, requests: list[Request]):
        check_id = 3473
        if check_id in [_.id for _ in requests]:
            print(self.env.now)


class LLMWorker(Worker):
    def __init__(
        self,
        env: simpy.Environment,
        id: int,
        engine: LLMEngine,
        psla_config: PSLAConfig,
        worker_config: WorkerConfig,
        block_size: int,
        batching: str,
        swap_policy: SwapPolicy,
        roofline: TransformerRoofline,
        max_parallem_sum: int,
        max_occupy_ratio: float = 1,
        pp_dim: int = 1,
        wrapped_llmcompass_vars: Tuple[
            LLMCompassSystem, TransformerBlockInitComputationTP, TransformerBlockAutoRegressionTP
        ] = None,
    ):
        super().__init__(env, id)
        self.llmcm_system, self.llmcm_prefill, self.llmcm_decode = None, None, None
        if wrapped_llmcompass_vars is not None:
            self.llmcm_system, self.llmcm_prefill, self.llmcm_decode = wrapped_llmcompass_vars
        self.roofline = roofline
        self.engine = engine

        self.role = worker_config.role
        self.hardware = worker_config.hardware
        self.network = worker_config.network
        self.nettype = worker_config.nettype
        self.model = psla_config.model

        self.block_size = block_size
        self.swap_policy = swap_policy

        self.cache_config = CacheConfig(self.block_size, self.hardware, self.model, self.roofline)

        self.preempted_cnt = 0

        self.pp_dim = pp_dim

        self.mems = [{}, {}, {}, {}]

        # FIXME: max_parallem_sum should be calculated based on accelerator type
        if batching == "paged-attn":
            if False:
                # if self.role == "prompt":
                self.scheduler = LLMPromptScheduler(max_parallem_sum=200)
            else:
                self.scheduler = LLMPagedAttnScheduler(
                    self.id,
                    cache_config=self.cache_config,
                    lazy_swap=self.lazy_swap,
                    max_parallem_sum=max_parallem_sum,
                    max_occupy_ratio=max_occupy_ratio if self.role != "prompt" else 1,
                )
        elif batching == "static":
            self.scheduler = LLMStaticScheduler(max_parallem_sum=max_parallem_sum)
        else:
            self.scheduler = LLMDynamicScheduler(max_parallem_sum=max_parallem_sum)

        self.status = WorkerStatus.WAITING
        self.action = self.env.process(self.run())

    @property
    def lazy_swap(self):
        return self.swap_policy == SwapPolicy.lazy

    def workload(self):
        return self.scheduler.workload()

    def cpu_workload(self):
        return self.scheduler.cpu_workload()

    def trace_memory(self):
        self.memory.append((self.env.now, self.workload()))
        self.cpu_mem.append((self.env.now, self.cpu_workload()))

    def step(self, requests: list[Request], latency: float):
        for request in requests:
            request.step(self.env, latency, len(requests))

    def swap_eager(self, swap_metadata: SwapMetadata):
        hardwares = self.roofline.hardwares
        links = self.roofline.links

        if self.id == swap_metadata.remote_id:
            link = links[hardwares[self.hardware].Pcie]
            Latency = link.Latency
            BW = link.UniBW
        else:
            swap_worker = self.engine.workers[swap_metadata.remote_id]
            if self.network == swap_worker.network:
                Latency = max(
                    links[hardwares[self.hardware].Nvlink].Latency,
                    links[hardwares[swap_worker.hardware].Nvlink].Latency,
                )
                BW = min(
                    links[hardwares[self.hardware].Nvlink].UniBW,
                    links[hardwares[swap_worker.hardware].Nvlink].UniBW,
                )
            else:
                Latency = max(links[self.nettype].Latency, links[swap_worker.nettype].Latency)
                BW = min(links[self.nettype].UniBW, links[swap_worker.nettype].UniBW)

        swap_overhead_in = (
            Latency
            + swap_metadata.num_blocks_to_swap_in
            * self.cache_config.block_size
            * self.cache_config.size_per_token
            / _GB
            / BW
        )
        swap_overhead_out = (
            Latency
            + swap_metadata.num_blocks_to_swap_out
            * self.cache_config.block_size
            * self.cache_config.size_per_token
            / _GB
            / BW
        )

        return max(swap_overhead_in, swap_overhead_out)

    def run(self):
        while True:
            if self.status == WorkerStatus.RUNNING and not self.msg_queue.items:
                self.send_task(self, Task.STEP)
            msg = yield self.msg_queue.get()

            try:
                match msg.task:
                    case Task.ADD:
                        self.scheduler.add_requests(msg.requests)
                        self.status = WorkerStatus.RUNNING
                    case Task.STEP:
                        assert self.status == WorkerStatus.RUNNING
                        running, swapped, swap_metadata = self.scheduler.schedule()

                        if swapped and swap_metadata:
                            if self.lazy_swap:
                                self.send_task(self.engine, Task.SWAP, swapped, swap_metadata)
                            else:
                                latency = self.swap_eager(swap_metadata)
                                self.preempted_cnt += 1
                                yield self.env.timeout(latency)

                        if running:
                            latency = self.dynamic_batch(running)
                            yield self.env.timeout(latency)

                            self.step(running, latency)

                            running = self.scheduler.update(running)
                            self.trace_memory()

                            self.engine.trace_all_memory_usage()
                            if running and self.role == "prompt":
                                swapped, swap_metadata = self.scheduler.schedule_prompt()
                                if swapped and swap_metadata:
                                    self.send_task(self.engine, Task.SWAP, swapped, swap_metadata)

                        if not running and not self.scheduler.waiting and not swapped:
                            self.status = WorkerStatus.WAITING

                    case Task.SWAP_LOCAL:
                        if msg.requests:
                            for req in msg.requests:
                                self.scheduler._swap_out(req)
                                self.scheduler.swapped.append(req)

                            latency = self.swap_eager(msg.metadata)
                            yield self.env.timeout(latency)
                            self.status = WorkerStatus.RUNNING

                    case Task.SWAP_OUT_REMOTE:
                        if msg.requests:
                            self.env.process(self.swap_out(msg))

                    case Task.SWAP_IN_REMOTE:
                        if msg.requests:
                            self.env.process(self.swap_in(msg))

                    case Task.SWAP_IN_REMOTE_DONE:
                        metadata = msg.metadata
                        self.scheduler.running.extend(metadata.to_running)
                        self.scheduler.swapped.extend(metadata.to_swapped)
                        self.status = WorkerStatus.RUNNING

                    case Task.STOP:
                        self.status = WorkerStatus.STOPPED
                        break
            except RuntimeError:
                self.send_task(self.engine, Task.STOP)

    def swap_out(self, msg):
        for req in msg.requests:
            self.scheduler._remote_swap_out(req)
        msg.metadata.event.succeed()
        latency = self.swap_eager(msg.metadata)
        yield self.env.timeout(latency)
        self.status = WorkerStatus.RUNNING

    def swap_in(self, msg):
        yield msg.metadata.event
        to_running, to_swapped = self.scheduler._remote_swap_in(msg.requests)
        latency = self.swap_eager(msg.metadata)
        yield self.env.timeout(latency)
        self.send_task(self, Task.SWAP_IN_REMOTE_DONE, None, SwapInMetadata(to_running, to_swapped))
        self.status = WorkerStatus.RUNNING

    def dynamic_batch(self, requests: list[Request]) -> float:
        if self.llmcm_system is None:
            return self.dynamic_batch_roofline(requests)
        else:
            try:
                return self.dynamic_batch_llmcompass(requests)
            except Exception as e:
                print(f"Error occured in LLMCompass: {e}")
                return self.dynamic_batch_roofline(requests)

    def dynamic_batch_roofline(self, requests: list[Request]) -> float:
        batch_size = len(requests)
        sum_attn_latency = 0
        proj_latency = 0
        # Prefill Stage
        if requests[0].is_prompt:
            total_prompt_len = sum([req.prompt_len for req in requests])
            total_prompt_len = int(total_prompt_len / 128) * 128 + 128  # Calibration
            proj_latency, _ = self.roofline.Compute_Timebreakdown_Iteration(
                total_prompt_len,
                requests[0].generation_idx,
                1,
                self.model,
                self.hardware,
                Pipeline_Stage=self.pp_dim,
            )
        # Generation Stage
        else:
            proj_latency, _ = self.roofline.Compute_Timebreakdown_Iteration(
                requests[0].prompt_len,
                requests[0].generation_idx,
                batch_size,
                self.model,
                self.hardware,
                Pipeline_Stage=self.pp_dim,
            )
        for req in requests:
            _, attn_latency = self.roofline.Compute_Timebreakdown_Iteration(
                req.prompt_len,
                req.generation_idx,
                1,
                self.model,
                self.hardware,
                Pipeline_Stage=self.pp_dim,
            )
            sum_attn_latency += attn_latency

        # Calibration
        total_time = proj_latency + sum_attn_latency

        if requests[0].is_prompt:
            total_time = total_time * 1.41 + 0.009
        else:
            total_time = total_time / 0.55
        return total_time

    def dynamic_batch_llmcompass(self, requests: list[Request]) -> float:
        batch_size = len(requests)
        sum_attn_latency = 0
        proj_latency = 0

        if requests[0].is_prompt:
            total_prompt_len = sum([req.prompt_len for req in requests])
            total_prompt_len = int(total_prompt_len / 128) * 128 + 128  # Calibration
            if total_prompt_len in self.mems[0]:
                proj_latency = self.mems[0][total_prompt_len]
            else:
                _ = self.llmcm_prefill(Tensor([1, total_prompt_len, 4096], data_type_dict["fp16"]))
                proj_latency, _ = self.llmcm_prefill.compile_and_simulate_proj_attn(
                    self.llmcm_system, "heuristic-GPU"
                )
                self.mems[0][total_prompt_len] = proj_latency
            proj_latency *= 40  # layer num
            for req in requests:
                if req.prompt_len in self.mems[1]:
                    attn = self.mems[1][req.prompt_len]
                else:
                    _ = self.llmcm_prefill(
                        Tensor([1, req.prompt_len, 4096], data_type_dict["fp16"])
                    )
                    _, attn = self.llmcm_prefill.compile_and_simulate_proj_attn(
                        self.llmcm_system, "heuristic-GPU"
                    )
                    self.mems[1][req.prompt_len] = attn
                sum_attn_latency += attn * 40  # layer num
        else:
            if (batch_size, max([req.generation_idx for req in requests])) in self.mems[2]:
                proj_latency = self.mems[2][
                    (batch_size, max([req.generation_idx for req in requests]))
                ]
            else:
                _ = self.llmcm_decode(
                    Tensor([batch_size, 1, 4096], data_type_dict["fp16"]),
                    max([req.generation_idx for req in requests]),
                )
                proj_latency, _ = self.llmcm_decode.compile_and_simulate_proj_attn(
                    self.llmcm_system, "heuristic-GPU"
                )
                self.mems[2][
                    (batch_size, max([req.generation_idx for req in requests]))
                ] = proj_latency
            proj_latency *= 40  # layer num
            for req in requests:
                if req.prompt_len + req.generation_idx in self.mems[3]:
                    attn = self.mems[3][req.prompt_len + req.generation_idx]
                else:
                    _ = self.llmcm_decode(
                        Tensor([1, 1, 4096], data_type_dict["fp16"]),
                        req.prompt_len + req.generation_idx,
                    )
                    _, attn = self.llmcm_decode.compile_and_simulate_proj_attn(
                        self.llmcm_system, "heuristic-GPU"
                    )
                    self.mems[3][req.prompt_len + req.generation_idx] = attn
                sum_attn_latency += attn * 40
        # Calibration
        total_time = proj_latency + sum_attn_latency
        return total_time


class WorkerPool(ABC):
    def __init__(self, workers: list[LLMWorker]):
        self.workers = workers
        self.start_idx = workers[0].id

    @abstractmethod
    def schedule(self) -> LLMWorker:
        pass

    def get(self, id: int) -> LLMWorker:
        return self.workers[id - self.start_idx]

    def __len__(self) -> int:
        return len(self.workers)


class DePool(WorkerPool):
    def __init__(self, workers: list[LLMWorker]):
        super().__init__(workers)
        self.cur_idx = 0
        self.num_workers = len(workers)

    def schedule(self):
        i = self.cur_idx
        self.cur_idx = (self.cur_idx + 1) % self.num_workers
        return self.workers[i]


class CePool(WorkerPool):
    def workloads(self):
        return [worker.workload() for worker in self.workers]

    def schedule(self):
        workloads = self.workloads()
        i = workloads.index(min(workloads))
        return self.workers[i]


class LLMEngine(Worker):
    def __init__(
        self,
        env: simpy.Environment,
        block_size: int,
        batching: str,
        swap_policy: SwapPolicy,
        psla_config: PSLAConfig,
        cluster_config: ClusterConfig,
        roofline: TransformerRoofline,
        pworker_pool_type: str,
        gworker_pool_type: str,
        max_parallem_sum: int,
        max_occupy_ratio: float = 1,
        pp_dim: int = 1,
        wrapped_llmcompass_vars: Tuple[
            LLMCompassSystem, TransformerBlockInitComputationTP, TransformerBlockAutoRegressionTP
        ] = None,
    ):
        super().__init__(env, -1)

        pworker_pool_type = pworker_pool_type.lower()
        gworker_pool_type = gworker_pool_type.lower()

        pool_type_map = {"depool": DePool, "cepool": CePool}

        self.workers = [
            LLMWorker(
                env,
                id,
                self,
                psla_config,
                worker_config,
                block_size,
                batching,
                swap_policy,
                roofline,
                max_parallem_sum,
                max_occupy_ratio,
                pp_dim,
                wrapped_llmcompass_vars,
            )
            for id, worker_config in enumerate(cluster_config.workers())
        ]
        self.pworkers = pool_type_map[pworker_pool_type](
            [worker for worker in self.workers if worker.role == "homo" or worker.role == "prompt"]
        )
        self.gworkers = pool_type_map[gworker_pool_type](
            [
                worker
                for worker in self.workers
                if worker.role == "homo" or worker.role == "generation"
            ]
        )

        self.memory_usage_trace = []

        self.action = self.env.process(self.run())

    def trace_all_memory_usage(self):
        """记录所有worker的内存使用情况"""
        current_time = self.env.now
        memory_snapshot = {"time": current_time, "workers": []}

        for worker in self.workers:
            if hasattr(worker.scheduler, "block_manager"):
                (
                    free_blocks,
                    used_blocks,
                    total_blocks,
                ) = worker.scheduler.block_manager.get_gpu_status()
                usage_percent = (used_blocks / total_blocks) * 100 if total_blocks > 0 else 0
                memory_snapshot["workers"].append(
                    {
                        "worker_id": worker.id,
                        "role": worker.role,
                        "free_blocks": free_blocks,
                        "used_blocks": used_blocks,
                        "total_blocks": total_blocks,
                        "usage_percent": usage_percent,
                    }
                )

        self.memory_usage_trace.append(memory_snapshot)

    def run(self):
        while True:
            msg = yield self.msg_queue.get()
            match msg.task:
                case Task.ADD:
                    worker = self.pworkers.schedule()
                    self.send_task(worker, Task.ADD, msg.requests)
                    self.trace_all_memory_usage()
                # 3. 内存交换
                case Task.SWAP:
                    if msg.sender_id < len(self.pworkers):
                        # prompt ==> generation
                        sender = self.pworkers.get(msg.sender_id)
                        # Balance SWAP from prefill to generation
                        num_requests = len(msg.requests)
                        num_per_worker = math.ceil(num_requests / len(self.gworkers))
                        for i in range(0, num_requests, num_per_worker):
                            worker = self.gworkers.schedule()
                            requests = msg.requests[i : i + num_per_worker]
                            if sender is worker:
                                self.send_task(worker, Task.SWAP_LOCAL, requests, msg.metadata)
                            else:
                                num_blocks_to_swap = msg.metadata.num_blocks_to_swap_out
                                event = self.env.event()
                                self.send_task(
                                    sender,
                                    Task.SWAP_OUT_REMOTE,
                                    requests,
                                    SwapMetadata(worker.id, 0, num_blocks_to_swap, event),
                                )
                                self.send_task(
                                    worker,
                                    Task.SWAP_IN_REMOTE,
                                    requests,
                                    SwapMetadata(sender.id, num_blocks_to_swap, 0, event),
                                )
                    else:
                        # generation ==> generation
                        sender = self.gworkers.get(msg.sender_id)
                        worker = self.gworkers.schedule()
                        if sender is worker:
                            self.send_task(worker, Task.SWAP_LOCAL, msg.requests, msg.metadata)
                        else:
                            num_blocks_to_swap = msg.metadata.num_blocks_to_swap_out
                            self.send_task(
                                sender,
                                Task.SWAP_OUT_REMOTE,
                                msg.requests,
                                SwapMetadata(worker.id, 0, num_blocks_to_swap),
                            )
                            self.send_task(
                                worker,
                                Task.SWAP_IN_REMOTE,
                                msg.requests,
                                SwapMetadata(sender.id, num_blocks_to_swap, 0),
                            )
                    self.trace_all_memory_usage()
                case Task.STOP:
                    for worker in self.workers:
                        self.send_task(worker, Task.STOP)
                    break

    def add_requests(self, requests: list[Request]):
        self.send_task(self, Task.ADD, requests)

    def add_requests_burst(self, requests: list[Request]):
        worker = self.pworkers.schedule()
        self.send_task(worker, Task.ADD, requests)
