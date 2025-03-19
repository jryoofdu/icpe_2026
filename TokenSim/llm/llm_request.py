import simpy
from dataclasses import dataclass, field
from pydantic import BaseModel
from enum import Enum, auto

from TokenSim.block.block import LogicalTokenBlock, PhysicalTokenBlock


class RequestStatus(Enum):
    """Status of a sequence."""

    WAITING = auto()
    RUNNING = auto()
    SWAPPED = auto()
    SWAPPED_REMOTE = auto()
    FINISHED_STOPPED = auto()


class RequestTime(BaseModel):
    id: int
    time: list[float]
    service_time: list[float]
    batch: list[int]

    @property
    def request_time(self):
        return sum(self.time)

    @property
    def prompt_time(self):
        return self.time[0]

    @property
    def decode_time(self):
        return sum(self.time[1:]) / (len(self.time) - 1)

    @property
    def decode_max_time(self):
        return max(self.time[1:])

    @property
    def prompt_batch(self):
        return self.batch[0]

    @property
    def decode_batch(self):
        return sum(self.batch[1:]) / (len(self.batch) - 1)

    @property
    def prompt_util(self):
        return self.service_time[0] / self.time[0]

    @property
    def decode_util(self):
        return sum(self.service_time[1:]) / sum(self.time[1:])

    @property
    def prompt_idle(self):
        return 1 - self.prompt_util

    @property
    def decode_idle(self):
        return 1 - self.decode_util


class LLMTime(BaseModel):
    time: list[RequestTime] = []

    @property
    def request_time(self):
        return [t.request_time for t in self.time]

    @property
    def prompt_time(self):
        return [t.prompt_time for t in self.time]

    @property
    def decode_time(self):
        return [t.decode_time for t in self.time]

    @property
    def decode_max_time(self):
        return [t.decode_max_time for t in self.time]

    @property
    def prompt_util(self):
        return [t.prompt_util for t in self.time]

    @property
    def decode_util(self):
        return [t.decode_util for t in self.time]

    @property
    def prompt_idle(self):
        return [t.prompt_idle for t in self.time]

    @property
    def decode_idle(self):
        return [t.decode_idle for t in self.time]

    @property
    def prompt_batch(self):
        return [t.prompt_batch for t in self.time]

    @property
    def decode_batch(self):
        return [t.decode_batch for t in self.time]


g_time = LLMTime()


@dataclass
class Request:
    id: int
    prompt_len: int
    generation_len: int
    block_size: int = field(repr=False)
    generation_idx: int = 0
    status: RequestStatus = field(default=RequestStatus.WAITING, repr=False)
    time: list[float] = field(default_factory=list, repr=False)
    service_time: list[float] = field(default_factory=list, repr=False)
    batch: list[int] = field(default_factory=list, repr=False)
    tqdm_submit_func = None

    def __post_init__(self):
        self._physical_token_blocks: list[PhysicalTokenBlock] = []
        self._logical_token_blocks: list[LogicalTokenBlock] = []
        self._append_tokens(self.prompt_len)

    @property
    def is_prompt(self) -> bool:
        return self.generation_idx == 0

    @property
    def is_done(self) -> bool:
        return self.generation_idx == self.generation_len

    @property
    def num_physical_token_blocks(self):
        return len(self._physical_token_blocks)

    @property
    def num_logical_token_blocks(self):
        return len(self._logical_token_blocks)

    def _append_physical_block(self, block) -> None:
        self._physical_token_blocks.append(block)

    def _append_logical_block(self) -> None:
        """Create a new logical block and append it at the end of all blocks."""
        block = LogicalTokenBlock(
            block_id=self.num_logical_token_blocks,
            block_size=self.block_size,
        )
        self._logical_token_blocks.append(block)

    def _append_tokens(self, num_tokens: int) -> None:
        cursor = 0
        while cursor < num_tokens:
            if not self._logical_token_blocks:
                self._append_logical_block()

            last_block = self._logical_token_blocks[-1]
            if last_block.is_full():
                self._append_logical_block()
                last_block = self._logical_token_blocks[-1]

            num_empty_slots = last_block.get_num_empty_slots()
            last_block.append_tokens(min(num_empty_slots, num_tokens - cursor))
            cursor += num_empty_slots

    @property
    def arrival_time(self):
        return self.time[0]

    def arrive(self, env: simpy.Environment):
        self.time.append(env.now)

    def step(self, env: simpy.Environment, latency: float, batch: int):
        self.generation_idx += 1
        self._append_tokens(1)
        self.time.append(env.now)
        self.service_time.append(latency)
        self.batch.append(batch)
        if self.is_done:
            self.time = [self.time[i + 1] - self.time[i] for i in range(self.generation_len)]
            g_time.time.append(
                RequestTime(
                    id=self.id, time=self.time, service_time=self.service_time, batch=self.batch
                )
            )
            if self.tqdm_submit_func:
                self.tqdm_submit_func(1)
