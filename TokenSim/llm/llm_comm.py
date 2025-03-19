from dataclasses import dataclass

import simpy
from TokenSim.llm.llm_request import Request


@dataclass
class SwapMetadata:
    remote_id: int
    num_blocks_to_swap_in: int
    num_blocks_to_swap_out: int
    event: simpy.Event = None


@dataclass
class SwapInMetadata:
    to_running: list[Request]
    to_swapped: list[Request]
