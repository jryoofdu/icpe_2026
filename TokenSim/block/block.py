"""Token blocks."""

from typing import List
from enum import Enum, auto


class Device(Enum):
    GPU = auto()
    CPU = auto()


class LogicalTokenBlock:
    """A block that stores a contiguous chunk of tokens from left to right.

    Logical blocks are used to represent the states of the corresponding
    physical blocks in the KV cache.
    """

    def __init__(
        self,
        block_id: int,
        block_size: int,
    ) -> None:
        self.block_id = block_id
        self.block_size = block_size

        self.num_tokens = 0

    def is_empty(self) -> bool:
        return self.num_tokens == 0

    def get_num_empty_slots(self) -> int:
        return self.block_size - self.num_tokens

    def is_full(self) -> bool:
        return self.num_tokens == self.block_size

    def append_tokens(self, num_new_tokens) -> None:
        assert num_new_tokens <= self.get_num_empty_slots()
        self.num_tokens += num_new_tokens


class PhysicalTokenBlock:
    """Represents the state of a block in the KV cache."""

    def __init__(
        self,
        device: Device,
        block_id: int,
        block_size: int,
    ) -> None:
        self.device = device
        self.block_number = block_id
        self.block_size = block_size

        self.ref_count = 0
        self.tokens = None  # Store the token sequence for cache matching

    def __repr__(self) -> str:
        return (
            f"PhysicalTokenBlock(device={self.device}, "
            f"block_number={self.block_number}, "
            f"ref_count={self.ref_count})"
        )
