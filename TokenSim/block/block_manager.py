from TokenSim.llm.llm_request import Request
from TokenSim.block.block import Device, PhysicalTokenBlock
from typing import Tuple


class BlockAllocator:
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
    ) -> None:
        self.device = device
        self.block_size: int = block_size
        self.num_blocks: int = num_blocks

        # Initialize the free blocks.
        self.num_free_blocks: int = self.num_blocks

    def allocate(self) -> PhysicalTokenBlock:
        if self.num_free_blocks == 0:
            raise ValueError("Out of memory! No free blocks are available.")
        self.num_free_blocks -= 1
        return PhysicalTokenBlock(self.device, 0, self.block_size)

    def free(self) -> None:
        self.num_free_blocks += 1

    def get_num_free_blocks(self) -> int:
        return self.num_free_blocks

    def get_num_allocated_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    def set_num_free_blocks(self, block_num):
        self.num_free_blocks = block_num

    def get_status(self) -> Tuple[int, int, int]:
        return (
            self.num_free_blocks,
            self.num_blocks - self.num_free_blocks,
            self.num_blocks,
        )


class BlockManager:
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
    ):
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.gpu_allocator = BlockAllocator(Device.GPU, block_size, num_gpu_blocks)
        self.cpu_allocator = BlockAllocator(Device.CPU, block_size, num_cpu_blocks)

    def can_allocate(self, req: Request) -> bool:
        num_required_blocks = req.num_logical_token_blocks
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        return num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks

    def allocate(self, req: Request):
        # Allocate new physical token blocks that will store the prompt&generated tokens.
        for _ in range(req.num_logical_token_blocks):
            block = self.gpu_allocator.allocate()
            req._append_physical_block(block)

    def get_gpu_status(self) -> Tuple[int, int, int]:
        return self.gpu_allocator.get_status()

    def free(self, req: Request):
        # only handle when request finishes normally.
        # If abnormal case happens, e.g., aborted during SWAPPED status, block_table is needed.
        for _ in range(req.num_physical_token_blocks):
            self.gpu_allocator.free()

    def can_append_slot(self, swapiness: int) -> bool:
        """
        swapiness is the number of free blocks before activating swap
        Simple heuristic:
        - eager swap: if there is at least one free block for current reqeust, we can append.
        - lazy swap: running queue will not be changed in this step as we have watermark
            protection. Swap happens ONLY when watermark for all running requets is exceeded.
        """
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        return num_free_gpu_blocks >= 1 + swapiness

    def append_slot(self, req: Request):
        """Allocate a physical slot for a new token."""
        if req.num_physical_token_blocks < req.num_logical_token_blocks:
            # The request has a new logical block, which
            # happens in Scheduler.update_output_tokens().
            # Allocate a new physical block.
            block = self.gpu_allocator.allocate()
            req._append_physical_block(block)

    def can_swap_in(self, req: Request, watermark: int | None = None) -> bool:
        num_required_blocks = req.num_physical_token_blocks
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        if watermark:
            return num_free_gpu_blocks - num_required_blocks >= watermark
        else:
            return num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks

    def swap_in(self, req: Request):
        # CPU block -> GPU block.
        for _ in range(req.num_physical_token_blocks):
            self.gpu_allocator.allocate()
            self.cpu_allocator.free()

    def remote_swap_in_swapped(self, req: Request):
        # Other GPU block -> This GPU block.
        for _ in range(req.num_physical_token_blocks):
            self.cpu_allocator.allocate()

    def remote_swap_in_running(self, req: Request):
        # Other GPU block -> This GPU block.
        for _ in range(req.num_physical_token_blocks):
            self.gpu_allocator.allocate()

    def can_swap_out(self, req: Request) -> bool:
        return req.num_physical_token_blocks <= self.cpu_allocator.get_num_free_blocks()

    def swap_out(self, req: Request):
        # GPU block -> CPU block.
        for _ in range(req.num_physical_token_blocks):
            self.cpu_allocator.allocate()
            self.gpu_allocator.free()

    def remote_swap_out(self, req: Request):
        # This GPU block -> Other GPU block.
        for _ in range(req.num_physical_token_blocks):
            self.gpu_allocator.free()

    def try_allocate(self, num_to_swap_out: int):
        for _ in range(num_to_swap_out):
            self.gpu_allocator.allocate()
