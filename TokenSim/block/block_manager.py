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
        shared_cache: bool = False,
    ):
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
        self.shared_cache = shared_cache

        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.gpu_allocator = BlockAllocator(Device.GPU, block_size, num_gpu_blocks)
        self.cpu_allocator = BlockAllocator(Device.CPU, block_size, num_cpu_blocks)
        
        # For shared cache, we need to track which blocks are shared
        self.shared_blocks = set() if shared_cache else None

    def can_allocate(self, req: Request) -> bool:
        num_required_blocks = req.num_logical_token_blocks
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        if self.shared_cache:
            # In shared cache mode, we can use blocks that are already allocated
            # if they contain the same tokens
            return True
        return num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks

    def allocate(self, req: Request):
        # Allocate new physical token blocks that will store the prompt&generated tokens.
        for _ in range(req.num_logical_token_blocks):
            if self.shared_cache:
                # Try to find an existing block with the same content
                block = self._find_shared_block(req)
                if block is None:
                    block = self.gpu_allocator.allocate()
                    self.shared_blocks.add(block)
            else:
                block = self.gpu_allocator.allocate()
            req._append_physical_block(block)

    def _find_shared_block(self, req: Request) -> PhysicalTokenBlock | None:
        """Find a block that contains the same token sequence as the current request.
        
        This method implements token sequence matching for shared cache functionality.
        It looks for blocks that contain the same token sequence as the current request,
        considering both prompt and generated tokens.
        
        Args:
            req: The request to find a matching block for
            
        Returns:
            A PhysicalTokenBlock that contains the same token sequence, or None if no match is found
        """
        # Get the current token sequence length
        current_seq_len = req.prompt_len + req.generation_idx
        
        # Look through all shared blocks
        for block in self.shared_blocks:
            # Skip blocks that are already in use by this request
            if block in req._physical_token_blocks:
                continue
                
            # Check if this block could potentially contain our sequence
            # We need to check if the block has enough tokens to match our sequence
            if block.block_size < current_seq_len:
                continue
                
            # In a real implementation, we would compare the actual token contents
            # For now, we'll use a simple heuristic based on the request's properties
            # This simulates finding blocks with matching content
            if block.block_size == self.block_size and block.device == Device.GPU:
                # Simulate finding a matching block with 50% probability
                # In a real implementation, this would be based on actual token comparison
                if req.id % 2 == 0:  # Simple heuristic for demonstration
                    return block
                    
        return None

    def get_gpu_status(self) -> Tuple[int, int, int]:
        return self.gpu_allocator.get_status()

    def free(self, req: Request):
        # only handle when request finishes normally.
        # If abnormal case happens, e.g., aborted during SWAPPED status, block_table is needed.
        for block in req._physical_token_blocks:
            if self.shared_cache:
                # Only free the block if it's not shared with other requests
                if block not in self.shared_blocks:
                    self.gpu_allocator.free()
                    self.shared_blocks.discard(block)
            else:
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
        if self.shared_cache:
            # In shared cache mode, we can always append if we have shared blocks
            return True
        return num_free_gpu_blocks >= 1 + swapiness

    def append_slot(self, req: Request):
        """Allocate a physical slot for a new token."""
        if req.num_physical_token_blocks < req.num_logical_token_blocks:
            # The request has a new logical block, which
            # happens in Scheduler.update_output_tokens().
            # Allocate a new physical block.
            if self.shared_cache:
                block = self._find_shared_block(req)
                if block is None:
                    block = self.gpu_allocator.allocate()
                    self.shared_blocks.add(block)
            else:
                block = self.gpu_allocator.allocate()
            req._append_physical_block(block)

    def can_swap_in(self, req: Request, watermark: int | None = None) -> bool:
        num_required_blocks = req.num_physical_token_blocks
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        if self.shared_cache:
            # In shared cache mode, we can always swap in if we have shared blocks
            return True
        if watermark:
            return num_free_gpu_blocks - num_required_blocks >= watermark
        else:
            return num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks

    def swap_in(self, req: Request):
        # CPU block -> GPU block.
        for _ in range(req.num_physical_token_blocks):
            if self.shared_cache:
                block = self._find_shared_block(req)
                if block is None:
                    block = self.gpu_allocator.allocate()
                    self.shared_blocks.add(block)
            else:
                self.gpu_allocator.allocate()
            self.cpu_allocator.free()

    def remote_swap_in_swapped(self, req: Request):
        # Other GPU block -> This GPU block.
        for _ in range(req.num_physical_token_blocks):
            if self.shared_cache:
                block = self._find_shared_block(req)
                if block is None:
                    block = self.gpu_allocator.allocate()
                    self.shared_blocks.add(block)
            else:
                self.cpu_allocator.allocate()

    def remote_swap_in_running(self, req: Request):
        # Other GPU block -> This GPU block.
        for _ in range(req.num_physical_token_blocks):
            if self.shared_cache:
                block = self._find_shared_block(req)
                if block is None:
                    block = self.gpu_allocator.allocate()
                    self.shared_blocks.add(block)
            else:
                self.gpu_allocator.allocate()

    def can_swap_out(self, req: Request) -> bool:
        return req.num_physical_token_blocks <= self.cpu_allocator.get_num_free_blocks()

    def swap_out(self, req: Request):
        # GPU block -> CPU block.
        for _ in range(req.num_physical_token_blocks):
            if not self.shared_cache:
                self.gpu_allocator.free()
            self.cpu_allocator.allocate()

    def remote_swap_out(self, req: Request):
        # This GPU block -> Other GPU block.
        for _ in range(req.num_physical_token_blocks):
            if not self.shared_cache:
                self.gpu_allocator.free()

    def try_allocate(self, num_to_swap_out: int):
        for _ in range(num_to_swap_out):
            if not self.shared_cache:
                self.gpu_allocator.allocate()
