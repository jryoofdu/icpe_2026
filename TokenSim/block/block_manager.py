from TokenSim.llm.llm_request import Request
from TokenSim.block.block import Device, PhysicalTokenBlock
from TokenSim.block.shared_cache import SharedMemoryCache
from typing import Tuple, Dict, Set, Optional


class BlockAllocator:
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        num_blocks: int,
    ) -> None:
        self.num_blocks: int = num_blocks
        self.free_blocks = set(range(num_blocks))

    def allocate(self) -> int | None:
        """Allocate a block ID.
        
        Returns:
            int | None: The allocated block ID, or None if no blocks are available.
        """
        if not self.free_blocks:
            return None
        return self.free_blocks.pop()

    def free(self, block_id: int) -> None:
        """Free a block ID.
        
        Args:
            block_id: The ID of the block to free.
        """
        self.free_blocks.add(block_id)

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks.
        
        Returns:
            int: The number of free blocks.
        """
        return len(self.free_blocks)

    def get_num_allocated_blocks(self) -> int:
        """Get the number of allocated blocks.
        
        Returns:
            int: The number of allocated blocks.
        """
        return self.num_blocks - len(self.free_blocks)

    def get_status(self) -> tuple[int, int, int]:
        """Get the status of the allocator.
        
        Returns:
            tuple[int, int, int]: A tuple containing (free_blocks, allocated_blocks, total_blocks).
        """
        return (
            len(self.free_blocks),
            self.num_blocks - len(self.free_blocks),
            self.num_blocks,
        )


class BlockManager:
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        shared_cache: bool = False,
        debug: bool = False,
    ):
        self.block_size = block_size
        self.shared_cache = shared_cache
        self.debug = debug

        # Convert to integers to avoid type errors
        num_gpu_blocks = int(num_gpu_blocks)
        num_cpu_blocks = int(num_cpu_blocks)

        # Initialize GPU and CPU blocks
        self.gpu_blocks = {
            i: PhysicalTokenBlock(Device.GPU, i, block_size)
            for i in range(num_gpu_blocks)
        }
        self.cpu_blocks = {
            i: PhysicalTokenBlock(Device.CPU, i, block_size)
            for i in range(num_cpu_blocks)
        }

        # Initialize allocators
        self.gpu_allocator = BlockAllocator(num_gpu_blocks)
        self.cpu_allocator = BlockAllocator(num_cpu_blocks)

        # Initialize SharedMemoryCache for LRU eviction
        if self.shared_cache:
            # Use GPU blocks as cache capacity
            self.shared_memory_cache = SharedMemoryCache(capacity=num_gpu_blocks)
            # Map token sequences to list of block IDs for tracking
            self.token_to_block_map: Dict[tuple, list[int]] = {}
        else:
            self.shared_memory_cache = None
            self.token_to_block_map = {}

        # Legacy cache statistics (kept for backward compatibility)
        self.cache_hits = 0
        self.cache_misses = 0

        if self.debug or True:
            print(f"[BlockManager] shared_cache={self.shared_cache}, debug={self.debug}")

    def get_cache_stats(self) -> dict:
        """Get cache hit/miss statistics.
        
        Returns:
            dict: A dictionary containing cache statistics including:
                - hits: Number of cache hits
                - misses: Number of cache misses
                - hit_rate: Cache hit rate as a percentage
                - total: Total number of allocations
                - evictions: Number of cache evictions (if using SharedMemoryCache)
        """
        if self.shared_cache and self.shared_memory_cache:
            # Use SharedMemoryCache statistics
            cache_stats = self.shared_memory_cache.get_stats()
            return {
                "hits": cache_stats.hits,
                "misses": cache_stats.misses,
                "hit_rate": cache_stats.hit_rate * 100,  # Convert to percentage
                "total": cache_stats.total_accesses,
                "evictions": cache_stats.evictions
            }
        else:
            # Use legacy statistics
            total = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
            
            return {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": hit_rate,
                "total": total,
                "evictions": 0
            }

    def get_num_free_blocks(self, device: Device) -> int:
        return sum(1 for b in self.device_blocks[device] if b.ref_count == 0)

    def can_allocate(self, req: Request) -> bool:
        """Check if we can allocate blocks for a request."""
        num_blocks = (req.prompt_len + req.block_size - 1) // req.block_size
        if self.debug:
            print(f"can_allocate: required_blocks={num_blocks}, free_gpu_blocks={self.gpu_allocator.get_num_free_blocks()}")
        
        # If using SharedMemoryCache, we can always allocate (eviction will handle capacity)
        if self.shared_cache and self.shared_memory_cache:
            return True
        
        # Otherwise, check if we have enough free blocks
        return self.gpu_allocator.get_num_free_blocks() >= num_blocks

    def allocate(self, req: Request) -> None:
        """Allocate blocks for a request."""
        if self.debug:
            print(f"allocate: allocating blocks for request {req.id}")
            print(f"Prompt length: {req.prompt_len}, Block size: {req.block_size}")

        if self.shared_cache and self.shared_memory_cache:
            # Use SharedMemoryCache for LRU eviction
            self._allocate_with_shared_cache(req)
        else:
            # Use legacy allocation method
            self._allocate_legacy(req)

    def _allocate_with_shared_cache(self, req: Request) -> None:
        """Allocate blocks using SharedMemoryCache with LRU eviction."""
        req_tokens = req.get_token_sequence()
        
        # Try to get from cache first
        cached_tokens = self.shared_memory_cache.get(req_tokens)
        if cached_tokens is not None:
            # Cache hit - reuse existing blocks
            if self.debug:
                print(f"Cache hit for request {req.id}")
            
            # Find the blocks that were previously allocated for this token sequence
            if req_tokens in self.token_to_block_map:
                block_ids = self.token_to_block_map[req_tokens]
                for block_id in block_ids:
                    if block_id in self.gpu_blocks:
                        block = self.gpu_blocks[block_id]
                        req._append_physical_block(block)
                        block.ref_count += 1
                req.cache_hit = True
                return
            
            # If block not found, treat as miss
            if self.debug:
                print(f"Block not found for cached tokens, treating as miss")
        
        # Cache miss - allocate new blocks
        if self.debug:
            print(f"Cache miss for request {req.id}, allocating new blocks")
        
        num_blocks = (req.prompt_len + req.block_size - 1) // req.block_size
        
        # Check if we need to evict blocks due to capacity constraints
        if self.gpu_allocator.get_num_free_blocks() < num_blocks:
            # Need to evict some blocks to make space
            blocks_to_evict = num_blocks - self.gpu_allocator.get_num_free_blocks()
            self._evict_blocks(blocks_to_evict)
        
        # Allocate new blocks
        allocated_block_ids = []
        for _ in range(num_blocks):
            block_id = self.gpu_allocator.allocate()
            if block_id is None:
                raise RuntimeError("Failed to allocate GPU block")
            allocated_block_ids.append(block_id)
            block = self.gpu_blocks[block_id]
            req._append_physical_block(block)
            block.ref_count = 1
            block.tokens = req_tokens
        
        # Store the list of block IDs for this token sequence
        if allocated_block_ids:
            self.token_to_block_map[req_tokens] = allocated_block_ids
        
        # Add to shared cache
        self.shared_memory_cache.put(req_tokens, allocated_block_ids)
        req.cache_hit = False

    def _allocate_legacy(self, req: Request) -> None:
        """Legacy allocation method without SharedMemoryCache."""
        # First try to find a shared block
        shared_block = self._find_shared_block(req)
        if shared_block is not None:
            if self.debug:
                print(f"Found shared block {shared_block.block_number}")
            self.cache_hits += 1
            req._append_physical_block(shared_block)
            shared_block.ref_count += 1
            req.cache_hit = True
            return

        # If no shared block found, allocate new blocks
        self.cache_misses += 1
        req.cache_hit = False
        num_blocks = (req.prompt_len + req.block_size - 1) // req.block_size
        if self.debug:
            print(f"Allocating {num_blocks} new blocks")

        for _ in range(num_blocks):
            block_id = self.gpu_allocator.allocate()
            if block_id is None:
                raise RuntimeError("Failed to allocate GPU block")
            block = self.gpu_blocks[block_id]
            req._append_physical_block(block)
            block.ref_count = 1
            block.tokens = req.get_token_sequence()  # Store the token sequence

    def _evict_blocks(self, num_blocks_to_evict: int) -> None:
        """Evict blocks using LRU policy from SharedMemoryCache."""
        if not self.shared_cache or not self.shared_memory_cache:
            return
        
        # Get the LRU list from the cache
        lru_list = self.shared_memory_cache.lru.copy()
        
        # Evict the least recently used blocks
        evicted_blocks = 0
        i = 0
        while evicted_blocks < num_blocks_to_evict and i < len(lru_list):
            token_sequence = lru_list[i]
            
            # Remove from cache
            if token_sequence in self.shared_memory_cache.cache:
                del self.shared_memory_cache.cache[token_sequence]
            
            # Remove from LRU list
            if token_sequence in self.shared_memory_cache.lru:
                self.shared_memory_cache.lru.remove(token_sequence)
            
            # Free all associated blocks
            if token_sequence in self.token_to_block_map:
                block_ids = self.token_to_block_map[token_sequence]
                for block_id in block_ids:
                    if block_id in self.gpu_blocks:
                        block = self.gpu_blocks[block_id]
                        if block.ref_count > 0:
                            block.ref_count -= 1
                            if block.ref_count == 0:
                                self.gpu_allocator.free(block_id)
                                evicted_blocks += 1
                # Remove from token mapping
                del self.token_to_block_map[token_sequence]
            
            if self.debug:
                print(f"Evicted block(s) for token sequence {token_sequence}")
            i += 1

    def _find_shared_block(self, req: Request) -> PhysicalTokenBlock | None:
        """Find a shared block that can be used for this request."""
        if not self.shared_cache:
            return None

        req_tokens = req.get_token_sequence()
        for block in self.gpu_blocks.values():
            if block.ref_count > 0 and getattr(block, 'tokens', None) == req_tokens:
                return block
        return None

    def can_append_slot(self, num_blocks: int = 0) -> bool:
        """Check if we can append a slot to the request."""
        if self.debug:
            print(f"can_append_slot: num_blocks={num_blocks}, free_gpu_blocks={self.gpu_allocator.get_num_free_blocks()}")
        return self.gpu_allocator.get_num_free_blocks() >= num_blocks

    def append_slot(self, req: Request) -> None:
        """Append a new slot to the request."""
        if self.debug:
            print(f"append_slot: appending slot for request {req.id}")

        block_id = self.gpu_allocator.allocate()
        if block_id is None:
            raise RuntimeError("Failed to allocate GPU block")
        block = self.gpu_blocks[block_id]
        req._append_physical_block(block)
        block.ref_count = 1

    def can_swap_in(self, req: Request, watermark: int | None = None) -> bool:
        """Check if we can swap in blocks for a request."""
        num_required_blocks = req.num_physical_token_blocks
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        
        if self.debug:
            print(f"can_swap_in: required_blocks={num_required_blocks}, free_gpu_blocks={num_free_gpu_blocks}")
        
        if self.shared_cache:
            # In shared cache mode, we can always swap in if we have shared blocks
            return True
            
        if watermark is not None:
            return num_free_gpu_blocks - num_required_blocks >= watermark
        else:
            return num_free_gpu_blocks - num_required_blocks >= self.gpu_allocator.get_num_free_blocks()

    def swap_in(self, req: Request) -> None:
        """Swap in blocks from CPU to GPU."""
        if self.debug:
            print(f"swap_in: swapping in blocks for request {req.id}")
            print(f"Number of blocks to swap in: {len(req._physical_token_blocks)}")

        for block in req._physical_token_blocks:
            if block.device == Device.CPU:
                self.cpu_allocator.free(block.block_number)
                block_id = self.gpu_allocator.allocate()
                if block_id is None:
                    raise RuntimeError("Failed to allocate GPU block")
                block = self.gpu_blocks[block_id]
                block.ref_count = 1

    def remote_swap_in_swapped(self, req: Request) -> None:
        """Handle remote swap in for swapped requests."""
        if self.debug:
            print(f"remote_swap_in_swapped: handling remote swap in for request {req.id}")

        for block in req._physical_token_blocks:
            if block.device == Device.CPU:
                self.cpu_allocator.free(block.block_number)
                block_id = self.gpu_allocator.allocate()
                if block_id is None:
                    raise RuntimeError("Failed to allocate GPU block")
                block = self.gpu_blocks[block_id]
                block.ref_count = 1

    def remote_swap_in_running(self, req: Request) -> None:
        """Handle remote swap in for running requests."""
        if self.debug:
            print(f"remote_swap_in_running: handling remote swap in for request {req.id}")

        for block in req._physical_token_blocks:
            if block.device == Device.CPU:
                self.cpu_allocator.free(block.block_number)
                block_id = self.gpu_allocator.allocate()
                if block_id is None:
                    raise RuntimeError("Failed to allocate GPU block")
                block = self.gpu_blocks[block_id]
                block.ref_count = 1

    def swap_out(self, req: Request) -> None:
        """Swap out blocks from GPU to CPU."""
        if self.debug:
            print(f"swap_out: swapping out blocks for request {req.id}")
            print(f"Number of blocks to swap out: {len(req._physical_token_blocks)}")

        for block in req._physical_token_blocks:
            if block.device == Device.GPU:
                self.gpu_allocator.free(block.block_number)
                block_id = self.cpu_allocator.allocate(block.block_number)
                if block_id is None:
                    raise RuntimeError("Failed to allocate CPU block")
                block = self.cpu_blocks[block_id]
                block.ref_count = 1

    def remote_swap_out(self, req: Request) -> None:
        """Handle remote swap out."""
        if self.debug:
            print(f"remote_swap_out: handling remote swap out for request {req.id}")

        for block in req._physical_token_blocks:
            if block.device == Device.GPU:
                self.gpu_allocator.free(block.block_number)
                block_id = self.cpu_allocator.allocate(block.block_number)
                if block_id is None:
                    raise RuntimeError("Failed to allocate CPU block")
                block = self.cpu_blocks[block_id]
                block.ref_count = 1

    def try_allocate(self, num_blocks: int) -> None:
        """Try to allocate a specific number of blocks."""
        if self.debug:
            print(f"try_allocate: attempting to allocate {num_blocks} blocks")
            print(f"Free GPU blocks: {self.gpu_allocator.get_num_free_blocks()}")

        for _ in range(num_blocks):
            block_id = self.gpu_allocator.allocate()
            if block_id is None:
                break

    def get_gpu_status(self) -> tuple[int, int, int]:
        """Get GPU memory status."""
        free_blocks = self.gpu_allocator.get_num_free_blocks()
        used_blocks = len(self.gpu_blocks) - free_blocks
        total_blocks = len(self.gpu_blocks)
        if self.debug:
            print(f"get_gpu_status: free={free_blocks}, used={used_blocks}, total={total_blocks}")
        return free_blocks, used_blocks, total_blocks

    def free(self, req: Request) -> None:
        """Free blocks allocated to a request."""
        if self.debug:
            print(f"free: freeing blocks for request {req.id}")
            print(f"Number of blocks to free: {len(req._physical_token_blocks)}")

        for block in req._physical_token_blocks:
            block.ref_count -= 1
            if block.ref_count == 0:
                if block.device == Device.GPU:
                    self.gpu_allocator.free(block.block_number)
                    
                    # If using SharedMemoryCache, check if we need to remove from cache
                    if self.shared_cache and self.shared_memory_cache and hasattr(block, 'tokens'):
                        token_sequence = block.tokens
                        
                        # Check if this token sequence is still referenced by other blocks
                        still_referenced = False
                        if token_sequence in self.token_to_block_map:
                            block_ids = self.token_to_block_map[token_sequence]
                            for block_id in block_ids:
                                if block_id != block.block_number:
                                    other_block = self.gpu_blocks.get(block_id)
                                    if other_block and other_block.ref_count > 0:
                                        still_referenced = True
                                        break
                        
                        # If no other blocks reference this token sequence, remove from cache
                        if not still_referenced:
                            if token_sequence in self.shared_memory_cache.cache:
                                del self.shared_memory_cache.cache[token_sequence]
                            if token_sequence in self.shared_memory_cache.lru:
                                self.shared_memory_cache.lru.remove(token_sequence)
                            if token_sequence in self.token_to_block_map:
                                del self.token_to_block_map[token_sequence]
                            
                            if self.debug:
                                print(f"Removed token sequence {token_sequence} from cache")
                else:
                    self.cpu_allocator.free(block.block_number)
        req._physical_token_blocks.clear()
