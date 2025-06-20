#!/usr/bin/env python3
"""
Test script to demonstrate LRU eviction functionality in SharedMemoryCache.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'TokenSim'))

from TokenSim.block.block_manager import BlockManager
from TokenSim.llm.llm_request import Request

def test_lru_eviction():
    """Test LRU eviction when cache capacity is exceeded."""
    
    print("Testing LRU eviction functionality...")
    
    # Create a BlockManager with very limited GPU blocks to force eviction
    block_size = 128
    num_gpu_blocks = 3  # Very small capacity to force eviction
    num_cpu_blocks = 10
    
    print(f"Creating BlockManager with limited capacity:")
    print(f"  - Block size: {block_size}")
    print(f"  - GPU blocks: {num_gpu_blocks} (limited)")
    print(f"  - CPU blocks: {num_cpu_blocks}")
    print(f"  - Shared cache: True")
    
    block_manager = BlockManager(
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=num_cpu_blocks,
        shared_cache=True,
        debug=True
    )
    
    # Create requests that will exceed the cache capacity
    requests = [
        Request(id=0, prompt_len=100, generation_len=50, block_size=block_size),   # 1 block
        Request(id=1, prompt_len=150, generation_len=75, block_size=block_size),   # 2 blocks
        Request(id=2, prompt_len=200, generation_len=100, block_size=block_size),  # 2 blocks (will cause eviction)
        Request(id=3, prompt_len=100, generation_len=50, block_size=block_size),   # 1 block (should be cache hit)
        Request(id=4, prompt_len=300, generation_len=150, block_size=block_size),  # 3 blocks (will cause more evictions)
    ]
    
    print(f"\nTesting with {len(requests)} requests that will exceed cache capacity...")
    
    # Allocate blocks for each request
    for i, req in enumerate(requests):
        print(f"\n--- Allocating blocks for Request {i} ---")
        print(f"Request {i}: prompt_len={req.prompt_len}, generation_len={req.generation_len}")
        
        # Check if we can allocate
        can_allocate = block_manager.can_allocate(req)
        print(f"Can allocate: {can_allocate}")
        
        if can_allocate:
            block_manager.allocate(req)
            print(f"Allocated {req.num_physical_token_blocks} blocks")
            print(f"Cache hit: {req.cache_hit}")
            
            # Show current cache statistics
            cache_stats = block_manager.get_cache_stats()
            print(f"Current cache stats - Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}, Evictions: {cache_stats.get('evictions', 0)}")
        else:
            print("Cannot allocate - insufficient blocks")
            break
    
    # Get final cache statistics
    print(f"\n--- Final Cache Statistics ---")
    cache_stats = block_manager.get_cache_stats()
    print(f"Hits: {cache_stats['hits']}")
    print(f"Misses: {cache_stats['misses']}")
    print(f"Hit Rate: {cache_stats['hit_rate']:.2f}%")
    print(f"Total: {cache_stats['total']}")
    print(f"Evictions: {cache_stats.get('evictions', 0)}")
    
    # Test GPU status
    print(f"\n--- Final GPU Status ---")
    free_blocks, used_blocks, total_blocks = block_manager.get_gpu_status()
    print(f"Free blocks: {free_blocks}")
    print(f"Used blocks: {used_blocks}")
    print(f"Total blocks: {total_blocks}")
    
    # Test accessing the first request again (should be a cache miss due to eviction)
    print(f"\n--- Testing re-access of Request 0 (should be evicted) ---")
    req0_again = Request(id=10, prompt_len=100, generation_len=50, block_size=block_size)
    if block_manager.can_allocate(req0_again):
        block_manager.allocate(req0_again)
        print(f"Re-access cache hit: {req0_again.cache_hit}")
        
        # Show updated cache statistics
        cache_stats = block_manager.get_cache_stats()
        print(f"Updated cache stats - Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}, Evictions: {cache_stats.get('evictions', 0)}")
    
    print(f"\n✓ LRU eviction test completed successfully!")

if __name__ == "__main__":
    test_lru_eviction() 