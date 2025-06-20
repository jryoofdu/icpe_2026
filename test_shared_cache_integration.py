#!/usr/bin/env python3
"""
Test script to verify SharedMemoryCache integration with BlockManager.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'TokenSim'))

from TokenSim.block.block_manager import BlockManager
from TokenSim.llm.llm_request import Request
from TokenSim.block.block import Device

def test_shared_cache_integration():
    """Test the integration of SharedMemoryCache with BlockManager."""
    
    print("Testing SharedMemoryCache integration with BlockManager...")
    
    # Create a BlockManager with shared cache enabled
    block_size = 128
    num_gpu_blocks = 10
    num_cpu_blocks = 20
    
    print(f"Creating BlockManager with:")
    print(f"  - Block size: {block_size}")
    print(f"  - GPU blocks: {num_gpu_blocks}")
    print(f"  - CPU blocks: {num_cpu_blocks}")
    print(f"  - Shared cache: True")
    
    block_manager = BlockManager(
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=num_cpu_blocks,
        shared_cache=True,
        debug=True
    )
    
    # Verify SharedMemoryCache is initialized
    assert block_manager.shared_memory_cache is not None, "SharedMemoryCache should be initialized"
    print("✓ SharedMemoryCache initialized successfully")
    
    # Create test requests with different token sequences
    requests = [
        Request(id=0, prompt_len=100, generation_len=50, block_size=block_size),
        Request(id=1, prompt_len=150, generation_len=75, block_size=block_size),
        Request(id=2, prompt_len=200, generation_len=100, block_size=block_size),
        Request(id=3, prompt_len=100, generation_len=50, block_size=block_size),  # Same as request 0
        Request(id=4, prompt_len=300, generation_len=150, block_size=block_size),
    ]
    
    print(f"\nTesting with {len(requests)} requests...")
    
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
        else:
            print("Cannot allocate - insufficient blocks")
            break
    
    # Get cache statistics
    print(f"\n--- Cache Statistics ---")
    cache_stats = block_manager.get_cache_stats()
    print(f"Hits: {cache_stats['hits']}")
    print(f"Misses: {cache_stats['misses']}")
    print(f"Hit Rate: {cache_stats['hit_rate']:.2f}%")
    print(f"Total: {cache_stats['total']}")
    print(f"Evictions: {cache_stats.get('evictions', 0)}")
    
    # Free some requests to test cache cleanup
    print(f"\n--- Freeing Request 1 ---")
    block_manager.free(requests[1])
    
    # Get updated cache statistics
    print(f"\n--- Updated Cache Statistics ---")
    cache_stats = block_manager.get_cache_stats()
    print(f"Hits: {cache_stats['hits']}")
    print(f"Misses: {cache_stats['misses']}")
    print(f"Hit Rate: {cache_stats['hit_rate']:.2f}%")
    print(f"Total: {cache_stats['total']}")
    print(f"Evictions: {cache_stats.get('evictions', 0)}")
    
    # Test GPU status
    print(f"\n--- GPU Status ---")
    free_blocks, used_blocks, total_blocks = block_manager.get_gpu_status()
    print(f"Free blocks: {free_blocks}")
    print(f"Used blocks: {used_blocks}")
    print(f"Total blocks: {total_blocks}")
    
    print(f"\n✓ SharedMemoryCache integration test completed successfully!")

if __name__ == "__main__":
    test_shared_cache_integration() 