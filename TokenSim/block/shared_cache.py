from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import logging
from TokenSim.block.block import PhysicalTokenBlock
from TokenSim.llm.llm_request import Request

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """Statistics for cache performance monitoring"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    
    times: List[float] = None   # will hold per-access durations

    def __post_init__(self):
        if self.times is None:
            self.times = []

    @property
    def total_accesses(self) -> int:
        return self.hits + self.misses
    
    @property
    def hit_rate(self) -> float:
        if self.total_accesses == 0:
            return 0.0
        return self.hits / self.total_accesses
    
    @property
    def miss_rate(self) -> float:
        if self.total_accesses == 0:
            return 0.0
        return self.misses / self.total_accesses

    def quantiles(self) -> Dict[str, float]:
        """Return p50, p99, and max of recorded times (ms)."""
        import numpy as _np
        arr = _np.array(self.times)
        if arr.size == 0:
            return {"p50": 0.0, "p99": 0.0, "max": 0.0}
        p50, p99 = _np.percentile(arr, [50, 99])
        return {"p50": float(p50), "p99": float(p99), "max": float(arr.max())}


class SharedMemoryCache:
    """Implements a shared memory cache with LRU eviction policy for token sequences"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: Dict[Tuple, List[int]] = {}  # token_sequence -> [block_ids]
        self.lru: List[Tuple] = []  # Tracks LRU order of token sequences
        self.stats = CacheStats()
        
    def get(self, token_sequence: Tuple) -> Optional[List[int]]:

        
        """Get token sequence from cache, updating LRU order"""
        if token_sequence in self.cache:
            print(f"Cache hit for token sequence: {token_sequence}")
            print(f"Cache hit for token sequence: {token_sequence}")
            self.stats.hits += 1
            self._update_lru(token_sequence)
            return self.cache[token_sequence]
        self.stats.misses += 1


        return None
        
    def put(self, token_sequence: Tuple, block_ids: List[int]) -> Optional[Tuple]:
        
        """Add token sequence to cache, evicting LRU if needed"""
        if token_sequence in self.cache:
            print(f"Updating cache for token sequence in self cache: {token_sequence}")
            print(f"Updating cache for token sequence in self cache: {token_sequence}")
            self._update_lru(token_sequence)
            self.cache[token_sequence] = block_ids
            return None
            
        evicted_sequence = None
        if len(self.cache) >= self.capacity:
            print(f"Cache full, evicting least recently used token sequence")
            print(f"Cache full, evicting least recently used token sequence")
            evicted_sequence = self.lru.pop(0)
            del self.cache[evicted_sequence]
            self.stats.evictions += 1
            
        self.cache[token_sequence] = block_ids
        self.lru.append(token_sequence)

        return evicted_sequence
        
    def _update_lru(self, token_sequence: Tuple):
        """Update LRU order for accessed token sequence"""
        if token_sequence in self.lru:
            self.lru.remove(token_sequence)
        self.lru.append(token_sequence)
        
    def clear(self):
        """Clear cache contents and reset stats"""
        self.cache.clear()
        self.lru.clear()
        self.stats = CacheStats()
        
    def get_stats(self) -> CacheStats:
        """Get current cache statistics"""
        return self.stats

class SharedMemoryCacheLFU:
    """Implements a shared memory cache with LFU eviction policy"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: Dict[int, List[int]] = {}          # block_id -> [token_ids]
        self.freq: Dict[int, int] = {}                 # block_id -> frequency count
        self.stats = CacheStats()

    def get(self, block_id: int) -> Optional[List[int]]:
        """Get block from cache, updating usage frequency"""
        if block_id in self.cache:
            self.stats.hits += 1
            self.freq[block_id] += 1
            return self.cache[block_id]
        self.stats.misses += 1
        return None

    def put(self, block_id: int, tokens: List[int]) -> Optional[int]:
        """Add block to cache with LFU eviction if needed"""
        evicted_block = None

        if block_id in self.cache:
            self.cache[block_id] = tokens
            self.freq[block_id] += 1
            return None

        if len(self.cache) >= self.capacity:
            # Find block with the lowest frequency
            min_freq = min(self.freq.values())
            # Among those, evict the first block with min frequency
            for bid in self.cache:
                if self.freq[bid] == min_freq:
                    evicted_block = bid
                    break
            del self.cache[evicted_block]
            del self.freq[evicted_block]
            self.stats.evictions += 1

        self.cache[block_id] = tokens
        self.freq[block_id] = 1
        return evicted_block

    def clear(self):
        """Clear cache contents and reset stats"""
        self.cache.clear()
        self.freq.clear()
        self.stats = CacheStats()

    def get_stats(self) -> CacheStats:
        return self.stats
