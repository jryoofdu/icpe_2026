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
            self.stats.hits += 1
            self._update_lru(token_sequence)
            return self.cache[token_sequence]
        self.stats.misses += 1
        return None
        
    def put(self, token_sequence: Tuple, block_ids: List[int]) -> Optional[Tuple]:
        """Add token sequence to cache, evicting LRU if needed"""
        if token_sequence in self.cache:
            self._update_lru(token_sequence)
            self.cache[token_sequence] = block_ids
            return None
            
        evicted_sequence = None
        if len(self.cache) >= self.capacity:
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
