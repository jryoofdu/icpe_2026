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
    """Implements a shared memory cache with LRU eviction policy"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: Dict[int, List[int]] = {}  # block_id -> [token_ids]
        self.lru: List[int] = []  # Tracks LRU order of blocks
        self.stats = CacheStats()
        
    def get(self, block_id: int) -> Optional[List[int]]:
        """Get block from cache, updating LRU order"""
        if block_id in self.cache:
            self.stats.hits += 1
            self._update_lru(block_id)
            return self.cache[block_id]
        self.stats.misses += 1
        return None
        
    def put(self, block_id: int, tokens: List[int]) -> Optional[int]:
        """Add block to cache, evicting LRU if needed"""
        if block_id in self.cache:
            self._update_lru(block_id)
            self.cache[block_id] = tokens
            return None
            
        evicted_block = None
        if len(self.cache) >= self.capacity:
            evicted_block = self.lru.pop(0)
            del self.cache[evicted_block]
            self.stats.evictions += 1
            
        self.cache[block_id] = tokens
        self.lru.append(block_id)
        return evicted_block
        
    def _update_lru(self, block_id: int):
        """Update LRU order for accessed block"""
        self.lru.remove(block_id)
        self.lru.append(block_id)
        
    def clear(self):
        """Clear cache contents and reset stats"""
        self.cache.clear()
        self.lru.clear()
        self.stats = CacheStats()
        
    def get_stats(self) -> CacheStats:
        """Get current cache statistics"""
        return self.stats 