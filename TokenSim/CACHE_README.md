# Shared Cache System Documentation

## Overview

The Shared Cache System provides a comprehensive cache hit rate calculation and performance monitoring solution for the TokenSim framework. It implements an LRU (Least Recently Used) cache with detailed statistics tracking.

## Features

- **LRU Cache Implementation**: Efficient memory management with least recently used eviction policy
- **Real-time Statistics**: Track cache hits, misses, evictions, and hit rates
- **Performance Monitoring**: Detailed latency and throughput metrics
- **Easy Integration**: Simple API for adding cache functionality to existing systems
- **Comprehensive Logging**: Detailed event tracking for debugging and analysis

## Architecture

### Core Components

1. **SharedMemoryCache**: Main cache implementation with LRU eviction
2. **CacheStats**: Statistics tracking and reporting
3. **BlockManager Integration**: Seamless integration with existing block management
4. **Performance Monitoring**: Real-time cache performance metrics

### File Structure

```
TokenSim/
├── block/
│   ├── shared_cache.py      # Main cache implementation
│   ├── block_manager.py     # Cache-integrated block manager
│   └── block.py            # Block definitions
├── config/
│   ├── config.py           # Cache configuration options
│   └── psla_config.py      # PSLA cache settings
├── llm/
│   ├── llm_engine.py       # Engine with cache support
│   ├── llm_scheduler.py    # Scheduler with cache integration
│   └── llm_request.py      # Request handling with cache
└── CACHE_README.md         # This documentation
```

## Usage

### Basic Cache Usage

```python
from TokenSim.block.shared_cache import SharedMemoryCache

# Initialize cache with capacity
cache = SharedMemoryCache(capacity=1000)

# Store data in cache
cache.put(block_id=1, tokens=[1, 2, 3, 4])

# Retrieve data from cache
tokens = cache.get(block_id=1)

# Get cache statistics
stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.2f}%")
print(f"Total accesses: {stats.total_accesses}")
```

### Integration with BlockManager

```python
from TokenSim.block.block_manager import BlockManager

# Initialize block manager with cache enabled
block_manager = BlockManager(
    capacity=1000,
    shared_cache=True,  # Enable cache
    debug=True
)

# Cache is automatically used for block operations
block_id = block_manager.allocate_block(request_id=1, token_ids=[1, 2, 3])
```

### Configuration

Enable cache in your PSLA configuration:

```python
from TokenSim.config.psla_config import PSLAConfig

config = PSLAConfig(
    name="llama-7b",
    shared_cache=True,  # Enable shared cache
    # ... other configuration options
)
```

## API Reference

### SharedMemoryCache

#### Constructor
```python
SharedMemoryCache(capacity: int)
```
- `capacity`: Maximum number of blocks the cache can hold

#### Methods

##### `get(block_id: int) -> Optional[List[int]]`
Retrieve a block from cache.
- **Returns**: Token list if found, None if miss
- **Side effects**: Updates LRU order, increments hit/miss counters

##### `put(block_id: int, tokens: List[int]) -> Optional[int]`
Store a block in cache.
- **Returns**: Evicted block ID if eviction occurred, None otherwise
- **Side effects**: May evict LRU block, updates LRU order

##### `clear()`
Clear all cache contents and reset statistics.

##### `get_stats() -> CacheStats`
Get current cache performance statistics.

### CacheStats

#### Properties

- `hits: int` - Number of cache hits
- `misses: int` - Number of cache misses
- `evictions: int` - Number of evictions
- `total_accesses: int` - Total number of cache accesses
- `hit_rate: float` - Cache hit rate (0.0 to 1.0)
- `miss_rate: float` - Cache miss rate (0.0 to 1.0)

## Performance Monitoring

### Cache Statistics

The system automatically tracks:
- **Hit Rate**: Percentage of successful cache retrievals
- **Miss Rate**: Percentage of failed cache retrievals
- **Eviction Count**: Number of blocks evicted due to capacity limits
- **Total Accesses**: Total number of cache operations

### Integration with Results

Cache statistics are automatically included in benchmark results:

```python
# Cache stats are automatically collected and reported
print("Cache Statistics:")
print(f"Cache Hits: {cache_stats['hits']}")
print(f"Cache Misses: {cache_stats['misses']}")
print(f"Cache Hit Rate: {cache_stats['hit_rate']:.2f}%")
print(f"Total Allocations: {cache_stats['total']}")
```

## Configuration Options

### BlockManager Configuration

```python
BlockManager(
    capacity=1000,           # Cache capacity
    shared_cache=True,       # Enable/disable cache
    debug=False             # Enable debug logging
)
```

### PSLA Configuration

```python
PSLAConfig(
    name="model-name",
    shared_cache=True,       # Enable shared cache
    # ... other options
)
```

## Best Practices

### Capacity Planning
- Set cache capacity based on available memory
- Monitor eviction rates to optimize capacity
- Consider workload patterns when sizing cache

### Performance Optimization
- Monitor hit rates to identify optimization opportunities
- Use debug mode for detailed performance analysis
- Regularly clear cache statistics for accurate measurements

### Integration Guidelines
- Enable cache in configuration before running benchmarks
- Monitor cache statistics during performance testing
- Use cache statistics to optimize block allocation strategies

## Troubleshooting

### Common Issues

1. **Low Hit Rate**: Consider increasing cache capacity or optimizing access patterns
2. **High Eviction Rate**: Cache capacity may be too small for workload
3. **Performance Degradation**: Monitor cache overhead vs. benefits

### Debug Mode

Enable debug logging for detailed cache operations:

```python
block_manager = BlockManager(
    capacity=1000,
    shared_cache=True,
    debug=True  # Enable debug logging
)
```

## Examples

### Basic Cache Usage Example

```python
from TokenSim.block.shared_cache import SharedMemoryCache

# Initialize cache
cache = SharedMemoryCache(capacity=100)

# Simulate cache operations
for i in range(10):
    cache.put(i, list(range(i*10, (i+1)*10)))
    
# Access some blocks
for i in range(5):
    result = cache.get(i)
    print(f"Block {i}: {result}")

# Check statistics
stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.2f}%")
print(f"Total accesses: {stats.total_accesses}")
```

### Integration Example

```python
from TokenSim.config.psla_config import PSLAConfig
from TokenSim.llm.llm_engine import LLMEngine

# Configure with cache enabled
config = PSLAConfig(
    name="llama-7b",
    shared_cache=True
)

# Engine automatically uses cache
engine = LLMEngine(
    psla_config=config,
    # ... other parameters
)

# Cache statistics are automatically collected
# and reported in benchmark results
```

## Performance Considerations

### Memory Usage
- Cache memory usage is proportional to capacity
- Each cached block stores token IDs
- Monitor memory usage in production environments

### CPU Overhead
- LRU updates have O(1) amortized complexity
- Statistics tracking adds minimal overhead
- Debug logging may impact performance

### Scalability
- Cache is designed for single-worker scenarios
- For multi-worker setups, consider separate caches per worker
- Monitor cache performance as workload scales

## Future Enhancements

### Planned Features
- Multi-level cache support
- Cache warming strategies
- Advanced eviction policies
- Distributed cache support

### Performance Optimizations
- Lock-free cache operations
- Memory pool optimization
- Predictive cache prefetching
- Adaptive cache sizing 