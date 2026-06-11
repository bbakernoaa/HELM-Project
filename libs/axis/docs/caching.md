# Weight Caching {#caching}

Weight generation is computationally expensive for high-resolution grids. The
`WeightCache` component enables serialization of precomputed `InterpolationMatrix`
objects to a compact binary format, allowing reuse across timesteps without
recomputation.

## Overview

The weight cache uses a self-describing binary format with:
- 40-byte header (magic, version, endianness, dimensions)
- Contiguous IEEE-754 arrays (factor_list, factor_row, factor_col, frac_a, frac_b, area_a, area_b)

AXIS performs no file I/O — the cache operates on raw byte buffers. The caller
(AMIO, Python, or application code) owns persistence.

## Serialization

```cpp
#include <axis/solver/weight_cache.hpp>

// Generate weights (expensive)
auto matrix = gen.generate(src_mesh, dst_mesh, config);

// Query required buffer size
auto buf_size = axis::solver::WeightCache::serialize(matrix, nullptr, 0);

// Allocate and serialize
std::vector<char> buffer(buf_size);
axis::solver::WeightCache::serialize(matrix, buffer.data(), buffer.size());

// Now write buffer to disk/network via AMIO or your own I/O
```

## Deserialization

```cpp
// Read buffer from disk (via AMIO or your own I/O)
std::vector<char> buffer = read_from_file("weights.bin");

// Deserialize — validates header and dimensions
auto matrix = axis::solver::WeightCache::deserialize<Kokkos::HostSpace>(
    buffer.data(), buffer.size());

// Use immediately
axis::solver::apply(matrix, src_field, dst_field);
```

## Binary Format

```
Offset  Size  Field
------  ----  -----
0       4     Magic bytes: "AXW\0"
4       4     Format version (uint32_t, currently 1)
8       1     Endianness marker (0x01 = little-endian, 0x02 = big-endian)
9       7     Reserved padding (zeros)
16      8     n_src (uint64_t)
24      8     n_dst (uint64_t)
32      8     nnz (uint64_t)
40      ...   factor_list[nnz]  (double, IEEE-754)
              factor_row[nnz]   (index_t)
              factor_col[nnz]   (index_t)
              frac_a[n_src]     (double)
              frac_b[n_dst]     (double)
              area_a[n_src]     (double)
              area_b[n_dst]     (double)
```

## Error Handling

| Condition | Behavior |
|-----------|----------|
| Version mismatch | `throw std::runtime_error("weight cache version mismatch: expected N, got M")` |
| Buffer too small | `throw std::runtime_error("weight cache buffer size inconsistent with dimensions")` |
| Null pointer | `throw std::runtime_error("weight cache: null buffer")` |

## Round-Trip Guarantee

Serialization followed by deserialization produces **bitwise-identical** apply results:

```cpp
auto matrix_a = gen.generate(src_mesh, dst_mesh, config);
axis::solver::apply(matrix_a, src_field, dst_a);

auto buf_size = axis::solver::WeightCache::serialize(matrix_a, nullptr, 0);
std::vector<char> buf(buf_size);
axis::solver::WeightCache::serialize(matrix_a, buf.data(), buf.size());

auto matrix_b = axis::solver::WeightCache::deserialize<Kokkos::HostSpace>(
    buf.data(), buf.size());
axis::solver::apply(matrix_b, src_field, dst_b);

// dst_a and dst_b are bitwise identical
```

## Python Usage

```python
import axis

matrix = axis.WeightGenerator.generate(src_mesh, dst_mesh, config)

# Serialize to bytes
blob = matrix.to_bytes()

# Save to file
with open("weights.bin", "wb") as f:
    f.write(blob)

# Reload
with open("weights.bin", "rb") as f:
    blob = f.read()
matrix2 = axis.InterpolationMatrix.from_bytes(blob)
```

## Best Practices

1. **Cache invalidation** — recompute weights when source or destination grid geometry changes
2. **Endianness** — the cache records endianness for cross-platform detection; consumers
   should check the header byte if transferring between architectures
3. **Versioning** — the format version allows AXIS upgrades to detect and reject stale caches
4. **Performance** — serialization cost is O(nnz) memcpy, far cheaper than weight generation
