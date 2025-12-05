# Technical Architecture Documentationi

## System Design

### Overview

The CUDA Neural Network Inference Accelerator is designed with modularity and clarity in mind. Each neural network operation is implemented as an independent CUDA kernel, allowing for easy testing, benchmarking, and future optimization.

### Component Architecture

```
┌────────────────────────────────────────────────────────┐
│                   Neural Network Class                 │
│                     (network.py)                       │
└───────────┬────────────────────────────────────────────┘
            │
            ├─────────────────────┬──────────────────────┐
            │                     │                      │
            v                     v                      v
    ┌───────────────┐    ┌──────────────┐      ┌──────────────┐
    │ Matrix Mult   │    │   Layers     │      │   Weights    │
    │  (optimized)  │    │ (ReLU, etc)  │      │   Loader     │
    └───────────────┘    └──────────────┘      └──────────────┘
            │                     │                      │
            v                     v                      v
    ┌──────────────────────────────────────────────────────┐
    │              CUDA Kernels on GPU                     │
    │  - matmul_optimized_kernel                           │
    │  - relu_kernel                                       │
    │  - add_bias_kernel                                   │
    │  - softmax_max_kernel                                │
    │  - softmax_exp_sum_kernel                            │
    │  - softmax_divide_kernel                             │
    └──────────────────────────────────────────────────────┘
```

## Kernel Implementations

### 1. Matrix Multiplication

**File:** `src/matmul_optimized.py`

**Algorithm:** Tiled matrix multiplication with shared memory

**Key Parameters:**
- Tile size: 16×16
- Block dimensions: (16, 16) threads
- Grid dimensions: Calculated based on output matrix size

**Memory Pattern:**
```
Global Memory (A, B)
        ↓
Shared Memory Tiles (sA, sB)
        ↓
Register Accumulation
        ↓
Global Memory (C)
```

**Performance Characteristics:**
- Memory bandwidth: ~16× reduction through tiling
- Occupancy: High (256 threads per block)
- Divergence: Minimal (only at boundaries)

**Code Structure:**
```python
@cuda.jit
def matmul_optimized_kernel(A, B, C, M, N, K):
    # Allocate shared memory
    sA = cuda.shared.array((TILE_SIZE, TILE_SIZE), dtype=np.float32)
    sB = cuda.shared.array((TILE_SIZE, TILE_SIZE), dtype=np.float32)
    
    # Calculate output position
    row = by * TILE_SIZE + ty
    col = bx * TILE_SIZE + tx
    
    # Loop over tiles
    for tile_idx in range(num_tiles):
        # Load tiles into shared memory
        # Synchronize threads
        # Compute partial products
        # Synchronize before next tile
    
    # Write final result
```

### 2. ReLU Activation

**File:** `src/layers.py`

**Algorithm:** Element-wise maximum operation

**Mathematical Definition:**
```
f(x) = max(0, x)
```

**Implementation:**
- 1D thread grid mapping directly to array elements
- Thread index calculation: `idx = cuda.grid(1)`
- Single memory access per element (read and write)
- No thread synchronization required

**Performance:**
- Memory-bound operation
- Achieves near-peak memory bandwidth
- Minimal arithmetic intensity

### 3. Bias Addition

**File:** `src/layers.py`

**Algorithm:** Broadcasting addition across batch dimension

**Mathematical Definition:**
```
output[i,j] = input[i,j] + bias[j]
```

**Implementation:**
- 2D thread grid: (batch_size, features)
- Each thread handles one output element
- Bias vector loaded once per row of threads
- Coalesced memory access for input/output

**Memory Access Pattern:**
```
Input:  Sequential across features (coalesced)
Bias:   Broadcast read (cached effectively)
Output: Sequential across features (coalesced)
```

### 4. Softmax Activation

**File:** `src/layers.py`

**Algorithm:** Three-stage reduction with numerical stability

**Mathematical Definition:**
```
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

**Implementation Stages:**

**Stage 1: Find Maximum**
```python
@cuda.jit
def softmax_max_kernel(x, max_vals, batch_size, features):
    # One thread per row
    # Sequential reduction within row
    # Write max value for each row
```

**Stage 2: Compute Exponentials and Sum**
```python
@cuda.jit
def softmax_exp_sum_kernel(x, max_vals, exp_vals, sum_vals, ...):
    # Compute exp(x - max) for numerical stability
    # Accumulate sum of exponentials
```

**Stage 3: Normalize**
```python
@cuda.jit
def softmax_divide_kernel(exp_vals, sum_vals, out, ...):
    # 2D grid for parallel division
    # Each thread divides one element
```

**Numerical Stability:**
The subtraction of max(x) prevents overflow in the exponential:
- Without: exp(1000) = overflow
- With: exp(1000 - 1000) = exp(0) = 1.0

## Data Flow

### Forward Pass Execution

```
Input Batch (CPU)
    │
    ├─> Copy to GPU
    │
    ├─> Layer 1: matmul(x, w1.T)    [GPU Kernel]
    ├─> Layer 1: add_bias(z1, b1)   [GPU Kernel]
    ├─> Layer 1: relu(z1)           [GPU Kernel]
    │   Copy intermediate to CPU
    │   Copy back to GPU for next layer
    │
    ├─> Layer 2: matmul(a1, w2.T)   [GPU Kernel]
    ├─> Layer 2: add_bias(z2, b2)   [GPU Kernel]
    ├─> Layer 2: relu(z2)           [GPU Kernel]
    │   Copy intermediate to CPU
    │   Copy back to GPU for next layer
    │
    ├─> Layer 3: matmul(a2, w3.T)   [GPU Kernel]
    ├─> Layer 3: add_bias(z3, b3)   [GPU Kernel]
    ├─> Layer 3: softmax(z3)        [GPU Kernel]
    │
    └─> Copy to CPU
        │
        └─> Output Predictions
```

**Optimization Opportunity:** Current implementation copies data between CPU and GPU for each layer. Keeping all intermediate activations on GPU would eliminate this overhead.

## Memory Hierarchy Utilization

### Global Memory
- Input data, weights, and output
- Bandwidth: ~732 GB/s (Tesla P100)
- Latency: ~400-800 cycles

### Shared Memory
- Matrix tiles (16×16 elements)
- Bandwidth: ~9000 GB/s
- Latency: ~30 cycles
- Size per SM: 64 KB

### Registers
- Accumulation variables
- Latency: 1 cycle
- Per-thread limit: 255 registers

### L1/L2 Cache
- Automatic caching by hardware
- L1: 24 KB per SM (data cache)
- L2: 4 MB shared across GPU

## Thread Organization

### Matrix Multiplication
```
Grid:  (ceil(N/16), ceil(M/16)) blocks
Block: (16, 16) threads
Total threads per block: 256
```

### Element-wise Operations (ReLU, Bias)
```
Grid:  (ceil(N/256), 1) blocks (1D)
       or (ceil(rows/16), ceil(cols/16)) blocks (2D)
Block: 256 threads (1D)
       or (16, 16) threads (2D)
```

### Softmax
```
Stage 1 (max): 1 thread per row
Stage 2 (sum): 1 thread per row
Stage 3 (divide): 2D grid, 1 thread per element
```

## Performance Bottlenecks

### Current Limitations

1. **CPU-GPU Data Transfer**
   - Impact: High
   - Each layer copies data back to CPU
   - PCIe bandwidth: ~16 GB/s (much slower than GPU compute)

2. **Kernel Launch Overhead**
   - Impact: Medium
   - Multiple kernel launches per forward pass
   - Each launch: ~5-10 microseconds overhead

3. **Memory Bandwidth**
   - Impact: Medium
   - Element-wise operations are memory-bound
   - Not utilizing full compute capacity

4. **No Kernel Fusion**
   - Impact: High
   - matmul + bias + ReLU could be single kernel
   - Would eliminate intermediate memory transfers

### Theoretical Performance Ceiling

**Tesla P100 Specifications:**
- Peak FP32: 9.3 TFLOPS
- Memory Bandwidth: 732 GB/s
- Memory per operation: 12 bytes (3 floats: read A, read B, write C)

**Matrix Multiplication Performance:**
- Current: ~165 GFLOPS (batch 1024)
- Theoretical peak: ~9,300 GFLOPS
- Efficiency: 1.8%
- Limiting factor: Memory bandwidth and kernel overhead

**Achievable with Optimizations:**
- Kernel fusion: 3-5× improvement
- Persistent data on GPU: 2-3× improvement
- Combined: 6-15× improvement possible
- Realistic target: 1,000-1,500 GFLOPS

## Numerical Precision

### Float32 Accuracy

All computations use single-precision floating point (float32):
- Range: ±3.4e38
- Precision: ~7 decimal digits
- Epsilon: 1.19e-07

### Accuracy Verification

Comparison with PyTorch (using same float32):
- Maximum probability difference: 0.000002
- Mean absolute error: <1e-6
- All predictions match exactly

### Sources of Error

1. **Rounding in multiplication:** Accumulation of many small errors
2. **Exponential computation:** Loss of precision for large inputs
3. **Division in softmax:** Can amplify errors if denominator is small

**Mitigation:** Numerically stable softmax prevents most issues.

## Testing Strategy

### Unit Tests

Each kernel tested independently:
```python
# Test correctness against NumPy
result_gpu = kernel(input)
result_cpu = numpy_implementation(input)
assert np.allclose(result_gpu, result_cpu, rtol=1e-4)
```

### Integration Tests

Full network tested against PyTorch:
```python
cuda_output = cuda_network.forward(input)
pytorch_output = pytorch_network.forward(input)
assert accuracy_match > 99.9%
```

### Performance Tests

Benchmarks across multiple configurations:
- Different batch sizes (1, 16, 64, 256, 1024)
- Different matrix sizes
- Comparison with CPU and PyTorch GPU

## Build and Deployment

### Compilation

Numba uses JIT (Just-In-Time) compilation:
1. First kernel call: Compile CUDA code (~100ms)
2. Subsequent calls: Use cached compiled code
3. Compilation triggered per kernel signature

### Caching

Compiled kernels cached in:
```
~/.numba/cache/
```

Clear cache to force recompilation:
```bash
rm -rf ~/.numba/cache/
```

### GPU Compatibility

Code compatible with:
- Compute Capability 3.5+ (Kepler and newer)
- Tested on Pascal (Tesla P100)
- Should work on Volta, Turing, Ampere, Ada, Hopper

### Production Considerations

For production deployment:
- Pre-compile kernels to avoid first-call latency
- Pin memory for faster CPU-GPU transfers
- Use CUDA streams for overlapping compute and transfer
- Consider batching multiple inference requests

## References

Internal documentation:
- `src/layers.py` - Activation and normalization kernels
- `src/matmul_optimized.py` - Matrix multiplication implementation
- `src/network.py` - Integration and pipeline
- `benchmarks/benchmark.py` - Performance measurement

External resources:
- NVIDIA CUDA C Programming Guide
- Numba CUDA Documentation
- "Programming Massively Parallel Processors" by Kirk and Hwu
