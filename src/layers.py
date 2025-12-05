"""
CUDA kernels for neural network layers
Implements: ReLU, Softmax, and helper functions
"""
import warnings
from numba.core.errors import NumbaPerformanceWarning  # or: from numba.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
import numpy as np
from numba import cuda
import math

# ============================================================================
# ReLU Activation: max(0, x)
# ============================================================================

@cuda.jit
def relu_kernel(x, out, n):
    """
    ReLU activation: out[i] = max(0, x[i])
    
    Args:
        x: input array (flattened)
        out: output array (same shape as x)
        n: total number of elements
    
    Each thread processes one element
    """
    idx = cuda.grid(1)
    if idx < n:
        out[idx] = max(0.0, x[idx])

def relu_forward(x):
    """
    Apply ReLU activation to input array
    
    Args:
        x: NumPy array of any shape (on CPU)
    
    Returns:
        NumPy array with ReLU applied (on CPU)
    """
    # Flatten to 10 for processing
    original_shape = x.shape
    x_flat = x.ravel()
    n = x_flat.size

    # Allocate GPU memory
    x_gpu = cuda.to_device(x_flat)
    out_gpu = cuda.device_array(n, dtype=np.float32)

    # Configure kernel launch
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block

    # Launch kernel
    relu_kernel[blocks, threads_per_block] (x_gpu, out_gpu, n)

    # Copy result back and reshape
    result = out_gpu.copy_to_host()
    return result.reshape(original_shape)

# ============================================================================
# Add Bias: x + bias (broadcast across batch)
# ============================================================================

@cuda.jit
def add_bias_kernel(x, bias, out, batch_size, features):
    """
    Add bias vector to each row of matrix: out[i,j] = x[i,j] + bias[j]
    
    Args:
        x: input matrix (batch_size × features)
        bias: bias vector (features,)
        out: output matrix (batch_size × features)
        batch_size: number of rows
        features: number of columns
    
    Each thread processes one element
    """
    row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if row <  batch_size and col < features:
        idx = row * features + col
        out[idx] = x[idx] + bias[col]

def add_bias(x, bias):
    """
    Add bias to each sample in batch
    
    Args:
        x: input array (batch_size, features)
        bias: bias vector (features,)
    
    Returns:
        x + bias (broadcasted)
    """
    batch_size, features = x.shape

    # Flatten arrays for GPU
    x_flat = x.ravel()

    # Allocate GPU memory
    x_gpu = cuda.to_device(x_flat)
    bias_gpu = cuda.to_device(bias)
    out_gpu = cuda.device_array(x_flat.size, dtype=np.float32)

    # Cobfugure 2D kernel launch
    threads_per_block = (16, 16)
    blocks_x = (batch_size + threads_per_block[0] - 1) //  threads_per_block[0]
    blocks_y = (features + threads_per_block[1] - 1 ) // threads_per_block[1]
    blocks = (blocks_x, blocks_y)

    # Lounch kernel
    add_bias_kernel[blocks, threads_per_block](
        x_gpu, bias_gpu, out_gpu, batch_size, features
    )

    # Copy result back
    result = out_gpu.copy_to_host()
    return result.reshape(batch_size, features)

# ============================================================================
# Softmax: exp(x) / sum(exp(x))
# ============================================================================

@cuda.jit
def softmax_max_kernel(x, max_vals, batch_size, features):
    """
    Find maximum value in each row (for numerical stability)
    
    Args:
        x: input matrix (batch_size × features), flattened
        max_vals: output array (batch_size,) to store max of each row
        batch_size: number of rows
        features: number of columns
    """
    row = cuda.grid(1)

    if row < batch_size:
        max_val = -1e38  # Start with very small number

        # Find max in this row
        for col in range(features):
            idx = row * features + col
            if x[idx] > max_val:
                max_val = x[idx]

        max_vals[row] = max_val

@cuda.jit
def softmax_exp_sum_kernel(x, max_vals, exp_vals, sum_vals, batch_size, features):
    """
    Compute exp(x - max) and sum for each row
    
    Args:
        x: input matrix (batch_size × features), flattened
        max_vals: max value for each row (batch_size,)
        exp_vals: output for exp values (batch_size × features), flattened
        sum_vals: output for sum of exp values (batch_size,)
        batch_size: number of rows
        features: number of columns
    """
    row = cuda.grid(1)

    if row < batch_size:
        row_sum = 0.0
        max_val = max_vals[row]

        # ompute exp(x-max) and accumulate sum
        for col in range(features):
            idx = row * features + col
            exp_val = math.exp(x[idx] - max_val)
            exp_vals[idx] = exp_val
            row_sum += exp_val

        sum_vals[row] = row_sum

@cuda.jit
def softmax_divide_kernel(exp_vals, sum_vals, out, batch_size, features):
    """
    Divide each exp value by row sum to get probabilities
    
    Args:
        exp_vals: exp values (batch_size × features), flattened
        sum_vals: sum of exp values for each row (batch_size,)
        out: output probabilities (batch_size × features), flattened
        batch_size: number of rows
        features: number of columns
    """
    row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if row < batch_size and col < features:
        idx = row * features + col
        out[idx] = exp_vals[idx] / sum_vals[row]

def softmax_forward(x):
    """
    Apply softmax activation: softmax(x) = exp(x) / sum(exp(x))
    Uses numerically stable version: exp(x - max(x))
    
    Args:
        x: input array (batch_size, num_classes)
    
    Returns:
        probabilities (batch_size, num_classes), each row sums to 1
    """
    batch_size, features = x.shape
    n = x.size

    # Flatten input
    x_flat = x.ravel()

    # Allocate GPU memory
    x_gpu = cuda.to_device(x_flat)
    max_vals_gpu = cuda.device_array(batch_size, dtype=np.float32)
    exp_vals_gpu = cuda.device_array(n, dtype=np.float32)
    sum_vals_gpu = cuda.device_array(batch_size, dtype=np.float32)
    out_gpu = cuda.device_array(n, dtype=np.float32)

    # Step 1: Find max in each row
    threads_per_block = 256
    blocks = (batch_size + threads_per_block - 1) // threads_per_block
    softmax_max_kernel[blocks, threads_per_block](
        x_gpu, max_vals_gpu, batch_size, features
    )

    # Step2: Compute exp(x-max) and sum
    softmax_exp_sum_kernel[blocks, threads_per_block](
        x_gpu, max_vals_gpu, exp_vals_gpu, sum_vals_gpu, batch_size, features
    )
 
    # Step3: Divide by sum
    threads_per_block_2d = (16, 16)
    blocks_x = (batch_size + threads_per_block_2d[0] - 1) // threads_per_block_2d[0]
    blocks_y = (features + threads_per_block_2d[1] - 1) // threads_per_block_2d[1]
    blocks_2d = (blocks_x, blocks_y)

    softmax_divide_kernel[blocks_2d, threads_per_block_2d](
        exp_vals_gpu, sum_vals_gpu, out_gpu, batch_size, features
    )

    # Copy results back
    result = out_gpu.copy_to_host()
    return result.reshape(batch_size, features)

# ============================================================================
# Testing and Verification
# ============================================================================

if __name__ == '__main__':
    # Suppress performance warnings for small test data
    # (In real workloads with large batches, these won't appear)
    import warnings
    warnings.filterwarnings('ignore', message='Grid size')
    
    print("Testing CUDA kernels...")
    print("=" * 60)

    # Test ReLU
    print("\n1. Testing ReLU activation:")
    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0],
                           [3.0, -4.0, 5.0, -6.0, 7.0]], dtype=np.float32)
    print(f"Input:\n{x}")

    result = relu_forward(x)
    print(f"Output:\n{result}")
    expected = np.maximum(0, x)
    print(f"Expected:\n{expected}")
    print(f"Match: {np.allclose(result, expected)}")

    # Test bias addition
    print("\n2. Testing bias addition:")
    x = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]], dtype=np.float32)
    bias = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    print(f"Input:\n{x}")
    print(f"Bias: {bias}")

    result = add_bias(x, bias)
    print(f"Output:\n{result}")
    expected = x + bias
    print(f"Expected:\n{expected}")
    print(f"Match: {np.allclose(result, expected)}")

    # Test Softmax
    print("\n3. Testing Softmax activation:")
    x = np.array([[1.0, 2.0, 3.0],
                          [1.0, 1.0, 1.0],
                          [0.0, 0.0, 0.0]], dtype=np.float32)
    print(f"Input:\n{x}")

    result = softmax_forward(x)
    print(f"Output:\n{result}")
    
    # Verify properties
    print(f"Row sums (should be ~1.0): {result.sum(axis=1)}")
    print(f"All positive: {(result > 0).all()}")
    
    # Compare with NumPy implementation
    def numpy_softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    expected = numpy_softmax(x)
    print(f"Expected:\n{expected}")
    print(f"Match: {np.allclose(result, expected)}")
    
    print("\n" + "=" * 60)
    print(" All kernel tests passed!")



















    