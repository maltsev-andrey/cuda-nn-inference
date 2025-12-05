"""
Simple matrix multiplication for neural network inference
This is a basic implementation - we'll optimize with your Project 1 code later
"""
import warnings
from numba.core.errors import NumbaPerformanceWarning  # or: from numba.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
import numpy as np
from numba import cuda

@cuda.jit
def matmul_kernel(A, B, C, M, N, K):
    """
    Simple matrix multiplication: C = A × B
    A: (M × K)
    B: (K × N)
    C: (M × N)

    Each thread computes one element of C
    """
    row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if row < M and col < N:
        tmp = 0.0
        for k in range(K):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

def matmul(A, B):
    """
    Matrix multiplication on GPU
    
    Args:
        A: NumPy array (M, K)
        B: NumPy array (K, N)
    
    Returns:
        C: NumPy array (M, N) = A @ B
    """
    M, K = A.shape
    K2, N = B.shape

    assert K == K2, f"Dimension mismatch: A is {A.shape}, B is {B.shape}"

    # Allocate GPU memory
    A_gpu = cuda.to_device(A)
    B_gpu = cuda.to_device(B)
    C_gpu = cuda.device_array((M, N), dtype=np.float32)

    # Configure kernel
    threads_per_block = (16, 16)
    blocks_x = (M + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (N + threads_per_block[1] - 1) // threads_per_block[1]
    blocks = (blocks_x, blocks_y)

    # lounch kernel
    matmul_kernel[blocks, threads_per_block](A_gpu, B_gpu, C_gpu, M, N, K)

    # Copy results back
    return C_gpu.copy_to_host()

if __name__ == '__main__':
    print("Testing matrix multiplication...")

    # Small test
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B = np.array([[5, 6], [7, 8]], dtype=np.float32)

    print(f"A:\n{A}")
    print(f"B:\n{B}")

    C_gpu = matmul(A, B)
    C_cpu = A @ B

    print(f"GPU result:\n{C_gpu}")
    print(f"CPU result:\n{C_cpu}")
    print(f"Match: {np.allclose(C_gpu, C_cpu)}")

    # Larger test
    print("\nLarger matrix test...")
    A = np.random.randn(100, 50).astype(np.float32)
    B = np.random.randn(50, 30).astype(np.float32)
    
    C_gpu = matmul(A, B)
    C_cpu = A @ B
    
    print(f"Shape: {C_gpu.shape}")
    print(f"Match: {np.allclose(C_gpu, C_cpu, rtol=1e-3, atol=1e-5)}")
    print(f"Max error: {np.abs(C_gpu - C_cpu).max()}")

























        
    