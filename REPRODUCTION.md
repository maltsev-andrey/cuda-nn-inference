# Reproduction Guide

This document provides step-by-step instructions to reproduce all results from the project.

## Prerequisites

### Hardware Requirements

- NVIDIA GPU with CUDA support
- Minimum: Compute Capability 3.5 (Kepler)
- Recommended: Compute Capability 6.0+ (Pascal or newer)
- GPU Memory: 2GB minimum, 8GB+ recommended

**Tested Configuration:**
- GPU: NVIDIA Tesla P100-PCIE-16GB
- Compute Capability: 6.0
- CUDA Cores: 3584
- Memory: 16GB HBM2

### Software Requirements

- Operating System: Linux (tested on RHEL 9), Windows 10+, or macOS
- CUDA Toolkit: 11.0 or higher
- Python: 3.8, 3.9, 3.10, or 3.11
- pip: Latest version
- Git: For cloning repository

### Verify CUDA Installation

```bash
# Check NVIDIA driver
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.xx       Driver Version: 525.xx       CUDA Version: 12.x    |
# +-----------------------------------------------------------------------------+

# Check CUDA compiler (optional)
nvcc --version
```

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/maltsev-andrey/cuda-nn-inference.git
cd cuda-nn-inference
```

### Step 2: Create Virtual Environment

```bash
# Create environment
python3 -m venv cuda_env

# Activate environment
source cuda_env/bin/activate  # Linux/macOS
# OR
cuda_env\Scripts\activate  # Windows
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

**Installation time:** 2-5 minutes depending on internet speed

### Step 4: Verify Installation

```bash
# Test Numba CUDA
python3 -c "from numba import cuda; print('CUDA available:', cuda.is_available()); print('GPU:', cuda.gpus)"

# Expected output:
# CUDA available: True
# GPU: <CUDA device 0 'Tesla P100-PCIE-16GB'>
```

## Reproducing Results

### Phase 1: Train PyTorch Baseline Model

This step trains the reference PyTorch model that provides weights for the CUDA implementation.

```bash
# Run training script
python3 src/train_pytorch_model.py
```

**Expected Runtime:** 5-10 minutes on Tesla P100

**Expected Output:**
```
============================================================
Training MNIST Classifier
============================================================

Loading MNIST dataset...
Training samples: 60000
Test samples: 10000

Model architecture:
SimpleNet(
  (fc1): Linear(in_features=784, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=10, bias=True)
)

Total parameters: 235,146

Starting training...
Epoch 1/10: 100%|████████| 469/469 [00:07<00:00, loss=0.2698, acc=91.89%]
...
Epoch 10/10: 100%|████████| 469/469 [00:07<00:00, loss=0.0164, acc=99.44%]

============================================================
Training complete! Best test accuracy: 97.82%
============================================================

Extracting weights for CUDA implementation...
Weights saved to models/mnist_weights.npz
```

**Output Files:**
- `models/mnist_model.pth` - PyTorch model checkpoint
- `models/mnist_weights.npz` - Weights for CUDA inference
- `data/MNIST/` - Downloaded MNIST dataset

**Troubleshooting:**

If training fails with CUDA out of memory:
```bash
# Edit src/train_pytorch_model.py
# Change: BATCH_SIZE = 128
# To:     BATCH_SIZE = 64
```

If dataset download fails:
```bash
# Manually download MNIST
mkdir -p data
# Visit http://yann.lecun.com/exdb/mnist/
# Download and place in data/MNIST/raw/
```

### Phase 2: Test Individual Kernels

Verify each CUDA kernel works correctly before integration.

#### Test 2a: Matrix Multiplication

```bash
python3 src/matmul_optimized.py
```

**Expected Output:**
```
============================================================
Optimized Matrix Multiplication Test
============================================================

[1] Correctness test (small matrix):
  Shape: (64, 32) × (32, 48) = (64, 48)
  Optimized vs NumPy - Match: True
  Optimized vs Simple - Match: True
  Max error: 3.814697e-06

[2] Performance benchmark (neural network sizes):

     1 ×  784 ×  256:
    Optimized:  1.122 ms  (   0.4 GFLOPS)
    Simple:     1.219 ms  (   0.3 GFLOPS)
    Speedup:   1.09×

  1024 ×  784 ×  256:
    Optimized:  2.484 ms  ( 165.4 GFLOPS)
    Simple:     5.217 ms  (  78.8 GFLOPS)
    Speedup:   2.10×

[3] Large matrix benchmark:
  1024 × 1024:    5.83 ms |   368.6 GFLOPS |   4.0% of peak
  2048 × 2048:   36.56 ms |   469.9 GFLOPS |   5.1% of peak
```

#### Test 2b: Activation Functions

```bash
python3 src/layers.py
```

**Expected Output:**
```
Testing CUDA kernels...
============================================================

1. Testing ReLU activation:
Input:
[[-2. -1.  0.  1.  2.]
 [ 3. -4.  5. -6.  7.]]
Output:
[[0. 0. 0. 1. 2.]
 [3. 0. 5. 0. 7.]]
Match: True

2. Testing bias addition:
Match: True

3. Testing Softmax activation:
Row sums (should be ~1.0): [1. 1. 1.]
Match: True

============================================================
All kernel tests passed!
```

### Phase 3: Test Integrated Network

Verify the complete neural network implementation.

```bash
python3 src/network.py
```

**Expected Output:**
```
Using optimized matrix multiplication
============================================================
Neural Network Inference Test
============================================================

[1] Loading neural network...
Loading weights from models/mnist_weights.npz...
Weights loaded successfully!

[2] Loading test data...
  Test images: (10000, 784)
  Test labels: (10000,)

[3] Testing single image inference...
  Predicted digit: 7
  Actual digit: 7
  Correct: True

[4] Testing batch inference...
  Batch size: 100
  Accuracy: 100.00%

[5] Testing full test set...
  Total images: 10000
  Accuracy: 97.82%

[6] Comparing with PyTorch baseline...
  PyTorch accuracy: 97.82%
  CUDA accuracy:    97.82%
  Predictions match: 100.00%
  Max probability difference: 0.000002
```

**Key Metrics to Verify:**
- Accuracy: 97.82% (matches PyTorch)
- Predictions match: 100.00%
- Max probability difference: <0.00001

### Phase 4: Performance Benchmarking

Measure throughput and latency across different batch sizes.

```bash
python3 benchmarks/benchmark.py
```

**Expected Runtime:** 2-3 minutes

**Expected Output:**
```
============================================================
CUDA Neural Network - Performance Benchmark
============================================================

CUDA Implementation Benchmark
============================================================

Batch size: 1
  Throughput:         126 images/sec
  Latency:          7.948 ms/image
  Accuracy:        100.00%

Batch size: 16
  Throughput:       2,009 images/sec
  Latency:          0.498 ms/image
  Accuracy:         97.12%

Batch size: 64
  Throughput:       8,066 images/sec
  Latency:          0.124 ms/image
  Accuracy:         97.28%

Batch size: 256
  Throughput:      29,616 images/sec
  Latency:          0.034 ms/image
  Accuracy:         97.82%

Batch size: 1024
  Throughput:      88,300 images/sec
  Latency:          0.011 ms/image
  Accuracy:         97.86%

PyTorch GPU Benchmark
============================================================
[Similar output for PyTorch comparison]

CPU (NumPy) Benchmark
============================================================
[Similar output for CPU baseline]

Results saved to results/benchmark_results.txt
```

**Output Files:**
- `results/benchmark_results.txt` - Detailed benchmark results

## Expected Performance by GPU

Results will vary by GPU architecture. Here are approximate expectations:

### Tesla P100 (Tested Configuration)
- Batch 1024 throughput: 80,000-90,000 img/sec
- Batch 1024 latency: 0.011-0.012 ms/img
- Matrix multiply: 150-170 GFLOPS

### RTX 3090
- Batch 1024 throughput: 150,000-200,000 img/sec
- Matrix multiply: 300-400 GFLOPS

### RTX 4090
- Batch 1024 throughput: 250,000-350,000 img/sec
- Matrix multiply: 500-700 GFLOPS

### GTX 1660
- Batch 1024 throughput: 40,000-50,000 img/sec
- Matrix multiply: 80-100 GFLOPS

## Analyzing Results

### Accuracy Verification

The CUDA implementation should match PyTorch exactly:

```python
# Quick verification script
from network import NeuralNetwork
import torch
import numpy as np

# Load both models
cuda_net = NeuralNetwork('models/mnist_weights.npz')
pytorch_net = torch.load('models/mnist_model.pth')

# Generate test input
test_input = np.random.randn(10, 784).astype(np.float32)

# Compare outputs
cuda_out = cuda_net.predict_proba(test_input)
pytorch_out = pytorch_net(torch.from_numpy(test_input)).softmax(dim=1).numpy()

print(f"Max difference: {np.abs(cuda_out - pytorch_out).max()}")
# Should be < 0.00001
```

### Performance Analysis

Check if performance scales correctly with batch size:

```python
import matplotlib.pyplot as plt

batch_sizes = [1, 16, 64, 256, 1024]
throughputs = [126, 2009, 8066, 29616, 88300]  # Your results

plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, throughputs, marker='o')
plt.xlabel('Batch Size')
plt.ylabel('Throughput (images/sec)')
plt.title('CUDA Implementation Scaling')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.savefig('results/scaling.png')
```

Expected pattern: Throughput should increase roughly linearly with batch size until memory becomes constraint.

## Common Issues and Solutions

### Issue 1: CUDA Not Available

**Symptom:**
```
from numba import cuda; cuda.is_available()
False
```

**Solution:**
```bash
# Check CUDA toolkit installation
ls /usr/local/cuda/

# Install CUDA toolkit if missing
# Visit: https://developer.nvidia.com/cuda-downloads

# Verify driver
nvidia-smi
```

### Issue 2: Out of Memory

**Symptom:**
```
numba.cuda.cudadrv.driver.CudaAPIError: [OUT_OF_MEMORY]
```

**Solution:**
- Reduce batch size in benchmarks
- Close other GPU applications
- Use smaller test dataset

### Issue 3: Slow Performance

**Symptom:** Throughput much lower than expected

**Checklist:**
- GPU not in performance mode (check nvidia-smi power state)
- CPU throttling due to thermal constraints
- Running in virtual machine (GPU passthrough required)
- Multiple processes using GPU simultaneously

### Issue 4: Accuracy Mismatch

**Symptom:** CUDA accuracy differs from PyTorch

**Solution:**
- Verify weights loaded correctly: check file sizes
- Ensure float32 precision throughout pipeline
- Check for NaN values in intermediate computations:
  ```python
  # Add to network.py forward pass
  assert not np.isnan(z1).any(), "NaN detected in layer 1"
  ```

### Issue 5: Import Errors

**Symptom:**
```
ModuleNotFoundError: No module named 'numba'
```

**Solution:**
```bash
# Verify virtual environment is activated
which python  # Should point to cuda_env

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

## Validation Checklist

Use this checklist to verify successful reproduction:

- [ ] CUDA available and GPU detected
- [ ] PyTorch model trained (97-98% accuracy)
- [ ] Weights file created (models/mnist_weights.npz)
- [ ] Matrix multiplication tests pass
- [ ] Activation function tests pass
- [ ] Network accuracy matches PyTorch (97.82%)
- [ ] Benchmark completes without errors
- [ ] Results file generated (results/benchmark_results.txt)

## Performance Baseline Comparison

Compare your results with the baseline:

| Metric                  | Baseline (Tesla P100)| Your Result |
|-------------------------|----------------------|-------------|
| Training Accuracy       | 97.82%               | ___________ |
| CUDA Accuracy           | 97.82%               | ___________ |
| Throughput (batch 1024) | 88,300 img/sec       | ___________ |
| Latency (batch 1024)    | 0.011 ms             | ___________ |
| Max Prob Difference     | 0.000002             | ___________ |

## Additional Experiments

### Experiment 1: Different Network Architectures

Modify layer sizes in `train_pytorch_model.py`:
```python
# Try different architectures
self.fc1 = nn.Linear(784, 512)  # Wider layers
self.fc2 = nn.Linear(512, 256)
self.fc3 = nn.Linear(256, 10)
```

Re-run training and benchmarks to see impact.

### Experiment 2: Different Tile Sizes

Modify `TILE_SIZE` in `matmul_optimized.py`:
```python
TILE_SIZE = 32  # Try 8, 16, 32
```

Benchmark to find optimal tile size for your GPU.

### Experiment 3: Mixed Precision

Test float16 vs float32 performance:
```python
# In network.py
data = np.load(weights_path)
self.w1 = data['w1'].astype(np.float16)  # Try float16
```

Note: May impact accuracy.

## Documentation

After successful reproduction, the following files contain results:

- `results/benchmark_results.txt` - Performance measurements
- `models/mnist_model.pth` - Trained PyTorch model
- `models/mnist_weights.npz` - Exported weights
- Console output - Accuracy verification

## Support

For issues not covered in this guide:
- Check GitHub Issues page
- Review ARCHITECTURE.md for technical details
- Consult Numba CUDA documentation
- Verify hardware compatibility

## Next Steps

After reproducing baseline results:
1. Experiment with optimizations
2. Try different datasets (Fashion-MNIST, CIFAR-10)
3. Implement additional layer types (convolution, pooling)
4. Profile with NVIDIA Nsight for bottleneck analysis
