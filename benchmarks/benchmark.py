"""
Performance Benchmarking for CUDA Neural Network Inference

Measures:
1. Throughput (images/second)
2. Latency (milliseconds/image)
3. Comparison: CUDA vs PyTorch GPU vs CPU
4. Batch size analysis
"""

import numpy as np
import time
import sys
import os

# Add parent directory to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from src.network import NeuralNetwork
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

def load_test_data(n_samples=10000):
    """Load MNIST test data"""
    try:
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

        images = []
        labels = []
        for i, (img, label) in enumerate(test_dataset):
            if i >= n_samples:
                break
            images.append(img.numpy().flatten())
            labels.append(label)

        return np.array(images, dtype=np.float32), np.array(labels)

    except ImportError:
        print("Warning: PyTorch not available, using random data")
        return np.random.randn(n_samples, 784).astype(np.float32), np.random.randint(0, 10, n_samples)

def benchmark_cuda(net, images, labels, batch_sizes=[1, 16, 64, 256, 1024]):
    """
    Benchmark CUDA implementation with different batch sizes
    
    Returns:
        dict: Results for each batch size
    """
    print("\n" + "="*60)
    print("CUDA Implementation Benchmark")
    print("="*60)

    results = {}

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")

        # Prepare data
        n_batches = min(len(images) // batch_size, 100) # Limit to 100 barches for speed
        total_images = n_batches * batch_size

        # Warmup (first run is slower due to JIT compilation)
        warmup_batch = images[:batch_size]
        _=net.predict(warmup_batch)

        # Measure throughput
        start_time = time.time()

        for i in range(n_batches):
            batch = images[i * batch_size:(i+1)*batch_size]
            predictions = net.predict(batch)

        end_time = time.time()
        elapsed = end_time - start_time

        # Calculate metrics
        throughput = total_images / elapsed
        latency_ms = (elapsed / total_images) * 1000

        # Calculate accuracy
        all_predictions = []
        all_labels = []
        for i in range(n_batches):
            batch = images[i*batch_size:(i+1)*batch_size]
            batch_labels = labels[i*batch_size:(i+1)*batch_size]
            predictions = net.predict(batch)
            all_predictions.extend(predictions)
            all_labels.extend(batch_labels)

        accuracy = (np.array(all_predictions) == np.array(all_labels)).mean() * 100

        results[batch_size] = {
            'throughput': throughput,
            'latency_ms': latency_ms,
            'accuracy': accuracy,
            'total_time': elapsed,
            'total_images': total_images
        }

        print(f"  Throughput:  {throughput:>10,.0f} images/sec")
        print(f"  Latency:     {latency_ms:>10.3f} ms/image")
        print(f"  Accuracy:    {accuracy:>10.2f}%")
        print(f"  Total time:  {elapsed:>10.2f} seconds")

    return results

def benchmark_pytorch_gpu(images, labels, batch_sizes=[1, 16, 64, 256, 1024]):
    """
    Benchmark PyTorch GPU implementation
    
    Returns:
        dict: Results for each batch size
    """
    print("\n" + "="*60)
    print("PyTorch GPU Benchmark")
    print("="*60)
    
    try:
        import torch
        import torch.nn as nn
        
        if not torch.cuda.is_available():
            print("CUDA not available for PyTorch, skipping GPU benchmark")
            return None

        # Load model
        class SimpleNet(nn.Module):
            def __init__(self):
                super(SimpleNet, self).__init__()
                self.fc1 = nn.Linear(784, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, 10)

            def forward(self, x):
                x = x.view(-1, 784)
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        device = torch.device('cuda')
        model = SimpleNet().to(device)
        model.load_state_dict(torch.load('models/mnist_model.pth', weights_only=True))
        model.eval()

        results = {}

        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}")

            n_batches = min(len(images) // batch_size, 100)
            total_images = n_batches * batch_size

            # Warmup
            warmup_batch = torch.from_numpy(images[:batch_size]).to(device)
            with torch.no_grad():
                _=model(warmup_batch)
            torch.cuda.synchronize()

            # Benchmark
            start_time = time.time()

            with torch.no_grad():
                for i in range(n_batches):
                    batch = torch.from_numpy(
                        images[i*batch_size:(i+1)*batch_size]
                    ).to(device)
                    outputs = model(batch)
                    _ = outputs.argmax(dim=1).cpu()

            torch.cuda.synchronize()
            end_time = time.time()
            elapsed = end_time - start_time

            throughput = total_images / elapsed
            latency_ms = (elapsed / total_images) * 1000

            results[batch_size] = {
                'throughput': throughput,
                'latency_ms': latency_ms,
                'total_time': elapsed,
                'total_images': total_images
            }
            
            print(f"  Throughput:  {throughput:>10,.0f} images/sec")
            print(f"  Latency:     {latency_ms:>10.3f} ms/image")
            print(f"  Total time:  {elapsed:>10.2f} seconds")
        
        return results

    except Exception as e:
        print(f"PyTorch GPU benchmark failed: {e}")
        return None

def benchmark_cpu(images, labels, batch_sizes=[1, 16, 64, 256]):
    """
    Benchmark CPU (NumPy) implementation
    
    Returns:
        dict: Results for each batch size
    """
    print("\n" + "="*60)
    print("CPU (NumPy) Benchmark")
    print("="*60)

    # Load weights
    data = np.load('models/mnist_weights.npz')
    w1 = data['w1'].astype(np.float32)
    b1 = data['b1'].astype(np.float32)
    w2 = data['w2'].astype(np.float32)
    b2 = data['b2'].astype(np.float32)
    w3 = data['w3'].astype(np.float32)
    b3 = data['b3'].astype(np.float32)

    def cpu_forward(x):
        """Pure NumPy forward pass"""
        # Layer 1
        z1 = x @ w1.T + b1
        a1 = np.maximum(0, z1)

        # Layer 2
        z2 = a1 @ w2.T + b2
        a2 = np.maximum(0, z2)

        # Layer 3
        z3 = a2 @ w3.T + b3

        # Sortmax
        exp_z3 = np.exp(z3 - np.max(z3, axis=1, keepdims=True))
        probs = exp_z3 / np.sum(exp_z3, axis=1, keepdims=True)

        return probs

    results = {}

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")

        # Limit batches for CPU (slower)
        n_batches = min(len(images) // batch_size, 50)
        total_images = n_batches * batch_size

        # Warmup
        _ = cpu_forward(images[:batch_size])

        # Benchmark
        start_time = time.time()

        for i in range(n_batches):
            batch = images[i*batch_size:(i+1)*batch_size]
            probs = cpu_forward(batch)
            predictions = np.argmax(probs, axis=1)

        end_time = time.time()
        elapsed = end_time - start_time

        throughput = total_images / elapsed
        latency_ms = (end_time - start_time) * 1000

        results[batch_size] = {
            'throughput': throughput,
            'latency_ms': latency_ms,
            'total_time': elapsed,
            'total_images': total_images
        }

        print(f"  Throughput:  {throughput:>10,.0f} images/sec")
        print(f"  Latency:     {latency_ms:>10.3f} ms/image")
        print(f"  Total time:  {elapsed:>10.2f} seconds")
    
    return results

def print_comparasion(cuda_results, pytorch_results, cpu_results):
    """Print comparison table"""
    print("\n" + "="*60)
    print("Performance Comparison Summary")
    print("="*60)
    
    batch_sizes = sorted(cuda_results.keys())

    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")

        cuda_throughput = cuda_results[batch_size]['throughput']
        cuda_latency = cuda_results[batch_size]['latency_ms']

        print(f" CUDA:  {cuda_throughput:>10,.0f} img/sec  |  {cuda_latency:>6.3f} ms/img")

        if pytorch_results and batch_size in pytorch_results:
            pt_throughput = pytorch_results[batch_size]['throughput']
            pt_latency = pytorch_results[batch_size]['latency_ms']
            speedup = cuda_throughput / pt_throughput
            print(f"PyTorch GPU:  {pt_throughput:>10,.0f} img/sec  |  {pt_latency:>6.3f} ms/img  (CUDA is {speedup:.2f}x)")
        
        if cpu_results and batch_size in cpu_results:
            cpu_throughput = cpu_results[batch_size]['throughput']
            cpu_latency = cpu_results[batch_size]['latency_ms']
            speedup = cuda_throughput / cpu_throughput
            print(f"CPU:          {cpu_throughput:>10,.0f} img/sec  |  {cpu_latency:>6.3f} ms/img  (CUDA is {speedup:.2f}x)")


def save_results(cuda_results, pytorch_results, cpu_results, output_file='results/benchmark_results.txt'):
    """Save results to file"""
    os.makedirs('results', exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("CUDA Neural Network Inference - Benchmark Results\n")
        f.write("="*60 + "\n\n")

        # CUDA results
        f.write("CUDA Implementation;\n")
        f.write("-"*60 + "\n")
        for batch_size, results in sorted(cuda_results.items()):
            f.write(f"Batch {batch_size:4d}: {results['throughput']:10,.0f} img/sec | "
                   f"{results['latency_ms']:6.3f} ms/img | Acc: {results['accuracy']:.2f}%\n")

        # PyTorch results
        if pytorch_results:
            f.write("\nPyTorch GPU:\n")
            f.write("-"*60 + "\n")
            for batch_size, results in sorted(cpu_results.items()):
                f.write(f"Batch {batch_size:4d}: {results['throughput']:10,.0f} img/sec | "
                       f"{results['latency_ms']:6.3f} mc/img\n")

        # Speedups
        f.write("\n" + "="*60 + "\n")
        f.write("Speedup Summary (Best Case):\n")
        f.write("-"*60 + "\n")
        
        # Find best CUDA result
        best_cuda = max(cuda_results.items(), key=lambda x: x[1]['throughput'])
        f.write(f"Best CUDA: Batch {best_cuda[0]} = {best_cuda[1]['throughput']:,.0f} img/sec\n")
        
        if pytorch_results:
            best_pt = max(pytorch_results.items(), key=lambda x: x[1]['throughput'])
            speedup = best_cuda[1]['throughput'] / best_pt[1]['throughput']
            f.write(f"vs PyTorch GPU: {speedup:.2f}x\n")
        
        if cpu_results:
            best_cpu = max(cpu_results.items(), key=lambda x: x[1]['throughput'])
            speedup = best_cuda[1]['throughput'] / best_cpu[1]['throughput']
            f.write(f"vs CPU: {speedup:.2f}x\n")
    
    print(f"\nResults saved to {output_file}")

def main():
    print("="*60)
    print("CUDA Neural Network - Performance Benchmark")
    print("="*60)
    
    # Load network
    print("\nLoading neural network...")
    net = NeuralNetwork('models/mnist_weights.npz')
    
    # Load test data
    print("Loading test data...")
    images, labels = load_test_data(n_samples=10000)
    print(f"Loaded {len(images)} test images")
    
    # Define batch sizes to test
    batch_sizes = [1, 16, 64, 256, 1024]
    
    # Run benchmarks
    cuda_results = benchmark_cuda(net, images, labels, batch_sizes)
    pytorch_results = benchmark_pytorch_gpu(images, labels, batch_sizes)
    cpu_results = benchmark_cpu(images, labels, batch_sizes[:4])  # Skip large batches for CPU
    
    # Print comparison
    print_comparasion(cuda_results, pytorch_results, cpu_results)
    
    # Save results
    save_results(cuda_results, pytorch_results, cpu_results)
    
    print("\n" + "="*60)
    print("  Benchmark complete!")
    print("="*60)


if __name__ == '__main__':
    main()