"""
Neural Network Inference Engine using CUDA

Loads pre-trained weights and performs fast GPU inference
Architecture: 784 → 256 → 128 → 10 (MNIST classifier)
"""

import numpy as np
from numba import cuda
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# Import our CUDA kernels
from layers import relu_forward, add_bias, softmax_forward
try:
    from matmul_optimized import matmul
    print("Using optimized matrix multiplication")
except ImportError:
    from matmul import matmul
    print("Warning: Using simple matrix multiplication (slower)")


class NeuralNetwork:
    """
    GPU-accelerated neural network for inference
    
    Architecture:
        Input (784) → FC(256) + ReLU → FC(128) + ReLU → FC(10) + Softmax → Output
    """
    
    def __init__(self, weights_path='models/mnist_weights.npz'):
        """
        Load pre-trained weights from file
        
        Args:
            weights_path: path to .npz file with weights
        """
        print(f"Loading weights from {weights_path}...")
        
        # Load weights
        data = np.load(weights_path)
        
        # Store weights (keep on CPU, will transfer per batch)
        # Note: PyTorch stores weights as (out_features, in_features)
        # For matmul: output = input @ weights.T + bias
        self.w1 = data['w1'].astype(np.float32)  # (256, 784)
        self.b1 = data['b1'].astype(np.float32)  # (256,)
        self.w2 = data['w2'].astype(np.float32)  # (128, 256)
        self.b2 = data['b2'].astype(np.float32)  # (128,)
        self.w3 = data['w3'].astype(np.float32)  # (10, 128)
        self.b3 = data['b3'].astype(np.float32)  # (10,)
        
        print(f"  Layer 1: {self.w1.shape} + bias {self.b1.shape}")
        print(f"  Layer 2: {self.w2.shape} + bias {self.b2.shape}")
        print(f"  Layer 3: {self.w3.shape} + bias {self.b3.shape}")
        print("Weights loaded successfully!")
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: input batch (batch_size, 784) - flattened images
        
        Returns:
            probabilities (batch_size, 10) - one probability per digit class
        """
        # Ensure input is float32 and 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x = x.astype(np.float32)
        
        # Layer 1: Linear + ReLU
        # x: (batch, 784) @ w1.T: (784, 256) → (batch, 256)
        z1 = matmul(x, self.w1.T)
        z1 = add_bias(z1, self.b1)
        a1 = relu_forward(z1)
        
        # Layer 2: Linear + ReLU
        # a1: (batch, 256) @ w2.T: (256, 128) → (batch, 128)
        z2 = matmul(a1, self.w2.T)
        z2 = add_bias(z2, self.b2)
        a2 = relu_forward(z2)
        
        # Layer 3: Linear + Softmax
        # a2: (batch, 128) @ w3.T: (128, 10) → (batch, 10)
        z3 = matmul(a2, self.w3.T)
        z3 = add_bias(z3, self.b3)
        output = softmax_forward(z3)
        
        return output
    
    def predict(self, x):
        """
        Predict class labels for input batch
        
        Args:
            x: input batch (batch_size, 784) or single image (784,)
        
        Returns:
            predicted class labels (batch_size,) - digits 0-9
        """
        probs = self.forward(x)
        return np.argmax(probs, axis=1)
    
    def predict_proba(self, x):
        """
        Get probability distribution for input batch
        
        Args:
            x: input batch (batch_size, 784)
        
        Returns:
            probabilities (batch_size, 10)
        """
        return self.forward(x)


def load_test_data():
    """Load MNIST test data for verification"""
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
        
        # Convert to numpy arrays
        images = []
        labels = []
        for img, label in test_dataset:
            images.append(img.numpy().flatten())
            labels.append(label)
        
        return np.array(images, dtype=np.float32), np.array(labels)
    
    except ImportError:
        print("PyTorch not available, generating random test data")
        return np.random.randn(100, 784).astype(np.float32), np.random.randint(0, 10, 100)


# ============================================================================
# Testing and Verification
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Neural Network Inference Test")
    print("=" * 60)
    
    # Load network
    print("\n[1] Loading neural network...")
    net = NeuralNetwork('models/mnist_weights.npz')
    
    # Load test data
    print("\n[2] Loading test data...")
    images, labels = load_test_data()
    print(f"  Test images: {images.shape}")
    print(f"  Test labels: {labels.shape}")
    
    # Test single image
    print("\n[3] Testing single image inference...")
    single_image = images[0:1]  # Keep 2D: (1, 784)
    probs = net.forward(single_image)
    pred = net.predict(single_image)
    print(f"  Input shape: {single_image.shape}")
    print(f"  Output probabilities: {probs[0]}")
    print(f"  Predicted digit: {pred[0]}")
    print(f"  Actual digit: {labels[0]}")
    print(f"  Correct: {pred[0] == labels[0]}")
    
    # Test batch inference
    print("\n[4] Testing batch inference...")
    batch_size = 100
    batch_images = images[:batch_size]
    batch_labels = labels[:batch_size]
    
    predictions = net.predict(batch_images)
    accuracy = (predictions == batch_labels).mean() * 100
    
    print(f"  Batch size: {batch_size}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    # Test larger batch
    print("\n[5] Testing full test set...")
    all_predictions = net.predict(images)
    full_accuracy = (all_predictions == labels).mean() * 100
    
    print(f"  Total images: {len(images)}")
    print(f"  Accuracy: {full_accuracy:.2f}%")
    
    # Compare with PyTorch (if available)
    print("\n[6] Comparing with PyTorch baseline...")
    try:
        import torch
        import torch.nn as nn
        
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
        
        # Load PyTorch model
        pytorch_model = SimpleNet()
        pytorch_model.load_state_dict(torch.load('models/mnist_model.pth', weights_only=True))
        pytorch_model.eval()
        
        # Run PyTorch inference
        with torch.no_grad():
            pytorch_input = torch.from_numpy(images)
            pytorch_output = pytorch_model(pytorch_input)
            pytorch_probs = torch.softmax(pytorch_output, dim=1).numpy()
            pytorch_preds = pytorch_output.argmax(dim=1).numpy()
        
        pytorch_accuracy = (pytorch_preds == labels).mean() * 100
        
        print(f"  PyTorch accuracy: {pytorch_accuracy:.2f}%")
        print(f"  CUDA accuracy:    {full_accuracy:.2f}%")
        print(f"  Predictions match: {(all_predictions == pytorch_preds).mean() * 100:.2f}%")
        
        # Compare probability outputs
        cuda_probs = net.predict_proba(images)
        max_diff = np.abs(cuda_probs - pytorch_probs).max()
        print(f"  Max probability difference: {max_diff:.6f}")
        
    except Exception as e:
        print(f"  PyTorch comparison skipped: {e}")
    
    print("\n" + "=" * 60)
    print("✓ Neural network inference test complete!")
    print("=" * 60)
