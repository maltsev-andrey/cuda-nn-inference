"""
Train a simple fully-connected neural network on MNIST
This will serve as a baseline and provide weights for CUDA inference
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import numpy as np
from tqdm import tqdm
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

# Network architectura (most much my CUDA implementation)
class SimpleNet(nn.Module):
    """
    Simple 3-layer fully-connected network
    Architecture: 784 → 256 → 128 → 10
    """
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # Flatten input
        x = x.view(-1, 784)

        # Layer 1: Linear + ReLU
        x = torch.relu(self.fc1(x))

        # Layer 2: Linear +  ReLU
        x = torch.relu(self.fc2(x))

        # Layer 3: Linear (no activation, we'll apply softmax during inference)
        x = self.fc3(x)

        return x

def get_data_loaders():
    """Load MNIST dataset"""
    # Transform: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Download and load training data
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # Download  and load test data
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
    for batch_idx, (data, target) in enumerate(pbar):
        # Move data to device
        data, target = data.to(device), target.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

         # Calculate accuracy
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        total_loss += loss.item()

        # Update preogress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}'
        })

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def test(model, test_loader, criterion):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0
    correct =  0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc = 'Testing'):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Calculate loss
            test_loss += criterion(output, target).item()

            # Calculate accuracy
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy

def save_weights_for_cuda(model, filepath='models/mnist_weights.npz'):
    """
    Save model weights in format suitable for CUDA implementation
    Extracts weights and biases as NumPy arrays
    """
    os.makedirs('models', exist_ok=True)
    
    # Extract weights and biases
    weights = {}

    # Layer 1: fc1
    weights['w1'] = model.fc1.weight.data.cpu().numpy()  # Shape: (256, 784)
    weights['b1'] = model.fc1.bias.data.cpu().numpy()    # Shape: (256,)
    
    # Layer 2: fc2
    weights['w2'] = model.fc2.weight.data.cpu().numpy()  # Shape: (128, 256)
    weights['b2'] = model.fc2.bias.data.cpu().numpy()    # Shape: (128,)
    
    # Layer 3: fc3
    weights['w3'] = model.fc3.weight.data.cpu().numpy()  # Shape: (10, 128)
    weights['b3'] = model.fc3.bias.data.cpu().numpy()    # Shape: (10,)
    
    # Save as .npz file
    np.savez(filepath, **weights)
    print(f"\nWeights saved to {filepath}")

    # Print shapes for verification
    print("\nWeight shapes:")
    for name, weight in weights.items():
        print(f" {name}: {weight.shape}")

def main():
    print("="*60)
    print("Training MNIST Classifier")
    print("="*60)

    # Get data loaders
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = get_data_loaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = SimpleNet().to(device)
    print(f"\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("\nStartin training...")
    best_accurasy = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
        test_loss, test_acc = test(model, test_loader, criterion)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f" Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f" Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Save for best model
        if test_acc > best_accurasy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'models/mnist_model.pth')
            print(f" Best model saved (accuracy: {test_acc:.2f}%)")

    print("\n" + "="*60)
    print(f"Training complete! Best test accuracy: {best_accuracy:.2f}%")
    print("="*60)
    
    # Save weights for CUDA implementation
    print("\nExtracting weights for CUDA implementation...")
    save_weights_for_cuda(model)

    print("\n All done+ you can use the weights in your CUDA implementation.")

if __name__ == '__main__':
    main()
    





















 