import os
import torch
import numpy as np
import json
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score


# Dataset Class
class PileupDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        PyTorch Dataset for loading pileup images and corresponding genotypes.

        :param data_dir: Directory containing the .npy files and metadata.json
        :param transform: Image transformations (augmentations)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        # Load metadata
        metadata_path = os.path.join(data_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Collect samples with file paths and labels
        for entry in metadata:
            img_path = entry["npy_path"]
            genotype = entry["genotype"]
            if os.path.exists(img_path):
                self.samples.append((img_path, genotype))

        # Get unique genotype labels
        self.genotype_classes = sorted(set(gt for _, gt in self.samples))
        self.genotype_to_idx = {gt: i for i, gt in enumerate(self.genotype_classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Loads an image and returns it as a tensor with the corresponding label.
        """
        img_path, genotype = self.samples[idx]

        # Load the .npy image
        image = np.load(img_path).astype(np.float32)  # Convert to float for CNN
        image = torch.tensor(image).permute(2, 0, 1)  # Convert to (C, H, W) format

        # Normalize image (optional)
        if self.transform:
            image = self.transform(image)

        # Convert genotype to numerical label
        label = self.genotype_to_idx[genotype]
        return image, label


# Function to create DataLoader
def get_dataloader(data_dir, batch_size=32, shuffle=True, num_workers=2):
    """
    Returns a DataLoader for the pileup dataset.

    :param data_dir: Directory containing the data
    :param batch_size: Batch size
    :param shuffle: Whether to shuffle data
    :param num_workers: Number of workers for loading data
    :return: PyTorch DataLoader and genotype class mapping
    """
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize if needed
    ])

    dataset = PileupDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader, dataset.genotype_to_idx


# Load GoogleNet Model
def load_googlenet(num_classes):
    model = models.googlenet(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# Train Function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    """
    Trains the model and validates after each epoch.

    :param model: The GoogleNet model
    :param train_loader: Training DataLoader
    :param val_loader: Validation DataLoader
    :param criterion: Loss function
    :param optimizer: Optimizer
    :param device: Device to run on (CPU/GPU)
    :param num_epochs: Number of epochs
    """
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Compute validation accuracy
        val_acc = evaluate_model(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Validation Accuracy: {val_acc:.2f}%")


# Evaluation Function
def evaluate_model(model, data_loader, device):
    """
    Evaluates model performance on validation/test set.

    :param model: The trained model
    :param data_loader: DataLoader for validation/test data
    :param device: Device to run on (CPU/GPU)
    :return: Accuracy percentage
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds) * 100


# Main Function
if __name__ == "__main__":
    data_dir = "output_pileups"  # Change this to your dataset path
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_loader, genotype_map = get_dataloader(data_dir, batch_size, shuffle=True)
    val_loader, _ = get_dataloader(data_dir, batch_size, shuffle=False)  # Using same set for validation for simplicity

    # Print genotype mapping
    print("Genotype Classes:", genotype_map)

    # Load and modify GoogleNet
    model = load_googlenet(num_classes=len(genotype_map))
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)

    # Final evaluation
    test_acc = evaluate_model(model, val_loader, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
