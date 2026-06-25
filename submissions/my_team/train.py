from pathlib import Path
import sys

# Add the project root directory to Python's import path.
# This allows importing base_model.py and labels.py from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from base_model import ImageNetSubset
from model import ModelArchitecture


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATA_ROOT = PROJECT_ROOT / "dataset"
OUTPUT = Path("weights.joblib")

IMAGE_SIZE = 64
BATCH_SIZE = 128
EPOCHS = 5
LR = 0.0001


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch."""

    # Set the model to training mode.
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # tqdm shows a progress bar for the current epoch.
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for images, labels in progress_bar:
        # Move the current batch to the selected device.
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients from the previous optimization step.
        optimizer.zero_grad()

        # Forward pass.
        outputs = model(images)

        # Compute the classification loss.
        loss = criterion(outputs, labels)

        # Backward pass.
        loss.backward()

        # Update model parameters.
        optimizer.step()

        # Accumulate total loss.
        running_loss += loss.item() * images.size(0)

        # Compute predictions and update accuracy counters.
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Display current batch statistics in the progress bar.
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / total:.4f}",
        )

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model on the validation set."""

    # Set the model to evaluation mode.
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient calculation during validation.
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)

        for images, labels in progress_bar:
            # Move the current batch to the selected device.
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass.
            outputs = model(images)

            # Compute validation loss.
            loss = criterion(outputs, labels)

            # Accumulate total loss.
            running_loss += loss.item() * images.size(0)

            # Compute predictions and update accuracy counters.
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Display current validation statistics in the progress bar.
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct / total:.4f}",
            )

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """Plot and save loss and accuracy curves."""

    epochs = range(1, len(train_losses) + 1)

    # Plot training and validation loss.
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, marker="o", label="Train")
    plt.plot(epochs, val_losses, marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Across Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.close()

    # Plot training and validation accuracy.
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_accs, marker="o", label="Train")
    plt.plot(epochs, val_accs, marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Across Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_curve.png")
    plt.close()


def main():
    """Full training pipeline that creates weights.joblib."""

    # Select GPU if available, otherwise use CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Training transformations include simple augmentations for robustness.
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor(),
    ])

    # Validation transformations should not include random augmentation.
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # Load the existing train and validation folders.
    # ImageNetSubset expects the dataset root and the split name separately.
    train_dataset = ImageNetSubset(
        DATA_ROOT,
        split="train",
        transform=train_transform,
    )

    val_dataset = ImageNetSubset(
        DATA_ROOT,
        split="validation",
        transform=val_transform,
    )

    # Create data loaders for mini-batch training and validation.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"Training samples   : {len(train_dataset)}")
    print(f"Validation samples : {len(val_dataset)}")

    # Initialize the model from scratch.
    model = ModelArchitecture(num_classes=20).to(device)

    # Cross-entropy is suitable for multi-class classification.
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer with a small learning rate.
    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
    )

    # Store metrics for plotting after training.
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Main training loop.
    for epoch in range(EPOCHS):
        print("\n" + "=" * 60)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("=" * 60)

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_acc:.4f}"
        )

        print(
            f"Validation Loss: {val_loss:.4f} | "
            f"Validation Accuracy: {val_acc:.4f}"
        )

    # Save loss and accuracy plots.
    plot_training_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
    )

    print("\nSaved loss_curve.png and accuracy_curve.png")

    # Move model to CPU before saving for hardware-independent loading.
    state_dict = model.cpu().state_dict()
    joblib.dump(state_dict, OUTPUT)

    print(f"Saved trained weights to {OUTPUT}")


if __name__ == "__main__":
    main()