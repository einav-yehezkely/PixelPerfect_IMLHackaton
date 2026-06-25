from pathlib import Path
import sys
import time

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

DATA_ROOT = PROJECT_ROOT / "dataset" / "train_set"

OUTPUT = Path(__file__).resolve().parent / "weights.joblib"

IMAGE_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001


def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}m {seconds}s"


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=True)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        current_acc = correct / total

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{current_acc:.4f}",
        )

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=True)

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            current_acc = correct / total

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{current_acc:.4f}",
            )

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    output_dir = Path(__file__).resolve().parent

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, marker="o", label="Train")
    plt.plot(epochs, val_losses, marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Across Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_accs, marker="o", label="Train")
    plt.plot(epochs, val_accs, marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Across Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png")
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"Using device: {device}")
    print("=" * 60)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

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
    print(f"Image size         : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size         : {BATCH_SIZE}")
    print(f"Epochs             : {EPOCHS}")
    print(f"Learning rate      : {LR}")
    print("=" * 60)

    model = ModelArchitecture(num_classes=20).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
    )

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    total_start_time = time.time()
    epoch_times = []

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()

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

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        epochs_left = EPOCHS - (epoch + 1)
        estimated_time_left = avg_epoch_time * epochs_left

        total_elapsed = time.time() - total_start_time

        print("\nEpoch summary:")
        print(f"Train Loss          : {train_loss:.4f}")
        print(f"Train Accuracy      : {train_acc:.4f} ({train_acc * 100:.2f}%)")
        print(f"Validation Loss     : {val_loss:.4f}")
        print(f"Validation Accuracy : {val_acc:.4f} ({val_acc * 100:.2f}%)")

        print("\nTime:")
        print(f"Epoch time          : {format_time(epoch_time)}")
        print(f"Total elapsed       : {format_time(total_elapsed)}")
        print(f"Estimated time left : {format_time(estimated_time_left)}")

    plot_training_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
    )

    print("\nSaved loss_curve.png and accuracy_curve.png")

    state_dict = model.cpu().state_dict()
    joblib.dump(state_dict, OUTPUT)

    total_time = time.time() - total_start_time

    print("\n" + "=" * 60)
    print("Training finished!")
    print(f"Total training time : {format_time(total_time)}")
    print(f"Saved trained weights to: {OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()