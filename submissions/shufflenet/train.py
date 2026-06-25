from pathlib import Path
import sys
import copy
import random

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from base_model import ImageNetSubset
from model import ModelArchitecture


DATA_ROOT = PROJECT_ROOT / "dataset"
OUTPUT = Path("weights.joblib")

IMAGE_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 15
LR = 0.01
SUBSET_FRACTION = 1


val_losses = []
val_accuracies = []


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    for images, labels in tqdm(dataloader, desc="Train"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, desc="Eval"):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc=desc):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def create_datasets(train_transform, val_transform):
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

    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    return train_dataset, val_dataset


def create_random_train_loader(train_dataset, subset_fraction):
    num_train = len(train_dataset)
    num_subset = int(subset_fraction * num_train)

    indices = random.sample(range(num_train), num_subset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=SubsetRandomSampler(indices),
        num_workers=0,
    )

    print(f"Training this epoch on {num_subset}/{num_train} images")
    print(f"Train batches this epoch: {len(train_loader)}")

    return train_loader


def create_val_loader(val_dataset):
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"Validation batches: {len(val_loader)}")

    return val_loader


def plot_training_progress():
    epochs = list(range(len(val_losses)))

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_progress.png")
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomChoice([
            transforms.RandomRotation(10),

            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
            ),

            transforms.GaussianBlur(kernel_size=3),

            transforms.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
            ),

            transforms.RandomGrayscale(p=1.0),
        ]),

        transforms.ToTensor(),

        transforms.RandomErasing(p=0.3),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset, val_dataset = create_datasets(
        train_transform,
        val_transform,
    )

    val_loader = create_val_loader(val_dataset)

    model = ModelArchitecture(num_classes=20).to(device)

    criterion = nn.CrossEntropyLoss()

    initial_val_loss, initial_val_acc = evaluate(
        model,
        val_loader,
        criterion,
        device,
        desc="Initial Val Eval",
    )

    val_losses.append(initial_val_loss)
    val_accuracies.append(initial_val_acc)

    print("\nInitial performance before training:")
    print(f"Initial Val Loss: {initial_val_loss:.4f} | Initial Val Acc: {initial_val_acc:.4f}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5,
    )

    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 20)

        train_loader = create_random_train_loader(
            train_dataset,
            subset_fraction=SUBSET_FRACTION,
        )

        train_one_epoch(
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
            desc="Val Eval",
        )

        scheduler.step()

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    plot_training_progress()

    state_dict = model.state_dict()
    joblib.dump(state_dict, OUTPUT)

    output_txt = Path(__file__).resolve().parent / "output.txt"

    with open(output_txt, "w") as f:
        f.write("=== Training Summary ===\n\n")
        f.write(f"Device: {device}\n")
        f.write(f"Image size: {IMAGE_SIZE}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Learning rate: {LR}\n")
        f.write(f"Optimizer: AdamW\n")
        f.write(f"Weight decay: 1e-4\n")
        f.write(f"Subset fraction per epoch: {SUBSET_FRACTION}\n\n")

        f.write(f"Train images total: {len(train_dataset)}\n")
        f.write(f"Train images per epoch: {int(SUBSET_FRACTION * len(train_dataset))}\n")
        f.write(f"Validation images: {len(val_dataset)}\n\n")

        f.write(f"Initial validation loss: {initial_val_loss:.4f}\n")
        f.write(f"Initial validation accuracy: {initial_val_acc:.4f}\n\n")

        f.write("Validation results by epoch:\n")
        for epoch, (loss, acc) in enumerate(zip(val_losses[1:], val_accuracies[1:]), start=1):
            f.write(
                f"Epoch {epoch:2d}: "
                f"Loss = {loss:.4f}, "
                f"Accuracy = {acc:.4f}\n"
            )

        f.write(f"\nBest validation accuracy: {best_val_acc:.4f}\n")

    print(f"\nSaved model to {OUTPUT}")
    print("Saved plot to training_progress.png")
    print(f"Saved run summary to {output_txt}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()