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

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from tqdm import tqdm

from base_model import ImageNetSubset
from model import ModelArchitecture


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATA_ROOT = PROJECT_ROOT / "dataset" / "train_set"
AUG_ROOT = PROJECT_ROOT / "dataset" / "augmentations" / "augmentations"

OUTPUT = Path(__file__).resolve().parent / "weights.joblib"

IMAGE_SIZE = 64
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

    return running_loss / total, correct / total


def evaluate(model, dataloader, criterion, device, name="Validation"):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=name, leave=True)

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

    return running_loss / total, correct / total


def plot_training_curves(
    train_losses,
    clean_val_losses,
    robust_val_losses,
    train_accs,
    clean_val_accs,
    robust_val_accs,
):
    epochs = range(1, len(train_losses) + 1)
    output_dir = Path(__file__).resolve().parent

    # Loss graph
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, clean_val_losses, marker="o", label="Clean Validation Loss")
    plt.plot(epochs, robust_val_losses, marker="o", label="Robust Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Across Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()

    # Accuracy graph
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_accs, marker="o", label="Train Accuracy")
    plt.plot(epochs, clean_val_accs, marker="o", label="Clean Validation Accuracy")
    plt.plot(epochs, robust_val_accs, marker="o", label="Robust Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Across Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png")
    plt.close()

    # Clean validation accuracy only
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, clean_val_accs, marker="o", label="Clean Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Clean Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "clean_validation_accuracy_curve.png")
    plt.close()

    # Robust validation accuracy only
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, robust_val_accs, marker="o", label="Robust Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Robust Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "robust_validation_accuracy_curve.png")
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"Using device: {device}")
    print("=" * 60)

    # Stronger train augmentations.
    # These are used while training, so the model sees many versions of the same image.
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.04,
        ),
        transforms.ToTensor(),
    ])

    # Clean validation transform.
    # No random changes here, so we measure normal validation accuracy.
    clean_val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # Robust validation transform.
    # This checks whether the model still works after visual changes.
    robust_val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.25,
            hue=0.05,
        ),
        transforms.ToTensor(),
    ])

    # Regular training dataset
    regular_train_dataset = ImageNetSubset(
        DATA_ROOT,
        split="train",
        transform=train_transform,
    )

    train_datasets = [regular_train_dataset]

    # Add provided augmentation folders into training, if they exist.
    # This exposes the model to already-manipulated images.
    color_jitter_path = AUG_ROOT / "color_jitter"
    random_rotation_path = AUG_ROOT / "random_rotation"

    if color_jitter_path.exists():
        color_jitter_dataset = ImageNetSubset(
            AUG_ROOT,
            split="color_jitter",
            transform=train_transform,
        )
        train_datasets.append(color_jitter_dataset)

    if random_rotation_path.exists():
        random_rotation_dataset = ImageNetSubset(
            AUG_ROOT,
            split="random_rotation",
            transform=train_transform,
        )
        train_datasets.append(random_rotation_dataset)

    train_dataset = ConcatDataset(train_datasets)

    # Clean validation dataset
    clean_val_dataset = ImageNetSubset(
        DATA_ROOT,
        split="validation",
        transform=clean_val_transform,
    )

    # Robust validation dataset: same validation images, but with visual changes
    robust_val_dataset = ImageNetSubset(
        DATA_ROOT,
        split="validation",
        transform=robust_val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    clean_val_loader = DataLoader(
        clean_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    robust_val_loader = DataLoader(
        robust_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"Training samples          : {len(train_dataset)}")
    print(f"Clean validation samples  : {len(clean_val_dataset)}")
    print(f"Robust validation samples : {len(robust_val_dataset)}")
    print(f"Image size                : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size                : {BATCH_SIZE}")
    print(f"Epochs                    : {EPOCHS}")
    print(f"Learning rate             : {LR}")
    print("=" * 60)

    model = ModelArchitecture(num_classes=20).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
    )

    train_losses = []
    clean_val_losses = []
    robust_val_losses = []

    train_accs = []
    clean_val_accs = []
    robust_val_accs = []

    best_score = 0.0
    best_clean_acc = 0.0
    best_robust_acc = 0.0

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

        clean_val_loss, clean_val_acc = evaluate(
            model,
            clean_val_loader,
            criterion,
            device,
            name="Clean Validation",
        )

        robust_val_loss, robust_val_acc = evaluate(
            model,
            robust_val_loader,
            criterion,
            device,
            name="Robust Validation",
        )

        train_losses.append(train_loss)
        clean_val_losses.append(clean_val_loss)
        robust_val_losses.append(robust_val_loss)

        train_accs.append(train_acc)
        clean_val_accs.append(clean_val_acc)
        robust_val_accs.append(robust_val_acc)

        # We save according to a combined score:
        # clean accuracy + robust accuracy.
        # This matches the challenge idea: good standard accuracy and robustness.
        combined_score = 0.5 * clean_val_acc + 0.5 * robust_val_acc

        if combined_score > best_score:
            best_score = combined_score
            best_clean_acc = clean_val_acc
            best_robust_acc = robust_val_acc

            state_dict = model.cpu().state_dict()
            joblib.dump(state_dict, OUTPUT)
            model.to(device)

            print("\nNew best model saved!")
            print(f"Clean Validation Accuracy  : {clean_val_acc:.4f} ({clean_val_acc * 100:.2f}%)")
            print(f"Robust Validation Accuracy : {robust_val_acc:.4f} ({robust_val_acc * 100:.2f}%)")
            print(f"Combined Score             : {combined_score:.4f}")

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        epochs_left = EPOCHS - (epoch + 1)
        estimated_time_left = avg_epoch_time * epochs_left

        total_elapsed = time.time() - total_start_time

        print("\nEpoch summary:")
        print(f"Train Loss                 : {train_loss:.4f}")
        print(f"Train Accuracy             : {train_acc:.4f} ({train_acc * 100:.2f}%)")
        print(f"Clean Validation Loss      : {clean_val_loss:.4f}")
        print(f"Clean Validation Accuracy  : {clean_val_acc:.4f} ({clean_val_acc * 100:.2f}%)")
        print(f"Robust Validation Loss     : {robust_val_loss:.4f}")
        print(f"Robust Validation Accuracy : {robust_val_acc:.4f} ({robust_val_acc * 100:.2f}%)")
        print(f"Combined Score             : {combined_score:.4f}")
        print(f"Best Combined Score        : {best_score:.4f}")

        print("\nTime:")
        print(f"Epoch time                 : {format_time(epoch_time)}")
        print(f"Total elapsed              : {format_time(total_elapsed)}")
        print(f"Estimated time left        : {format_time(estimated_time_left)}")

    plot_training_curves(
        train_losses,
        clean_val_losses,
        robust_val_losses,
        train_accs,
        clean_val_accs,
        robust_val_accs,
    )

    total_time = time.time() - total_start_time

    print("\nSaved graphs:")
    print("loss_curve.png")
    print("accuracy_curve.png")
    print("clean_validation_accuracy_curve.png")
    print("robust_validation_accuracy_curve.png")

    print("\n" + "=" * 60)
    print("Training finished!")
    print(f"Total training time        : {format_time(total_time)}")
    print(f"Best combined score        : {best_score:.4f}")
    print(f"Best clean validation acc  : {best_clean_acc:.4f} ({best_clean_acc * 100:.2f}%)")
    print(f"Best robust validation acc : {best_robust_acc:.4f} ({best_robust_acc * 100:.2f}%)")
    print(f"Best weights saved to      : {OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()