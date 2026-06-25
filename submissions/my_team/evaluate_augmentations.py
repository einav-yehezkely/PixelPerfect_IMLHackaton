from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import joblib
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from base_model import ImageNetSubset
from model import ModelArchitecture


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# This matches your folder structure:
# dataset/augmentations/augmentations/color_jitter
# dataset/augmentations/augmentations/random_rotation
AUG_ROOT = PROJECT_ROOT / "dataset" / "augmentations" / "augmentations"

WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.joblib"

IMAGE_SIZE = 64
BATCH_SIZE = 64


def evaluate(model, dataloader, criterion, device, name):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Evaluating {name}", leave=True)

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            acc = correct / total

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{acc:.4f}",
            )

    avg_loss = running_loss / total
    avg_acc = correct / total

    return avg_loss, avg_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"Using device: {device}")
    print(f"Augmentation root: {AUG_ROOT}")
    print(f"Weights path: {WEIGHTS_PATH}")
    print("=" * 60)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    model = ModelArchitecture(num_classes=20).to(device)

    state_dict = joblib.load(WEIGHTS_PATH)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    augmentation_splits = [
        "color_jitter",
        "random_rotation",
    ]

    results = {}

    for split_name in augmentation_splits:
        print("\n" + "=" * 60)
        print(f"Checking augmentation: {split_name}")
        print("=" * 60)

        dataset = ImageNetSubset(
            AUG_ROOT,
            split=split_name,
            transform=transform,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        print(f"Loaded {len(dataset)} images for {split_name}")

        loss, acc = evaluate(
            model,
            dataloader,
            criterion,
            device,
            split_name,
        )

        results[split_name] = {
            "loss": loss,
            "accuracy": acc,
        }

        print(f"\n{split_name} results:")
        print(f"Loss     : {loss:.4f}")
        print(f"Accuracy : {acc:.4f} ({acc * 100:.2f}%)")

    print("\n" + "=" * 60)
    print("Final robustness results")
    print("=" * 60)

    for name, result in results.items():
        print(
            f"{name:15s} | "
            f"Loss: {result['loss']:.4f} | "
            f"Accuracy: {result['accuracy']:.4f} "
            f"({result['accuracy'] * 100:.2f}%)"
        )


if __name__ == "__main__":
    main()