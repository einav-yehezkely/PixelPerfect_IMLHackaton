"""
Run:
  python evaluate_augmentations.py
"""

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from labels import (
    HF_INDEX_TO_NAME,
    HF_INDEX_TO_IDX,
    TARGET_HF_INDICES,
)

# ── editable ──────────────────────────────────────────────────────────────────
AUG_ROOT = Path("dataset") / "augmentations"
TEAM_DIR = Path("submissions") / "shufflenet"   # change if your folder has another name
BATCH_SIZE = 64
WEIGHTS_FILENAME = "weights.joblib"
# ──────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class ImageNetSubset(Dataset):
    def __init__(self, root: Path, split: str, transform=None):
        self.transform = transform
        self.samples = []

        split_root = root / split

        if not split_root.exists():
            raise FileNotFoundError(
                f"Augmentation folder not found: {split_root}\n"
                f"Expected structure: {root}/{split}/<class_name>/*.jpg"
            )

        for hf_idx in sorted(TARGET_HF_INDICES):
            class_name = HF_INDEX_TO_NAME[hf_idx]
            class_dir = split_root / class_name

            if not class_dir.exists():
                raise FileNotFoundError(f"Class folder not found: {class_dir}")

            local_idx = HF_INDEX_TO_IDX[hf_idx]

            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((img_path, local_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def load_aug_loader(split_name: str):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    dataset = ImageNetSubset(
        AUG_ROOT,
        split=split_name,
        transform=transform,
    )

    print(
        f"Loaded {len(dataset)} images from {split_name} "
        f"across {len(TARGET_HF_INDICES)} classes."
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )


def load_submission(team_dir: Path):
    predict_path = team_dir / "predict.py"
    model_path = team_dir / "model.py"
    weights_path = team_dir / WEIGHTS_FILENAME

    if not predict_path.exists():
        raise FileNotFoundError(f"Missing predict.py in {team_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.py in {team_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {WEIGHTS_FILENAME} in {team_dir}")

    sys.path.insert(0, str(team_dir))
    sys.modules.pop("model", None)

    try:
        spec = importlib.util.spec_from_file_location(
            f"{team_dir.name}_predict",
            predict_path,
        )

        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import predict.py from {team_dir}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "Model"):
            raise AttributeError(
                f"predict.py in {team_dir} must define a class named Model"
            )

        model = module.Model()
        model.load(str(weights_path))

    finally:
        sys.path.pop(0)
        sys.modules.pop("model", None)

    return model


@torch.no_grad()
def evaluate(model, loader):
    correct = 0
    total = 0

    for x, y in loader:
        preds = model.predict(x)

        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total


def main():
    print("Loading submission...")
    model = load_submission(TEAM_DIR)

    augmentation_splits = [
        "color_jitter",
        "random_rotation",
    ]

    results = []

    for split_name in augmentation_splits:
        print("\n" + "=" * 60)
        print(f"Evaluating augmentation: {split_name}")
        print("=" * 60)

        loader = load_aug_loader(split_name)
        acc = evaluate(model, loader)

        results.append((split_name, acc))
        print(f"{split_name} accuracy: {acc:.4f} ({acc * 100:.2f}%)")

    print("\n--- Augmentation robustness results ---")
    for name, acc in results:
        print(f"{name:<20} {acc:.4f} ({acc * 100:.2f}%)")


if __name__ == "__main__":
    main()