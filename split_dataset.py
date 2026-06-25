from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

# Root directory containing the dataset
dataset_root = Path("dataset")

# Original training directory
train_root = dataset_root / "train"

# Validation directory to be created
val_root = dataset_root / "validation"
val_root.mkdir(exist_ok=True)

# Iterate over all class folders
for class_dir in tqdm(list(train_root.iterdir()), desc="Processing classes"):
    if not class_dir.is_dir():
        continue

    # Get all images belonging to the current class
    images = list(class_dir.glob("*"))

    # Split into 80% train and 20% validation
    train_imgs, val_imgs = train_test_split(
        images,
        test_size=0.2,
        random_state=42
    )

    # Create the corresponding validation class folder
    val_class_dir = val_root / class_dir.name
    val_class_dir.mkdir(exist_ok=True)

    # Move validation images from train to validation
    for img in tqdm(
        val_imgs,
        desc=f"Moving {class_dir.name}",
        leave=False
    ):
        shutil.move(
            str(img),
            str(val_class_dir / img.name)
        )

print("Dataset successfully split into train and validation.")