import os
import openml
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from core.dataloading import ImageDataset

# -------------------------
# Dataset setup, splits, transforms, dataloaders
# -------------------------

# Set up the birds dataset from OpenML
BRD = openml.datasets.get_dataset(46770, download_all_files=True)
data_dir = Path(openml.config.get_cache_directory()) / 'datasets' / str(BRD.dataset_id) / "BRD_Extended" / "images"
X_all, y_all, categorical_indicator, attribute_names = BRD.get_data(target=BRD.default_target_attribute)
X_all["file_path"] = X_all["FILE_NAME"].apply(lambda x: os.path.join(str(data_dir), x))
print("The dataset has {} images of birds and {} classes".format(X_all.shape[0], len(np.unique(y_all))))

# Map labels to integers
label_to_idx = {label: idx for idx, label in enumerate(sorted(y_all.unique()))}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
y_all = y_all.map(label_to_idx)

# Train-test split and data loaders
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Transform for training data with augmentation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),  # Small rotation for augmentation
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Color augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
])

# Transform for validation/test data
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
])

# Create datasets and data loaders
train_ds = ImageDataset(X_train, y_train, transform=transform)
val_ds = ImageDataset(X_val, y_val, transform=test_transform)
test_ds = ImageDataset(X_test, y_test, transform=test_transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# For OpenML setup
def get_openml_transform():
    """Returns the transform to be used with OpenML data module"""
    return transforms.Compose([
        transforms.ToPILImage(),
        # Lambda function to convert to RGB will be provided by OpenML's convert_to_rgb
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

def get_data_dir():
    """Returns the data directory path for the birds dataset"""
    return str(data_dir)