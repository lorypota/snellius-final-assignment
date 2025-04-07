import os
import random
import numpy as np
import torch
from pathlib import Path

import openml
from torch.utils.data import DataLoader
from torchvision import transforms

from core.dataloading import ImageDataset
from core.fewshot import SubsetByClass, EpisodicSampler

# Define results directory
RESULTS_DIR = os.path.join(os.getcwd(), "results")
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")
OUTPUTS_DIR = os.path.join(RESULTS_DIR, "outputs")

# Create directories if they don't exist
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Load flower dataset
FLW = openml.datasets.get_dataset(44283, download_all_files=True)
data_dir_flower = Path(openml.config.get_cache_directory())/'datasets'/str(FLW.dataset_id)/"FLW_Mini"/"images"
Xi_all, yi_all, categorical_indicator, attribute_names = FLW.get_data(target=FLW.default_target_attribute)
Xi_all["file_path"] = Xi_all["FILE_NAME"].apply(lambda x: os.path.join(data_dir_flower, x))
print("The dataset has {} images of flowers and {} classes".format(Xi_all.shape[0], len(np.unique(yi_all))))

# Map labels to integers
flower_to_idx = {label: idx for idx, label in enumerate(sorted(yi_all.unique()))}
idx_to_flower = {idx: label for label, idx in flower_to_idx.items()}
yi_all = yi_all.map(flower_to_idx)

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# Create the dataset
flower_ds = ImageDataset(Xi_all, yi_all, transform=train_transform)

# Split classes into train, validation, and test sets
all_classes = flower_ds.classes
random.shuffle(all_classes)

num_classes = len(all_classes)
train_classes = all_classes[:int(0.6 * num_classes)]
val_classes = all_classes[int(0.6 * num_classes):int(0.8 * num_classes)]
test_classes = all_classes[int(0.8 * num_classes):]

# Create subsets based on class splits
train_dataset = SubsetByClass(flower_ds, train_classes)
val_dataset = SubsetByClass(flower_ds, val_classes)
test_dataset = SubsetByClass(flower_ds, test_classes)

train_sampler = EpisodicSampler(train_dataset, episodes_per_epoch=100, N_way=5, K_shot=5, Q_query=5)
val_sampler = EpisodicSampler(val_dataset, episodes_per_epoch=50, N_way=5, K_shot=5, Q_query=5)
test_sampler = EpisodicSampler(test_dataset, episodes_per_epoch=100, N_way=5, K_shot=5, Q_query=5)

# Print stats about the data splits
print(f"Total classes: {num_classes}")
print(f"Training classes: {len(train_classes)}")
print(f"Validation classes: {len(val_classes)}")
print(f"Test classes: {len(test_classes)}")

# Create a custom DataLoader wrapper for EpisodicSampler
class EpisodicDataLoader:
    def __init__(self, episodic_sampler):
        self.episodic_sampler = episodic_sampler
        
    def __iter__(self):
        return iter(self.episodic_sampler)
    
    def __len__(self):
        return len(self.episodic_sampler)