import os
import openml
import numpy as np
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from core.dataloading import ImageDataset

# -------------------------
# Dataset setup, splits, transforms, dataloaders
# -------------------------

# Do not change this code and don't overwrite the data splits created below.
FLW = openml.datasets.get_dataset(44283, download_all_files=True)
data_dir_flower = Path(openml.config.get_cache_directory())/'datasets'/str(FLW.dataset_id)/"FLW_Mini"/"images"
Xi_all, yi_all, categorical_indicator, attribute_names = FLW.get_data(target=FLW.default_target_attribute)
Xi_all["file_path"] = Xi_all["FILE_NAME"].apply(lambda x: os.path.join(str(data_dir_flower), x))
print("The dataset has {} images of flowers and {} classes".format(Xi_all.shape[0], len(np.unique(yi_all))))

# Map labels to integers
flower_to_idx = {label: idx for idx, label in enumerate(sorted(yi_all.unique()))}
idx_to_flower = {idx: label for label, idx in flower_to_idx.items()}
yi_all = yi_all.map(flower_to_idx)

# Transform for training data. Feel free to change (e.g. add data augmentation).
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),  # Added some rotation for augmentation
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Added color jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# Transform for test data. Leave this unchanged.
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# Create the complete dataset
flower_ds = ImageDataset(Xi_all, yi_all, transform=transform)

# Class to create subsets by class
class SubsetByClass(Dataset):
    def __init__(self, dataset, class_list):
        self.dataset = dataset
        self.indices = []
        
        # Get indices of samples belonging to the specified classes
        for i in range(len(dataset)):
            _, label = dataset[i]
            if label in class_list:
                self.indices.append(i)
        
        self.classes = class_list
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

# Episodic sampler for few-shot learning
class EpisodicSampler:
    def __init__(self, dataset, episodes_per_epoch=100, N_way=5, K_shot=5, Q_query=5):
        """
        Args:
            dataset: Dataset to sample from
            episodes_per_epoch: Number of episodes per epoch
            N_way: Number of classes per episode
            K_shot: Number of support examples per class
            Q_query: Number of query examples per class
        """
        self.dataset = dataset
        self.episodes_per_epoch = episodes_per_epoch
        self.N_way = N_way
        self.K_shot = K_shot
        self.Q_query = Q_query
        
        # Group samples by class
        self.samples_by_class = {}
        for i in range(len(dataset)):
            img, label = dataset[i]
            if label not in self.samples_by_class:
                self.samples_by_class[label] = []
            self.samples_by_class[label].append(i)
    
    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            # Randomly select N classes
            selected_classes = random.sample(self.dataset.classes, self.N_way)
            
            support_x, support_y = [], []
            query_x, query_y = [], []
            
            # For each selected class
            for class_idx, cls in enumerate(selected_classes):
                # Get indices for this class
                indices = self.samples_by_class[cls]
                
                # Ensure we have enough samples
                if len(indices) < (self.K_shot + self.Q_query):
                    # If not enough samples, sample with replacement
                    selected_indices = random.choices(indices, k=(self.K_shot + self.Q_query))
                else:
                    # Sample without replacement
                    selected_indices = random.sample(indices, k=(self.K_shot + self.Q_query))
                
                # Split into support and query sets
                support_indices = selected_indices[:self.K_shot]
                query_indices = selected_indices[self.K_shot:self.K_shot + self.Q_query]
                
                # Add to our support and query sets
                for idx in support_indices:
                    img, _ = self.dataset[idx]
                    support_x.append(img)
                    support_y.append(class_idx)  # Use relative class index within episode
                
                for idx in query_indices:
                    img, _ = self.dataset[idx]
                    query_x.append(img)
                    query_y.append(class_idx)  # Use relative class index within episode
            
            # Convert to numpy arrays or tensors as needed
            yield support_x, support_y, query_x, query_y, selected_classes
    
    def __len__(self):
        return self.episodes_per_epoch

# Function to visualize images
def visualize_random(dataset, idx_to_label, num_samples=5):
    """Visualize random samples from the dataset"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i in range(num_samples):
        idx = random.randint(0, len(dataset)-1)
        img, label = dataset[idx]
        # Convert tensor to numpy for visualization
        img = img.permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)  # denormalize
        
        axes[i].imshow(img)
        axes[i].set_title(f"Class: {idx_to_label[label]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_sets(images, labels, idx_to_label, selected_classes, title):
    """Visualize support or query sets"""
    import matplotlib.pyplot as plt
    
    n_classes = len(selected_classes)
    n_samples = len(images) // n_classes
    
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(n_samples*3, n_classes*3))
    
    for i, (img, label) in enumerate(zip(images, labels)):
        row = label
        col = i % n_samples
        
        # Convert tensor to numpy for visualization
        img = img.permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)  # denormalize
        
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"Class: {idx_to_label[selected_classes[label]]}")
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# Split classes into train, validation, and test sets
all_classes = list(flower_to_idx.values())
random.seed(42)  # For reproducibility
random.shuffle(all_classes)

num_classes = len(all_classes)
train_classes = all_classes[:int(0.6 * num_classes)]
val_classes = all_classes[int(0.6 * num_classes):int(0.8 * num_classes)]
test_classes = all_classes[int(0.8 * num_classes):]

print(f"Classes split: {len(train_classes)} for training, {len(val_classes)} for validation, {len(test_classes)} for testing")

# Create datasets based on class splits
train_dataset = SubsetByClass(flower_ds, train_classes)
val_dataset = SubsetByClass(flower_ds, val_classes)
test_dataset = SubsetByClass(flower_ds, test_classes)

# Create episodic samplers
train_sampler = EpisodicSampler(train_dataset, episodes_per_epoch=100, N_way=5, K_shot=5, Q_query=5)
val_sampler = EpisodicSampler(val_dataset, episodes_per_epoch=100, N_way=5, K_shot=5, Q_query=5)
test_sampler = EpisodicSampler(test_dataset, episodes_per_epoch=100, N_way=5, K_shot=5, Q_query=5)

# Create a standard dataloader for the complete dataset if needed
flower_loader = DataLoader(flower_ds, batch_size=32, shuffle=True)