import random
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# For evaluation of few shot learning in task 3
# Create subsets of examples for specific subsets of classes
class SubsetByClass(Dataset):
    def __init__(self, base_dataset, allowed_classes):
        """
        Parameters:
          base_dataset (Dataset): The original dataset.
          allowed_classes (iterable): List or set of allowed class labels.
        """
        self.base_dataset = base_dataset
        self.allowed_classes = set(allowed_classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(allowed_classes)}

        # Filter indices that belong to allowed classes
        self.filtered_indices = [
            idx for idx in range(len(base_dataset))
            if int(base_dataset[idx][1]) in self.allowed_classes
        ]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, i):
        if isinstance(i, torch.Tensor):
            i = i.item()
        
        real_idx = self.filtered_indices[i]
        img, label = self.base_dataset[real_idx]
        label = int(label)  # Ensure label is an integer
        mapped_label = self.class_to_idx[label]
        return img, mapped_label

    def get_images_for_class(self, cls):
        return [
            i for i, idx in enumerate(self.filtered_indices)
            if int(self.base_dataset[idx][1]) == cls
        ]

    @property
    def classes(self):
        return list(self.allowed_classes)


# An episodic data loader that yields multiple episodes per batch.
class EpisodicSampler:
    def __init__(self, dataset, episodes_per_epoch, N_way, K_shot, Q_query):
        """
        Parameters:
          dataset (Dataset): The dataset to sample from (should have a 'classes' property).
          episodes_per_epoch (int): Number of episodes per epoch.
          N_way (int): Number of classes per episode.
          K_shot (int): Number of support examples per class.
          Q_query (int): Number of query examples per class.
        """
        self.dataset = dataset
        self.episodes_per_epoch = episodes_per_epoch
        self.N_way = N_way
        self.K_shot = K_shot
        self.Q_query = Q_query

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            yield self.sample_episode(self.dataset, self.N_way, self.K_shot, self.Q_query)

    def __len__(self):
        return self.episodes_per_epoch

    # Creates support and query sets for one episode.
    def sample_episode(self, dataset, N_way, K_shot, Q_query):
        # Randomly sample N classes
        selected_classes = random.sample(dataset.classes, N_way)

        support_x, support_y = [], []
        query_x, query_y = [], []

        # Get K images of each class for support set and Q images for query set
        for new_label, cls in enumerate(selected_classes):
            indices = dataset.get_images_for_class(cls)
            sampled_indices = random.sample(indices, K_shot + Q_query)

            support_idx = sampled_indices[:K_shot]
            query_idx = sampled_indices[K_shot:]

            for idx in support_idx:
                img, _ = dataset[idx]
                support_x.append(img)
                support_y.append(new_label)

            for idx in query_idx:
                img, _ = dataset[idx]
                query_x.append(img)
                query_y.append(new_label)

        support_x = torch.stack(support_x)
        support_y = torch.tensor(support_y)
        query_x = torch.stack(query_x)
        query_y = torch.tensor(query_y)

        return support_x, support_y, query_x, query_y, selected_classes


def visualize_sets(x, y, idx_to_label, selected_classes, title=""):
    """
    Visualizes a grid of images with labels.
    
    Parameters:
      x (Tensor): Batch of images.
      y (Tensor): Batch of labels.
      idx_to_label (dict): Mapping from label indices to label names.
      selected_classes (list): The classes selected for the episode.
      title (str): Title for the plot.
    """
    fig, axes = plt.subplots(5, 5, figsize=(10, 10), layout='constrained')
    fig.suptitle(title, fontsize=20)
    for i, (xi, yi) in enumerate(zip(x, y)):
        # Denormalize assuming normalization with mean=0.5 and std=0.5
        np_img = (xi * 0.5 + 0.5).permute(1, 2, 0).numpy()
        # Determine which subplot to use
        ax = axes[yi][i % 5]
        ax.imshow(np_img)
        # Get a short label for display
        label_text = idx_to_label.get(selected_classes[yi.item()], "N/A").split('(')[0][:20]
        ax.set_xlabel(label_text, fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
