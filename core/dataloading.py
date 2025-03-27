from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from collections import defaultdict

# Simple Image Dataset
class ImageDataset(Dataset):
    """ Helper class for loading images into a Torch dataset. """
    def __init__(self, X, y, transform=None):
        self.file_paths = X["file_path"].values
        self.labels = y.values
        self.transform = transform or transforms.ToTensor()
        
        # Few few-shot learning only
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_to_indices[label].append(idx)
        self.classes = list(self.class_to_indices.keys())

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label).long()
    
    # Utility function to get all images for a given class
    # Useful for few shot learning
    def get_images_for_class(self, class_label):
        return self.class_to_indices[class_label]