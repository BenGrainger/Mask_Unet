import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import glob
import tifffile as tiff

class SegmentationDataset(Dataset):
    def __init__(self, path_name, transforms=None):
        super().__init__()
        
        # Get all image and mask paths
        image_paths_all = glob.glob(f"{path_name}/images/*")
        mask_paths_all = glob.glob(f"{path_name}/masks/*")

        # Build stem: path dict
        image_dict = {Path(p).stem: p for p in image_paths_all}
        mask_dict = {Path(p).stem: p for p in mask_paths_all}

        # Intersect on stem
        common_stems = set(image_dict.keys()) & set(mask_dict.keys())

        self.image_paths = [image_dict[stem] for stem in common_stems]
        self.masks_paths = [mask_dict[stem] for stem in common_stems]
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = tiff.imread(self.image_paths[idx])
        mask = tiff.imread(self.masks_paths[idx])

        if image is None or mask is None:
            raise ValueError(f"Failed to read image or mask at index {idx}: {self.image_paths[idx]}, {self.masks_paths[idx]}")

        image = image.astype("float32") / 255.0
        mask = (mask > 0).astype("float32")

        image = np.expand_dims(image, axis=-1)
        mask = np.expand_dims(mask, axis=-1)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        mask = np.transpose(mask, (2, 0, 1))  # [1, H, W]

        return image, mask
    

class DebrisDataset(Dataset):
    def __init__(self, debris_dir, no_debris_dir, transform=None):
        """
        Args:
            debris_dir (str): Path to the directory with debris images.
            no_debris_dir (str): Path to the directory with no debris images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.debris_dir = debris_dir
        self.no_debris_dir = no_debris_dir
        self.transform = transform

        # List of image filenames in each directory
        self.debris_images = os.listdir(debris_dir)
        self.no_debris_images = os.listdir(no_debris_dir)

        # Combine the image names and their corresponding labels
        self.image_paths = []
        self.labels = []
        
        # Add "debris" images
        for img_name in self.debris_images:
            self.image_paths.append(os.path.join(debris_dir, img_name))
            self.labels.append(1)  # Label for "debris" images is 1
        
        # Add "no_debris" images
        for img_name in self.no_debris_images:
            self.image_paths.append(os.path.join(no_debris_dir, img_name))
            self.labels.append(0)  # Label for "no_debris" images is 0

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB format

        # Get the corresponding label
        label = self.labels[idx]

        # Apply the transformation (if any)
        if self.transform:
            image = self.transform(image)

        return image, label
    
