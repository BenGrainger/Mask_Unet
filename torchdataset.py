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
    

class SegmentationDatasetMulti(Dataset):
    """Create Semantic Segmentation Dataset. Read images, apply augmentations, and process transformations

    Args:
        Dataset (image): Aerial Drone Images
    """
    # CLASSES = {'building': 44, 'land': 91, 'road':172, 'vegetation':212, 'water':171, 'unlabeled':155}
    # CLASSES_KEYS = list(CLASSES.keys())
    
    def __init__(self, path_name) -> None:
        super().__init__()
        self.image_names = os.listdir(f"{path_name}/images")
        self.image_paths = [f"{path_name}/images/{i}" for i in self.image_names]
        self.masks_names = os.listdir(f"{path_name}/masks")
        self.masks_paths = [f"{path_name}/masks/{i}" for i in self.masks_names]
        
        # filter all images that do not exist in both folders
        self.img_stem = [Path(i).stem for i in self.image_paths]
        self.msk_stem = [Path(i).stem for i in self.masks_paths]
        self.img_msk_stem = set(self.img_stem) & set(self.msk_stem)
        self.image_paths = [i for i in self.image_paths if (Path(i).stem in self.img_msk_stem)]


    def convert_mask(self, mask):
        mask[mask == 155] = 0  # unlabeled
        mask[mask == 44] = 1  # building
        mask[mask == 91] = 2  # land
        mask[mask == 171] = 3  # water
        mask[mask == 172] = 4  # road
        mask[mask == 212] = 5  # vegetation
        return mask   

    def __len__(self):
        return len(self.img_msk_stem)
    
    # def __getitem__(self, index):
    #     image = cv2.imread(self.image_paths[index])
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     image = image.transpose((2, 0, 1))  #structure: BS, C, H, W
    #     mask =  cv2.imread(self.masks_paths[index], 0)
    #     mask = self.convert_mask(mask)
    #     return image, mask
    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0  # normalize + cast
        image = image.transpose((2, 0, 1))  # (3, H, W)

        mask = cv2.imread(self.masks_paths[index], 0)
        mask = self.convert_mask(mask)
        mask = mask.astype(np.int64)

        return torch.tensor(image), torch.tensor(mask)