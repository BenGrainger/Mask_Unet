import os
import sys

# Add the project root (parent of this file's directory) to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("PROJECT_ROOT added to sys.path:", PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader
from mask_unet.torchdataset import SegmentationDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn, optim
from mask_unet.models import get_segmentation_model
from mask_unet.training.train_segmentor import train_segmentator


def get_transforms():
    return A.Compose([
        A.Resize(384, 512),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

train_dir = "debris_data/train/"
val_dir = "debris_data/val/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using', DEVICE)

dataset = SegmentationDataset(train_dir, transforms=get_transforms())
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
validation = SegmentationDataset(val_dir, transforms=get_transforms())
val_loader = DataLoader(validation, batch_size=4, shuffle=True)

model = get_segmentation_model().to(device)

criterion = nn.BCEWithLogitsLoss()  # <-- ORIGINAL (binary)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 5

model_save_path = "models/best_debris_segmentation_50.pth"

train_segmentator(model, optimizer, train_loader, val_loader, criterion, DEVICE, num_epochs, model_save_path)

torch.save(model.state_dict(), "models/final_debris_segmentation_50.pth")



