#%%
import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchdataset import SegmentationDatasetMulti, SegmentationDataset
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn, optim
from pathlib import Path
import cv2

#%%
    

# -------------------
# Transforms
# -------------------

def get_transforms():
    return A.Compose([
        A.Resize(320, 320),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

# -------------------
# Model
# -------------------

def get_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,  # grayscale
        classes=6,      # was binary
        activation=None
    )

# -------------------
# Training
# -------------------

def train(model, loader, optimizer, criterion, device):
    running_loss = 0.0
    model.train()
    for i, (images, masks) in enumerate(loader):
        print('batch', i)

        images, masks = images.to(device), masks.to(device)
        masks = masks.long()  # Convert masks to long type # needed for multi type

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader)  # mean loss

# -------------------
# Main
# -------------------


def main():
    train_dir = "sample_data/train/"

    val_dir = "sample_data/val/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using', device)

    dataset = SegmentationDatasetMulti(train_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    validation = SegmentationDatasetMulti(val_dir)
    val_loader = DataLoader(validation, batch_size=4, shuffle=True)

    model = get_model().to(device)

    # criterion = nn.BCEWithLogitsLoss()  # <-- ORIGINAL (binary)
    criterion = nn.CrossEntropyLoss()     # <-- UPDATED (multi-class)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    n_epochs = 50
    for epoch in range(n_epochs):
        print(epoch)
        loss = train(model, dataloader, optimizer, criterion, device)
        print(f"Epoch [{epoch+1}/{n_epochs}] - Loss: {loss:.4f}")
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "models/unet_multiclass_segmentation.pth")

#%%

if __name__ == "__main__":
    main()

# %%
