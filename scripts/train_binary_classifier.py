import os
import sys

# Add the project root (parent of this file's directory) to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("PROJECT_ROOT added to sys.path:", PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
from mask_unet.torchdataset import DebrisDataset
from mask_unet.models import DebrisClassifier
from mask_unet.training.train_classifier import train_classifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transforms():
    return transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
    ])

criterion = nn.BCEWithLogitsLoss()

debris_dir = '/ceph/zoo/users/debris/data/masks_no_masks/debris'
no_debris_dir = '/ceph/zoo/users/debris/data/masks_no_masks/nodebris'

transform = get_transforms()

dataset = DebrisDataset(debris_dir, no_debris_dir, transform=transform)


dataset_size = len(dataset)
val_size = int(0.2 * dataset_size)  
train_size = dataset_size - val_size  


train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


model = DebrisClassifier()
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 5
best_val_loss = float('inf')  
model_save_path = 'models/binary/best_debris_classifier.pth'  


train_classifier(model, optimizer, train_loader, val_loader, criterion, DEVICE, num_epochs, model_save_path)


torch.save(model.state_dict(), 'models/binary/final_debris_classifier.pth')

