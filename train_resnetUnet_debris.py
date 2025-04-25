#%%
import torch
from torch.utils.data import DataLoader
from torchdataset import SegmentationDataset
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn, optim


#%%
    

# -------------------
# Transforms
# -------------------

def get_transforms():
    return A.Compose([
        A.Resize(384, 512),
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
        in_channels=1,  # grayscale
        classes=1,      # was binary
        activation=None
    )

# -------------------
# Training
# -------------------

def train(model, loader, optimizer, criterion, device):
    running_loss = 0.0
    model.train()
    for i, (images, masks) in enumerate(loader):

        images, masks = images.to(device), masks.to(device)

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
#%%

def main():
    train_dir = "debris_data/train/"

    val_dir = "debris_data/val/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using', device)

    dataset = SegmentationDataset(train_dir, transforms=get_transforms())
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    validation = SegmentationDataset(val_dir, transforms=get_transforms())
    val_loader = DataLoader(validation, batch_size=4, shuffle=True)

    model = get_model().to(device)

    criterion = nn.BCEWithLogitsLoss()  # <-- ORIGINAL (binary)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    n_epochs = 5
    best_val_loss = float('inf') 

    for epoch in range(n_epochs):
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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "models/best_debris_segmentation_50.pth")  # Save model with best validation loss
                print(f"Saved model with validation loss: {val_loss/len(val_loader):.4f}")

    torch.save(model.state_dict(), "models/final_debris_segmentation_50.pth")

#%%

if __name__ == "__main__":
    main()

# %%
