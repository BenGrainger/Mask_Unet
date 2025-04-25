#%%
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
#%%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def get_transforms():
    return transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
    ])


class DebrisClassifier(nn.Module):
    def __init__(self, backbone='resnet34', pretrained=True):
        super(DebrisClassifier, self).__init__()
        if backbone == 'resnet34':
            self.encoder = models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet50':
            self.encoder = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError("Unsupported backbone")

        # Modify the final fully connected layer for binary classification
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 1)  # Binary output
        self.sigmoid = nn.Sigmoid()  # Apply sigmoid to the output

    def forward(self, x):
        x = self.encoder(x)
        x = self.sigmoid(x)  # Sigmoid to output probability for binary classification
        return x

#%%
criterion = nn.BCEWithLogitsLoss()


debris_dir = '/ceph/zoo/users/debris/data/masks_no_masks/debris'
no_debris_dir = '/ceph/zoo/users/debris/data/masks_no_masks/nodebris'

# Create the dataset
transform = get_transforms()
dataset = DebrisDataset(debris_dir, no_debris_dir, transform=transform)


dataset_size = len(dataset)
val_size = int(0.2 * dataset_size)  # 20% for validation
train_size = dataset_size - val_size  # Remaining for training


train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


model = DebrisClassifier()
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 5
best_val_loss = float('inf')  
model_save_path = 'models/binary/best_debris_classifier.pth'  

#%%
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels.float())  
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = 0
        correct_preds = 0
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            val_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).long()  # Convert to 0 or 1
            correct_preds += (predicted == labels).sum().item()

        val_accuracy = correct_preds / len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.4f}")

        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)  # Save model with best validation loss
            print(f"Saved model with validation loss: {val_loss/len(val_loader):.4f}")


torch.save(model.state_dict(), 'models/binary/final_debris_classifier.pth')

# %%
