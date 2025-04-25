#%%
import os
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn



#%%

image_directory = '/ceph/zoo/users/debris/data/large_trial_downsample/test'  
model_path = 'models/binary/best_debris_classifier.pth'  


def get_transforms():
    return transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats, modify as needed
    ])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DebrisClassifier(nn.Module):
    def __init__(self, backbone='resnet34', pretrained=True):
        super(DebrisClassifier, self).__init__()
        if backbone == 'resnet34':
            self.encoder = models.resnet34(pretrained=pretrained)
        else:
            raise ValueError("Unsupported backbone")

        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 1)  # Binary output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = self.sigmoid(x)
        return x


model = DebrisClassifier()
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()


transform = get_transforms()


image_files = [f for f in os.listdir(image_directory) if f.endswith(('jpg', 'jpeg', 'png', 'tiff'))]

#%%

def display_model_predictions(image_dir, model, transform, device):
    plt.figure(figsize=(12, 12)) 
    
    image_files = os.listdir(image_dir)  # List all images in the directory

    for i, image_name in enumerate(image_files):
        image_path = os.path.join(image_dir, image_name)
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            prediction = (output.squeeze() > 0.5).cpu().numpy()  # Binary prediction

        # Create a subplot to display the image and its prediction
        plt.subplot(5, 5, i + 1)  # 5x5 grid for 25 images
        plt.imshow(image)
        plt.title(f"{'Debris' if prediction == 1 else 'No Debris'}")
        plt.axis('off')

        if i == 24: 
            break  # Stop after displaying 25 images

    plt.tight_layout()
    plt.show()


# %%


# Assuming the transformation and image directory are already set up
display_model_predictions(image_directory, model, transform, device)
# %%
