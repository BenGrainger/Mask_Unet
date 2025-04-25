#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchdataset import SegmentationDataset
import segmentation_models_pytorch as smp 
import torchmetrics
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import pandas as pd
#%%

def check_folder_exists(directory):

    if not os.path.exists(directory):

        try:
            # Create the folder if it doesn't exist
            os.makedirs(directory)
            print(f"Folder '{directory}' created successfully.")

        except OSError as e:
            print(f"Error creating folder '{directory}': {e}")
    else:
        print(f"Folder '{directory}' already exists.")

def get_transforms():
    return A.Compose([
        A.Resize(384, 512),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

test_ds = SegmentationDataset(path_name='debris_data/val', transforms=get_transforms())
test_dataloader = DataLoader(test_ds, batch_size=4, shuffle=True)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## Model setup
def get_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,  # grayscale
        classes=1,      # was binary
        activation=None
    )
model = get_model()
model.to(DEVICE)
#%%
## load weights
checkpoint_name = 'debris_segmentation_50.pth'
model.load_state_dict(torch.load('models/' +checkpoint_name))

out_dir= r'/ceph/zoo/users/debris/Mask_training_script_Unet/outputs/' + checkpoint_name
check_folder_exists(out_dir)

## model evaluation
pixel_accuracies = []
intersection_over_unions = []
metric_iou = torchmetrics.JaccardIndex(task='binary').to(DEVICE)

#%%
with torch.no_grad():
    for inputs, targets in test_dataloader:
        inputs = inputs.to(DEVICE).float()
        targets = targets.to(DEVICE).float()

        outputs = model(inputs)
        predicted = (torch.sigmoid(outputs) > 0.5).float()

        correct = (predicted == targets).sum().item()
        total = torch.numel(targets)
        pixel_accuracies.append(correct / total)

        iou = metric_iou(predicted, targets.int())
        intersection_over_unions.append(iou.item())
pixel_accuracy = np.median(pixel_accuracies) * 100
iou_scores = np.median(intersection_over_unions) * 100
print(f"Median Pixel Accuracy: {np.median(pixel_accuracies) * 100:.2f}%")
print(f"Median IoU: {np.median(intersection_over_unions) * 100:.2f}%")

scores = {
    "score_type": ["IOU", "pixel_acc"],
    "values": [np.mean(np.array(iou_scores)), np.mean(np.array(pixel_accuracy))]
}

df = pd.DataFrame(scores)
df.to_csv(out_dir + '/score.csv', index=False)
#%%
# --------------------
# Visualization
# --------------------
check_folder_exists(out_dir + "/images")

with torch.no_grad():
    for i, (image_test, mask) in enumerate(test_dataloader):
        image_test = image_test.float().to(DEVICE)
        output = model(image_test)
        predicted_mask = (torch.sigmoid(output) > 0.5).float()

        fig, axs = plt.subplots(1, 3, figsize=(10, 10))
        axs[0].imshow(image_test[0][0].cpu(), cmap='gray')
        axs[1].imshow(mask[0][0].cpu(), cmap='gray')
        axs[2].imshow(predicted_mask[0][0].cpu(), cmap='gray')
        axs[0].set_title("image")
        axs[1].set_title("True Mask")
        axs[2].set_title("Predicted Mask")

        filename = f"output{i:04d}.png"
        
        plt.savefig(os.path.join(out_dir + "/images", filename), bbox_inches='tight')


# %%
