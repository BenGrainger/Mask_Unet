#%%
import torch
import segmentation_models_pytorch as smp 
import albumentations as A
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import os
import tifffile as tiff
from skimage.transform import resize
import cv2
from torch import nn
import torchvision.models as models
import numpy as np
from PIL import Image
#%%


def get_transforms_seg():
    return A.Compose([
        A.Resize(384, 512),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

def get_transforms_binary():
    return transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats, modify as needed
    ])

def downsample_image(image):
    image_resized = resize(
        image, 
        (image.shape[0] // 8, image.shape[1] // 8), 
        anti_aliasing=True,
        preserve_range=True  # <- ✅ Keeps original value range
        ).astype(image.dtype)  # Convert back to original type
    return(image_resized)

def upsample_mask(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()  

    upsampled_mask = cv2.resize(mask, (mask.shape[1]*8, mask.shape[0]*8), interpolation=cv2.INTER_NEAREST)
    return upsampled_mask

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## seg models setup
def get_model_seg():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1, 
        classes=1,      
        activation=None
    )
seg_model = get_model_seg()
seg_model.to(DEVICE)

seg_checkpoint_name = 'debris_segmentation_50.pth'
seg_model.load_state_dict(torch.load('models/' + seg_checkpoint_name))

seg_transform = get_transforms_seg()

## bin model setup

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
    
bin_model = DebrisClassifier()
bin_model.to(DEVICE)

bin_checkpoint_name = 'models/binary/best_debris_classifier.pth'  
bin_model.load_state_dict(torch.load(bin_checkpoint_name))

bin_transform = get_transforms_binary()
bin_model.eval()


#%%

def stack_infernce(input_directory, output_directory):
    out_mask_dir = os.path.join(output_directory, 'masks')
    check_folder_exists(out_mask_dir)
    out_data_dir = os.path.join(output_directory, 'data')
    check_folder_exists(out_data_dir)
    image_names = sorted([f for f in os.listdir(input_directory) if f.endswith(('jpg', 'jpeg', 'png', 'tiff', 'tif'))])


    for name in image_names:
        out_mask_path = os.path.join(out_mask_dir, name)
        out_data_path = os.path.join(out_data_dir, name)
        if os.path.isfile(out_mask_path) or os.path.isfile(out_data_path) == False:
            image_path = os.path.join(input_directory, name)
            full_image = tiff.imread(image_path)
            down_image = downsample_image(full_image)
            

            down_image_rgb = down_image.astype(np.uint8)  # Ensure uint8 type
            pil_image = Image.fromarray(down_image_rgb).convert("RGB")  # Convert to PIL and ensure RGB mode
            debris_nodebris_image = bin_transform(pil_image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = bin_model(debris_nodebris_image)
                debris_nodebris = (output.squeeze() > 0.5).cpu().numpy()  
                if debris_nodebris == 0:
                    zero_image = np.ones(full_image.shape)
                    tiff.imwrite(out_mask_path, zero_image.astype("float32"))
                    tiff.imwrite(out_data_path, full_image.astype("float32"))
                    print('saving file', name)
                elif debris_nodebris == 1:
                    image = down_image.astype("float32") / 255.0
                    transformed = seg_transform(image=image)
                    tensor_image = transformed['image'].unsqueeze(0).to(DEVICE)  
                    output = seg_model(tensor_image)
                    predicted_mask = (torch.sigmoid(output) > 0.5).float()
                    predict = predicted_mask[0][0].cpu()
                    up_pred = upsample_mask(predict)
                    tiff.imwrite(out_mask_path, up_pred.astype("float32"))
                    tiff.imwrite(out_data_path, full_image.astype("float32"))
                    print('saving file', name)

# utility function
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


# %%

def main(input_loc, output_loc):
    check_folder_exists(output_loc)
    tile_list = sorted(os.listdir(input_loc))
    for tile in tile_list:
        input_directory = os.path.join(input_loc, tile)
        check_folder_exists(output_loc)
        stack_infernce(input_directory, output_loc)

# %%

input_loc =  r'/ceph/zoo/users/debris/data/full_res/data/'
output_loc = r'/ceph/zoo/users/debris/data/full_res/predict_mask/'

main(input_loc, output_loc)
# %%
