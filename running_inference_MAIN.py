#%%
import torch
import segmentation_models_pytorch as smp 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import tifffile as tiff
from skimage.transform import resize
import cv2
#%%
input_directory =  r'/ceph/zoo/users/debris/data/full_res_all_tiles/data'
output_directory = r'/ceph/zoo/users/debris/data/full_res_all_tiles/mask_predict'

def get_transforms():
    return A.Compose([
        A.Resize(384, 512),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
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

## Model setup
def get_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1, 
        classes=1,      
        activation=None
    )
model = get_model()
model.to(DEVICE)


checkpoint_name = 'debris_segmentation_50.pth'
model.load_state_dict(torch.load('models/' +checkpoint_name))


image_names = os.listdir(input_directory)
transform = get_transforms()
#%%

with torch.no_grad():
    for name in image_names:
        image_path = os.path.join(input_directory, name)
        image = tiff.imread(image_path)
        image = downsample_image(image)
        image = image.astype("float32") / 255.0
        transformed = transform(image=image)
        tensor_image = transformed['image'].unsqueeze(0).to(DEVICE)  
        output = model(tensor_image)
        predicted_mask = (torch.sigmoid(output) > 0.5).float()
        predict = predicted_mask[0][0].cpu()
        up_pred = upsample_mask(predict)
        tiff.imwrite(os.path.join(output_directory, name), up_pred.astype("float32"))

# %%
