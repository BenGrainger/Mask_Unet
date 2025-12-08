from skimage.transform import resize
import cv2
import torch
import os
import tifffile as tiff
import numpy as np
from PIL import Image

def downsample_image(image, factor):
    image_resized = resize(
        image, 
        (image.shape[0] // factor, image.shape[1] // factor), 
        anti_aliasing=True,
        preserve_range=True 
        ).astype(image.dtype)  
    return(image_resized)

def upsample_mask(mask, factor):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()  

    upsampled_mask = cv2.resize(mask, (mask.shape[1]*factor, mask.shape[0]*factor), interpolation=cv2.INTER_NEAREST)
    return upsampled_mask

def inference_segmentation(model, input_directory, transform, DEVICE, output_directory):
    image_names = os.listdir(input_directory)
    with torch.no_grad():
        for name in image_names:
            image_path = os.path.join(input_directory, name)
            image = tiff.imread(image_path)
            image = downsample_image(image, 8)
            image = image.astype("float32") / 255.0
            transformed = transform(image=image)
            tensor_image = transformed['image'].unsqueeze(0).to(DEVICE)  
            output = model(tensor_image)
            predicted_mask = (torch.sigmoid(output) > 0.5).float()
            predict = predicted_mask[0][0].cpu()
            up_pred = upsample_mask(predict, 8)
            tiff.imwrite(os.path.join(output_directory, name), up_pred.astype("float32"))


def inference_classifier_segmentation(seg_model, bin_model, input_directory, bin_transform, seg_transform, DEVICE, output_directory):
    image_names = sorted([f for f in os.listdir(input_directory) if f.endswith(('jpg', 'jpeg', 'png', 'tiff', 'tif'))])


    for name in image_names:
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
                tiff.imwrite(os.path.join(output_directory, name), zero_image.astype("float32"))
            elif debris_nodebris == 1:
                image = down_image.astype("float32") / 255.0
                transformed = seg_transform(image=image)
                tensor_image = transformed['image'].unsqueeze(0).to(DEVICE)  
                output = seg_model(tensor_image)
                predicted_mask = (torch.sigmoid(output) > 0.5).float()
                predict = predicted_mask[0][0].cpu()
                up_pred = upsample_mask(predict)
                tiff.imwrite(os.path.join(output_directory, name), up_pred.astype("float32"))