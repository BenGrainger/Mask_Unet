import os
import sys
# Add the project root (parent of this file's directory) to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("PROJECT_ROOT added to sys.path:", PROJECT_ROOT)

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from mask_unet.inference import inference_segmentation
from mask_unet.models import get_segmentation_model

input_directory =  r'/ceph/zoo/users/debris/data/full_res_all_tiles/data'
output_directory = r'/ceph/zoo/users/debris/data/full_res_all_tiles/mask_predict'

def get_transforms():
    return A.Compose([
        A.Resize(384, 512),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = get_segmentation_model()
model.to(DEVICE)


checkpoint_name = 'debris_segmentation_50.pth'
model.load_state_dict(torch.load('models/' +checkpoint_name))


image_names = os.listdir(input_directory)
transform = get_transforms()


inference_segmentation(model, input_directory, transform, DEVICE, output_directory)


