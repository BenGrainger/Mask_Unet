import os
import sys
# Add the project root (parent of this file's directory) to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("PROJECT_ROOT added to sys.path:", PROJECT_ROOT)

import torch
import albumentations as A
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import os
from mask_unet.models import get_segmentation_model, DebrisClassifier
from mask_unet.inference import inference_classifier_segmentation

input_directory =  r'/ceph/zoo/users/debris/data/full_res/data/t0000'
output_directory = r'/ceph/zoo/users/debris/data/full_res/predict_mask/t0000'

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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

seg_model = get_segmentation_model()
seg_model.to(DEVICE)

seg_checkpoint_name = 'debris_segmentation_50.pth'
seg_model.load_state_dict(torch.load('models/' + seg_checkpoint_name))

seg_transform = get_transforms_seg()


bin_model = DebrisClassifier()
bin_model.to(DEVICE)

bin_checkpoint_name = 'models/binary/best_debris_classifier.pth'  
bin_model.load_state_dict(torch.load(bin_checkpoint_name))

bin_transform = get_transforms_binary()
bin_model.eval()


inference_classifier_segmentation(seg_model, bin_model, input_directory, bin_transform, seg_transform, DEVICE, output_directory)

