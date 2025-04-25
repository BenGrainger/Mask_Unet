
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import os

# ---------------------
# CONFIG
# ---------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/unet_multiclass_segmentation.pth"
IMAGES_PATH = "sample_data/test/images"
OUTPUT_PATH = "outputs/multicalss"
INPUT_SIZE = (320, 320)

# ---------------------
# TRANSFORMS
# ---------------------

def get_transforms():
    return Compose([
        Resize(*INPUT_SIZE),
        Normalize(mean=(0.5,), std=(0.5,)),  # adjust for RGB if needed
        ToTensorV2()
    ])

# ---------------------
# LOAD MODEL
# ---------------------

def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,  # change to 3 if RGB
        classes=6,
        activation=None  # must be None if using CrossEntropyLoss
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ---------------------
# PREDICT
# ---------------------

def predict(model, image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0
    image = np.expand_dims(image, axis=-1)  # (H, W, 1)
    transforms = get_transforms()
    transformed = transforms(image=image)
    input_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)  # shape: (1, 6, H, W)
        preds = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    return preds

# ---------------------
# MAIN
# ---------------------

def main():
    model = load_model()
    for image in os.listdir(IMAGES_PATH):
        mask = predict(model, os.path.join(image, IMAGES_PATH))
        cv2.imwrite(OUTPUT_PATH, mask * 40)  # scale for visibility if label ids are 0–5
        print(f"Saved predicted mask to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()