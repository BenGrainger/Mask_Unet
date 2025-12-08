# Mask_Unet

Two-stage pipeline for automated debris detection and segmentation in electron microscopy tiles:
1) binary classifier (debris vs no debris)
2) U-Net segmentation model for debris masks.

## Overview

- Detects tiles containing debris using a CNN-based classifier.
- Segments debris pixels using a ResNet34 U-Net.
- Designed for large-scale EM datasets (trained on HPC), with small demo data and notebooks for local exploration.

## Features

- Binary debris classifier (`DebrisClassifier`).
- U-Net-based segmentation model (`get_segmentation_model`).
- Modular training loops for classifier and segmentor.
- Generic inference pipeline (classifier-only, segmentation-only, or full pipeline).
- Jupyter notebooks for qualitative evaluation. 


## Repository Structure

Mask_Unet/
  mask_unet/
    __init__.py
    torchdataset.py          # DebrisDataset, SegmentationDataset
    models.py                # DebrisClassifier, get_segmentation_model
    inference.py             # Core inference helpers (classifier, segmentor, full pipeline)
    training/
      __init__.py
      train_classifier.py          # train_classifier, evaluation for binary classifier
      train_segmentor.py           # train_segmentator, evaluation for segmentation model

  scripts/
    train_binary_classifier.py
    train_segmentation.py
    run_inference_binary_classifier_segmentation.py        
    run_inference_segmentation.py  

  notebooks/
    evaluate_classifier.ipynb
    evaluate_segmentation.ipynb

  debris_data/               # small demo dataset (not full training data)
    ...

  models/                    # trained weights (not tracked in git; add your own)
    binary/
    segmentation/

  config/                    # (planned) YAML configs for training/inference

  Legacy/                    # old scripts actully used to train and infer models

  README.md

## Installaton 

torch
conda install -c conda-forge albumentations opencv
torchmetrics
torchvision
segmentation_models_pytorch
numpy
PIL
tifffile

## Data

The repository assumes two types of data:

1. **Classifier data** (debris vs no-debris tiles)
   - Debris tiles in one directory
   - No-debris tiles in another

2. **Segmentation data** (images + binary masks)
   - Typically organised as `<root>/train` and `<root>/val`
   - Internally assumed layout is described in `mask_unet/datasets.py`.

A small **demo dataset** is provided under `debris_data/` for running the notebooks and example scripts.

Full training data used in my experiments is **not** included in the repo (large EM datasets on the cluster). To train on your own data, update the paths in:

- `scripts/train_binary_classifier.py`
- `scripts/train_segmentation.py`
- notebooks under `notebooks/`

## TODO / Future Work

- [ ] Add YAML-based configuration for data paths and hyperparameters.
- [ ] Finalise `scripts/run_inference.py` with CLI arguments.
- [ ] Add metrics (Dice / IoU) to segmentation evaluation.
- [ ] Package as an installable Python module (`pip install -e .`).

## Author

Ben Grainger  
5th year Neuroscience PhD (Sainsbury Wellcome Centre)  
Focus: large-scale EM data, automated debris detection & cleanup. 




  


