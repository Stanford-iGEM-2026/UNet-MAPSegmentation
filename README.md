# Keloid scar segmentation with U-Net++

This project trains a U-Net++ model to outline keloid scars in clinical skin photographs.

The model architecture comes from the [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) library. This repository is the training project: the dataset, training script, and results.

## Model

- Architecture: U-Net++ (`UnetPlusPlus`)
- Encoder: ResNet34, started from ImageNet weights
- Task: binary segmentation (scar vs background)
- Input: RGB images resized to 480×640

## Data

30 image–mask pairs:

- `Images/scars/` — RGB photographs (`001.jpg`–`030.jpg`)
- `Images/masks/` — matching masks (`001.png`–`030.png`), where white is scar and black is background

The training script uses a fixed 80/20 split (seed 42): 24 training images and 6 validation images.

## Results

Best checkpoint: epoch 7, saved as `outputs/scar_training/best_model.pth`.

| Set | Mean Dice | Mean IoU |
| --- | --- | --- |
| Validation (6 unseen images) | 0.829 | 0.717 |
| Training (24 images) | 0.903 | 0.826 |

Validation Dice on individual images ranged from 0.722 to 0.935. Training was run for 50 epochs, and validation scores peaked early, so the saved model is the epoch-7 checkpoint rather than the final epoch.

Plots and prediction panels are in `outputs/scar_training/`.

## Train

```bash
pip install -r requirements.txt
python train_scars.py
```

## Predict

```bash
python predict_scars.py
```

This loads `outputs/scar_training/best_model.pth` and writes a panel for every image.
