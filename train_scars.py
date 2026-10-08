"""
Train U-Net++ on scar segmentation images in Images/scars + Images/masks.
Saves model weights, metrics history, and training plots.
"""

import json
import logging
import os
import random
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset as BaseDataset
from tqdm import tqdm

import segmentation_models_pytorch as smp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d:%m:%Y %H:%M:%S",
)

# ----------------------------
# Config
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
IMAGES_DIR = PROJECT_ROOT / "Images" / "scars"
MASKS_DIR = PROJECT_ROOT / "Images" / "masks"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scar_training"

EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
ETA_MIN = 1e-5
VAL_FRACTION = 0.2
RANDOM_SEED = 42
INPUT_SIZE = (480, 640)  # (height, width), divisible by 32

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def get_device():
    return DEVICE


class ScarDataset(BaseDataset):
    def __init__(self, stems, images_dir, masks_dir, input_size, augmentation=None):
        self.stems = stems
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.input_size = input_size
        self.augmentation = augmentation

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, index):
        stem = self.stems[index]
        image = cv2.imread(str(self.images_dir / f"{stem}.jpg"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

        image = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        mask = cv2.resize(mask, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_NEAREST)

        if self.augmentation is not None:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
            image = cv2.resize(image, (self.input_size[1], self.input_size[0]))
            mask = cv2.resize(
                mask, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_NEAREST
            )

        image = torch.tensor(image).float().permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask).long()
        return image, mask


class SegmentationModel(torch.nn.Module):
    def __init__(self, arch, encoder_name, in_channels=3, out_classes=1):
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=out_classes,
        )

    def forward(self, image):
        image = (image - self.mean) / self.std
        return self.model(image)


def split_stems(stems, val_fraction, seed):
    rng = random.Random(seed)
    stems = sorted(stems)
    rng.shuffle(stems)
    val_count = max(1, int(len(stems) * val_fraction))
    val_stems = stems[:val_count]
    train_stems = stems[val_count:]
    return train_stems, val_stems


def compute_binary_metrics(outputs, masks):
    prob_mask = outputs.sigmoid().squeeze(1)
    pred_mask = (prob_mask > 0.5).long()
    tp, fp, fn, tn = smp.metrics.get_stats(pred_mask, masks, mode="binary")
    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
    dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()
    return iou, dice


def run_epoch(model, dataloader, optimizer, scheduler, loss_fn, device, train=True):
    model.train(train)
    total_loss = 0.0
    tp = fp = fn = tn = 0

    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        for images, masks in tqdm(dataloader, desc="Train" if train else "Valid", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            if train:
                loss.backward()
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()

            with torch.inference_mode():
                prob_mask = outputs.sigmoid().squeeze(1)
                pred_mask = (prob_mask > 0.5).long()
                batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                    pred_mask, masks, mode="binary"
                )
                tp += batch_tp.sum().item()
                fp += batch_fp.sum().item()
                fn += batch_fn.sum().item()
                tn += batch_tn.sum().item()

    avg_loss = total_loss / len(dataloader)
    iou = smp.metrics.iou_score(
        torch.tensor([tp]), torch.tensor([fp]), torch.tensor([fn]), torch.tensor([tn]), reduction="micro"
    ).item()
    dice = smp.metrics.f1_score(
        torch.tensor([tp]), torch.tensor([fp]), torch.tensor([fn]), torch.tensor([tn]), reduction="micro"
    ).item()
    return avg_loss, iou, dice


def plot_training_history(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"], label="Validation", marker="o", markersize=3)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Dice Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_iou"], label="Val IoU", color="green", marker="o", markersize=3)
    axes[1].set_title("Validation IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["val_dice"], label="Val Dice", color="purple", marker="o", markersize=3)
    axes[2].set_title("Validation Dice (F1)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Dice")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_metrics.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"], label="Val Loss")
    ax.plot(epochs, history["val_iou"], label="Val IoU")
    ax.plot(epochs, history["val_dice"], label="Val Dice")
    ax.set_xlabel("Epoch")
    ax.set_title("U-Net++ Scar Segmentation — All Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_summary.png", dpi=150)
    plt.close()


def save_prediction_samples(model, dataset, output_dir, device, prefix="val", count=3):
    model.eval()
    indices = list(range(min(count, len(dataset))))
    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.inference_mode():
        for row, idx in enumerate(indices):
            image, mask = dataset[idx]
            pred = model(image.unsqueeze(0).to(device)).sigmoid().squeeze().cpu().numpy()
            image_np = image.permute(1, 2, 0).numpy()
            mask_np = mask.numpy()

            axes[row, 0].imshow(image_np)
            axes[row, 0].set_title("Image")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(mask_np, cmap="gray")
            axes[row, 1].set_title("Ground Truth")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(pred, cmap="gray")
            axes[row, 2].set_title("Prediction")
            axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_predictions.png", dpi=150)
    plt.close()


def main():
    device = get_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Using device: %s", device)

    stems = [p.stem for p in sorted(IMAGES_DIR.glob("*.jpg"))]
    if not stems:
        raise FileNotFoundError(f"No .jpg images found in {IMAGES_DIR}")

    train_stems, val_stems = split_stems(stems, VAL_FRACTION, RANDOM_SEED)
    logging.info("Train: %d images | Val: %d images", len(train_stems), len(val_stems))

    train_aug = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5, border_mode=0
            ),
        ]
    )

    train_dataset = ScarDataset(train_stems, IMAGES_DIR, MASKS_DIR, INPUT_SIZE, train_aug)
    val_dataset = ScarDataset(val_stems, IMAGES_DIR, MASKS_DIR, INPUT_SIZE, None)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SegmentationModel("UnetPlusPlus", "resnet34", in_channels=3, out_classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(train_loader), eta_min=ETA_MIN
    )
    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_dice": [],
    }
    best_iou = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_iou, train_dice = run_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device, train=True
        )
        val_loss, val_iou, val_dice = run_epoch(
            model, val_loader, optimizer, scheduler, loss_fn, device, train=False
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)

        logging.info(
            "Epoch %d/%d | train loss %.4f | val loss %.4f | val IoU %.4f | val Dice %.4f",
            epoch,
            EPOCHS,
            train_loss,
            val_loss,
            val_iou,
            val_dice,
        )

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_iou": val_iou,
                    "val_dice": val_dice,
                    "arch": "UnetPlusPlus",
                    "encoder_name": "resnet34",
                },
                OUTPUT_DIR / "best_model.pth",
            )

    torch.save(model.state_dict(), OUTPUT_DIR / "last_model.pth")

    with open(OUTPUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(OUTPUT_DIR / "split.json", "w") as f:
        json.dump({"train": train_stems, "val": val_stems}, f, indent=2)

    plot_training_history(history, OUTPUT_DIR)
    save_prediction_samples(model, val_dataset, OUTPUT_DIR, device, prefix="val", count=3)

    logging.info("Best validation IoU: %.4f", best_iou)
    logging.info("Outputs saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
