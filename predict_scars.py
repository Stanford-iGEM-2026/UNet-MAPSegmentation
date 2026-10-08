"""
Generate scar segmentation predictions using best_model.pth.
Saves per-image panels and summary grids for train and val sets.
"""

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_scars import INPUT_SIZE, IMAGES_DIR, MASKS_DIR, OUTPUT_DIR, ScarDataset, SegmentationModel, get_device


def load_model(checkpoint_path: Path, device: str) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = checkpoint.get("arch", "UnetPlusPlus")
    encoder_name = checkpoint.get("encoder_name", "resnet34")

    model = SegmentationModel(arch, encoder_name, in_channels=3, out_classes=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_mask(model, image_tensor, device):
    with torch.inference_mode():
        logits = model(image_tensor.unsqueeze(0).to(device))
        prob = logits.sigmoid().squeeze().cpu().numpy()
    return prob


def compute_image_metrics(pred_mask, gt_mask):
    pred = (pred_mask > 0.5).astype(np.uint8)
    gt = gt_mask.astype(np.uint8)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / union if union > 0 else 1.0
    dice = (2 * intersection) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 1.0
    return iou, dice


def save_triplet(image_rgb, gt_mask, pred_prob, out_path, title, iou, dice):
    pred_mask = (pred_prob > 0.5).astype(np.float32)
    overlay = image_rgb.copy()
    overlay[pred_mask > 0] = (
        0.45 * overlay[pred_mask > 0] + 0.55 * np.array([255, 0, 0])
    ).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Image")
    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_prob, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Prediction")
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    fig.suptitle(f"{title} | IoU={iou:.3f} | Dice={dice:.3f}")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_grid(dataset, stems, model, device, out_path, title_prefix):
    n = len(stems)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    metrics = []
    for row, stem in enumerate(stems):
        idx = dataset.stems.index(stem)
        image_t, mask_t = dataset[idx]
        pred_prob = predict_mask(model, image_t, device)
        gt = mask_t.numpy()
        iou, dice = compute_image_metrics(pred_prob, gt)
        metrics.append({"stem": stem, "iou": iou, "dice": dice})

        image_rgb = (image_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pred_mask = (pred_prob > 0.5).astype(np.float32)
        overlay = image_rgb.copy()
        overlay[pred_mask > 0] = (
            0.45 * overlay[pred_mask > 0] + 0.55 * np.array([255, 0, 0])
        ).astype(np.uint8)

        axes[row, 0].imshow(image_rgb)
        axes[row, 0].set_title(f"{stem} | Image")
        axes[row, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground Truth")
        axes[row, 2].imshow(pred_prob, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title("Prediction")
        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title(f"Overlay | IoU={iou:.2f}")
        for col in range(4):
            axes[row, col].axis("off")

    fig.suptitle(title_prefix, fontsize=14, y=1.002)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return metrics


def main():
    device = get_device()
    checkpoint_path = OUTPUT_DIR / "best_model.pth"
    pred_dir = OUTPUT_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads((OUTPUT_DIR / "split.json").read_text())
    train_stems = split["train"]
    val_stems = split["val"]
    all_stems = sorted(train_stems + val_stems, key=lambda x: int(x))

    model = load_model(checkpoint_path, device)
    dataset = ScarDataset(all_stems, IMAGES_DIR, MASKS_DIR, INPUT_SIZE, augmentation=None)

    all_metrics = {"val": [], "train": []}

    for split_name, stems in [("val", val_stems), ("train", train_stems)]:
        split_dir = pred_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            idx = dataset.stems.index(stem)
            image_t, mask_t = dataset[idx]
            pred_prob = predict_mask(model, image_t, device)
            gt = mask_t.numpy()
            iou, dice = compute_image_metrics(pred_prob, gt)
            image_rgb = (image_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            save_triplet(
                image_rgb,
                gt,
                pred_prob,
                split_dir / f"{stem}.png",
                f"{split_name.upper()} {stem}",
                iou,
                dice,
            )
            all_metrics[split_name].append({"stem": stem, "iou": iou, "dice": dice})

    val_metrics = save_grid(
        dataset,
        sorted(val_stems, key=lambda x: int(x)),
        model,
        device,
        OUTPUT_DIR / "all_val_predictions_best.png",
        "Validation predictions (best_model.pth)",
    )
    train_metrics = save_grid(
        dataset,
        sorted(train_stems, key=lambda x: int(x)),
        model,
        device,
        OUTPUT_DIR / "all_train_predictions_best.png",
        "Training predictions (best_model.pth)",
    )

    all_metrics["val"] = val_metrics
    all_metrics["train"] = train_metrics
    (pred_dir / "per_image_metrics.json").write_text(json.dumps(all_metrics, indent=2))

    val_iou = np.mean([m["iou"] for m in val_metrics])
    val_dice = np.mean([m["dice"] for m in val_metrics])
    train_iou = np.mean([m["iou"] for m in train_metrics])
    train_dice = np.mean([m["dice"] for m in train_metrics])

    print(f"Saved predictions to: {pred_dir}")
    print(f"Val  mean IoU={val_iou:.3f} Dice={val_dice:.3f} ({len(val_stems)} images)")
    print(f"Train mean IoU={train_iou:.3f} Dice={train_dice:.3f} ({len(train_stems)} images)")
    print(f"Grids: {OUTPUT_DIR / 'all_val_predictions_best.png'}")
    print(f"       {OUTPUT_DIR / 'all_train_predictions_best.png'}")


if __name__ == "__main__":
    main()
