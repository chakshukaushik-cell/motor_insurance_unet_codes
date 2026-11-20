import csv
import json
import random
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("/home/caiu0184/Downloads/Car_Parts_seg_model")

DATA_DIR = ROOT / "fine_tune_41"
IMAGE_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"

MAPPING_PATH = ROOT / "unet_dataset" / "class_mapping.json"

BASE_CHECKPOINT = (
    ROOT / "runs" / "car_parts_unet_new" / "best_model.pth"
)

OUTPUT_DIR = ROOT / "runs" / "fine_tune_41"

BEST_PATH = OUTPUT_DIR / "best_model.pth"
LAST_PATH = OUTPUT_DIR / "last_model.pth"

IMAGE_SIZE = 512
NUM_CLASSES = 59
IGNORE_INDEX = 255

BATCH_SIZE = 1
MAX_EPOCHS = 40

ENCODER_LR = 0.00001
DECODER_LR = 0.00003

WEIGHT_DECAY = 0.0001

EARLY_STOPPING_PATIENCE = 8
MIN_DELTA = 0.003

VALIDATION_VEHICLE = "6GJ201084405"

SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp"
}

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET DISCOVERY
# ============================================================

def collect_samples():

    image_files = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    mask_files = {
        p.stem: p
        for p in MASK_DIR.glob("*.png")
    }

    samples = []
    missing_masks = []

    for image_path in image_files:

        mask_path = mask_files.get(image_path.stem)

        if mask_path is None:
            missing_masks.append(image_path.name)
            continue

        vehicle_id = image_path.stem.split("_")[0]

        samples.append({
            "image": image_path,
            "mask": mask_path,
            "vehicle": vehicle_id
        })

    if missing_masks:
        raise RuntimeError(
            "Images without matching masks:\n" +
            "\n".join(missing_masks)
        )

    if len(samples) != 41:
        raise RuntimeError(
            f"Expected exactly 41 matched samples, "
            f"found {len(samples)}."
        )

    return samples


def create_vehicle_split(samples):

    vehicle_counts = Counter(
        item["vehicle"] for item in samples
    )

    print("\nVehicle-wise image counts:")

    for vehicle, count in sorted(vehicle_counts.items()):
        print(f"{vehicle}: {count}")

    if VALIDATION_VEHICLE not in vehicle_counts:
        raise ValueError(
            f"Validation vehicle not found: {VALIDATION_VEHICLE}"
        )

    if vehicle_counts[VALIDATION_VEHICLE] != 8:
        raise ValueError(
            "Validation vehicle must have exactly 8 images."
        )

    train_samples = [
        item for item in samples
        if item["vehicle"] != VALIDATION_VEHICLE
    ]

    val_samples = [
        item for item in samples
        if item["vehicle"] == VALIDATION_VEHICLE
    ]

    train_vehicles = {
        item["vehicle"] for item in train_samples
    }

    val_vehicles = {
        item["vehicle"] for item in val_samples
    }

    if train_vehicles.intersection(val_vehicles):
        raise RuntimeError("Vehicle leakage detected.")

    if len(train_samples) != 33 or len(val_samples) != 8:
        raise RuntimeError(
            f"Unexpected split: train={len(train_samples)}, "
            f"validation={len(val_samples)}"
        )

    print("\nDataset split:")
    print("Training images:", len(train_samples))
    print("Validation images:", len(val_samples))
    print("Validation vehicle:", VALIDATION_VEHICLE)
    print("Vehicle leakage: NONE")

    return train_samples, val_samples


# ============================================================
# MAPPING VALIDATION
# ============================================================

def load_mapping():

    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)

    classes = mapping["classes"]

    if len(classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, got {len(classes)}"
        )

    ids = [item["id"] for item in classes]

    if sorted(ids) != list(range(NUM_CLASSES)):
        raise ValueError("Class IDs are not contiguous 0-58.")

    if mapping["ignore_index"] != IGNORE_INDEX:
        raise ValueError("Ignore index mismatch.")

    return mapping


# ============================================================
# IMAGE AND MASK PREPROCESSING
# ============================================================

def letterbox(image, mask):

    image = image.convert("RGB")
    mask = mask.convert("L")

    image = ImageOps.contain(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        method=Image.Resampling.BILINEAR
    )

    mask = ImageOps.contain(
        mask,
        (IMAGE_SIZE, IMAGE_SIZE),
        method=Image.Resampling.NEAREST
    )

    canvas_image = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        (114, 114, 114)
    )

    canvas_mask = Image.new(
        "L",
        (IMAGE_SIZE, IMAGE_SIZE),
        IGNORE_INDEX
    )

    x = (IMAGE_SIZE - image.width) // 2
    y = (IMAGE_SIZE - image.height) // 2

    canvas_image.paste(image, (x, y))
    canvas_mask.paste(mask, (x, y))

    return canvas_image, canvas_mask


class CarPartsDataset(Dataset):

    def __init__(self, samples):

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        item = self.samples[index]

        image = Image.open(item["image"])
        mask = Image.open(item["mask"])

        image, mask = letterbox(image, mask)

        image = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        image = (image - MEAN) / STD

        image = torch.from_numpy(
            image.transpose(2, 0, 1).copy()
        ).float()

        mask = np.asarray(
            mask,
            dtype=np.uint8
        ).copy()

        mask = torch.from_numpy(mask).long()

        return image, mask


# ============================================================
# MASK VALIDATION
# ============================================================

def validate_masks(samples):

    print("\nValidating masks...")

    errors = []
    total_valid_pixels = 0
    total_ignore_pixels = 0

    for item in samples:

        mask = np.asarray(
            Image.open(item["mask"]).convert("L")
        )

        invalid = (
            (mask != IGNORE_INDEX) &
            (mask >= NUM_CLASSES)
        )

        if invalid.any():
            errors.append(item["mask"].name)

        total_valid_pixels += int(
            ((mask != IGNORE_INDEX) &
             (mask < NUM_CLASSES)).sum()
        )

        total_ignore_pixels += int(
            (mask == IGNORE_INDEX).sum()
        )

    if errors:
        raise ValueError(
            "Invalid class IDs in masks:\n" +
            "\n".join(errors)
        )

    if total_valid_pixels == 0:
        raise ValueError("No valid labelled pixels found.")

    print("Mask validation: PASSED")
    print("Valid labelled pixels:", total_valid_pixels)
    print("Ignored pixels:", total_ignore_pixels)


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(samples):

    counts = np.zeros(
        NUM_CLASSES,
        dtype=np.float64
    )

    for item in samples:

        mask = np.asarray(
            Image.open(item["mask"]).convert("L")
        )

        valid = mask[
            (mask != IGNORE_INDEX) &
            (mask < NUM_CLASSES)
        ]

        counts += np.bincount(
            valid.astype(np.int64),
            minlength=NUM_CLASSES
        )

    weights = np.ones(
        NUM_CLASSES,
        dtype=np.float32
    )

    present = counts > 0

    if present.any():

        weights[present] = (
            counts[present].sum() /
            counts[present]
        ) ** 0.5

        weights[present] /= weights[present].mean()

    weights = np.clip(weights, 0.25, 5.0)

    print(
        "Classes present in training masks:",
        int(present.sum())
    )

    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# CONFUSION MATRIX AND METRICS
# ============================================================

def update_confusion(confusion, target, prediction):

    target = target.reshape(-1)
    prediction = prediction.reshape(-1)

    valid = (
        (target != IGNORE_INDEX) &
        (target >= 0) &
        (target < NUM_CLASSES)
    )

    target = target[valid]
    prediction = prediction[valid]

    values = target * NUM_CLASSES + prediction

    batch_confusion = torch.bincount(
        values,
        minlength=NUM_CLASSES * NUM_CLASSES
    ).reshape(NUM_CLASSES, NUM_CLASSES)

    confusion += batch_confusion


def calculate_metrics(confusion):

    confusion = confusion.double()

    tp = torch.diag(confusion)

    actual = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)

    union = actual + predicted - tp

    iou = tp / union.clamp_min(1)

    present_foreground = union[1:] > 0

    if present_foreground.any():
        miou = iou[1:][present_foreground].mean().item()
    else:
        miou = 0.0

    pixel_accuracy = (
        tp.sum() / confusion.sum().clamp_min(1)
    ).item()

    return {
        "miou": miou,
        "pixel_accuracy": pixel_accuracy,
        "iou": iou.cpu().numpy(),
        "support": actual.cpu().numpy()
    }


# ============================================================
# TRAIN / VALIDATION EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None
):

    training = optimizer is not None

    if training:
        model.train()

        # Keep encoder BatchNorm running statistics stable
        # for this small dataset and batch size 1.
        model.encoder.eval()

    else:
        model.eval()

    confusion = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=torch.int64
    )

    total_loss = 0.0
    total_batches = 0

    for images, masks in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            outputs = model(images)

            loss = criterion(outputs, masks)

            if training:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

        predictions = outputs.argmax(dim=1)

        update_confusion(
            confusion,
            masks.detach().cpu(),
            predictions.detach().cpu()
        )

        total_loss += loss.item()
        total_batches += 1

    metrics = calculate_metrics(confusion)

    metrics["loss"] = total_loss / max(total_batches, 1)

    return metrics, confusion


# ============================================================
# CHECKPOINT SAVE
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_miou,
    mapping,
    original_checkpoint,
    extra=None
):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_miou": best_miou,
        "num_classes": NUM_CLASSES,
        "class_mapping": original_checkpoint["class_mapping"],
        "class_names": original_checkpoint["class_names"],
        "class_colors": original_checkpoint["class_colors"],
        "image_size": IMAGE_SIZE,
        "encoder": "resnet34",
        "ignore_index": IGNORE_INDEX
    }

    if extra is not None:
        checkpoint["fine_tuning"] = extra

    torch.save(checkpoint, path)


# ============================================================
# CSV REPORTS
# ============================================================

def save_history(history):

    path = OUTPUT_DIR / "epoch_metrics.csv"

    if not history:
        return

    with open(path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(history[0].keys())
        )

        writer.writeheader()
        writer.writerows(history)


def save_per_class_iou(metrics, mapping):

    path = OUTPUT_DIR / "validation_per_class_iou.csv"

    class_info = {
        item["id"]: item
        for item in mapping["classes"]
    }

    with open(path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "class_id",
            "class_name",
            "hex_color",
            "iou",
            "support_pixels"
        ])

        for class_id in range(NUM_CLASSES):

            item = class_info[class_id]

            writer.writerow([
                class_id,
                item["name"],
                item["color"],
                float(metrics["iou"][class_id]),
                int(metrics["support"][class_id])
            ])


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(IMAGE_DIR)

    if not MASK_DIR.exists():
        raise FileNotFoundError(MASK_DIR)

    if not BASE_CHECKPOINT.exists():
        raise FileNotFoundError(BASE_CHECKPOINT)

    if not MAPPING_PATH.exists():
        raise FileNotFoundError(MAPPING_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if BEST_PATH.exists() or LAST_PATH.exists():
        raise FileExistsError(
            "Fine-tuning checkpoints already exist in:\n"
            f"{OUTPUT_DIR}\n"
            "Choose a new output folder before running again."
        )

    print("\n========== DEVICE ==========")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    # Workaround for the existing GTX 1050 Ti environment.
    torch.backends.cudnn.enabled = False

    mapping = load_mapping()

    original_checkpoint = torch.load(
        BASE_CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    if original_checkpoint["num_classes"] != NUM_CLASSES:
        raise ValueError("Checkpoint class count mismatch.")

    if original_checkpoint["encoder"] != "resnet34":
        raise ValueError("Checkpoint encoder mismatch.")

    if original_checkpoint["image_size"] != IMAGE_SIZE:
        raise ValueError("Checkpoint image size mismatch.")

    if original_checkpoint["ignore_index"] != IGNORE_INDEX:
        raise ValueError("Checkpoint ignore index mismatch.")

    if len(original_checkpoint["class_names"]) != NUM_CLASSES:
        raise ValueError("Checkpoint class names mismatch.")

    mapping_names = [
        item["name"] for item in mapping["classes"]
    ]

    if mapping_names != original_checkpoint["class_names"]:
        raise ValueError(
            "Mapping class names do not match checkpoint."
        )

    print("\nCheckpoint validation: PASSED")
    print("Original checkpoint epoch:", original_checkpoint["epoch"])
    print(
        "Original checkpoint mIoU:",
        f"{original_checkpoint['best_miou'] * 100:.2f}%"
    )
    print("Classes:", NUM_CLASSES)

    samples = collect_samples()

    train_samples, val_samples = create_vehicle_split(samples)

    validate_masks(samples)

    train_dataset = CarPartsDataset(train_samples)
    val_dataset = CarPartsDataset(val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\n========== LOADING EXISTING U-NET ==========")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES
    )

    model.load_state_dict(
        original_checkpoint["model_state_dict"],
        strict=True
    )

    model = model.to(device)

    print("Existing checkpoint weights loaded successfully.")

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_samples
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=IGNORE_INDEX
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": ENCODER_LR
            },
            {
                "params": model.decoder.parameters(),
                "lr": DECODER_LR
            },
            {
                "params": model.segmentation_head.parameters(),
                "lr": DECODER_LR
            }
        ],
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold=MIN_DELTA,
        threshold_mode="abs",
        min_lr=1e-7
    )

    # --------------------------------------------------------
    # BASELINE VALIDATION
    # --------------------------------------------------------

    print("\n========== BASELINE VALIDATION ==========")

    baseline_metrics, _ = run_epoch(
        model,
        val_loader,
        criterion,
        device,
        optimizer=None
    )

    baseline_miou = baseline_metrics["miou"]

    print(
        "Original checkpoint validation mIoU:",
        f"{baseline_miou * 100:.2f}%"
    )

    print(
        "Original checkpoint validation loss:",
        f"{baseline_metrics['loss']:.4f}"
    )

    # Keep original checkpoint as best until a meaningful
    # validation improvement is achieved.
    shutil.copy2(BASE_CHECKPOINT, BEST_PATH)

    best_miou = baseline_miou
    patience_best = baseline_miou
    bad_epochs = 0

    history = []

    print("\n========== FINE-TUNING SETTINGS ==========")
    print("Maximum epochs:", MAX_EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Encoder LR:", ENCODER_LR)
    print("Decoder/head LR:", DECODER_LR)
    print("Early stopping patience:", EARLY_STOPPING_PATIENCE)
    print("Minimum meaningful mIoU improvement:", MIN_DELTA)
    print("Best checkpoint initially preserved from original model.")

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(1, MAX_EPOCHS + 1):

        print(f"\n========== EPOCH {epoch}/{MAX_EPOCHS} ==========")

        train_metrics, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer
        )

        val_metrics, _ = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None
        )

        val_miou = val_metrics["miou"]

        scheduler.step(val_miou)

        current_encoder_lr = optimizer.param_groups[0]["lr"]
        current_decoder_lr = optimizer.param_groups[1]["lr"]

        print(
            f"Train Loss: {train_metrics['loss']:.4f}"
        )

        print(
            f"Train Pixel Accuracy: "
            f"{train_metrics['pixel_accuracy'] * 100:.2f}%"
        )

        print(
            f"Train mIoU: {train_metrics['miou'] * 100:.2f}%"
        )

        print(
            f"Validation Loss: {val_metrics['loss']:.4f}"
        )

        print(
            f"Validation Pixel Accuracy: "
            f"{val_metrics['pixel_accuracy'] * 100:.2f}%"
        )

        print(
            f"Validation mIoU: {val_miou * 100:.2f}%"
        )

        print(
            f"Encoder LR: {current_encoder_lr:.7f} | "
            f"Decoder LR: {current_decoder_lr:.7f}"
        )

        # ----------------------------------------------------
        # SAVE ONLY MEANINGFULLY IMPROVED BEST CHECKPOINT
        # ----------------------------------------------------

        meaningful_improvement = (
            val_miou > best_miou + MIN_DELTA
        )

        if meaningful_improvement:

            best_miou = val_miou

            save_checkpoint(
                BEST_PATH,
                model,
                optimizer,
                epoch,
                best_miou,
                mapping,
                original_checkpoint,
                extra={
                    "baseline_miou": baseline_miou,
                    "validation_vehicle": VALIDATION_VEHICLE,
                    "train_images": len(train_samples),
                    "validation_images": len(val_samples),
                    "min_delta": MIN_DELTA
                }
            )

            print(
                "\nBEST CHECKPOINT UPDATED:",
                BEST_PATH
            )

            print(
                f"Meaningful improvement: "
                f"{best_miou * 100:.2f}%"
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if val_miou > patience_best + MIN_DELTA:

            patience_best = val_miou
            bad_epochs = 0

            print("Meaningful validation improvement detected.")

        else:

            bad_epochs += 1

            print(
                f"No meaningful improvement: "
                f"{bad_epochs}/{EARLY_STOPPING_PATIENCE}"
            )

        # ----------------------------------------------------
        # SAVE LAST CHECKPOINT
        # ----------------------------------------------------

        save_checkpoint(
            LAST_PATH,
            model,
            optimizer,
            epoch,
            best_miou,
            mapping,
            original_checkpoint,
            extra={
                "baseline_miou": baseline_miou,
                "validation_vehicle": VALIDATION_VEHICLE,
                "train_images": len(train_samples),
                "validation_images": len(val_samples),
                "min_delta": MIN_DELTA
            }
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_pixel_accuracy": train_metrics["pixel_accuracy"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "train_miou": train_metrics["miou"],
            "val_miou": val_miou,
            "best_miou": best_miou,
            "encoder_lr": current_encoder_lr,
            "decoder_lr": current_decoder_lr,
            "early_stopping_counter": bad_epochs
        })

        save_history(history)

        if bad_epochs >= EARLY_STOPPING_PATIENCE:

            print("\n========== EARLY STOPPING ==========")
            print(
                "No meaningful validation improvement for",
                EARLY_STOPPING_PATIENCE,
                "consecutive epochs."
            )
            break

    # --------------------------------------------------------
    # FINAL BEST CHECKPOINT EVALUATION
    # --------------------------------------------------------

    print("\n========== FINAL BEST MODEL EVALUATION ==========")

    best_checkpoint = torch.load(
        BEST_PATH,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        best_checkpoint["model_state_dict"],
        strict=True
    )

    model.to(device)

    final_metrics, final_confusion = run_epoch(
        model,
        val_loader,
        criterion,
        device,
        optimizer=None
    )

    save_per_class_iou(final_metrics, mapping)

    print(
        "Baseline validation mIoU:",
        f"{baseline_miou * 100:.2f}%"
    )

    print(
        "Best checkpoint validation mIoU:",
        f"{final_metrics['miou'] * 100:.2f}%"
    )

    print(
        "Best checkpoint validation pixel accuracy:",
        f"{final_metrics['pixel_accuracy'] * 100:.2f}%"
    )

    print("\n========== OUTPUT FILES ==========")
    print("Best checkpoint:", BEST_PATH)
    print("Last checkpoint:", LAST_PATH)
    print("Epoch metrics:", OUTPUT_DIR / "epoch_metrics.csv")
    print(
        "Per-class IoU:",
        OUTPUT_DIR / "validation_per_class_iou.csv"
    )

    print("\nFine-tuning completed.")


if __name__ == "__main__":
    main()

