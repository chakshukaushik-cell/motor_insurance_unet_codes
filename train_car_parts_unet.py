
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp


# ============================================================
# CONFIGURATION
# ============================================================

IGNORE_INDEX = 255

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

IMAGENET_STD = np.array(
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
# LOAD ACTUAL CLASS MAPPING FORMAT
# ============================================================

def load_class_mapping(dataset_root):

    mapping_path = dataset_root / "class_mapping.json"

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Class mapping not found: {mapping_path}"
        )

    with open(mapping_path, "r") as f:
        data = json.load(f)

    if "classes" not in data:
        raise ValueError(
            "Invalid class mapping: classes list missing."
        )

    classes = data["classes"]

    if not isinstance(classes, list) or not classes:
        raise ValueError(
            "Classes list is empty or invalid."
        )

    mapping = {}
    class_names_by_id = {}
    colors_by_id = {}

    for item in classes:

        class_id = int(item["id"])
        class_name = str(item["name"])

        if class_id in class_names_by_id:
            raise ValueError(
                f"Duplicate class ID: {class_id}"
            )

        if class_name in mapping:
            raise ValueError(
                f"Duplicate class name: {class_name}"
            )

        mapping[class_name] = class_id
        class_names_by_id[class_id] = class_name

        colors_by_id[class_id] = item.get(
            "color",
            "#808080"
        )

    expected_ids = list(range(len(classes)))
    actual_ids = sorted(class_names_by_id.keys())

    if actual_ids != expected_ids:
        raise ValueError(
            f"Class IDs are not continuous: {actual_ids}"
        )

    class_names = [
        class_names_by_id[i]
        for i in expected_ids
    ]

    num_classes = len(class_names)

    background_id = int(
        data.get("mask_background_id", 0)
    )

    ignore_index = int(
        data.get("ignore_index", IGNORE_INDEX)
    )

    if background_id != 0:
        raise ValueError(
            "This training script expects background ID 0."
        )

    if ignore_index != IGNORE_INDEX:
        raise ValueError(
            f"Expected ignore index {IGNORE_INDEX}, "
            f"but mapping specifies {ignore_index}."
        )

    print("\n========== CLASS MAPPING ==========")
    print("Format:", data.get("format", "unknown"))
    print("Number of classes:", num_classes)
    print("Background ID:", background_id)
    print("Ignore index:", ignore_index)
    print("First classes:", class_names[:5])
    print("Last classes:", class_names[-5:])

    return mapping, class_names, colors_by_id, num_classes


# ============================================================
# IMAGE AND MASK LETTERBOX
# ============================================================

def letterbox(image, mask, image_size):

    image = image.convert("RGB")
    mask = mask.convert("L")

    image = ImageOps.contain(
        image,
        (image_size, image_size),
        method=Image.Resampling.BILINEAR
    )

    mask = ImageOps.contain(
        mask,
        (image_size, image_size),
        method=Image.Resampling.NEAREST
    )

    image_canvas = Image.new(
        "RGB",
        (image_size, image_size),
        (114, 114, 114)
    )

    mask_canvas = Image.new(
        "L",
        (image_size, image_size),
        IGNORE_INDEX
    )

    x = (image_size - image.width) // 2
    y = (image_size - image.height) // 2

    image_canvas.paste(image, (x, y))
    mask_canvas.paste(mask, (x, y))

    return image_canvas, mask_canvas


# ============================================================
# DATASET
# ============================================================

class CarPartsDataset(Dataset):

    def __init__(self, root, split, image_size):

        self.image_dir = root / "images" / split
        self.mask_dir = root / "masks" / split

        self.image_size = image_size

        if not self.image_dir.exists():
            raise FileNotFoundError(self.image_dir)

        if not self.mask_dir.exists():
            raise FileNotFoundError(self.mask_dir)

        image_paths = sorted(
            self.image_dir.glob("*.png")
        )

        self.samples = []

        missing_masks = []

        for image_path in image_paths:

            mask_path = self.mask_dir / image_path.name

            if mask_path.exists():

                self.samples.append(
                    (image_path, mask_path)
                )

            else:

                missing_masks.append(image_path.name)

        if missing_masks:

            raise RuntimeError(
                f"{len(missing_masks)} images have missing masks "
                f"in {split}. Example: {missing_masks[:5]}"
            )

        if not self.samples:

            raise RuntimeError(
                f"No image-mask pairs found in {split}"
            )

        print(
            f"{split}: {len(self.samples)} image-mask pairs"
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, mask_path = self.samples[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        mask = Image.open(
            mask_path
        ).convert("L")

        image, mask = letterbox(
            image,
            mask,
            self.image_size
        )

        image = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        image = (
            image - IMAGENET_MEAN
        ) / IMAGENET_STD

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

def validate_masks(dataset, num_classes):

    print("\nValidating masks...")

    errors = []

    for image_path, mask_path in dataset.samples:

        mask = np.asarray(
            Image.open(mask_path).convert("L")
        )

        invalid = np.unique(
            mask[
                (mask != IGNORE_INDEX) &
                (mask >= num_classes)
            ]
        )

        if len(invalid) > 0:

            errors.append(
                f"{mask_path.name}: {invalid.tolist()}"
            )

    if errors:

        print("\nInvalid masks found:")

        for error in errors[:20]:
            print(error)

        raise ValueError(
            f"Found {len(errors)} masks with invalid class IDs."
        )

    print("Mask validation successful.")


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(dataset, num_classes):

    print("\nCalculating class weights...")

    counts = np.zeros(
        num_classes,
        dtype=np.float64
    )

    for _, mask_path in dataset.samples:

        mask = np.asarray(
            Image.open(mask_path).convert("L")
        )

        valid = mask[
            (mask != IGNORE_INDEX) &
            (mask >= 0) &
            (mask < num_classes)
        ]

        counts += np.bincount(
            valid.astype(np.int64),
            minlength=num_classes
        )

    weights = np.ones(
        num_classes,
        dtype=np.float32
    )

    present = counts > 0

    if present.any():

        weights[present] = (
            counts[present].sum() /
            counts[present]
        ) ** 0.5

        weights[present] /= weights[present].mean()

    weights = np.clip(
        weights,
        0.25,
        5.0
    )

    print("Class weights calculated.")
    print("Classes with pixels:", int(present.sum()))

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def update_confusion(
    confusion,
    target,
    prediction,
    num_classes
):

    target = target.reshape(-1)
    prediction = prediction.reshape(-1)

    valid = (
        (target != IGNORE_INDEX) &
        (target >= 0) &
        (target < num_classes)
    )

    target = target[valid]
    prediction = prediction[valid]

    values = (
        target * num_classes + prediction
    )

    batch_confusion = torch.bincount(
        values,
        minlength=num_classes * num_classes
    ).reshape(
        num_classes,
        num_classes
    )

    confusion += batch_confusion


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(confusion):

    confusion = confusion.double()

    tp = torch.diag(confusion)

    actual = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)

    union = actual + predicted - tp

    iou = tp / union.clamp_min(1)

    precision = (
        tp / predicted.clamp_min(1)
    )

    recall = (
        tp / actual.clamp_min(1)
    )

    # Exclude background and absent classes from mIoU
    present_foreground = union[1:] > 0

    if present_foreground.any():

        miou = iou[1:][present_foreground].mean().item()

    else:

        miou = 0.0

    pixel_accuracy = (
        tp.sum() /
        confusion.sum().clamp_min(1)
    ).item()

    return {
        "miou": miou,
        "pixel_accuracy": pixel_accuracy,
        "iou": iou.cpu().numpy(),
        "precision": precision.cpu().numpy(),
        "recall": recall.cpu().numpy(),
        "support": actual.cpu().numpy()
    }


# ============================================================
# TRAIN / VALIDATION / TEST EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    device,
    num_classes,
    optimizer=None
):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    confusion = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64
    )

    for batch_index, (images, masks) in enumerate(loader):

        images = images.to(device)
        masks = masks.to(device)

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):

            outputs = model(images)

            loss = criterion(
                outputs,
                masks
            )

            if training:

                loss.backward()
                optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        update_confusion(
            confusion,
            masks.detach().cpu(),
            predictions.detach().cpu(),
            num_classes
        )

        if training and (batch_index + 1) % 25 == 0:

            print(
                f"Train batch {batch_index + 1}/{len(loader)}"
                f" | loss={loss.item():.4f}"
            )

    metrics = calculate_metrics(confusion)

    average_loss = (
        total_loss / max(len(loader), 1)
    )

    return average_loss, metrics, confusion


# ============================================================
# TRAINING CURVES
# ============================================================

def plot_training_history(history, output_dir):

    if not history:
        return

    epochs = [
        item["epoch"]
        for item in history
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10)
    )

    axes[0, 0].plot(
        epochs,
        [x["train_loss"] for x in history],
        label="Train Loss"
    )

    axes[0, 0].plot(
        epochs,
        [x["val_loss"] for x in history],
        label="Validation Loss"
    )

    axes[0, 0].set_title("Training and Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(
        epochs,
        [x["train_pixel_accuracy"] * 100 for x in history],
        label="Train Pixel Accuracy"
    )

    axes[0, 1].plot(
        epochs,
        [x["val_pixel_accuracy"] * 100 for x in history],
        label="Validation Pixel Accuracy"
    )

    axes[0, 1].set_title("Pixel Accuracy (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(
        epochs,
        [x["train_miou"] * 100 for x in history],
        label="Train mIoU"
    )

    axes[1, 0].plot(
        epochs,
        [x["val_miou"] * 100 for x in history],
        label="Validation mIoU"
    )

    axes[1, 0].set_title("mIoU (%)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(
        epochs,
        [x["learning_rate"] for x in history],
        label="Learning Rate"
    )

    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir / "training_curves.png",
        dpi=180
    )

    plt.close()


# ============================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================

def plot_confusion_matrix(
    confusion,
    class_names,
    output_dir
):

    confusion = confusion.astype(np.float64)

    normalized = confusion / np.maximum(
        confusion.sum(axis=1, keepdims=True),
        1
    )

    matrices = [
        (
            confusion,
            "confusion_matrix.png",
            "Confusion Matrix - Raw"
        ),
        (
            normalized,
            "confusion_matrix_normalized.png",
            "Confusion Matrix - Normalized"
        )
    ]

    for matrix, filename, title in matrices:

        fig, ax = plt.subplots(
            figsize=(22, 19)
        )

        image = ax.imshow(
            matrix,
            interpolation="nearest",
            aspect="auto",
            cmap="viridis"
        )

        ax.set_title(title)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")

        ax.set_xticks(
            range(len(class_names))
        )

        ax.set_yticks(
            range(len(class_names))
        )

        ax.set_xticklabels(
            class_names,
            rotation=90,
            fontsize=7
        )

        ax.set_yticklabels(
            class_names,
            fontsize=7
        )

        fig.colorbar(
            image,
            ax=ax
        )

        plt.tight_layout()

        plt.savefig(
            output_dir / filename,
            dpi=200
        )

        plt.close()

        print("Saved:", output_dir / filename)


# ============================================================
# PER-CLASS METRICS CSV
# ============================================================

def save_class_metrics(
    metrics,
    class_names,
    output_dir
):

    csv_path = (
        output_dir / "per_class_metrics.csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Class",
            "IoU",
            "Precision",
            "Recall",
            "Support"
        ])

        for i, name in enumerate(class_names):

            writer.writerow([
                name,
                float(metrics["iou"][i]),
                float(metrics["precision"][i]),
                float(metrics["recall"][i]),
                int(metrics["support"][i])
            ])

    print("Saved:", csv_path)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=60
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=512
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.001
    )

    args = parser.parse_args()

    if args.patience < 1:
        raise ValueError("Patience must be at least 1.")

    if args.min_delta < 0:
        raise ValueError("min-delta cannot be negative.")

    set_seed(args.seed)

    dataset_root = Path(args.dataset)

    output_path = Path(args.output)

    output_dir = output_path.parent

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if output_path.exists():

        raise FileExistsError(
            f"\nCheckpoint already exists: {output_path}\n"
            "Please back it up or choose another output filename. "
            "The existing checkpoint will not be overwritten."
        )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if torch.cuda.is_available():

        # GTX 1050 Ti cuDNN engine compatibility workaround
        torch.backends.cudnn.enabled = False

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")

    print("\n========== SYSTEM CONFIGURATION ==========")

    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print("PyTorch:", torch.__version__)
    print("SMP:", smp.__version__)

    print(
        "cuDNN enabled:",
        torch.backends.cudnn.enabled
    )

    # --------------------------------------------------------
    # CLASS MAPPING
    # --------------------------------------------------------

    mapping, class_names, colors_by_id, num_classes = (
        load_class_mapping(dataset_root)
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = CarPartsDataset(
        dataset_root,
        "train",
        args.image_size
    )

    val_dataset = CarPartsDataset(
        dataset_root,
        "val",
        args.image_size
    )

    test_dataset = CarPartsDataset(
        dataset_root,
        "test",
        args.image_size
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validate_masks(
        train_dataset,
        num_classes
    )

    validate_masks(
        val_dataset,
        num_classes
    )

    validate_masks(
        test_dataset,
        num_classes
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda")
    )

    # --------------------------------------------------------
    # CLASS WEIGHTS AND LOSS
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_dataset,
        num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=IGNORE_INDEX
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\n========== MODEL CONFIGURATION ==========")

    print("Architecture: U-Net")
    print("Encoder: ResNet34")
    print("Encoder weights: ImageNet pretrained")
    print("Number of classes:", num_classes)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes
    )

    model = model.to(device)

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0001
    )

    # --------------------------------------------------------
    # LEARNING RATE SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        threshold=0.001,
        threshold_mode="abs",
        min_lr=1e-7
    )

    # --------------------------------------------------------
    # EARLY STOPPING VARIABLES
    # --------------------------------------------------------

    best_miou = -1.0
    patience_best = -1.0

    bad_epochs = 0

    history = []

    print("\n========== TRAINING SETTINGS ==========")

    print("Maximum epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Image size:", args.image_size)
    print("Learning rate:", args.lr)

    print("Early stopping monitor: Validation mIoU")
    print("Early stopping patience:", args.patience)
    print("Minimum improvement:", args.min_delta)

    print("\n========== TRAINING STARTED ==========")

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(1, args.epochs + 1):

        print(
            f"\n========== EPOCH {epoch}/{args.epochs} =========="
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_metrics, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            num_classes,
            optimizer
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss, val_metrics, _ = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            num_classes
        )

        val_miou = val_metrics["miou"]

        # ----------------------------------------------------
        # LR SCHEDULER
        # ----------------------------------------------------

        scheduler.step(val_miou)

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print("\n---------- EPOCH RESULTS ----------")

        print(
            f"Train Loss:           {train_loss:.4f}"
        )

        print(
            f"Train Pixel Accuracy: "
            f"{train_metrics['pixel_accuracy'] * 100:.2f}%"
        )

        print(
            f"Train mIoU:           "
            f"{train_metrics['miou'] * 100:.2f}%"
        )

        print(
            f"Validation Loss:      {val_loss:.4f}"
        )

        print(
            f"Validation Pixel Acc: "
            f"{val_metrics['pixel_accuracy'] * 100:.2f}%"
        )

        print(
            f"Validation mIoU:      "
            f"{val_miou * 100:.2f}%"
        )

        print(
            f"Learning Rate:        {current_lr:.7f}"
        )

        # ----------------------------------------------------
        # SAVE BEST CHECKPOINT
        # ----------------------------------------------------

        if val_miou > best_miou:

            best_miou = val_miou

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_miou": best_miou,
                "num_classes": num_classes,
                "class_mapping": mapping,
                "class_names": class_names,
                "class_colors": colors_by_id,
                "image_size": args.image_size,
                "encoder": "resnet34",
                "ignore_index": IGNORE_INDEX
            }

            torch.save(
                checkpoint,
                output_path
            )

            print(
                "\nSaved best model:",
                output_path
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if val_miou > patience_best + args.min_delta:

            patience_best = val_miou

            bad_epochs = 0

            print(
                "Early stopping: meaningful validation "
                "mIoU improvement detected."
            )

        else:

            bad_epochs += 1

            print(
                f"Early stopping counter: "
                f"{bad_epochs}/{args.patience}"
            )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_pixel_accuracy":
                train_metrics["pixel_accuracy"],
            "val_pixel_accuracy":
                val_metrics["pixel_accuracy"],
            "train_miou":
                train_metrics["miou"],
            "val_miou":
                val_miou,
            "learning_rate":
                current_lr,
            "best_miou":
                best_miou,
            "early_stopping_counter":
                bad_epochs
        }

        history.append(epoch_record)

        with open(
            output_dir / "training_history.json",
            "w"
        ) as f:

            json.dump(
                history,
                f,
                indent=4
            )

        plot_training_history(
            history,
            output_dir
        )

        # ----------------------------------------------------
        # STOP CONDITION
        # ----------------------------------------------------

        if bad_epochs >= args.patience:

            print("\n========== EARLY STOPPING ==========")

            print(
                f"No meaningful validation mIoU improvement "
                f"for {args.patience} consecutive epochs."
            )

            print(
                f"Best Validation mIoU: "
                f"{best_miou * 100:.2f}%"
            )

            print("Training stopped automatically.")

            break

    # ========================================================
    # TRAINING FINISHED
    # ========================================================

    print("\n========== TRAINING FINISHED ==========")

    print(
        f"Best Validation mIoU: "
        f"{best_miou * 100:.2f}%"
    )

    # ========================================================
    # LOAD BEST CHECKPOINT
    # ========================================================

    print("\nLoading best checkpoint...")

    checkpoint = torch.load(
        output_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Best checkpoint epoch:",
        checkpoint["epoch"]
    )

    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    print("\nRunning final test evaluation...")

    test_loss, test_metrics, test_confusion = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        num_classes
    )

    print("\n========== FINAL TEST RESULTS ==========")

    print(
        f"Test Loss: {test_loss:.4f}"
    )

    print(
        f"Test Pixel Accuracy: "
        f"{test_metrics['pixel_accuracy'] * 100:.2f}%"
    )

    print(
        f"Test mIoU: "
        f"{test_metrics['miou'] * 100:.2f}%"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    confusion_np = test_confusion.numpy()

    plot_confusion_matrix(
        confusion_np,
        class_names,
        output_dir
    )

    # ========================================================
    # PER CLASS METRICS
    # ========================================================

    save_class_metrics(
        test_metrics,
        class_names,
        output_dir
    )

    # ========================================================
    # SAVE TEST RESULTS
    # ========================================================

    test_results = {
        "test_loss": test_loss,
        "test_pixel_accuracy":
            test_metrics["pixel_accuracy"],
        "test_miou":
            test_metrics["miou"],
        "best_validation_miou":
            best_miou,
        "best_checkpoint_epoch":
            checkpoint["epoch"],
        "confusion_matrix":
            confusion_np.tolist(),
        "class_names":
            class_names
    }

    with open(
        output_dir / "test_evaluation.json",
        "w"
    ) as f:

        json.dump(
            test_results,
            f,
            indent=4
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n========== ALL OUTPUTS SAVED ==========")

    print("Best model:", output_path)

    print(
        "Training history:",
        output_dir / "training_history.json"
    )

    print(
        "Training curves:",
        output_dir / "training_curves.png"
    )

    print(
        "Confusion matrix:",
        output_dir / "confusion_matrix.png"
    )

    print(
        "Normalized confusion matrix:",
        output_dir / "confusion_matrix_normalized.png"
    )

    print(
        "Per-class metrics:",
        output_dir / "per_class_metrics.csv"
    )

    print(
        "Test evaluation:",
        output_dir / "test_evaluation.json"
    )

    print("\nTraining pipeline completed successfully.")


if __name__ == "__main__":
    main()
