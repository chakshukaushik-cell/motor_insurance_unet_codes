from pathlib import Path
from collections import Counter
import csv
import json
import random
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIGURATION
# ============================================================
ROOT = Path("/home/caiu0184/Downloads/New_unet_car_detection")
IMAGE_DIR = ROOT / "images"
MASK_DIR = ROOT / "masks"
INITIAL_WEIGHTS = ROOT / "unet_car_seg2.pth"
OUT = ROOT / "runs" / "finetune_70_15_15"

WIDTH, HEIGHT = 512, 288
BATCH_SIZE = 2
EPOCHS = 50
PATIENCE = 8
LEARNING_RATE = 1e-5
NUM_WORKERS = 2
SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Assumed preprocessing: ImageNet normalization.
# If the original model used different normalization, match it.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PROFILES = [
    "front_left", "front_right", "back_left", "back_right",
    "front", "back", "left", "right"
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GTX 1050 Ti workaround: disable cuDNN; use the working CUDA convolution backend.
torch.backends.cudnn.enabled = False
AMP = DEVICE.type == "cuda"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET VALIDATION
# ============================================================
def discover_pairs():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Images folder missing: {IMAGE_DIR}")
    if not MASK_DIR.is_dir():
        raise FileNotFoundError(f"Masks folder missing: {MASK_DIR}")
    if not INITIAL_WEIGHTS.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {INITIAL_WEIGHTS}")

    images = {
        p.stem.lower(): p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in EXTS
    }
    masks = {
        p.stem.lower(): p for p in MASK_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in EXTS
    }

    print("\n========== DATASET CHECK ==========")
    print("Images:", len(images))
    print("Masks:", len(masks))
    print("Images without masks:", len(images.keys() - masks.keys()))
    print("Masks without images:", len(masks.keys() - images.keys()))

    if images.keys() != masks.keys():
        print("Unmatched image names:", sorted(images.keys() - masks.keys())[:20])
        print("Unmatched mask names:", sorted(masks.keys() - images.keys())[:20])
        raise RuntimeError("Image-mask filenames must match exactly.")

    pairs = [(images[k], masks[k]) for k in sorted(images)]
    if len(pairs) < 20:
        raise RuntimeError("Too few matched pairs for reliable splitting.")

    empty_masks = []
    foreground_fractions = []

    for img_path, mask_path in pairs:
        with Image.open(img_path) as im:
            image_size = im.size
        with Image.open(mask_path) as ma:
            mask_array = np.asarray(ma.convert("L"))
            mask_size = ma.size

        if image_size != mask_size:
            raise ValueError(
                f"Size mismatch: {img_path.name}: {image_size}, "
                f"{mask_path.name}: {mask_size}"
            )

        values = set(np.unique(mask_array).tolist())
        if not values.issubset({0, 1, 255}):
            raise ValueError(
                f"Unexpected mask values in {mask_path.name}: "
                f"{sorted(values)[:20]}"
            )

        fraction = float(np.mean(mask_array > 0))
        foreground_fractions.append(fraction)
        if fraction == 0:
            empty_masks.append(mask_path.name)

    print("Matched pairs:", len(pairs))
    print("Image-mask dimensions: OK")
    print("Empty masks:", len(empty_masks))
    print("Median foreground fraction:",
          round(float(np.median(foreground_fractions)), 4))

    if empty_masks:
        print("WARNING: Empty masks found:", empty_masks[:10])
    if np.median(foreground_fractions) < 0.01:
        print("WARNING: Check whether masks correctly mark the car.")

    return pairs


# ============================================================
# VEHICLE ID AND PROFILE
# Expected filename example:
# 0ZLW05074644_back_left_side_captured_image.jpeg
# ============================================================
def get_vehicle_id(image_path):
    return image_path.stem.split("_")[0].upper()


def get_profile(image_path):
    name = image_path.stem.lower()

    for profile in ("front_left", "front_right", "back_left", "back_right"):
        if profile in name:
            return profile

    for profile in ("front", "back", "left", "right"):
        if f"_{profile}_" in f"_{name}_":
            return profile

    return "unknown"


# ============================================================
# 70 / 15 / 15 VEHICLE-WISE SPLIT
# All images belonging to one vehicle stay in the same split.
# ============================================================
def make_splits(pairs):
    groups = {}
    for pair in pairs:
        groups.setdefault(get_vehicle_id(pair[0]), []).append(pair)

    vehicle_ids = sorted(groups)
    rng = random.Random(SEED)
    rng.shuffle(vehicle_ids)

    n = len(vehicle_ids)
    if n < 10:
        raise RuntimeError(
            f"Only {n} vehicle IDs found. Check filename-based grouping."
        )

    n_train = round(n * TRAIN_RATIO)
    n_val = round(n * VAL_RATIO)
    n_test = n - n_train - n_val

    # Ensure all three splits have at least one vehicle.
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise RuntimeError("Unable to create all three grouped splits.")

    train_ids = set(vehicle_ids[:n_train])
    val_ids = set(vehicle_ids[n_train:n_train + n_val])
    test_ids = set(vehicle_ids[n_train + n_val:])

    def flatten(ids):
        return [
            pair for vid in sorted(ids)
            for pair in groups[vid]
        ]

    train = flatten(train_ids)
    val = flatten(val_ids)
    test = flatten(test_ids)

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    print("\n========== 70 / 15 / 15 SPLIT CHECK ==========")
    print(f"{'Split':<12}{'Vehicles':>12}{'Images':>12}{'Image %':>12}")
    for name, data, ids in [
        ("Train", train, train_ids),
        ("Validation", val, val_ids),
        ("Test", test, test_ids),
    ]:
        print(
            f"{name:<12}{len(ids):>12}{len(data):>12}"
            f"{100 * len(data) / len(pairs):>11.1f}%"
        )

    print("Vehicle ID overlap:",
          len((train_ids & val_ids) |
              (train_ids & test_ids) |
              (val_ids & test_ids)))

    for name, data, ids in [
        ("TRAIN", train, train_ids),
        ("VALIDATION", val, val_ids),
        ("TEST", test, test_ids),
    ]:
        counts = Counter(get_profile(img) for img, _ in data)
        print(f"{name} profiles:", dict(counts))

        missing = [p for p in PROFILES if counts[p] == 0]
        if missing:
            print(f"WARNING: {name} is missing profiles: {missing}")

        if name == "VALIDATION" and len(ids) < 5:
            print("WARNING: Validation has fewer than 5 vehicles.")
        if name == "TEST" and len(ids) < 5:
            print("WARNING: Test has fewer than 5 vehicles.")
        if name == "VALIDATION" and len(data) < 20:
            print("WARNING: Validation has fewer than 20 images.")
        if name == "TEST" and len(data) < 20:
            print("WARNING: Test has fewer than 20 images.")

    manifest = OUT / "split_manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "vehicle_id", "profile", "image", "mask"])
        for name, data in [
            ("train", train), ("validation", val), ("test", test)
        ]:
            for img, mask in data:
                writer.writerow([
                    name, get_vehicle_id(img), get_profile(img),
                    img.name, mask.name
                ])

    print("Split manifest:", manifest)
    return train, val, test


# ============================================================
# DATASET
# ============================================================
class CarDataset(Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img_path, mask_path = self.pairs[index]

        with Image.open(img_path) as im:
            image = im.convert("RGB").resize(
                (WIDTH, HEIGHT), Image.Resampling.BILINEAR
            )

        with Image.open(mask_path) as ma:
            mask = ma.convert("L").resize(
                (WIDTH, HEIGHT), Image.Resampling.NEAREST
            )

        image = np.asarray(image, dtype=np.float32) / 255.0
        mask = np.asarray(mask, dtype=np.uint8)
        mask = (mask > 0).astype(np.int64)

        if self.augment and random.random() < 0.5:
            image = image[:, ::-1, :].copy()
            mask = mask[:, ::-1].copy()

        image = (image - MEAN) / STD
        image = torch.from_numpy(image.transpose(2, 0, 1).copy())
        mask = torch.from_numpy(mask.copy()).long()

        return image, mask, img_path.name


# ============================================================
# MODEL: EXACT CHECKPOINT ARCHITECTURE
# ============================================================
def build_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=2,
        activation=None,
    )


def load_initial_model():
    model = build_model()
    state = torch.load(
        INITIAL_WEIGHTS, map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE)


# ============================================================
# EVALUATION
# Confusion matrix: rows=actual, columns=predicted
# ============================================================
@torch.no_grad()
def evaluate(model, loader, save_predictions=False, tag="after"):
    model.eval()
    cm = np.zeros((2, 2), dtype=np.int64)

    pred_dir = OUT / "test_predictions" / tag
    if save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    for images, masks, names in loader:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        with torch.autocast(device_type=DEVICE.type, enabled=AMP):
            logits = model(images)

        preds = logits.argmax(dim=1)

        actual = masks.reshape(-1).cpu().numpy()
        predicted = preds.reshape(-1).cpu().numpy()
        cm += np.bincount(
            actual * 2 + predicted, minlength=4
        ).reshape(2, 2)

        if save_predictions:
            for i, name in enumerate(names):
                pred = preds[i].byte().cpu().numpy()
                pred_image = Image.fromarray(
                    (pred * 255).astype(np.uint8)
                ).resize(
                    Image.open(IMAGE_DIR / name).size,
                    Image.Resampling.NEAREST,
                )
                stem = Path(name).stem
                pred_image.save(pred_dir / f"{stem}_mask.png")

                with Image.open(IMAGE_DIR / name) as im:
                    original = im.convert("RGB")

                overlay = np.asarray(original).copy()
                binary = np.asarray(pred_image) > 0
                overlay[binary] = (
                    0.55 * overlay[binary]
                    + 0.45 * np.array([0, 255, 0])
                ).astype(np.uint8)
                Image.fromarray(overlay).save(
                    pred_dir / f"{stem}_overlay.png"
                )

    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])

    def ratio(a, b):
        return a / b if b else 0.0

    metrics = {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": ratio(tp, tp + fp),
        "Recall": ratio(tp, tp + fn),
        "Specificity": ratio(tn, tn + fp),
        "IoU": ratio(tp, tp + fp + fn),
        "Dice": ratio(2 * tp, 2 * tp + fp + fn),
        "Pixel_Accuracy": ratio(tp + tn, int(cm.sum())),
    }
    return cm, metrics


# ============================================================
# SAVE REPORTS
# ============================================================
def save_confusion_matrix(cm, path, title):
    im = Image.new("RGB", (700, 430), "white")
    d = ImageDraw.Draw(im)
    d.text((25, 20), title, fill="black")
    d.text((255, 65), "Predicted class", fill="black")
    d.text((200, 105), "Background", fill="black")
    d.text((390, 105), "Car", fill="black")
    d.text((20, 170), "Actual", fill="black")
    d.text((35, 165), "Background", fill="black")
    d.text((35, 245), "Car", fill="black")

    cells = [
        (170, 140, 350, 220, int(cm[0, 0])),
        (350, 140, 530, 220, int(cm[0, 1])),
        (170, 220, 350, 300, int(cm[1, 0])),
        (350, 220, 530, 300, int(cm[1, 1])),
    ]
    for x1, y1, x2, y2, value in cells:
        d.rectangle((x1, y1, x2, y2), outline="black", width=2)
        d.text((x1 + 25, y1 + 30), f"{value:,}", fill="black")

    d.text((25, 350), "Rows = actual | Columns = predicted", fill="black")
    im.save(path)


def save_metrics(tag, cm, metrics):
    with open(OUT / f"metrics_{tag}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())

    with open(OUT / f"confusion_matrix_{tag}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual_predicted", "background", "car"])
        writer.writerow(["background", int(cm[0, 0]), int(cm[0, 1])])
        writer.writerow(["car", int(cm[1, 0]), int(cm[1, 1])])

    save_confusion_matrix(
        cm, OUT / f"confusion_matrix_{tag}.png",
        f"Test confusion matrix - {tag}"
    )


def print_metrics(title, metrics):
    print(f"\n========== {title} ==========")
    for key, value in metrics.items():
        if key in ("TP", "FP", "FN", "TN"):
            print(f"{key:16}: {value:,}")
        else:
            print(f"{key:16}: {value:.5f} ({value * 100:.2f}%)")


# ============================================================
# MAIN TRAINING
# ============================================================
def main():
    print("PyTorch:", torch.__version__)
    print("SMP:", smp.__version__)
    print("Device:", DEVICE)
    if AMP:
        print("GPU:", torch.cuda.get_device_name(0))

    pairs = discover_pairs()
    train_pairs, val_pairs, test_pairs = make_splits(pairs)

    train_loader = DataLoader(
        CarDataset(train_pairs, augment=True),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=AMP
    )
    val_loader = DataLoader(
        CarDataset(val_pairs),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=AMP
    )
    test_loader = DataLoader(
        CarDataset(test_pairs),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=AMP
    )

    model = load_initial_model()
    print("\nCheckpoint architecture loaded strictly.")

    cm_before, metrics_before = evaluate(model, test_loader)
    save_metrics("before", cm_before, metrics_before)
    print_metrics("BEFORE FINE-TUNING", metrics_before)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)

    best_dice = -1.0
    best_epoch = 0
    no_improve = 0
    history = []

    best_path = OUT / "best_model.pth"
    last_path = OUT / "last_model.pth"

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        model.train()
        total_loss = 0.0
        total_count = 0

        for images, masks, _ in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=DEVICE.type, enabled=AMP):
                logits = model(images)
                ce_loss = F.cross_entropy(logits, masks)

                prob = torch.softmax(logits, dim=1)[:, 1]
                target = masks.float()
                intersection = (prob * target).sum(dim=(1, 2))
                denominator = (
                    prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
                )
                dice_loss = 1.0 - (
                    (2 * intersection + 1e-6)
                    / (denominator + 1e-6)
                ).mean()
                loss = 0.5 * ce_loss + 0.5 * dice_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            total_count += images.size(0)

        train_loss = total_loss / max(total_count, 1)
        _, val_metrics = evaluate(model, val_loader)
        val_dice = val_metrics["Dice"]
        scheduler.step(val_dice)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_dice": val_dice,
            "val_iou": val_metrics["IoU"],
            "val_precision": val_metrics["Precision"],
            "val_recall": val_metrics["Recall"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - start,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02}/{EPOCHS} | "
            f"loss={train_loss:.5f} | "
            f"val Dice={val_dice:.5f} | "
            f"val IoU={val_metrics['IoU']:.5f} | "
            f"lr={row['learning_rate']:.2e} | "
            f"time={row['seconds']:.1f}s",
            flush=True,
        )

        torch.save(model.state_dict(), last_path)

        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            print("  Best checkpoint saved.")
        else:
            no_improve += 1

        with open(OUT / "training_history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        if no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break

    best_model = build_model().to(DEVICE)
    best_model.load_state_dict(
        torch.load(best_path, map_location="cpu", weights_only=True),
        strict=True,
    )

    cm_after, metrics_after = evaluate(
        best_model, test_loader, save_predictions=True, tag="after"
    )
    save_metrics("after", cm_after, metrics_after)
    print_metrics("AFTER FINE-TUNING", metrics_after)

    print("\n========== TEST COMPARISON ==========")
    for key in ("Dice", "IoU", "Precision", "Recall", "Pixel_Accuracy"):
        before = metrics_before[key]
        after = metrics_after[key]
        print(
            f"{key:16}: {before:.5f} -> {after:.5f} "
            f"(change {after - before:+.5f})"
        )

    summary = {
        "best_epoch": best_epoch,
        "best_validation_dice": best_dice,
        "device": str(DEVICE),
        "input_size": [WIDTH, HEIGHT],
        "split": "vehicle-ID grouped 70/15/15",
        "preprocessing": "RGB / 255 + ImageNet mean/std",
        "before": metrics_before,
        "after": metrics_after,
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nAll results:", OUT)
    print("Best model:", best_path)
    print("Test masks and overlays:", OUT / "test_predictions" / "after")


if __name__ == "__main__":
    main()
