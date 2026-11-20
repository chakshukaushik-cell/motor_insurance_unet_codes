from pathlib import Path
import csv
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ================= CONFIGURATION =================
ROOT = Path("/home/caiu0184/Downloads/Unet_car_detection")
IMAGE_DIR = ROOT / "matched_original_images"
MASK_DIR = ROOT / "main_car_masks"
OVERLAY_DIR = ROOT / "main_car_overlays"
OUT_DIR = ROOT / "runs" / "unet_car"

IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 40
LEARNING_RATE = 1e-3
SEED = 42
DEVICE = torch.device("cpu")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ================= CONFUSION MATRIX =================
def save_confusion_matrix(cm, output_path, title):
    """
    Rows = Actual class
    Columns = Predicted class

    Class 0 = Background
    Class 1 = Car
    """
    cell_w = 300
    cell_h = 180
    left = 210
    top = 140
    right = 40
    bottom = 120

    width = left + 2 * cell_w + right
    height = top + 2 * cell_h + bottom

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    cv2.putText(
        canvas, title, (left, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2
    )
    cv2.putText(
        canvas, "Predicted class", (left + 140, 95),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2
    )
    cv2.putText(
        canvas, "Actual", (20, top + cell_h),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2
    )

    names = ["Background", "Car"]

    for col, name in enumerate(names):
        x = left + col * cell_w
        cv2.putText(
            canvas, name, (x + 45, top - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2
        )

    for row, name in enumerate(names):
        y = top + row * cell_h
        cv2.putText(
            canvas, name, (20, y + cell_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2
        )

        for col in range(2):
            x = left + col * cell_w
            value = int(cm[row, col])

            # Green = correct class, red = incorrect class
            color = (190, 225, 190) if row == col else (190, 190, 245)

            cv2.rectangle(
                canvas, (x, y),
                (x + cell_w, y + cell_h),
                color, thickness=-1
            )
            cv2.rectangle(
                canvas, (x, y),
                (x + cell_w, y + cell_h),
                (70, 70, 70), thickness=2
            )

            text = f"{value:,}"
            text_size = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
            )[0]

            tx = x + (cell_w - text_size[0]) // 2
            ty = y + cell_h // 2 + 10

            cv2.putText(
                canvas, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2
            )

    cv2.putText(
        canvas, "Pixel counts", (left, height - 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2
    )

    cv2.imwrite(str(output_path), canvas)
    print("Confusion matrix saved:", output_path)


def calculate_confusion_matrix(model, loader, output_path, title):
    # Matrix order: actual Background/Car, predicted Background/Car
    cm = np.zeros((2, 2), dtype=np.int64)
    model.eval()

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            predictions = (torch.sigmoid(logits) >= 0.5).long()
            actual = masks.long()

            actual = actual.view(-1).cpu().numpy()
            predictions = predictions.view(-1).cpu().numpy()

            # 0 = Background, 1 = Car
            combined = actual * 2 + predictions
            counts = np.bincount(combined, minlength=4)

            cm[0, 0] += counts[0]  # Background predicted Background
            cm[0, 1] += counts[1]  # Background predicted Car
            cm[1, 0] += counts[2]  # Car predicted Background
            cm[1, 1] += counts[3]  # Car predicted Car

    save_confusion_matrix(cm, output_path, title)

    total = cm.sum()
    if total:
        print(f"{title} pixel accuracy: {cm.trace() / total:.4%}")

    return cm


# ================= DATASET =================
def find_file_by_stem(folder, stem, extensions):
    matches = [
        p for p in folder.iterdir()
        if p.is_file()
        and p.stem == stem
        and p.suffix.lower() in extensions
    ]
    return sorted(matches)[0] if matches else None


images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
)

valid_pairs = []
missing_mask = []
missing_overlay = []
bad_files = []

for img_path in images:
    mask_path = find_file_by_stem(MASK_DIR, img_path.stem, {".png"})
    overlay_path = find_file_by_stem(
        OVERLAY_DIR, img_path.stem, {".jpg", ".jpeg", ".png"}
    )

    if mask_path is None:
        missing_mask.append(img_path.name)
        continue
    if overlay_path is None:
        missing_overlay.append(img_path.name)
        continue

    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    overlay = cv2.imread(str(overlay_path))

    if img is None or mask is None or overlay is None or mask.size == 0:
        bad_files.append(img_path.name)
        continue

    valid_pairs.append((img_path, mask_path, overlay_path))

print("\n========== DATASET CHECK ==========")
print("Original images:", len(images))
print("Complete sets:", len(valid_pairs))
print("Missing masks:", len(missing_mask))
print("Missing overlays:", len(missing_overlay))
print("Unreadable sets:", len(bad_files))

(OUT_DIR / "excluded_images.txt").write_text(
    "MISSING MASKS\n" + "\n".join(missing_mask)
    + "\n\nMISSING OVERLAYS\n" + "\n".join(missing_overlay)
    + "\n\nUNREADABLE FILES\n" + "\n".join(bad_files),
    encoding="utf-8"
)

if len(valid_pairs) != 89:
    raise SystemExit(
        f"Expected 89 complete sets, found {len(valid_pairs)}. "
        f"Check {OUT_DIR / 'excluded_images.txt'}"
    )

random.shuffle(valid_pairs)
train_pairs = valid_pairs[:65]
val_pairs = valid_pairs[65:78]
test_pairs = valid_pairs[78:89]

with open(OUT_DIR / "dataset_split.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["split", "image", "mask", "overlay"])
    for split_name, pairs in [
        ("train", train_pairs),
        ("validation", val_pairs),
        ("test", test_pairs),
    ]:
        for img, mask, overlay in pairs:
            writer.writerow([split_name, img.name, mask.name, overlay.name])

print("Split sizes:", len(train_pairs), len(val_pairs), len(test_pairs))
print("Device:", DEVICE)


class CarDataset(Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img_path, mask_path, _ = self.pairs[index]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(
            image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
        )
        mask = cv2.resize(
            mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST
        )
        mask = (mask > 127).astype(np.float32)

        if self.augment and random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = np.transpose(image, (2, 0, 1)).copy()

        return (
            torch.from_numpy(image),
            torch.from_numpy(mask[None, :, :].copy())
        )


# ================= U-NET MODEL =================
def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = double_conv(3, 16)
        self.enc2 = double_conv(16, 32)
        self.enc3 = double_conv(32, 64)
        self.enc4 = double_conv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = double_conv(128, 256)
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = double_conv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = double_conv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = double_conv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = double_conv(32, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def dice_iou_from_logits(logits, targets):
    preds = (torch.sigmoid(logits) >= 0.5).float()
    dims = (1, 2, 3)
    intersection = (preds * targets).sum(dims)
    pred_sum = preds.sum(dims)
    target_sum = targets.sum(dims)
    union = pred_sum + target_sum - intersection
    dice = (2 * intersection + 1e-6) / (
        pred_sum + target_sum + 1e-6
    )
    iou = (intersection + 1e-6) / (union + 1e-6)
    return dice.mean(), iou.mean()


def combined_loss(logits, targets):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    dice = (2 * intersection + 1e-6) / (
        probs.sum(dims) + targets.sum(dims) + 1e-6
    )
    return bce + (1 - dice.mean())


train_loader = DataLoader(
    CarDataset(train_pairs, augment=True),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
train_eval_loader = DataLoader(
    CarDataset(train_pairs),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
val_loader = DataLoader(
    CarDataset(val_pairs), batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    CarDataset(test_pairs), batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)


# ================= TRAINING =================
model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float("inf")
patience = 8
no_improvement = 0
history_path = OUT_DIR / "training_history.csv"

with open(history_path, "w", newline="") as f:
    csv.writer(f).writerow(
        ["epoch", "train_loss", "val_loss", "val_dice", "val_iou"]
    )

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss_sum = 0.0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = combined_loss(logits, y)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item() * x.size(0)

    train_loss = train_loss_sum / len(train_loader.dataset)

    model.eval()
    val_loss_sum = val_dice_sum = val_iou_sum = 0.0
    val_count = 0

    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x.to(DEVICE))
            loss = combined_loss(logits, y.to(DEVICE))
            dice, iou = dice_iou_from_logits(logits, y.to(DEVICE))
            n = x.size(0)
            val_loss_sum += loss.item() * n
            val_dice_sum += dice.item() * n
            val_iou_sum += iou.item() * n
            val_count += n

    val_loss = val_loss_sum / val_count
    val_dice = val_dice_sum / val_count
    val_iou = val_iou_sum / val_count

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
        f"val_Dice={val_dice:.4f} | val_IoU={val_iou:.4f}"
    )

    with open(history_path, "a", newline="") as f:
        csv.writer(f).writerow(
            [epoch, train_loss, val_loss, val_dice, val_iou]
        )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improvement = 0
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "img_size": IMG_SIZE,
            },
            OUT_DIR / "best_model.pth"
        )
        print("  Saved new best model.")
    else:
        no_improvement += 1

    if no_improvement >= patience:
        print("Early stopping.")
        break


# ================= EVALUATION =================
checkpoint = torch.load(
    OUT_DIR / "best_model.pth",
    map_location=DEVICE,
    weights_only=False
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Save confusion matrices with Background and Car labels.
train_cm = calculate_confusion_matrix(
    model,
    train_eval_loader,
    OUT_DIR / "confusion_matrix_train.png",
    "Training Confusion Matrix"
)

test_cm = calculate_confusion_matrix(
    model,
    test_loader,
    OUT_DIR / "confusion_matrix_test.png",
    "Test Confusion Matrix"
)

# Save individual test predictions.
pred_dir = OUT_DIR / "test_predictions"
pred_dir.mkdir(exist_ok=True)

test_dice_sum = test_iou_sum = 0.0
test_count = 0

with torch.no_grad():
    for batch_idx, (x, y) in enumerate(test_loader):
        logits = model(x.to(DEVICE))
        dice, iou = dice_iou_from_logits(logits, y.to(DEVICE))
        n = x.size(0)
        test_dice_sum += dice.item() * n
        test_iou_sum += iou.item() * n
        test_count += n

        preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy()
        for j in range(n):
            idx = batch_idx * BATCH_SIZE + j
            name = test_pairs[idx][0].stem
            pred_mask = (preds[j, 0] * 255).astype(np.uint8)
            cv2.imwrite(str(pred_dir / f"{name}_pred_mask.png"), pred_mask)

test_dice = test_dice_sum / test_count
test_iou = test_iou_sum / test_count

with open(OUT_DIR / "test_metrics.txt", "w") as f:
    f.write(f"Test images: {test_count}\n")
    f.write(f"Dice score: {test_dice:.4f}\n")
    f.write(f"IoU: {test_iou:.4f}\n")

print("\n========== TRAINING COMPLETE ==========")
print("Best model:", OUT_DIR / "best_model.pth")
print("Training confusion matrix:", OUT_DIR / "confusion_matrix_train.png")
print("Test confusion matrix:", OUT_DIR / "confusion_matrix_test.png")
print(f"Test Dice: {test_dice:.4f}")
print(f"Test IoU: {test_iou:.4f}")
