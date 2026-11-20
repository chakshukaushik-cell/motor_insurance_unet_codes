import json
import numpy as np
import torch
import segmentation_models_pytorch as smp

from pathlib import Path
from PIL import Image, ImageOps


# ============================================================
# PATHS
# ============================================================

ROOT = Path("/home/caiu0184/Downloads/Car_Parts_seg_model")

IMAGE_PATH = ROOT / "Screenshot 2026-11-10 142613.png"

CHECKPOINT_PATH = (
    ROOT / "runs/car_parts_unet_new/best_model.pth"
)

MAPPING_PATH = (
    ROOT / "unet_dataset/class_mapping.json"
)

OUTPUT_DIR = ROOT / "unseen_inference_original_model"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 512
NUM_CLASSES = 59
IGNORE_INDEX = 255

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL AND MAPPING
# ============================================================

def load_model():

    print("Loading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    if checkpoint["num_classes"] != NUM_CLASSES:
        raise ValueError("Checkpoint class count mismatch.")

    if checkpoint["encoder"] != "resnet34":
        raise ValueError("Unexpected encoder.")

    if checkpoint["image_size"] != IMAGE_SIZE:
        raise ValueError("Unexpected image size.")

    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)

    classes = sorted(
        mapping["classes"],
        key=lambda x: x["id"]
    )

    if len(classes) != NUM_CLASSES:
        raise ValueError("Class mapping mismatch.")

    if [x["name"] for x in classes] != checkpoint["class_names"]:
        raise ValueError("Checkpoint and mapping names differ.")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    print("Checkpoint loaded successfully.")
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Checkpoint mIoU:", checkpoint["best_miou"])
    print("Classes:", NUM_CLASSES)

    return model, classes


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    original_width, original_height = image.size

    resized = ImageOps.contain(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        method=Image.Resampling.BILINEAR
    )

    canvas = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        (114, 114, 114)
    )

    x = (IMAGE_SIZE - resized.width) // 2
    y = (IMAGE_SIZE - resized.height) // 2

    canvas.paste(resized, (x, y))

    image_array = np.asarray(
        canvas,
        dtype=np.float32
    ) / 255.0

    image_array = (
        image_array - MEAN
    ) / STD

    tensor = torch.from_numpy(
        image_array.transpose(2, 0, 1).copy()
    ).float().unsqueeze(0)

    return (
        tensor.to(DEVICE),
        (original_width, original_height),
        (resized.width, resized.height, x, y)
    )


# ============================================================
# RESTORE PREDICTION TO ORIGINAL IMAGE SIZE
# ============================================================

def restore_mask(prediction, original_size, resize_info):

    resized_width, resized_height, x, y = resize_info

    cropped = prediction[
        y:y + resized_height,
        x:x + resized_width
    ]

    mask_image = Image.fromarray(
        cropped.astype(np.uint8),
        mode="L"
    )

    mask_image = mask_image.resize(
        original_size,
        Image.Resampling.NEAREST
    )

    return np.asarray(mask_image, dtype=np.uint8)


# ============================================================
# COLOR MASK
# ============================================================

def create_colored_mask(mask, classes):

    height, width = mask.shape

    colored = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    for item in classes:

        class_id = item["id"]

        color = item["color"].lstrip("#")

        rgb = tuple(
            int(color[i:i + 2], 16)
            for i in (0, 2, 4)
        )

        colored[mask == class_id] = rgb

    # Ignore/padding pixels
    colored[mask == IGNORE_INDEX] = (255, 0, 255)

    return colored


# ============================================================
# MAIN INFERENCE
# ============================================================

def main():

    torch.backends.cudnn.enabled = False

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    model, classes = load_model()

    original_image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    print("\nOriginal image size:", original_image.size)

    input_tensor, original_size, resize_info = (
        preprocess_image(original_image)
    )

    print("Running inference...")

    with torch.no_grad():

        output = model(input_tensor)

        prediction = torch.argmax(
            output,
            dim=1
        ).squeeze(0).cpu().numpy()

    mask = restore_mask(
        prediction,
        original_size,
        resize_info
    )

    colored_mask = create_colored_mask(
        mask,
        classes
    )

    # --------------------------------------------------------
    # OVERLAY
    # --------------------------------------------------------

    original_array = np.asarray(
        original_image,
        dtype=np.uint8
    )

    overlay = (
        original_array.astype(np.float32) * 0.55
        + colored_mask.astype(np.float32) * 0.45
    ).clip(0, 255).astype(np.uint8)

    # Keep ignore pixels visually identifiable
    ignore_pixels = mask == IGNORE_INDEX
    overlay[ignore_pixels] = (255, 0, 255)

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    original_image.save(
        OUTPUT_DIR / "original_image.png"
    )

    Image.fromarray(mask, mode="L").save(
        OUTPUT_DIR / "predicted_mask.png"
    )

    Image.fromarray(colored_mask).save(
        OUTPUT_DIR / "colored_mask.png"
    )

    Image.fromarray(overlay).save(
        OUTPUT_DIR / "overlay.png"
    )

    # --------------------------------------------------------
    # CLASS PIXEL REPORT
    # --------------------------------------------------------

    total_pixels = mask.size

    print("\n========== PREDICTED CLASSES ==========")

    report_path = OUTPUT_DIR / "prediction_report.csv"

    import csv

    with open(report_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "class_id",
            "class_name",
            "color",
            "pixel_count",
            "pixel_percentage"
        ])

        for item in classes:

            class_id = item["id"]

            count = int(
                np.count_nonzero(mask == class_id)
            )

            percentage = (
                count / total_pixels
            ) * 100

            if count > 0:

                print(
                    f"{class_id:2d} | "
                    f"{item['name']:35s} | "
                    f"{percentage:.2f}%"
                )

            writer.writerow([
                class_id,
                item["name"],
                item["color"],
                count,
                round(percentage, 4)
            ])

    print("\n========== INFERENCE COMPLETED ==========")
    print("Output folder:", OUTPUT_DIR)
    print("Original image:", OUTPUT_DIR / "original_image.png")
    print("Predicted mask:", OUTPUT_DIR / "predicted_mask.png")
    print("Colored mask:", OUTPUT_DIR / "colored_mask.png")
    print("Overlay:", OUTPUT_DIR / "overlay.png")
    print("CSV report:", report_path)


if __name__ == "__main__":
    main()

