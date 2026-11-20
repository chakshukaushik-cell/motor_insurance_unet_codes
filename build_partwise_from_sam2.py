#!/usr/bin/env python3

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# =========================================================
# PATHS AND CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "images"
PARTS_DIR = ROOT / "parts_annotations"
FULL_MASK_DIR = ROOT / "output" / "sam2_outputs" / "full_masks"
MAPPING_FILE = ROOT / "car_parts_class_mapping.json"

OUTPUT_DIR = ROOT / "output" / "partwise_dataset_colored"

# SAM2 mask IDs
BACKGROUND_ID = 0

DEFECT_CLASSES = {
    1: "dent",
    2: "scratch",
    3: "broken",
}

# RGB colors
DEFECT_COLORS = {
    1: (255, 0, 0),      # Dent - Red
    2: (0, 255, 0),      # Scratch - Green
    3: (0, 80, 255),     # Broken - Blue
}

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
RANDOM_SEED = 42

LABEL_OVERRIDES = {
    "front_bumperr": {
        "canonical_name": "front_bumper",
        "class_id": 2,
    },
    "right_": {
        "canonical_name": None,
        "class_id": 255,
    },
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def safe_name(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value)
    return value.strip("_")


def find_image(stem):
    for ext in IMAGE_EXTENSIONS:
        path = IMAGE_DIR / (stem + ext)
        if path.exists():
            return path

    for path in IMAGE_DIR.iterdir():
        if path.is_file() and path.stem == stem:
            return path

    return None


def read_label_mapping():
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}

    for item in data.get("label_mappings", []):
        raw_label = item.get("raw_label")

        if raw_label is not None:
            mapping[raw_label] = {
                "canonical_name": item.get("canonical_name"),
                "class_id": item.get("class_id"),
            }

    mapping.update(LABEL_OVERRIDES)
    return mapping


def create_polygon_mask(points, width, height):
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)

    valid_points = []

    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue

        x = float(point[0])
        y = float(point[1])

        if np.isfinite(x) and np.isfinite(y):
            valid_points.append((x, y))

    if len(valid_points) < 3:
        return None

    draw.polygon(valid_points, fill=255)

    mask = np.array(canvas) > 0

    if not np.any(mask):
        return None

    return mask


def get_crop_box(mask, width, height, padding):
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(width, int(xs.max()) + padding + 1)
    y2 = min(height, int(ys.max()) + padding + 1)

    return x1, y1, x2, y2


def load_sam2_mask(stem, width, height):
    path = FULL_MASK_DIR / (stem + ".png")

    if not path.exists():
        return None

    mask = np.array(
        Image.open(path).convert("L"),
        dtype=np.uint8
    )

    if mask.shape != (height, width):
        raise ValueError(
            f"Mask dimension mismatch: {path.name}; "
            f"found {mask.shape}, expected {(height, width)}"
        )

    allowed_values = {0, 1, 2, 3}
    actual_values = set(np.unique(mask).tolist())

    if not actual_values.issubset(allowed_values):
        raise ValueError(
            f"Unexpected pixel IDs in {path.name}: {actual_values}"
        )

    return mask


def create_visualization(image_crop, mask_crop, part_name):
    visual = image_crop.copy()

    present_defects = []

    for class_id, defect_name in DEFECT_CLASSES.items():
        selected = mask_crop == class_id

        if not np.any(selected):
            continue

        color = np.array(
            DEFECT_COLORS[class_id],
            dtype=np.float32
        )

        original = visual[selected].astype(np.float32)

        blended = (
            original * 0.35
            + color * 0.65
        )

        visual[selected] = np.clip(
            blended, 0, 255
        ).astype(np.uint8)

        present_defects.append(defect_name)

    output = Image.fromarray(visual)
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()

    # Part name
    draw.rectangle(
        (0, 0, output.width, 18),
        fill=(25, 25, 25)
    )
    draw.text(
        (5, 4),
        "PART: " + str(part_name),
        fill=(255, 255, 255),
        font=font
    )

    # Defect names and color legend
    y = 24

    for class_id, defect_name in DEFECT_CLASSES.items():
        if defect_name not in present_defects:
            continue

        color = DEFECT_COLORS[class_id]

        draw.rectangle(
            (5, y, 17, y + 11),
            fill=color
        )

        draw.text(
            (23, y),
            defect_name.upper(),
            fill=(255, 255, 255),
            font=font
        )

        y += 15

    if not present_defects:
        draw.text(
            (5, 25),
            "NO DEFECT",
            fill=(255, 255, 255),
            font=font
        )

    return output, present_defects


def create_split_map(audit_ids):
    unique_ids = sorted(set(audit_ids))

    random.Random(RANDOM_SEED).shuffle(unique_ids)

    if len(unique_ids) < 3:
        raise RuntimeError(
            "At least 3 unique audit IDs are required."
        )

    total = len(unique_ids)

    train_count = max(1, round(total * 0.70))
    val_count = max(1, round(total * 0.15))

    if train_count + val_count >= total:
        train_count = total - 2
        val_count = 1

    result = {}

    for audit_id in unique_ids[:train_count]:
        result[audit_id] = "train"

    for audit_id in unique_ids[
        train_count:train_count + val_count
    ]:
        result[audit_id] = "val"

    for audit_id in unique_ids[train_count + val_count:]:
        result[audit_id] = "test"

    return result


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        rows = [{"message": "No records"}]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(rows)


# =========================================================
# MAIN DATASET BUILDER
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        action="store_true",
        help="Generate the actual dataset"
    )

    parser.add_argument(
        "--padding",
        type=int,
        default=5
    )

    args = parser.parse_args()

    required_paths = [
        IMAGE_DIR,
        PARTS_DIR,
        FULL_MASK_DIR,
        MAPPING_FILE,
    ]

    for path in required_paths:
        if not path.exists():
            raise SystemExit("Missing path: " + str(path))

    if args.padding < 0:
        raise SystemExit("Padding cannot be negative.")

    if args.run and OUTPUT_DIR.exists():
        raise SystemExit(
            "Output folder already exists:\n"
            + str(OUTPUT_DIR)
            + "\nExisting output will not be deleted. "
            "Rename the folder before a fresh run."
        )

    label_mapping = read_label_mapping()

    json_files = sorted(PARTS_DIR.glob("*.json"))

    records = []
    missing_images = []
    missing_masks = []
    json_errors = []

    print("\nChecking input files...")

    for json_path in json_files:
        stem = json_path.stem

        image_path = find_image(stem)
        mask_path = FULL_MASK_DIR / (stem + ".png")

        if image_path is None:
            missing_images.append({
                "image_stem": stem,
                "reason": "IMAGE_NOT_FOUND",
            })
            continue

        if not mask_path.exists():
            missing_masks.append({
                "image_stem": stem,
                "reason": "SAM2_MASK_NOT_FOUND",
            })
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        except Exception as exc:
            json_errors.append({
                "image_stem": stem,
                "reason": str(exc),
            })
            continue

        records.append(
            (json_path, image_path, json_data)
        )

    if not records:
        raise SystemExit(
            "No matching image + LabelMe JSON + SAM2 mask triplets found."
        )

    audit_ids = [
        json_path.stem.split("_")[0]
        for json_path, _, _ in records
    ]

    split_map = create_split_map(audit_ids)

    print("\nMatched triplets:", len(records))
    print("Missing images:", len(missing_images))
    print("Missing SAM2 masks:", len(missing_masks))
    print("JSON errors:", len(json_errors))

    print("\nClass mapping:")
    print("Background = 0")
    print("Dent       = 1 (Red)")
    print("Scratch    = 2 (Green)")
    print("Broken     = 3 (Blue)")

    print("\nMode:", "ACTUAL GENERATION" if args.run else "DRY RUN")
    print("Output:", OUTPUT_DIR)

    part_rows = []
    image_rows = []
    unknown_rows = []
    skipped_rows = []

    totals = Counter()
    split_counts = Counter()

    # -----------------------------------------------------
    # PROCESS EACH IMAGE
    # -----------------------------------------------------

    for index, (json_path, image_path, json_data) in enumerate(
        records, start=1
    ):
        stem = json_path.stem
        audit_id = stem.split("_")[0]
        split = split_map[audit_id]

        print(
            f"\n[{index}/{len(records)}] {stem} ({split})"
        )

        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)

        width, height = image.size

        json_width = json_data.get("imageWidth")
        json_height = json_data.get("imageHeight")

        if json_width and json_height:
            if (
                int(json_width) != width
                or int(json_height) != height
            ):
                raise RuntimeError(
                    f"Image/JSON dimensions mismatch: {stem}"
                )

        full_mask = load_sam2_mask(
            stem, width, height
        )

        if full_mask is None:
            continue

        source_counts = {
            name: int(np.count_nonzero(full_mask == class_id))
            for class_id, name in DEFECT_CLASSES.items()
        }

        # ---------------------------------------------
        # Read LabelMe part polygons
        # ---------------------------------------------

        parts = []

        for shape_index, shape in enumerate(
            json_data.get("shapes", [])
        ):
            raw_label = shape.get("label")
            shape_type = shape.get("shape_type")

            mapping = label_mapping.get(raw_label)

            if mapping is None:
                unknown_rows.append({
                    "image_stem": stem,
                    "shape_index": shape_index,
                    "raw_label": raw_label,
                    "reason": "UNKNOWN_LABEL",
                })
                continue

            canonical = mapping.get("canonical_name")
            part_class_id = mapping.get("class_id")

            if canonical is None or part_class_id is None:
                skipped_rows.append({
                    "image_stem": stem,
                    "shape_index": shape_index,
                    "raw_label": raw_label,
                    "reason": "IGNORED_MAPPING",
                })
                continue

            if int(part_class_id) == 255:
                skipped_rows.append({
                    "image_stem": stem,
                    "shape_index": shape_index,
                    "raw_label": raw_label,
                    "reason": "IGNORE_CLASS",
                })
                continue

            if shape_type != "polygon":
                skipped_rows.append({
                    "image_stem": stem,
                    "shape_index": shape_index,
                    "raw_label": raw_label,
                    "reason": "NOT_A_POLYGON",
                })
                continue

            polygon = create_polygon_mask(
                shape.get("points", []),
                width,
                height
            )

            if polygon is None:
                skipped_rows.append({
                    "image_stem": stem,
                    "shape_index": shape_index,
                    "raw_label": raw_label,
                    "reason": "INVALID_POLYGON",
                })
                continue

            box = get_crop_box(
                polygon,
                width,
                height,
                args.padding
            )

            if box is None:
                continue

            parts.append({
                "shape_index": shape_index,
                "raw_label": raw_label,
                "canonical": canonical,
                "part_class_id": int(part_class_id),
                "polygon": polygon,
                "box": box,
            })

        image_assigned = np.zeros(
            (height, width),
            dtype=bool
        )

        image_crop_count = 0

        # ---------------------------------------------
        # Generate one cutout for each part
        # ---------------------------------------------

        for part in parts:
            x1, y1, x2, y2 = part["box"]
            polygon = part["polygon"]

            # Crop the original image
            image_crop = image_array[
                y1:y2, x1:x2
            ].copy()

            # Crop the actual polygon mask
            polygon_crop = polygon[
                y1:y2, x1:x2
            ]

            # IMPORTANT:
            # Everything outside the part polygon is BLACK.
            image_crop[~polygon_crop] = 0

            # Keep only defects inside this exact part polygon.
            part_full_mask = np.where(
                polygon,
                full_mask,
                BACKGROUND_ID
            ).astype(np.uint8)

            mask_crop = part_full_mask[
                y1:y2, x1:x2
            ].copy()

            # Force outside polygon to background ID 0.
            mask_crop[~polygon_crop] = BACKGROUND_ID

            # Track defects that overlap recognized parts.
            image_assigned |= (
                polygon & (full_mask > BACKGROUND_ID)
            )

            defect_pixels = {
                name: int(np.count_nonzero(mask_crop == class_id))
                for class_id, name in DEFECT_CLASSES.items()
            }

            present_defects = [
                name
                for name, count in defect_pixels.items()
                if count > 0
            ]

            # Create colored segmentation visualization.
            visualization, _ = create_visualization(
                image_crop,
                mask_crop,
                part["canonical"]
            )

            output_stem = (
                safe_name(stem)
                + "__"
                + safe_name(part["canonical"])
                + "__part_"
                + f"{part['shape_index'] + 1:03d}"
            )

            part_rows.append({
                "image_stem": stem,
                "audit_id": audit_id,
                "split": split,
                "part": part["canonical"],
                "raw_label": part["raw_label"],
                "part_class_id": part["part_class_id"],
                "image_filename": output_stem + ".jpg",
                "mask_filename": output_stem + ".png",
                "visualization_filename": output_stem + ".jpg",
                "defect_types": ";".join(present_defects),
                "dent_pixels": defect_pixels["dent"],
                "scratch_pixels": defect_pixels["scratch"],
                "broken_pixels": defect_pixels["broken"],
                "total_defect_pixels": int(
                    np.count_nonzero(mask_crop)
                ),
                "status": "PENDING_MANUAL_REVIEW",
            })

            totals.update(defect_pixels)
            totals["part_crops"] += 1

            if present_defects:
                totals["defect_crops"] += 1
            else:
                totals["clean_crops"] += 1

            split_counts[split] += 1
            image_crop_count += 1

            # -----------------------------------------
            # Save output files
            # -----------------------------------------

            if args.run:
                image_out = (
                    OUTPUT_DIR / "images" / split
                    / (output_stem + ".jpg")
                )

                mask_out = (
                    OUTPUT_DIR / "masks" / split
                    / (output_stem + ".png")
                )

                visual_out = (
                    OUTPUT_DIR / "visualizations" / split
                    / (output_stem + ".jpg")
                )

                image_out.parent.mkdir(
                    parents=True, exist_ok=True
                )

                mask_out.parent.mkdir(
                    parents=True, exist_ok=True
                )

                visual_out.parent.mkdir(
                    parents=True, exist_ok=True
                )

                Image.fromarray(image_crop).save(
                    image_out,
                    quality=95
                )

                # Grayscale PNG: pixel IDs 0, 1, 2, 3.
                Image.fromarray(mask_crop).save(mask_out)

                visualization.save(
                    visual_out,
                    quality=95
                )

        # ---------------------------------------------
        # Image-level report
        # ---------------------------------------------

        unassigned = (
            (full_mask > BACKGROUND_ID)
            & (~image_assigned)
        )

        image_rows.append({
            "image_stem": stem,
            "audit_id": audit_id,
            "split": split,
            "parts_found": len(parts),
            "part_crops": image_crop_count,
            "source_dent_pixels": source_counts["dent"],
            "source_scratch_pixels": source_counts["scratch"],
            "source_broken_pixels": source_counts["broken"],
            "unassigned_defect_pixels": int(
                np.count_nonzero(unassigned)
            ),
            "status": "PROCESSED",
        })

        print(
            f"Parts: {len(parts)} | "
            f"Crops: {image_crop_count} | "
            f"Dent: {source_counts['dent']} | "
            f"Scratch: {source_counts['scratch']} | "
            f"Broken: {source_counts['broken']}"
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print("Matched images:", len(records))
    print("Part crops:", totals["part_crops"])
    print("Clean crops:", totals["clean_crops"])
    print("Defect crops:", totals["defect_crops"])

    print("Dent pixels:", totals["dent"])
    print("Scratch pixels:", totals["scratch"])
    print("Broken pixels:", totals["broken"])

    print("Train crops:", split_counts["train"])
    print("Val crops:", split_counts["val"])
    print("Test crops:", split_counts["test"])

    # =====================================================
    # SAVE REPORTS
    # =====================================================

    if args.run:
        reports = OUTPUT_DIR / "reports"

        save_csv(
            reports / "partwise_details.csv",
            part_rows
        )

        save_csv(
            reports / "image_summary.csv",
            image_rows
        )

        save_csv(
            reports / "unknown_labels.csv",
            unknown_rows
        )

        save_csv(
            reports / "skipped_shapes.csv",
            skipped_rows
        )

        save_csv(
            reports / "missing_images.csv",
            missing_images
        )

        save_csv(
            reports / "missing_masks.csv",
            missing_masks
        )

        save_csv(
            reports / "json_errors.csv",
            json_errors
        )

        config = {
            "dataset_status": "PENDING_MANUAL_REVIEW",
            "source_images": str(IMAGE_DIR),
            "source_parts": str(PARTS_DIR),
            "source_sam2_masks": str(FULL_MASK_DIR),
            "output": str(OUTPUT_DIR),
            "class_ids": {
                "background": 0,
                "dent": 1,
                "scratch": 2,
                "broken": 3,
            },
            "visualization_colors_rgb": {
                "dent": [255, 0, 0],
                "scratch": [0, 255, 0],
                "broken": [0, 80, 255],
            },
            "split": {
                "train": 0.70,
                "val": 0.15,
                "test": 0.15,
            },
            "padding": args.padding,
            "total_part_crops": totals["part_crops"],
        }

        with open(
            OUTPUT_DIR / "dataset_config.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(config, f, indent=4)

        print("\nDataset saved:", OUTPUT_DIR)
        print("Visualizations:", OUTPUT_DIR / "visualizations")
        print("Reports:", reports)

    else:
        print("\nDRY RUN COMPLETE.")
        print("No output files were created.")
        print("Check counts before running with --run.")


if __name__ == "__main__":
    main()
