#!/usr/bin/env python3
"""
Prepare a semantic-segmentation dataset from LabelMe JSON annotations and separate images.

- Matches JSON and image files by filename stem (not by list order).
- Applies the supplied canonical class mapping without editing source files.
- Keeps all views of the same vehicle together in train/val/test splits.
- Creates indexed PNG masks and colorful overlay previews.
- Writes audit reports for unmatched files and unknown labels.

Expected input:
  images/       (328 source images)
  annotations/  (246 LabelMe JSON files)
  car_parts_class_mapping.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IGNORE_INDEX = 255


def vehicle_id(stem: str) -> str:
    # Example: 0ZLW05074644_back_left_side_captured_image -> 0ZLW05074644
    return stem.split("_", 1)[0]


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def load_mapping(path: Path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    classes = cfg["classes"]
    class_ids = {int(c["id"]) for c in classes}
    if 0 not in class_ids or 255 in class_ids:
        raise ValueError("Class list must include background id 0 and must not use id 255.")
    mapping = {}
    for item in cfg["label_mappings"]:
        raw = str(item.get("raw_label", "")).strip()
        if not raw:
            continue
        mapping[raw] = item
    return cfg, mapping


def draw_shape(draw: ImageDraw.ImageDraw, shape: dict, fill: int, scale_x: float, scale_y: float):
    pts = shape.get("points") or []
    pts = [(float(p[0]) * scale_x, float(p[1]) * scale_y) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
    shape_type = (shape.get("shape_type") or "polygon").lower()

    if shape_type == "polygon":
        if len(pts) >= 3:
            draw.polygon(pts, fill=fill)
            return True
    elif shape_type == "rectangle":
        if len(pts) >= 2:
            draw.rectangle([pts[0], pts[1]], fill=fill)
            return True
    elif shape_type == "circle":
        if len(pts) >= 2:
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            r = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            draw.ellipse((x0-r, y0-r, x0+r, y0+r), fill=fill)
            return True
    return False


def make_overlay(image: Image.Image, mask: Image.Image, color_by_id: dict[int, tuple[int, int, int]], alpha=0.45):
    img = np.asarray(image.convert("RGB"), dtype=np.uint8)
    m = np.asarray(mask, dtype=np.uint8)
    color = np.zeros_like(img)
    for cid, rgb in color_by_id.items():
        if cid == 0 or cid == IGNORE_INDEX:
            continue
        color[m == cid] = rgb
    valid = (m != 0) & (m != IGNORE_INDEX)
    out = img.copy()
    out[valid] = (img[valid] * (1.0 - alpha) + color[valid] * alpha).astype(np.uint8)
    return Image.fromarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--class-map", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preview-limit", type=int, default=0,
                    help="0 = save overlay for every matched image; otherwise save first N per split")
    args = ap.parse_args()

    for folder in (args.images, args.annotations):
        if not folder.is_dir():
            raise SystemExit(f"Input folder does not exist: {folder}")

    cfg, label_map = load_mapping(args.class_map)
    class_by_id = {int(c["id"]): c for c in cfg["classes"]}
    color_by_id = {}
    for cid, c in class_by_id.items():
        h = str(c.get("color", "#808080")).lstrip("#")
        try:
            color_by_id[cid] = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        except Exception:
            color_by_id[cid] = (128, 128, 128)

    images = [x for x in args.images.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTS]
    jsons = sorted(args.annotations.glob("*.json"))
    image_index = defaultdict(list)
    for image_path in images:
        image_index[image_path.stem.casefold()].append(image_path)

    matched, unmatched_json, duplicate_image_stems = [], [], []
    used_images = set()
    for jp in jsons:
        candidates = image_index.get(jp.stem.casefold(), [])
        if len(candidates) == 1:
            matched.append((jp, candidates[0]))
            used_images.add(candidates[0].resolve())
        elif len(candidates) == 0:
            unmatched_json.append(jp.name)
        else:
            duplicate_image_stems.append(f"{jp.name}: " + "; ".join(x.name for x in candidates))

    unannotated_images = [x.name for x in images if x.resolve() not in used_images]
    audit_dir = args.output / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    write_csv(audit_dir / "unmatched_annotations.csv", ["json_file"], [[x] for x in unmatched_json])
    write_csv(audit_dir / "unannotated_images.csv", ["image_file"], [[x] for x in sorted(unannotated_images)])
    write_csv(audit_dir / "duplicate_image_stems.csv", ["problem"], [[x] for x in duplicate_image_stems])

    # Preflight all JSON files before writing any masks. Unknown labels are never silently remapped.
    records, unknown_counter, bad_json, dimension_mismatch = [], Counter(), [], []
    for jp, ip in matched:
        try:
            item = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as e:
            bad_json.append([jp.name, str(e)])
            continue
        shapes = item.get("shapes")
        if not isinstance(shapes, list):
            bad_json.append([jp.name, "Missing or invalid 'shapes' list"])
            continue
        for sh in shapes:
            raw = str(sh.get("label", "")).strip()
            if raw not in label_map:
                unknown_counter[raw or "<EMPTY_LABEL>"] += 1
        try:
            with Image.open(ip) as im:
                actual_size = im.size
            jw, jh = item.get("imageWidth"), item.get("imageHeight")
            if jw and jh and (int(jw), int(jh)) != actual_size:
                dimension_mismatch.append([jp.name, ip.name, f"JSON={jw}x{jh}", f"image={actual_size[0]}x{actual_size[1]}"])
                continue
        except Exception as e:
            bad_json.append([jp.name, f"Cannot open image {ip.name}: {e}"])
            continue
        records.append((jp, ip, item))

    write_csv(audit_dir / "invalid_annotations.csv", ["json_file", "reason"], bad_json)
    write_csv(audit_dir / "dimension_mismatches.csv",
              ["json_file", "image_file", "json_dimensions", "image_dimensions"], dimension_mismatch)
    write_csv(audit_dir / "unmapped_labels.csv", ["raw_label", "occurrences"],
              [[k, str(v)] for k, v in sorted(unknown_counter.items())])

    report = {
        "images_found": len(images),
        "json_annotations_found": len(jsons),
        "matched_json_image_pairs": len(matched),
        "valid_pairs_after_preflight": len(records),
        "unmatched_json_files": len(unmatched_json),
        "unannotated_images_not_used": len(unannotated_images),
        "duplicate_image_stem_groups": len(duplicate_image_stems),
        "invalid_json_or_image_records": len(bad_json),
        "dimension_mismatches": len(dimension_mismatch),
        "unmapped_raw_labels": dict(unknown_counter),
        "class_count_including_background": len(cfg["classes"]),
        "trainable_part_classes": len([c for c in cfg["classes"] if int(c["id"]) != 0]),
        "note": "Unannotated images are listed for audit only and are not assigned blank masks."
    }
    (audit_dir / "dataset_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n========== DATASET AUDIT ==========")
    print(json.dumps(report, indent=2))
    if duplicate_image_stems or unknown_counter or bad_json or dimension_mismatch:
        print("\nSTOP: Resolve duplicate stems, unknown labels, invalid records, or dimension mismatches.")
        print(f"Audit reports: {audit_dir}")
        sys.exit(2)
    if not records:
        raise SystemExit("No valid matched image/JSON pairs found.")

    # Group by vehicle ID to prevent the same vehicle leaking across splits.
    groups = defaultdict(list)
    for rec in records:
        groups[vehicle_id(rec[1].stem)].append(rec)
    group_names = list(groups)
    random.Random(args.seed).shuffle(group_names)
    # Greedy assignment by group size, maintaining approximate 70/15/15 image proportions.
    group_names.sort(key=lambda g: len(groups[g]), reverse=True)
    targets = {"train": 0.70 * len(records), "val": 0.15 * len(records), "test": 0.15 * len(records)}
    assigned = {"train": [], "val": [], "test": []}
    counts = Counter()
    for g in group_names:
        split = min(("train", "val", "test"), key=lambda s: (counts[s] - targets[s], counts[s] / max(targets[s], 1)))
        assigned[split].append(g)
        counts[split] += len(groups[g])
    split_for_vehicle = {g: split for split, gs in assigned.items() for g in gs}

    for split in ("train", "val", "test"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "masks" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "previews" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows, class_pixels, shape_counts = [], Counter(), Counter()
    preview_saved = Counter()
    for jp, ip, item in records:
        split = split_for_vehicle[vehicle_id(ip.stem)]
        with Image.open(ip) as src:
            image = src.convert("RGB")
        mask = Image.new("L", image.size, 0)
        drawer = ImageDraw.Draw(mask)
        # LabelMe's shape order is preserved; later shapes overwrite earlier overlaps.
        for sh in item["shapes"]:
            raw = str(sh.get("label", "")).strip()
            entry = label_map[raw]
            cid = int(entry.get("class_id", IGNORE_INDEX))
            if entry.get("action") == "ignore" or cid == IGNORE_INDEX:
                fill = IGNORE_INDEX
            else:
                fill = cid
            if draw_shape(drawer, sh, fill, 1.0, 1.0):
                shape_counts[raw] += 1
        mask_arr = np.asarray(mask, dtype=np.uint8)
        vals, cnts = np.unique(mask_arr, return_counts=True)
        for v, n in zip(vals.tolist(), cnts.tolist()):
            class_pixels[int(v)] += int(n)

        out_name = ip.stem + ".png"
        image.save(args.output / "images" / split / out_name)
        mask.save(args.output / "masks" / split / out_name)
        if args.preview_limit == 0 or preview_saved[split] < args.preview_limit:
            overlay = make_overlay(image, mask, color_by_id)
            overlay.save(args.output / "previews" / split / out_name)
            preview_saved[split] += 1
        manifest_rows.append([split, out_name, vehicle_id(ip.stem), ip.name, jp.name])

    write_csv(args.output / "manifest.csv",
              ["split", "output_name", "vehicle_id", "source_image", "source_json"], manifest_rows)
    (args.output / "class_mapping.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    pixel_rows = []
    for c in cfg["classes"]:
        cid = int(c["id"])
        pixel_rows.append([str(cid), c["name"], str(class_pixels.get(cid, 0))])
    pixel_rows.append([str(IGNORE_INDEX), "ignore", str(class_pixels.get(IGNORE_INDEX, 0))])
    write_csv(args.output / "class_pixel_counts.csv", ["class_id", "class_name", "pixel_count"], pixel_rows)

    summary = {
        **report,
        "generated_pairs": len(records),
        "split_image_counts": {s: sum(1 for r in manifest_rows if r[0] == s) for s in ("train", "val", "test")},
        "split_vehicle_counts": {s: len(assigned[s]) for s in ("train", "val", "test")},
        "vehicle_ids_are_grouped": True,
        "mask_background_id": 0,
        "ignore_index": IGNORE_INDEX,
        "shape_counts_by_raw_label": dict(shape_counts),
        "class_pixel_counts": {str(k): v for k, v in sorted(class_pixels.items())}
    }
    (args.output / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n========== MASK GENERATION COMPLETE ==========")
    print(f"Generated image-mask pairs: {len(records)}")
    print(f"Split counts: {summary['split_image_counts']}")
    print(f"Vehicle groups: {summary['split_vehicle_counts']}")
    print(f"Dataset output: {args.output}")
    print(f"Preview overlays: {args.output / 'previews'}")
    print(f"Audit report: {audit_dir / 'dataset_audit.json'}")


if __name__ == "__main__":
    main()
