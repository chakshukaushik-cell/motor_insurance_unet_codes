#!/usr/bin/env python3
"""
Create class-aware SAM2 defect masks and part-wise masked crops.

SAM2 receives each YOLO box as a BOX prompt. The defect class (dent/scratch/broken)
is preserved as metadata and used to write the correct target-mask class/color.
SAM2 itself is not a text-conditioned defect classifier.
"""
import argparse, json, csv, random
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np
import torch
torch.backends.cudnn.enabled = False
from PIL import Image

DEFECTS = {
    0: {"name": "dent", "mask_id": 1, "color": (255, 0, 0)},      # RGB red
    1: {"name": "scratch", "mask_id": 2, "color": (0, 255, 0)},   # RGB green
    2: {"name": "broken", "mask_id": 3, "color": (0, 0, 255)},    # RGB blue
}

def load_mapping(path):
    data = json.load(open(path, encoding="utf-8"))
    aliases = {}
    classes = {}
    for item in data.get("classes", []):
        cid = int(item["id"])
        classes[cid] = item["name"]
    for item in data.get("label_mappings", []):
        name = item.get("canonical_name")
        cid = item.get("class_id")
        if name and cid is not None and int(cid) != 255:
            aliases[item["raw_label"].strip().lower()] = (name, int(cid))
            aliases[name.strip().lower()] = (name, int(cid))
    # Also accept canonical names even if no explicit raw-label mapping exists.
    for cid, name in classes.items():
        if cid not in (0, 255):
            aliases[name.lower()] = (name, cid)
    return aliases, classes

def canonical_part(label, aliases):
    found = aliases.get(str(label).strip().lower())
    return found if found else (None, None)

def yolo_boxes(txt_path, width, height):
    rows = []
    if not txt_path.exists():
        return rows
    for line_no, line in enumerate(txt_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        vals = line.split()
        if len(vals) < 5:
            print(f"[WARN] {txt_path}:{line_no}: expected class cx cy w h")
            continue
        try:
            cls = int(float(vals[0]))
            cx, cy, bw, bh = map(float, vals[1:5])
        except ValueError:
            print(f"[WARN] bad YOLO row {txt_path}:{line_no}")
            continue
        if cls not in DEFECTS:
            print(f"[WARN] unknown defect class {cls} in {txt_path}; skipping")
            continue
        x1 = max(0, min(width - 1, int(round((cx - bw / 2) * width))))
        y1 = max(0, min(height - 1, int(round((cy - bh / 2) * height))))
        x2 = max(0, min(width - 1, int(round((cx + bw / 2) * width))))
        y2 = max(0, min(height - 1, int(round((cy + bh / 2) * height))))
        if x2 > x1 and y2 > y1:
            rows.append((cls, np.array([x1, y1, x2, y2], dtype=np.float32)))
    return rows

def labelme_part_masks(json_path, width, height, aliases):
    data = json.load(open(json_path, encoding="utf-8"))
    masks = {}
    part_ids = {}
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        name, cid = canonical_part(label, aliases)
        if name is None:
            print(f"[WARN] unmapped part label '{label}' in {json_path.name}; ignored")
            continue
        pts = np.asarray(shape.get("points", []), dtype=np.float32)
        if len(pts) < 3:
            continue
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        pts = np.round(pts).astype(np.int32)
        m = np.zeros((height, width), dtype=np.uint8)
        st = shape.get("shape_type", "polygon")
        if st == "rectangle" and len(pts) >= 2:
            cv2.rectangle(m, tuple(pts[0]), tuple(pts[1]), 1, -1)
        else:
            cv2.fillPoly(m, [pts], 1)
        if m.any():
            # Merge repeated polygons belonging to the same canonical part.
            masks[name] = cv2.bitwise_or(masks.get(name, np.zeros_like(m)), m)
            part_ids[name] = cid
    return masks, part_ids

def sam2_predictor(checkpoint, config, device):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(config, checkpoint, device=device)
    return SAM2ImagePredictor(model)

def color_overlay(rgb, defect_mask):
    out = rgb.copy()
    palette = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}
    for mid, color in palette.items():
        region = defect_mask == mid
        if region.any():
            out[region] = (0.45 * out[region] + 0.55 * np.array(color)).astype(np.uint8)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--defects", required=True)
    ap.add_argument("--parts", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-intersection-pixels", type=int, default=4)
    ap.add_argument("--min-mask-area", type=int, default=3)
    ap.add_argument("--sam2-multimask", action="store_true",
                    help="Ask SAM2 for 3 masks and select the highest predicted score.")
    ap.add_argument("--limit", type=int, default=0, help="0 = all matched images")
    args = ap.parse_args()

    images_dir, defects_dir = Path(args.images), Path(args.defects)
    parts_dir, out_dir = Path(args.parts), Path(args.out)
    for sub in ["sam2_outputs/full_masks", "sam2_outputs/overlays",
                "part_wise_dataset", "reports"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    aliases, _ = load_mapping(args.mapping)
    image_files = sorted([p for p in images_dir.iterdir()
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    matched = []
    for img_path in image_files:
        stem = img_path.stem
        txt = defects_dir / f"{stem}.txt"
        js = parts_dir / f"{stem}.json"
        if txt.exists() and js.exists():
            matched.append((img_path, txt, js))
    if args.limit:
        matched = matched[:args.limit]
    print(f"[INFO] matched image/YOLO/LabelMe triplets: {len(matched)}")
    if not matched:
        raise SystemExit("No matching basenames found. Check extensions and folders.")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable; using CPU (SAM2 will be slow).")
        device = "cpu"
    predictor = sam2_predictor(args.checkpoint, args.config, device)

    report_rows = []
    stats = Counter()
    for idx, (img_path, txt_path, json_path) in enumerate(matched, 1):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            stats["bad_images"] += 1
            continue
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        boxes = yolo_boxes(txt_path, w, h)
        parts, part_ids = labelme_part_masks(json_path, w, h, aliases)
        if not parts:
            stats["images_without_parts"] += 1
            continue

        predictor.set_image(rgb)
        full_defect = np.zeros((h, w), dtype=np.uint8)
        defect_records = []
        for defect_idx, (defect_cls, box) in enumerate(boxes):
            # SAM2 is prompted with the exact YOLO box. Class ID is retained separately.
            masks, scores, _ = predictor.predict(
                box=box,
                multimask_output=bool(args.sam2_multimask)
            )
            best = int(np.argmax(scores)) if len(scores) else 0
            sam_mask = masks[best].astype(bool)

            # Restrict SAM2 prediction to the original YOLO defect box.
            x1, y1, x2, y2 = np.round(box).astype(int)
            box_mask = np.zeros((h, w), dtype=bool)
            box_mask[y1:y2+1, x1:x2+1] = True
            sam_mask = sam_mask & box_mask

            if int(sam_mask.sum()) < args.min_mask_area:
                stats["empty_sam_masks"] += 1
                continue
            mask_id = DEFECTS[defect_cls]["mask_id"]
            # Later annotations do not overwrite already assigned pixels.
            full_defect[sam_mask & (full_defect == 0)] = mask_id
            defect_records.append((defect_cls, sam_mask, box))
            stats[f"sam2_{DEFECTS[defect_cls]['name']}"] += 1

        if not defect_records:
            stats["images_without_defects"] += 1

        # Save the full-image class mask (0 background, 1 dent, 2 scratch, 3 broken).
        cv2.imwrite(str(out_dir / "sam2_outputs/full_masks" / f"{img_path.stem}.png"),
                    full_defect)
        overlay = color_overlay(rgb, full_defect)
        cv2.imwrite(str(out_dir / "sam2_outputs/overlays" / f"{img_path.stem}.jpg"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # Part-wise intersection. Each source image/part produces one combined crop.
        for part_name, part_mask in parts.items():
            target_full = np.zeros((h, w), dtype=np.uint8)
            for defect_cls, sam_mask, box in defect_records:
                inter = sam_mask & (part_mask > 0)
                if int(inter.sum()) < args.min_intersection_pixels:
                    continue
                mid = DEFECTS[defect_cls]["mask_id"]
                target_full[inter & (target_full == 0)] = mid

            ys, xs = np.where(part_mask > 0)
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            # Crop around the part; zero pixels outside its polygon.
            crop_rgb = rgb[y1:y2, x1:x2].copy()
            crop_part = part_mask[y1:y2, x1:x2].astype(bool)
            crop_rgb[~crop_part] = 0
            crop_target = target_full[y1:y2, x1:x2].copy()
            crop_target[~crop_part] = 0

            part_dir = out_dir / "part_wise_dataset" / part_name
            for sub in ["images", "masks", "visualizations"]:
                (part_dir / sub).mkdir(parents=True, exist_ok=True)
            out_stem = f"{img_path.stem}__{part_name}"
            cv2.imwrite(str(part_dir / "images" / f"{out_stem}.jpg"),
                        cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(part_dir / "masks" / f"{out_stem}.png"), crop_target)
            vis = color_overlay(crop_rgb, crop_target)
            cv2.imwrite(str(part_dir / "visualizations" / f"{out_stem}.jpg"),
                        cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            for mid, name in [(1, "dent"), (2, "scratch"), (3, "broken")]:
                if np.any(crop_target == mid):
                    stats[f"part_crop_{name}"] += 1
            stats["part_crops"] += 1
            report_rows.append({
                "source_image": img_path.name, "part_name": part_name,
                "part_class_id": part_ids[part_name], "crop": out_stem,
                "dent_pixels": int((crop_target == 1).sum()),
                "scratch_pixels": int((crop_target == 2).sum()),
                "broken_pixels": int((crop_target == 3).sum())
            })

        if idx % 25 == 0 or idx == len(matched):
            print(f"[{idx}/{len(matched)}] {img_path.name}")

    report_path = out_dir / "reports" / "part_crop_manifest.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        fields = ["source_image", "part_name", "part_class_id", "crop",
                  "dent_pixels", "scratch_pixels", "broken_pixels"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report_rows)
    (out_dir / "reports" / "processing_stats.json").write_text(
        json.dumps(dict(stats), indent=2), encoding="utf-8")
    print("\n[DONE] Part-wise dataset created.")
    print("Manifest:", report_path)
    print(json.dumps(dict(stats), indent=2))
    print("Mask pixel IDs: 0=background, 1=dent, 2=scratch, 3=broken")

if __name__ == "__main__":
    main()
