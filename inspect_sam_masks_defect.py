from pathlib import Path
import cv2
import json
import numpy as np

root = Path("full_image_defect_masks/selected_masks")

for defect in ["dent", "scratch", "broken"]:
    folder = root / defect
    print(f"\n========== {defect.upper()} ==========")

    pngs = sorted(folder.glob("*.png"))
    jsons = sorted(folder.glob("*.json"))

    print("PNG masks:", len(pngs))
    print("JSON metadata:", len(jsons))

    for p in pngs[:5]:
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print("INVALID:", p.name)
            continue

        values = np.unique(mask)
        nonzero = int(np.count_nonzero(mask))
        print({
            "file": p.name,
            "shape": mask.shape,
            "values": values.tolist()[:20],
            "nonzero_pixels": nonzero,
            "coverage_percent": round(100 * nonzero / mask.size, 3)
        })

    for p in jsons[:2]:
        try:
            data = json.loads(p.read_text())
            print("\nJSON:", p.name)
            print("Keys:", list(data.keys()))
            print("Metadata:", json.dumps(data, indent=2)[:1200])
        except Exception as e:
            print("JSON ERROR:", p.name, e)

print("\n========== AUDIT COMPLETE ==========")
