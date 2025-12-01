# -*- coding: utf-8 -*-
import random
from pathlib import Path

import cv2
import numpy as np

# ========== CONFIG UTILISATEUR ==========
DATASET_ROOT = "data/solar_coco2"  # ou "data/solar_coco2"
SPLIT = "train"                   # "train", "valid", "val", "test"
MAX_IMAGES = 100                  # nb max d'images à parcourir
RANDOM_SEED = 42                  # None pour complètement aléatoire
SHOW_POLYGON_FILL = True          # remplissage semi-transparent
WINDOW_MAX_H = 900
# =======================================

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def stack_side_by_side(img1, img2, max_height=900):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    target_h = min(max(h1, h2), max_height)
    s1 = target_h / h1
    s2 = target_h / h2
    r1 = cv2.resize(img1, (int(w1 * s1), target_h))
    r2 = cv2.resize(img2, (int(w2 * s2), target_h))
    return np.hstack([r1, r2])


def collect_pairs(root, split):
    split_dir = Path(root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split '{split}' introuvable dans {root}")

    label_files = sorted(split_dir.glob("*.txt"))
    pairs = []
    for lab in label_files:
        stem = lab.stem
        img_path = None
        for ext in IMG_EXTS:
            p = split_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is not None:
            pairs.append((img_path, lab))

    return pairs


def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    pairs = collect_pairs(DATASET_ROOT, SPLIT)
    if not pairs:
        print("Aucune paire image/label trouvée.")
        return

    if MAX_IMAGES and MAX_IMAGES < len(pairs):
        pairs = random.sample(pairs, MAX_IMAGES)

    print(f"{len(pairs)} images sélectionnées pour visualisation.")

    cv2.namedWindow("GT | Annotated", cv2.WINDOW_NORMAL)

    for img_path, lab_path in pairs:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] impossible de lire {img_path}")
            continue
        h, w = img.shape[:2]
        annotated = img.copy()

        print(f"\nImage: {img_path.name}")
        print(f"Label: {lab_path.name}")

        with open(lab_path, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        for li, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 7:
                print(f"  ligne {li}: pas assez de valeurs -> {line}")
                continue

            try:
                cls_id = int(parts[0])
                vals = list(map(float, parts[1:]))
            except Exception as e:
                print(f"  ligne {li}: erreur parsing ({e}) -> {line}")
                continue

            cx, cy, bw, bh = vals[:4]
            poly = vals[4:]

            # bbox absolue
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # polygone absolu
            if len(poly) >= 6 and len(poly) % 2 == 0:
                xs = poly[0::2]
                ys = poly[1::2]
                pts = np.stack(
                    [
                        np.array(xs) * w,
                        np.array(ys) * h,
                    ],
                    axis=1,
                ).astype(np.int32)

                # debug console
                print(
                    f"  obj {li}: cls={cls_id}, n_pts={len(pts)}, "
                    f"bbox=({x1},{y1},{x2},{y2})"
                )

                # polygone (contour rouge)
                cv2.polylines(
                    annotated,
                    [pts.reshape(-1, 1, 2)],
                    isClosed=True,
                    color=(0, 0, 255),
                    thickness=2,
                )

                # optionnel : remplissage
                if SHOW_POLYGON_FILL:
                    overlay = annotated.copy()
                    cv2.fillPoly(
                        overlay,
                        [pts.reshape(-1, 1, 2)],
                        color=(0, 0, 255),
                    )
                    alpha = 0.3
                    annotated = cv2.addWeighted(
                        overlay, alpha, annotated, 1 - alpha, 0
                    )

            # bbox (vert)
            cv2.rectangle(
                annotated, (x1, y1), (x2, y2), (0, 255, 0), 2
            )

        combined = stack_side_by_side(img, annotated, max_height=WINDOW_MAX_H)
        cv2.imshow("GT | Annotated", combined)
        print("Touches : 'n' (suivante) · 'q' (quitter)")

        while True:
            k = cv2.waitKey(0) & 0xFF
            if k in (ord("n"), 13, 32):
                break
            if k in (ord("q"), 27):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
