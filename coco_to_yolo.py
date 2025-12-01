import os
from pathlib import Path
import json

# Chemin vers ton dataset COCO
DATA_ROOT = Path("data/solar_coco")  # adapte si besoin

MERGE_CLASSES = True  # True = tout devient "panel" (classe 0)


def convert_split(split: str):
    split_dir = DATA_ROOT / split
    coco_path = split_dir / "_annotations.coco.json"

    with open(coco_path, "r") as f:
        coco = json.load(f)

    # Map image_id -> infos
    id2img = {img["id"]: img for img in coco["images"]}

    # Map category_id -> id YOLO
    if MERGE_CLASSES:
        catid2yolo = {c["id"]: 0 for c in coco["categories"]}
        class_names = ["panel"]
    else:
        sorted_cats = sorted(coco["categories"], key=lambda c: c["id"])
        catid2yolo = {c["id"]: i for i, c in enumerate(sorted_cats)}
        class_names = [c["name"] for c in sorted_cats]

    labels_dir = split_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    img_to_lines = {}

    for ann in coco["annotations"]:
        img = id2img[ann["image_id"]]
        w, h = img["width"], img["height"]

        # bbox COCO -> centre + taille normalisés
        x, y, bw, bh = ann["bbox"]
        xc = (x + bw / 2) / w
        yc = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h

        segs = ann["segmentation"]
        if not segs:
            continue

        # Ici, Roboflow fournit un seul polygone
        poly = segs[0]
        if len(poly) < 6:  # au moins 3 points
            continue

        xs = poly[0::2]
        ys = poly[1::2]
        norm_coords = []
        for px, py in zip(xs, ys):
            norm_coords.append(px / w)
            norm_coords.append(py / h)

        cls = catid2yolo[ann["category_id"]]

        line = " ".join(
            [str(cls),
             f"{xc:.6f}", f"{yc:.6f}", f"{nw:.6f}", f"{nh:.6f}"]
            + [f"{v:.6f}" for v in norm_coords]
        )

        img_to_lines.setdefault(img["file_name"], []).append(line)

    # Écriture fichiers .txt
    for fname, lines in img_to_lines.items():
        stem = Path(fname).stem
        label_path = labels_dir / f"{stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    print(
        f"{split}: {len(coco['images'])} images, "
        f"{len(coco['annotations'])} annotations, "
        f"{len(list(labels_dir.glob('*.txt')))} fichiers labels."
    )
    return class_names


if __name__ == "__main__":
    all_classes = set()
    for sp in ["train", "valid", "test"]:
        names = convert_split(sp)
        all_classes.update(names)

    print("Classes utilisées :", sorted(all_classes))
