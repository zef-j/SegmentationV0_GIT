from pathlib import Path

LABEL_DIRS = [
    "data/solar_coco/train",
    "data/solar_coco2/train",
]

def inspect_label(path):
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cls_id = int(parts[0])
            floats = list(map(float, parts[1:]))

            cx, cy, w, h = floats[:4]
            poly = floats[4:]
            if len(poly) < 8:
                continue  # pas assez de points

            xs = poly[0::2]
            ys = poly[1::2]

            # bbox déduite du polygone
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            bx = (min_x + max_x) / 2
            by = (min_y + max_y) / 2
            bw = max_x - min_x
            bh = max_y - min_y

            # si poly ~ bbox → segmentation = gros rectangle
            if abs(bx - cx) < 1e-3 and abs(by - cy) < 1e-3 \
               and abs(bw - w) < 1e-3 and abs(bh - h) < 1e-3:
                return "rect_like"
            else:
                return "true_poly"
    return "empty"

for folder in LABEL_DIRS:
    folder = Path(folder)
    txts = list(folder.glob("*.txt"))
    print(f"\n=== {folder} ===")
    for t in txts[:20]:  # on teste quelques fichiers
        res = inspect_label(t)
        print(t.name, "->", res)
