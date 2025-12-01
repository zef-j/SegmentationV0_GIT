import cv2
import numpy as np
from pathlib import Path

# ====== À ADAPTER SI BESOIN ======
SPLIT_DIR = Path("data/solar_coco2/train")  # ou valid / test
MAX_HEIGHT = 900
# ================================


def parse_yolo_segmentation_label(label_path, img_w, img_h):
    """
    Lit un fichier YOLO segmentation (format:
    cls cx cy w h x1 y1 x2 y2 ... ) et renvoie une liste de polygones en pixels.
    """
    polys = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= 5:
                continue
            # cls = int(parts[0])  # on ne l'utilise pas ici
            # bbox cx,cy,w,h = parts[1:5]  # pas nécessaire pour l'affichage du polygone

            coords = list(map(float, parts[5:]))  # x1 y1 x2 y2 ...
            if len(coords) % 2 != 0:
                continue

            pts = []
            for x_norm, y_norm in zip(coords[0::2], coords[1::2]):
                x = int(x_norm * img_w)
                y = int(y_norm * img_h)
                pts.append([x, y])

            if len(pts) >= 3:
                polys.append(np.array(pts, dtype=np.int32))

    return polys


def stack_images_side_by_side(img_left, img_right, max_height=800):
    """Met deux images côte à côte, redimensionnées à la même hauteur."""
    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]

    target_h = min(max(h1, h2), max_height)
    scale1 = target_h / h1
    scale2 = target_h / h2

    left_resized = cv2.resize(img_left, (int(w1 * scale1), target_h))
    right_resized = cv2.resize(img_right, (int(w2 * scale2), target_h))

    return np.hstack([left_resized, right_resized])


def main():
    if not SPLIT_DIR.exists():
        print(f"Dossier split introuvable : {SPLIT_DIR}")
        return

    # liste des images
    image_files = sorted(
        [p for p in SPLIT_DIR.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )

    if not image_files:
        print(f"Aucune image trouvée dans {SPLIT_DIR}")
        return

    print(f"{len(image_files)} images trouvées dans {SPLIT_DIR}")

    cv2.namedWindow("GT | Original vs Segmentation", cv2.WINDOW_NORMAL)

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Impossible de lire {img_path}")
            continue

        h, w = img.shape[:2]
        annotated = img.copy()

        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            print(f"[{img_path.name}] Aucun label (.txt) trouvé.")
        else:
            polys = parse_yolo_segmentation_label(label_path, w, h)
            if not polys:
                print(f"[{img_path.name}] Label trouvé mais aucun polygone valide.")
            else:
                print(f"[{img_path.name}] {len(polys)} polygone(s) trouvé(s).")

                # créer un overlay de couleur
                overlay = np.zeros_like(annotated, dtype=np.uint8)
                for poly in polys:
                    # remplir le polygone en bleu
                    cv2.fillPoly(overlay, [poly.reshape(-1, 1, 2)], (255, 0, 0))
                    # et tracer le contour en rouge
                    cv2.polylines(annotated, [poly.reshape(-1, 1, 2)], True, (0, 0, 255), 2)

                # mélange avec transparence
                annotated = cv2.addWeighted(annotated, 1.0, overlay, 0.3, 0)

        combined = stack_images_side_by_side(img, annotated, max_height=MAX_HEIGHT)

        cv2.imshow("GT | Original vs Segmentation", combined)
        print("Image:", img_path.name, "| 'n' = next, 'q' = quit")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("n"), 13, 32):  # n, Enter, Space
                break
            elif key in (ord("q"), 27):  # q, ESC
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
