# -*- coding: utf-8 -*-
import random
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# ============== CONFIG UTILISATEUR ==============
MODEL_PATH = "runs/segment/train_modelm_1_cont/weights/last.pt"  # poids du modèle
DATASETS = [
    "data/solar_coco",
    "data/solar_coco2",      # ajoute/enlève ce que tu veux
]
SPLIT = "test"               # "test" ou "valid" (ou "val" si ton dataset s'appelle ainsi)
NUM_SAMPLES = 50             # nb d'images à tirer aléatoirement (global, tous datasets confondus)
DEVICE = "mps"               # "mps" (Mac M1/M2), "cuda", ou "cpu"
CONF_THRESHOLD = 0.25
IMG_SIZE = 640               # taille d'inférence
MAX_WINDOW_HEIGHT = 900      # hauteur max de la fenêtre affichée
DRAW_BOXES = True
DRAW_MASKS = True
MASK_ALPHA = 0.35            # transparence des masques
LINE_THICKNESS = 2           # épaisseur des rectangles
RANDOM_SEED = 42             # fixe pour reproductibilité (ou None)
SAVE_ANNOTATED = False       # True => sauvegarde les images annotées à côté des originales
# =================================================

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def stack_images_side_by_side(img_left, img_right, max_height=900):
    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]
    target_h = min(max(h1, h2), max_height)
    scale1 = target_h / h1
    scale2 = target_h / h2
    left_resized = cv2.resize(img_left, (int(w1 * scale1), target_h))
    right_resized = cv2.resize(img_right, (int(w2 * scale2), target_h))
    return np.hstack([left_resized, right_resized])

def collect_images(datasets, split):
    all_images = []
    for ds in datasets:
        split_dir = Path(ds) / split
        if not split_dir.exists():
            # essaie "val" si "valid" ou "test" ne sont pas présents
            if split == "valid" and (Path(ds) / "val").exists():
                split_dir = Path(ds) / "val"
            elif split == "val" and (Path(ds) / "valid").exists():
                split_dir = Path(ds) / "valid"
            elif split == "test" and not split_dir.exists():
                # laisse tel quel si test n'existe pas
                pass
        if split_dir.exists():
            imgs = sorted([p for p in split_dir.glob("*") if p.suffix.lower() in IMG_EXTS])
            all_images.extend(imgs)
    return all_images

def random_colors(n):
    rng = np.random.default_rng(12345)
    cols = []
    for _ in range(n):
        c = rng.integers(0, 255, size=3, dtype=np.uint8)
        cols.append((int(c[0]), int(c[1]), int(c[2])))
    return cols

def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    model = YOLO(MODEL_PATH)

    # Récupère un pool d'images depuis tous les datasets pour le split demandé
    all_imgs = collect_images(DATASETS, SPLIT)
    if not all_imgs:
        print(f"Aucune image trouvée dans {DATASETS} pour le split '{SPLIT}'.")
        return

    # Tirage aléatoire
    if NUM_SAMPLES and NUM_SAMPLES < len(all_imgs):
        sample_paths = random.sample(all_imgs, NUM_SAMPLES)
    else:
        sample_paths = all_imgs

    print(f"{len(sample_paths)} images sélectionnées (sur {len(all_imgs)} trouvées).")
    cv2.namedWindow("Original | Annotated", cv2.WINDOW_NORMAL)

    # Inference (on peut streamer ou faire une par une; ici on va itérer pour garder affichage/contrôle)
    for img_path in sample_paths:
        # Lance la prédiction sur une seule image pour pouvoir afficher pas à pas
        results = model.predict(
            source=str(img_path),
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )
        result = results[0]

        # Images
        orig = result.orig_img.copy()
        annotated = orig.copy()

        boxes = result.boxes
        masks = result.masks
        names = result.names  # dict id->classname

        # Titre console
        print(f"\nImage: {img_path}")

        # Dessin
        if boxes is None or len(boxes) == 0:
            print("  Aucun panneau détecté.")
        else:
            # Couleurs aléatoires par instance (pour distinguer les masques)
            cols = random_colors(len(boxes)) if DRAW_MASKS else [(0, 255, 0)] * len(boxes)

            # Dessin des masques d'abord (pour être sous les boîtes/texte)
            if DRAW_MASKS and masks is not None and masks.data is not None:
                mdata = masks.data.cpu().numpy()  # [N, H, W]
                for det_id, m in enumerate(mdata):
                    color = cols[det_id]
                    overlay = np.zeros_like(annotated, dtype=np.uint8)
                    overlay[m > 0.5] = (int(color[2]), int(color[1]), int(color[0]))  # BGR
                    annotated = cv2.addWeighted(annotated, 1.0, overlay, MASK_ALPHA, 0)

            # Puis les boîtes et labels
            if DRAW_BOXES:
                for det_id, box in enumerate(boxes):
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    cls_name = names.get(cls_id, str(cls_id))
                    color = cols[det_id] if DRAW_MASKS else (0, 255, 0)

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, LINE_THICKNESS)
                    label = f"ID {det_id} | {cls_name} {conf:.2f}"
                    cv2.putText(
                        annotated, label, (x1 + 5, y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
                    )
                    print(f"  Det {det_id}: cls={cls_name}, conf={conf:.3f}, box=({x1},{y1},{x2},{y2})")

        combined = stack_images_side_by_side(orig, annotated, max_height=MAX_WINDOW_HEIGHT)
        cv2.imshow("Original | Annotated", combined)

        print(f"Image : {img_path.name}")
        print("  boxes:", 0 if boxes is None else len(boxes))


        if SAVE_ANNOTATED:
            out_path = Path(str(img_path).rsplit(".", 1)[0] + "_annotated.jpg")
            cv2.imwrite(str(out_path), annotated)

        print("Touches : 'n' (suivante) · 'q' (quitter)")
        while True:
            k = cv2.waitKey(0) & 0xFF
            if k in (ord("n"), 13, 32):  # n, Enter, Space
                break
            if k in (ord("q"), 27):      # q, Esc
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
