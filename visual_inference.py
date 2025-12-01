import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ====== À ADAPTER ======
MODEL_PATH = "runs/segment/train3/weights/best.pt"  # chemin vers ton modèle entraîné
SOURCE_DIR = "data/solar_coco/test"                 # dossier avec les images à visualiser
DEVICE = "mps"                                      # "mps" sur Mac M1, sinon "cpu"
CONF_THRESHOLD = 0.25
# ========================


def stack_images_side_by_side(img_left, img_right, max_height=800):
    """Met deux images côte à côte, en les redimensionnant si nécessaire."""
    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]

    # même hauteur pour les deux
    target_h = min(max(h1, h2), max_height)
    scale1 = target_h / h1
    scale2 = target_h / h2

    img_left_resized = cv2.resize(img_left, (int(w1 * scale1), target_h))
    img_right_resized = cv2.resize(img_right, (int(w2 * scale2), target_h))

    return np.hstack([img_left_resized, img_right_resized])


def main():
    model = YOLO(MODEL_PATH)
    source_path = Path(SOURCE_DIR)

    # Liste des images dans le dossier source
    image_files = sorted(
        [p for p in source_path.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )

    if not image_files:
        print(f"Aucune image trouvée dans {SOURCE_DIR}")
        return

    print(f"{len(image_files)} images trouvées. Lancement de l'inférence...")

    # Utilisation du mode stream pour traiter les images une par une
    results = model(
        [str(p) for p in image_files],
        imgsz=640,
        conf=CONF_THRESHOLD,
        device=DEVICE,
        stream=True,
        verbose=False,
    )

    cv2.namedWindow("Original | Annotated", cv2.WINDOW_NORMAL)

    for img_path, result in zip(image_files, results):
        # Image originale (BGR)
        orig = result.orig_img.copy()

        # Annotated = copie de l'originale
        annotated = orig.copy()

        # Récupération des infos de détection
        boxes = result.boxes  # Boxes object
        masks = result.masks  # Masks object ou None
        names = result.names  # dict id -> class name

        print(f"\nImage : {img_path.name}")
        if boxes is None or len(boxes) == 0:
            print("  Aucun panneau détecté.")
        else:
            for det_id, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = xyxy.astype(int)
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cls_name = names.get(cls_id, str(cls_id))

                # Dessin du rectangle
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Texte à l'intérieur : ID, classe, conf
                label = f"ID {det_id} | {cls_name} {conf:.2f}"
                # Position du texte : en haut à gauche du rectangle
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                # Affiche aussi les infos en console
                print(f"  Det {det_id}: cls={cls_name}, conf={conf:.3f}, box=({x1},{y1},{x2},{y2})")

            # Optionnel : overlay des masques de segmentation
            if masks is not None:
                # masks.data : [N, H, W]
                mask_data = masks.data.cpu().numpy()
                for det_id, m in enumerate(mask_data):
                    colored_mask = np.zeros_like(annotated, dtype=np.uint8)
                    colored_mask[m > 0.5] = (0, 0, 255)  # rouge pour le masque
                    # mélange avec alpha pour transparence
                    annotated = cv2.addWeighted(annotated, 1.0, colored_mask, 0.3, 0)

        # Met les deux images côte à côte
        combined = stack_images_side_by_side(orig, annotated)

        cv2.imshow("Original | Annotated", combined)
        print("Appuie sur 'n' pour image suivante, 'q' pour quitter.")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("n"), 13, 32):  # 'n', Enter, Space
                break
            elif key in (ord("q"), 27):  # 'q' ou ESC
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
