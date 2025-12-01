from ultralytics import YOLO
from pathlib import Path

def main():
    model = YOLO("runs/segment/train3/weights/best.pt")  # adapte le chemin si besoin

    source = "data/solar_coco/test"  # dossier ou image unique
    model.predict(
        source=source,
        imgsz=640,
        device="mps",
        save=True,          # sauvegarde les images avec masques
        save_txt=False
    )

if __name__ == "__main__":
    main()
