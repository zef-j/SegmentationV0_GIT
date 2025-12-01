from ultralytics import YOLO

def main():
    # Modèle nano segmentation (léger, OK pour Raspberry Pi plus tard)
    model = YOLO("yolov8n-seg.pt")

    model.train(
        data="panel_seg.yaml",
        epochs=20,       # pour un premier test
        imgsz=640,
        device="mps",    # ou "cpu" si MPS n'est pas dispo
        batch=16          # ajuste si tu as des soucis de RAM
    )

if __name__ == "__main__":
    main()

