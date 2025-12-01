from ultralytics import YOLO
from pathlib import Path

def main():
    # === PARAMÈTRES À ADAPTER FACILEMENT ===
    RUN_NAME = "train_l4_v1"     # nom du nouveau run
    BASE_MODEL = "yolov8m-seg.pt"  # modèle de départ
    DATA_YAML = "panel_seg.yaml"
    EPOCHS = 50
    IMG_SIZE = 640

    # GPU: 0 = premier GPU, "cuda" marche aussi
    DEVICE = 0

    # Commence prudemment, on augmentera si la VRAM le permet
    BATCH_SIZE = 16
    WORKERS = 0

    print(f"[INFO] Training YOLOv8 segmentation sur GPU={DEVICE}")
    print(f"       Run name : {RUN_NAME}")
    print(f"       Model    : {BASE_MODEL}")
    print(f"       Epochs   : {EPOCHS}, batch={BATCH_SIZE}, imgsz={IMG_SIZE}")

    model = YOLO(BASE_MODEL)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device=DEVICE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        name=RUN_NAME,
        project="runs/segment",   # dossier classique d'Ultralytics
        cache=False,              # on pourra mettre True plus tard si le disque suit
        amp=True,                 # mixed precision pour aller plus vite
    )

    # Petit résumé
    best = Path("runs/segment") / RUN_NAME / "weights" / "best.pt"
    print(f"[INFO] Training terminé. Poids : {best}")

if __name__ == "__main__":
    main()
