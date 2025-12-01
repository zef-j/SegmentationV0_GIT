from pathlib import Path
from ultralytics import YOLO

# ================== CONFIG UTILISATEUR ==================
#BASE_MODEL = "yolov8n-seg.pt"  #Nano
BASE_MODEL = "yolov8m-seg.pt"
DATA_YAML = "panel_seg.yaml"
DEVICE = "mps"
BATCH = 10
IMG_SIZE = 640

run_name = "train_modelm_1"
RESUME_FROM = "train_modelm_1"         # run à reprendre

USE_LAST_CHECKPOINT = True
EXTRA_EPOCHS = 50
# ========================================================

def main():
    runs_dir = Path("runs/segment")

    if RESUME_FROM is not None:
        ckpt_name = "last.pt" if USE_LAST_CHECKPOINT else "best.pt"
        ckpt_path = runs_dir / RESUME_FROM / "weights" / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable : {ckpt_path}")

        print(f"[INFO] On repart de : {ckpt_path}")
        model = YOLO(str(ckpt_path))
        run_name = f"{RESUME_FROM}_cont"


    else:
        print(f"[INFO] Nouveau training à partir de {BASE_MODEL}")
        model = YOLO(BASE_MODEL)



    model.train(
        data=DATA_YAML,
        epochs=EXTRA_EPOCHS,
        imgsz=IMG_SIZE,
        device=DEVICE,
        batch=BATCH,
        project="runs/segment",
        name=run_name,
        exist_ok=True,
        # pas de loggers=...
    )

if __name__ == "__main__":
    main()
