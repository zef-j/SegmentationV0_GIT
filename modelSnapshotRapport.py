# -*- coding: utf-8 -*-
"""
One-shot training run report for Ultralytics YOLOv8-seg.
- Prints framework details, model architecture, dataset structure & format,
  final training metrics, model size, and an inference speed benchmark.
- Copy the console output into your report.

Setup:
  - Put this file at project root (same level as 'runs/' and your data YAML).
  - pip install ultralytics pyyaml pandas opencv-python

Configure at the top ↓ then Run from PyCharm (no CLI needed).
"""

from pathlib import Path
import sys, os, time, platform, json
import yaml
import pandas as pd
import torch
import cv2

from ultralytics import YOLO
import ultralytics


# ─────────────────────────────────────────────────────────────
#  USER CONFIG
# ─────────────────────────────────────────────────────────────
RUNS_DIR = Path("runs/segment")   # Ultralytics default for segmentation
RUN_NAME = "train_modelm_1"               # e.g. "train", "train2", "train3", ...
AUTO_PICK_LATEST = False          # if True, ignores RUN_NAME and uses the latest train run
DO_SPEED_TEST = True              # benchmark inference speed on a small sample
SPEED_MAX_IMAGES = 30             # number of images to time (from test split)
SPEED_DEVICE = None               # None=auto; or "mps" | "cuda" | "cpu"
SAVE_REPORT_TO_FILE = True        # also write a text file next to the run
# ─────────────────────────────────────────────────────────────


def detect_device():
    if SPEED_DEVICE:
        return SPEED_DEVICE
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def find_run_dir():
    if AUTO_PICK_LATEST:
        # pick the latest folder named train*, by modification time
        cand = [p for p in RUNS_DIR.glob("train*") if p.is_dir()]
        if not cand:
            raise FileNotFoundError(f"No train* directories under {RUNS_DIR}")
        latest = max(cand, key=lambda p: p.stat().st_mtime)
        return latest
    else:
        rd = RUNS_DIR / RUN_NAME
        if not rd.exists():
            raise FileNotFoundError(f"Run directory not found: {rd}")
        return rd


def load_yaml_safe(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pretty_bytes(n_bytes: int) -> str:
    return f"{n_bytes/1_000_000:.2f} MB"


def list_images(folder: Path):
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])


def expand_split_dirs(data_yaml_path: Path, data_yaml: dict, key: str):
    """
    Resolve split entries (str or list) to absolute directories.
    Honors 'path:' in YAML (root) and YAML file's own folder if relative.
    """
    dirs = []
    root_from_yaml = data_yaml.get("path", None)
    if isinstance(data_yaml.get(key), list):
        items = data_yaml[key]
    else:
        items = [data_yaml.get(key)]

    base_dir = data_yaml_path.parent
    root_path = None
    if root_from_yaml:
        root_path = Path(root_from_yaml)
        if not root_path.is_absolute():
            root_path = (base_dir / root_path).resolve()

    for item in items:
        if item is None:
            continue
        p = Path(item)
        if not p.is_absolute():
            if root_path is not None:
                p = (root_path / p).resolve()
            else:
                p = (base_dir / p).resolve()
        dirs.append(p)
    return dirs


def count_images_and_labels(dirs):
    total_imgs = 0
    total_lbls = 0
    missing = 0
    orphans = 0
    for d in dirs:
        imgs = list_images(d)
        lbls = sorted(list(d.glob("*.txt")))
        total_imgs += len(imgs)
        total_lbls += len(lbls)
        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in lbls}
        missing += len(img_stems - lbl_stems)
        orphans += len(lbl_stems - img_stems)
    return total_imgs, total_lbls, missing, orphans


def detect_annotation_format(sample_label: Path) -> str:
    """
    Peek one .txt and infer whether it's YOLO seg with polygons.
    Returns a short description string.
    """
    if not sample_label.exists():
        return "No label file to inspect"
    with open(sample_label, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                # YOLO seg: cls cx cy w h x1 y1 x2 y2 ...
                try:
                    cls = int(parts[0])
                    floats = list(map(float, parts[1:]))
                except:
                    continue
                poly = floats[4:]
                if len(poly) >= 6 and len(poly) % 2 == 0 and all(0.0 <= v <= 1.0 for v in floats):
                    return "YOLOv8 segmentation (bbox + polygon, normalized)"
            # otherwise keep looking
    return "Unknown / not YOLO-seg (or empty)"


def read_results_csv(run_dir: Path):
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None, None
    df = pd.read_csv(csv_path)
    last = df.iloc[-1].to_dict()
    return df, last


def model_architecture_summary(model):
    """
    Build a compact layer table: idx, module class, params.
    """
    rows = []
    total_params = 0
    for idx, m in enumerate(model.modules()):
        # skip the root nn.Module (idx == 0)
        if idx == 0:
            continue
        p = sum(p.numel() for p in m.parameters(recurse=False))
        total_params += p
        rows.append((idx, m.__class__.__name__, p))
    return rows, total_params


def pick_test_dirs(data_yaml_path: Path, data_yaml: dict):
    # prefer 'test' if present, else 'val'/'valid'
    if data_yaml.get("test") is not None:
        return expand_split_dirs(data_yaml_path, data_yaml, "test")
    if data_yaml.get("val") is not None:
        return expand_split_dirs(data_yaml_path, data_yaml, "val")
    if data_yaml.get("valid") is not None:
        return expand_split_dirs(data_yaml_path, data_yaml, "valid")
    return []


def measure_inference_speed(yolo_model: YOLO, img_dirs, imgsz=640, device="cpu", max_images=30):
    # collect up to max_images
    images = []
    for d in img_dirs:
        images.extend(list_images(d))
        if len(images) >= max_images:
            break
    images = images[:max_images]
    if not images:
        return None, None, None

    # warmup one image
    try:
        yolo_model.predict(source=str(images[0]), imgsz=imgsz, device=device, verbose=False)
    except Exception:
        pass

    t0 = time.perf_counter()
    yolo_model.predict(source=[str(p) for p in images], imgsz=imgsz, device=device, stream=False, verbose=False)
    dt = time.perf_counter() - t0
    ips = len(images) / dt if dt > 0 else None
    ms_per_img = 1000.0 * dt / len(images) if len(images) > 0 else None
    return len(images), ips, ms_per_img


def main():
    run_dir = find_run_dir()
    print("=" * 80)
    print(f"RUN REPORT  ·  {run_dir}")
    print("=" * 80)

    # --- framework details
    device = detect_device()
    print("\n[Framework]")
    print(f"Python        : {sys.version.split()[0]}  ({platform.platform()})")
    print(f"Ultralytics   : {ultralytics.__version__}")
    print(f"Torch         : {torch.__version__}")
    print(f"Device        : {device}  "
          f"(MPS={getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()}, "
          f"CUDA={torch.cuda.is_available()})")

    # --- load args.yaml / hyp.yaml
    args_yaml_path = None
    for name in ("args.yaml", "opt.yaml"):
        p = run_dir / name
        if p.exists():
            args_yaml_path = p
            break
    args_yaml = load_yaml_safe(args_yaml_path) if args_yaml_path else {}
    hyp_yaml_path = run_dir / "hyp.yaml"
    hyp_yaml = load_yaml_safe(hyp_yaml_path) if hyp_yaml_path.exists() else {}

    # --- weights
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    weights_path = best_path if best_path.exists() else (last_path if last_path.exists() else None)
    model_size = pretty_bytes(weights_path.stat().st_size) if weights_path else "N/A"

    # --- model architecture
    print("\n[Model]")
    print(f"Weights       : {weights_path if weights_path else 'N/A'}  ({model_size})")
    print(f"Model family  : {args_yaml.get('model', 'unknown')}")
    print(f"Epochs/imgsz  : {args_yaml.get('epochs')} / {args_yaml.get('imgsz')}")
    print(f"Batch/optimizer/lr0/momentum/wd : {args_yaml.get('batch')} / "
          f"{args_yaml.get('optimizer', 'auto')} / {args_yaml.get('lr0')} / "
          f"{args_yaml.get('momentum')} / {args_yaml.get('weight_decay')}")
    if hyp_yaml:
        print(f"Hyp overrides : {', '.join(sorted(hyp_yaml.keys()))}")

    # load model and summarize
    arch_rows = []
    total_params = None
    try:
        yolo = YOLO(str(weights_path)) if weights_path else YOLO(args_yaml.get("model", "yolov8n-seg.pt"))
        rows, total_params = model_architecture_summary(yolo.model)
        arch_rows = rows[:]
    except Exception as e:
        print(f"Architecture  : (failed to introspect) {e}")

    if total_params is not None:
        print(f"Parameters    : {total_params:,d}")
        print("\n[Layers]")
        print("idx  module                              params")
        print("-" * 60)
        # print first ~40 lines, then ellipsis
        MAX_LINES = 60
        for idx, (i, name, p) in enumerate(arch_rows, 1):
            if idx > MAX_LINES:
                print("... (truncated)")
                break
            print(f"{i:>3}  {name:<35} {p:>10,d}")

    # --- dataset architecture
    data_yaml_path = None
    data_yaml = {}
    try:
        data_ref = args_yaml.get("data")
        data_yaml_path = Path(data_ref) if data_ref else None
        if data_yaml_path and not data_yaml_path.is_absolute():
            # if training was launched from project root, relative path should resolve
            data_yaml_path = (Path.cwd() / data_yaml_path).resolve()
        if data_yaml_path and data_yaml_path.exists():
            data_yaml = load_yaml_safe(data_yaml_path)
    except Exception:
        pass

    print("\n[Dataset]")
    if data_yaml:
        print(f"Data YAML     : {data_yaml_path}")
        # resolve splits
        for split_key in ("train", "val", "valid", "test"):
            dirs = expand_split_dirs(data_yaml_path, data_yaml, split_key) if data_yaml.get(split_key) is not None else []
            if not dirs:
                continue
            n_img, n_lbl, miss, orph = count_images_and_labels(dirs)
            print(f"{split_key:<6} dirs :")
            for d in dirs:
                print(f"  - {d}")
            print(f"    images={n_img}, labels={n_lbl}, missing_labels={miss}, labels_wo_images={orph}")
        # infer label format from one label file in 'train'
        train_dirs = expand_split_dirs(data_yaml_path, data_yaml, "train")
        sample_label = None
        for d in train_dirs:
            lbls = sorted(d.glob("*.txt"))
            if lbls:
                sample_label = lbls[0]
                break
        fmt = detect_annotation_format(sample_label) if sample_label else "No label to inspect"
        print(f"Annotation    : {fmt}")
        # class names
        names = data_yaml.get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
        if isinstance(names, list):
            print(f"Classes       : {len(names)} -> {names}")
    else:
        print("Data YAML     : N/A (could not load).")

    # --- training results
    print("\n[Training Results]")
    df, last = read_results_csv(run_dir)
    if df is None:
        print("results.csv   : not found")
    else:
        # show last-epoch metrics succinctly
        # pick common columns if present
        cols_of_interest = [c for c in df.columns if any(k in c.lower() for k in [
            "box_loss", "seg_loss", "cls_loss", "dfl_loss",
            "metrics", "map", "precision", "recall"
        ])]
        print(f"results.csv   : {len(df)} rows (showing last epoch)")
        for k in cols_of_interest:
            v = last.get(k, None)
            if v is None:
                continue
            if isinstance(v, float):
                print(f"  {k:<25} {v:.5f}")
            else:
                print(f"  {k:<25} {v}")

    # --- speed benchmark
    if DO_SPEED_TEST:
        test_dirs = pick_test_dirs(data_yaml_path, data_yaml) if data_yaml else []
        imgsz = args_yaml.get("imgsz", 640)
        if test_dirs:
            n, ips, ms = measure_inference_speed(yolo, test_dirs, imgsz=imgsz, device=detect_device(), max_images=SPEED_MAX_IMAGES)
            if n is None:
                print("\n[Speed] No test images found to measure.")
            else:
                print("\n[Speed]")
                print(f"Images tested : {n}")
                print(f"Throughput    : {ips:.2f} img/s")
                print(f"Latency       : {ms:.1f} ms/img  (imgsz={imgsz}, device={detect_device()})")
        else:
            print("\n[Speed] No test or val directories found in data YAML.")

    # --- save report to txt
    if SAVE_REPORT_TO_FILE:
        out_path = run_dir / "run_report.txt"
        # simple approach: re-run main printing to a buffer is heavy; instead ask user to copy console
        # but at least we write key pointers
        with open(out_path, "w") as f:
            f.write("Copy console output for full report. Key pointers:\n")
            f.write(f"Run dir: {run_dir}\n")
            f.write(f"Weights: {weights_path} ({model_size})\n")
            f.write(f"Data   : {data_yaml_path}\n")
        print(f"\n[Saved] Minimal pointers written to: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
