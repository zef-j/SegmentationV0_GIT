from pathlib import Path

# Adapte ce chemin si besoin
DATA_ROOT = Path("data/solar_coco")

def move_labels(split: str):
    split_dir = DATA_ROOT / split
    labels_dir = split_dir / "labels"

    if not labels_dir.exists():
        print(f"[{split}] Pas de dossier 'labels', rien à faire.")
        return

    txt_files = list(labels_dir.glob("*.txt"))
    print(f"[{split}] {len(txt_files)} fichiers .txt trouvés dans {labels_dir}")

    for lbl in txt_files:
        dest = split_dir / lbl.name
        if dest.exists():
            print(f"  ATTENTION: {dest} existe déjà, je ne l'écrase pas.")
            continue
        print(f"  Déplacement: {lbl.name} -> {dest}")
        lbl.rename(dest)

    # Optionnel : supprimer le dossier labels s'il est vide
    try:
        labels_dir.rmdir()
        print(f"[{split}] Dossier 'labels' supprimé (vide).")
    except OSError:
        print(f"[{split}] Dossier 'labels' non vide, je le laisse.")

if __name__ == "__main__":
    for sp in ["train", "valid", "test"]:
        move_labels(sp)
