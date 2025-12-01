from pathlib import Path
import json
import argparse


def list_categories(data_root: Path):
    """Scan train/valid/test COCO files and list all category ids/names."""
    cat_id_to_name = {}
    for split in ["train", "valid", "test"]:
        coco_path = data_root / split / "_annotations.coco.json"
        if not coco_path.exists():
            print(f"[{split}] Pas de fichier {coco_path}, on passe.")
            continue
        with open(coco_path, "r") as f:
            coco = json.load(f)
        for c in coco.get("categories", []):
            cat_id_to_name[c["id"]] = c["name"]
    if not cat_id_to_name:
        print("Aucune catégorie trouvée. Vérifie le chemin et les fichiers COCO.")
        return
    print("Catégories trouvées (id: name) :")
    for cid, name in sorted(cat_id_to_name.items()):
        print(f"  {cid}: {name}")


def convert_split(data_root: Path, split: str, keep_names_lower: set):
    """Convertit un split COCO en labels YOLOv8 segmentation.

    - data_root/split/_annotations.coco.json : annotations COCO
    - data_root/split/*.jpg|png               : images
    - Sortie : data_root/split/*.txt          : labels YOLO (bbox + polygon)
    """
    split_dir = data_root / split
    coco_path = split_dir / "_annotations.coco.json"
    if not coco_path.exists():
        print(f"[{split}] Pas de fichier {coco_path}, split ignoré.")
        return

    with open(coco_path, "r") as f:
        coco = json.load(f)

    id2img = {img["id"]: img for img in coco.get("images", [])}
    id2cat = {c["id"]: c["name"] for c in coco.get("categories", [])}

    img_to_lines = {}

    for ann in coco.get("annotations", []):
        img = id2img.get(ann["image_id"])
        if img is None:
            continue

        cat_name = id2cat.get(ann["category_id"], "")
        if keep_names_lower and cat_name.lower() not in keep_names_lower:
            # on ignore les annotations qui ne sont pas des panneaux
            continue

        w, h = img["width"], img["height"]

        # bbox COCO -> centre + taille normalisés
        x, y, bw, bh = ann["bbox"]
        xc = (x + bw / 2) / w
        yc = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h

        segs = ann.get("segmentation", [])
        if not segs:
            continue

        # on prend le premier polygone (Roboflow fournit généralement un seul)
        poly = segs[0]
        if len(poly) < 6:
            continue  # au moins 3 points (x,y)

        xs = poly[0::2]
        ys = poly[1::2]
        norm_coords = []
        for px, py in zip(xs, ys):
            norm_coords.append(px / w)
            norm_coords.append(py / h)

        # On force tout dans la classe 0 ("panel")
        cls = 0

        line = " ".join(
            [str(cls),
             f"{xc:.6f}", f"{yc:.6f}", f"{nw:.6f}", f"{nh:.6f}"]
            + [f"{v:.6f}" for v in norm_coords]
        )

        img_to_lines.setdefault(img["file_name"], []).append(line)

    # Écriture des fichiers .txt (un par image avec au moins un panneau)
    n_labels = 0
    for fname, lines in img_to_lines.items():
        stem = Path(fname).stem
        label_path = split_dir / f"{stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
        n_labels += 1

    print(
        f"[{split}] {len(coco.get('images', []))} images, "
        f"{len(coco.get('annotations', []))} annotations COCO, "
        f"{n_labels} fichiers labels YOLO créés (panneaux uniquement)."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convertir un dataset COCO Segmentation (Roboflow) vers YOLOv8 segmentation, "
                    "en ne gardant que les panneaux."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Chemin vers la racine du dataset (contenant train/, valid/, test/).",
    )
    parser.add_argument(
        "--panel-cats",
        type=str,
        default=None,
        help=(
            "Liste de noms de catégories à garder comme 'panel', séparés par des virgules. "
            "Exemple: --panel-cats 'solar,clean_solar,good_panel,defective_panel'. "
            "Si non fourni, toutes les catégories seront conservées comme panneaux."
        ),
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Lister uniquement les catégories trouvées dans les fichiers COCO, puis quitter.",
    )

    args = parser.parse_args()
    data_root = Path(args.root)

    if not data_root.exists():
        raise SystemExit(f"Chemin racine introuvable: {data_root}")

    if args.inspect:
        list_categories(data_root)
        return

    if args.panel_cats is None:
        # On garde toutes les catégories
        keep_names_lower = set()  # set vide => aucune filtration par nom
        print(
            "Aucun --panel-cats fourni : toutes les catégories COCO seront converties "
            "et mappées sur la classe YOLO 'panel' (id 0)."
        )
    else:
        keep = [n.strip() for n in args.panel_cats.split(",") if n.strip()]
        keep_names_lower = {n.lower() for n in keep}
        print("Catégories conservées comme panneaux (comparaison insensible à la casse) :")
        for n in sorted(keep_names_lower):
            print(f"  - {n}")

    for split in ["train", "valid", "test"]:
        convert_split(data_root, split, keep_names_lower)


if __name__ == "__main__":
    main()


