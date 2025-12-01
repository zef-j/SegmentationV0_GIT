from pathlib import Path

ROOTS = [
    Path("data/solar_coco/train"),
    Path("data/solar_coco2/train"),
]

def check_root(root: Path, max_files=20):
    print(f"\n=== Vérification dans {root} ===")
    if not root.exists():
        print(f"⚠️ Dossier introuvable, on skip.")
        return

    txt_files = sorted(root.glob("*.txt"))
    if not txt_files:
        print("⚠️ Aucun .txt trouvé.")
        return

    ok_count = 0
    for i, lbl_path in enumerate(txt_files):
        if i >= max_files:
            break
        with open(lbl_path, "r") as f:
            for line_num, line in enumerate(f, start=1):
                parts = line.strip().split()
                if len(parts) < 7:
                    print(f"  ⚠️ {lbl_path} ligne {line_num}: moins de 7 valeurs -> {parts}")
                    continue

                try:
                    cls = int(parts[0])
                except ValueError:
                    print(f"  ⚠️ {lbl_path} ligne {line_num}: classe non entière -> {parts[0]}")
                    continue

                if cls != 0:
                    print(f"  ⚠️ {lbl_path} ligne {line_num}: classe != 0 -> {cls}")

                vals = []
                try:
                    vals = list(map(float, parts[1:]))
                except ValueError:
                    print(f"  ⚠️ {lbl_path} ligne {line_num}: valeurs non float -> {parts[1:]}")
                    continue

                # 4 premiers = cx, cy, w, h
                if len(vals) < 4:
                    print(f"  ⚠️ {lbl_path} ligne {line_num}: pas assez de valeurs après cls.")
                    continue

                cx, cy, w, h = vals[:4]
                poly = vals[4:]

                # Check [0,1]
                for v in vals:
                    if not (0.0 <= v <= 1.0):
                        print(f"  ⚠️ {lbl_path} ligne {line_num}: valeur {v} hors [0,1].")

                # Polygone: nombre pair (x,y)
                if len(poly) % 2 != 0:
                    print(
                        f"  ⚠️ {lbl_path} ligne {line_num}: "
                        f"nombre de coords polygone impair ({len(poly)})."
                    )

        ok_count += 1

    print(f"  Vérification terminée sur {ok_count} fichiers (max_files={max_files}).")


if __name__ == "__main__":
    for root in ROOTS:
        check_root(root)
    print("\nTerminé.")
