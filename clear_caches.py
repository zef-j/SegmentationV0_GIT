from pathlib import Path

DATA_ROOT = Path("data/solar_coco")

for cache_file in DATA_ROOT.glob("*.cache"):
    print(f"Suppression de {cache_file}")
    cache_file.unlink()
