"""Leak-free train/val/test split for LeafDoc (tomato, potato, pepper)."""
import hashlib
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path.home() / "Downloads" / "archive (8)" / "PlantVillage"
OUT = Path("data/splits")
MAX_PER_CLASS = 300
SEED = 42
EXTS = {".jpg", ".jpeg", ".png"}

random.seed(SEED)

rows = []
# "PlantVillage"
class_dirs = sorted(d for d in ROOT.iterdir() if d.is_dir() and d.name != "PlantVillage")

for d in class_dirs:
    files = [p for p in d.iterdir() if p.suffix.lower() in EXTS]
    random.shuffle(files)
    seen, kept = set(), 0
    for p in files:
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in seen:          # exact duplicate, skip
            continue
        seen.add(h)
        rows.append({"path": str(p), "label": d.name, "hash": h})
        kept += 1
        if kept >= MAX_PER_CLASS:
            break
    print(f"{d.name}: {kept} images")

df = pd.DataFrame(rows)
print("Total:", len(df), "| classes:", df.label.nunique())

train, temp = train_test_split(df, test_size=0.3, stratify=df.label, random_state=SEED)
val, test = train_test_split(temp, test_size=0.5, stratify=temp.label, random_state=SEED)

# Leakage check (acceptance criteria)
assert not set(train.hash) & set(val.hash)
assert not set(train.hash) & set(test.hash)
assert not set(val.hash) & set(test.hash)
print("Leakage check passed")

OUT.mkdir(parents=True, exist_ok=True)
for name, part in [("train", train), ("val", val), ("test", test)]:
    part.to_csv(OUT / f"{name}.csv", index=False)
    print(name, len(part))