"""
Import an image-classification dataset that is organised as one folder per
class (no train/val/test split), e.g. a Kaggle download like:

    src/
      ok/*.jpg
      spaghetti/*.jpg
      bed_not_stick/*.jpg

into the shared classification dataset:

    out/
      train/<class>/  val/<class>/  test/<class>/
      manifest.csv

What it does:
  * renames classes with --map (e.g. ok=normal) so all sources share names
  * splits each class into train/val/test (default 80/10/10, fixed seed)
  * prefixes every file with --source so names from different datasets never
    collide, and so you can tell where each image came from
  * appends one row per image to out/manifest.csv
  * refuses to import the same --source twice (no accidental duplicates)

Usage:
    python import_class_folders.py SRC OUT --source kaggle_defects \
        --map ok=normal --map bed_not_stick=detachment
Options:
    --split 0.8 0.1 0.1   train/val/test fractions
    --seed 42             random seed for the split
    --dry-run             only print what would happen
"""

import argparse
import csv
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def parse_map(items):
    m = {}
    for it in items:
        if "=" not in it:
            sys.exit(f"--map expects old=new, got '{it}'")
        old, new = it.split("=", 1)
        m[old.strip()] = new.strip()
    return m


def already_imported(manifest: Path, source: str) -> bool:
    if not manifest.exists():
        return False
    with open(manifest, newline="") as f:
        return any(row.get("source_dataset") == source for row in csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--source", required=True, help="short name of this dataset, e.g. kaggle_defects")
    ap.add_argument("--map", action="append", default=[], help="rename a class: old=new")
    ap.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude", type=Path,
                    help="text file of image paths to skip (from dedupe_check.py)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if abs(sum(args.split) - 1) > 1e-6:
        sys.exit("--split fractions must sum to 1")
    rename = parse_map(args.map)
    manifest = args.out / "manifest.csv"
    if already_imported(manifest, args.source):
        sys.exit(f"'{args.source}' is already in {manifest}. Use a different --source "
                 "or remove the old rows/files first.")

    class_dirs = sorted(d for d in args.src.iterdir() if d.is_dir())
    if not class_dirs:
        sys.exit(f"No class folders found in {args.src}")
    unknown = set(rename) - {d.name for d in class_dirs}
    if unknown:
        sys.exit(f"--map refers to folders that don't exist: {sorted(unknown)}")

    excluded = set()
    if args.exclude:
        excluded = {str(Path(l.strip()).resolve()) for l in args.exclude.read_text().splitlines() if l.strip()}
    rng = random.Random(args.seed)
    plan = []  # (src_path, split, class)
    for d in class_dirs:
        imgs = sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT
                      and str(p.resolve()) not in excluded)
        rng.shuffle(imgs)
        n = len(imgs)
        n_train = round(n * args.split[0])
        n_val = round(n * args.split[1])
        cls = rename.get(d.name, d.name)
        for i, p in enumerate(imgs):
            split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
            plan.append((p, split, cls))

    stats = defaultdict(Counter)
    for _, split, cls in plan:
        stats[split][cls] += 1
    classes = sorted({c for _, _, c in plan})
    print(f"\nSource '{args.source}': {len(plan)} images\n")
    w = max([14] + [len(c) + 2 for c in classes])
    print(f"{'split':<8}" + "".join(f"{c:>{w}}" for c in classes))
    for s in SPLITS:
        print(f"{s:<8}" + "".join(f"{stats[s][c]:>{w}}" for c in classes))

    existing = set()
    for s in SPLITS:
        if (args.out / s).is_dir():
            existing |= {d.name for d in (args.out / s).iterdir() if d.is_dir()}
    new_classes = set(classes) - existing
    if existing and new_classes:
        print(f"\nNOTE: new classes not in the existing dataset: {sorted(new_classes)}")

    if args.dry_run:
        print("\n(dry run — nothing copied)")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    write_header = not manifest.exists()
    with open(manifest, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["file", "split", "class", "source_dataset", "original_path"])
        for p, split, cls in plan:
            dst_dir = args.out / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{args.source}__{p.name}"
            k = 1
            while dst.exists():  # same file name in two sub-folders
                dst = dst_dir / f"{args.source}__{p.stem}_{k}{p.suffix}"
                k += 1
            shutil.copy2(p, dst)
            w.writerow([str(dst.relative_to(args.out)), split, cls, args.source,
                        str(p.relative_to(args.src))])
    print(f"\nCopied into {args.out} and appended to {manifest.name}")


if __name__ == "__main__":
    main()
