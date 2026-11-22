"""
Convert a YOLO detection dataset (e.g. a Roboflow "YOLOv8" export) into an
image-classification dataset (one folder per class), in the layout that
Ultralytics YOLO classification expects:

    out_dir/
      train/<class>/*.jpg
      val/<class>/*.jpg
      test/<class>/*.jpg

Rules for each image:
  * no boxes in its label file      -> "normal"
  * boxes of exactly one class      -> that class
  * boxes of several classes        -> depends on --multi:
        "skip"     (default) leave the image out
        "first"    use the class that appears first in data.yaml
        "majority" use the class with the most boxes

Every copied file is prefixed with --source, and one row per image is appended
to out_dir/manifest.csv (same columns as import_class_folders.py), so images
from several datasets can live side by side and be traced back.

Usage:
    python yolo_det_to_cls.py SRC_DIR OUT_DIR --source roboflow_spaghetti_v3
                              [--multi skip|first|majority]
                              [--normal-name normal] [--move]

SRC_DIR is the folder that contains data.yaml (Roboflow's dataset.location).
"""

import argparse
import csv
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Roboflow uses "valid", Ultralytics classification expects "val".
SPLIT_NAMES = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def load_class_names(src: Path) -> list[str]:
    with open(src / "data.yaml") as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, dict):  # {0: "a", 1: "b"}
        names = [names[k] for k in sorted(names)]
    return [str(n) for n in names]


def classes_in_label(label_file: Path) -> Counter:
    counts = Counter()
    if not label_file.exists():
        return counts
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if parts:
            counts[int(float(parts[0]))] += 1
    return counts


def choose_class(counts: Counter, names, normal_name, multi):
    if not counts:
        return normal_name
    if len(counts) == 1:
        return names[next(iter(counts))]
    if multi == "first":
        return names[min(counts)]
    if multi == "majority":
        return names[counts.most_common(1)[0][0]]
    return None  # skip


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--source", required=True, help="short name of this dataset")
    ap.add_argument("--map", action="append", default=[],
                    help="rename a class: old=new (repeatable)")
    ap.add_argument("--exclude", type=Path,
                    help="text file of image paths to skip (from dedupe_check.py)")
    ap.add_argument("--multi", choices=["skip", "first", "majority"], default="skip")
    ap.add_argument("--normal-name", default="normal")
    ap.add_argument("--move", action="store_true",
                    help="move files instead of copying (saves space, destroys source)")
    args = ap.parse_args()

    names = load_class_names(args.src)
    rename = {}
    for it in args.map:
        old, _, new = it.partition("=")
        if not new or old not in names:
            sys.exit(f"--map '{it}': expected old=new with old one of {names}")
        rename[old] = new
    names = [rename.get(n, n) for n in names]
    excluded = set()
    if args.exclude:
        excluded = {str(Path(l.strip()).resolve()) for l in args.exclude.read_text().splitlines() if l.strip()}
    n_excluded = 0
    if args.normal_name in names:
        sys.exit(f"'{args.normal_name}' is already a class name; pick another --normal-name")

    op = shutil.move if args.move else shutil.copy2
    stats = defaultdict(Counter)
    skipped = 0
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.csv"
    if manifest.exists():
        with open(manifest, newline="") as f:
            if any(r.get("source_dataset") == args.source for r in csv.DictReader(f)):
                sys.exit(f"'{args.source}' is already in {manifest}.")
    write_header = not manifest.exists()

    with open(manifest, "a", newline="") as mf:
        writer = csv.writer(mf)
        if write_header:
            writer.writerow(["file", "split", "class", "source_dataset", "original_path"])

        for split_dir in sorted(p for p in args.src.iterdir() if p.is_dir()):
            split = SPLIT_NAMES.get(split_dir.name)
            img_dir = split_dir / "images"
            if split is None or not img_dir.is_dir():
                continue
            lbl_dir = split_dir / "labels"

            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in IMG_EXT:
                    continue
                if str(img.resolve()) in excluded:
                    n_excluded += 1
                    continue
                counts = classes_in_label(lbl_dir / f"{img.stem}.txt")
                cls = choose_class(counts, names, args.normal_name, args.multi)
                if cls is None:
                    skipped += 1
                    continue
                dst = args.out / split / cls
                dst.mkdir(parents=True, exist_ok=True)
                target = dst / f"{args.source}__{img.name}"
                op(str(img), str(target))
                writer.writerow([str(target.relative_to(args.out)), split, cls,
                                 args.source, str(img.relative_to(args.src))])
                stats[split][cls] += 1

    all_classes = sorted({c for s in stats.values() for c in s})
    print(f"\nWrote classification dataset to {args.out}\n")
    w = max([14] + [len(c) + 2 for c in all_classes])
    print(f"{'split':<8}" + "".join(f"{c:>{w}}" for c in all_classes))
    for split in ("train", "val", "test"):
        if split in stats:
            print(f"{split:<8}" + "".join(f"{stats[split][c]:>{w}}" for c in all_classes))
    if n_excluded:
        print(f"\nExcluded {n_excluded} duplicate images (--exclude).")
    if skipped:
        print(f"\nSkipped {skipped} images containing several classes (see --multi).")
    if len(all_classes) < 2:
        print("\nWARNING: only one class was produced. A classifier needs at least two "
              "(e.g. add 'normal' images with no failures).")


if __name__ == "__main__":
    main()
