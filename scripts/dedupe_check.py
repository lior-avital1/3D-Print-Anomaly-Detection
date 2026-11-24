"""
Find duplicate / near-duplicate images with a perceptual hash (dHash).
Robust to resizing, re-compression and small brightness changes, so it also
catches the same photo re-exported by a different Roboflow/Kaggle user.
Two modes:

1) New source vs. what we already have  (run BEFORE every import)
       python dedupe_check.py REF_DIR --new NEW_DIR --out dupes.txt
   Writes the paths of images in NEW_DIR that already exist in REF_DIR.
   Pass that file to the importer with --exclude dupes.txt.

2) Inside an existing dataset  (finds train/val/test leakage)
       python dedupe_check.py CLS_DIR --out internal_dupes.txt
   Lists groups of near-identical images; flags groups that span splits.

Options:
    --threshold 5   max differing bits out of 64 (0 = identical, 5 = default)

Only needs Pillow and numpy.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(path: Path) -> int | None:
    try:
        with Image.open(path) as im:
            g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    except Exception as e:
        print(f"  ! cannot read {path}: {e}")
        return None
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hash_dir(root: Path):
    paths, hashes = [], []
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)
    print(f"Hashing {len(files)} images in {root} ...")
    for i, p in enumerate(files, 1):
        h = dhash(p)
        if h is not None:
            paths.append(p)
            hashes.append(h)
        if i % 1000 == 0:
            print(f"  {i}/{len(files)}")
    return paths, np.array(hashes, dtype=np.uint64)


def distances(h: np.uint64, arr: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(arr, h).view(np.uint8).reshape(-1, 8)
    return POPCOUNT[x].sum(axis=1)


def split_of(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    return parts[0] if parts and parts[0] in {"train", "val", "test"} else "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", type=Path, help="dataset we already have (e.g. classification/)")
    ap.add_argument("--new", type=Path, help="new source to check against ref")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=int, default=5)
    args = ap.parse_args()

    ref_paths, ref_h = hash_dir(args.ref)

    if args.new:
        new_paths, new_h = hash_dir(args.new)
        dupes = []
        for p, h in zip(new_paths, new_h):
            d = distances(h, ref_h)
            j = int(d.argmin()) if len(d) else -1
            if j >= 0 and d[j] <= args.threshold:
                dupes.append((p, ref_paths[j], int(d[j])))
        args.out.write_text("".join(f"{p}\n" for p, _, _ in dupes))
        print(f"\n{len(dupes)} of {len(new_paths)} new images already exist "
              f"({100 * len(dupes) / max(len(new_paths), 1):.1f}%).")
        for p, r, d in dupes[:10]:
            print(f"  {p.name}  ~  {r.relative_to(args.ref)}  (diff {d})")
        if len(dupes) > 10:
            print("  ...")
        print(f"List written to {args.out} -> pass it to the importer with --exclude")
        return

    # internal mode: group near-identical images
    seen = np.zeros(len(ref_h), dtype=bool)
    groups = []
    for i in range(len(ref_h)):
        if seen[i]:
            continue
        idx = np.nonzero(distances(ref_h[i], ref_h) <= args.threshold)[0]
        idx = [k for k in idx if not seen[k]]
        seen[idx] = True
        if len(idx) > 1:
            groups.append(idx)

    leaks = [g for g in groups if len({split_of(ref_paths[k], args.ref) for k in g}) > 1]
    with open(args.out, "w") as f:
        for g in groups:
            splits = {split_of(ref_paths[k], args.ref) for k in g}
            f.write(("LEAK " if len(splits) > 1 else "") + " | ".join(
                str(ref_paths[k].relative_to(args.ref)) for k in g) + "\n")
    extra = sum(len(g) - 1 for g in groups)
    print(f"\n{len(groups)} groups of duplicates ({extra} redundant images).")
    print(f"{len(leaks)} groups span more than one split (train/val/test leakage).")
    print(f"Details in {args.out}")


if __name__ == "__main__":
    main()
