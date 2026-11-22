# Data Sources

All images are stored in Google Drive (`Final_Project/DataSet`), not in this repository.
Each image file in `classification/` is prefixed with its source name (e.g. `purvi_failure_v4__xyz.jpg`),
and `classification/manifest.csv` maps every file back to its original path.

## Image datasets

| Source name (prefix) | Dataset | Version | Original format | License | Classes we took (original → ours) |
|---|---|---|---|---|---|
| `roboflow_spaghetti_v3`* | [Spaghetti – aiot-innowork](https://universe.roboflow.com/aiot-innowork/spaghetti-mckhg) (Roboflow) | v3 | YOLOv8 detection | _CC BY 4.0_ | spaghetti, stringing, zits |
| `kaggle_defects` | _TODO: Kaggle link_ ("3d printing defects") | – | one folder per class | _TODO_ | ok → normal, spaghetti, bed_not_stick → detachment |
| `purvi_failure_v4` | [3d print failure detection – Purvi Rathore](https://universe.roboflow.com/purvi-rathore-5amqh/3d-print-failure-detection-efvsh/dataset/4) (Roboflow) | v4 (no augmentation) | YOLOv8 detection | CC BY 4.0 | spaghetti, stringing, warping |
| `fyp_defects_noaug` | [3D printing defects – fyp](https://universe.roboflow.com/fyp-ul2aa/3d-printing-defects-wptsr) (Roboflow), forked and regenerated without augmentation | fork v1 | YOLOv8 detection | CC BY 4.0 | stinging → stringing, poor first layer → poor_first_layer, over extrusion → over_extrusion, under extrusion → under_extrusion, spaghetti |

\* Imported before file prefixes were introduced: its files have no prefix, and its manifest is `manifest_roboflow_old.csv`.

## Processing applied to every source

1. **Duplicate removal** – perceptual hash (dHash, ≤ 5 of 64 bits different) against everything already imported (`scripts/dedupe_check.py`).
2. **Leakage check** – no image may appear in more than one of train / val / test.
3. **Class mapping** – source class names renamed to our class list (see README).
4. **Images with several classes** are skipped (classification needs one label per image).

## Candidate sources (not imported yet)

| Dataset | Why | License |
|---|---|---|
| [3D-printing errors 2 – Yawllen](https://universe.roboflow.com/yawllen-jectr/3d-printing-errors-2) (Roboflow) | Delamination, warping, under-extrusion | CC BY 4.0 |
| [3d printing warp](https://universe.roboflow.com/dev-vbetg/3d-printing-warp) (Roboflow) | Largest warping set found | not stated |
| [DCASE 2024 Task 2 – 3D printer sounds](https://zenodo.org/records/11259435) (Zenodo) | Normal / anomalous printer audio for the acoustic channel | CC BY-NC-SA 4.0 |
