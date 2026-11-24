# Edge-Cloud Anomaly Detection for Additive Manufacturing

## Project Overview
This project is an autonomous, distributed AI system designed to detect physical failures in 3D printing processes (e.g., layer shifting, spaghetti failures) in real-time. It utilizes an Edge device (Raspberry Pi) for local acoustic processing and machine control via active G-code injection, paired with a Cloud server running a fine-tuned YOLO computer vision model.

## Team Members
- Lior Avital
- Matan Yehezkel
(Group 102)

## Advisors
- Nir Sweed (Advisor)
- Gal Fiebelman (Mentor)

## Setup Instructions
(Detailed setup instructions for the Edge client and Cloud server environments will be added in upcoming milestones.)

## Repository structure
| Folder | Contents |
|---|---|
| `notebooks/` | Data pipeline and model training (Google Colab) |
| `scripts/` | Dataset import and duplicate-detection tools |
| `cloud/` | FastAPI inference server |
| `edge/` | Raspberry Pi capture and printer control |
| `docs/` | Data sources and licenses |

## Data
Images are stored in Google Drive (`Final_Project/DataSet`), not in this repo. Sources and licenses: [`docs/data_sources.md`](docs/data_sources.md)

- 4 public sources (Roboflow ×3, Kaggle ×1), ~8,700 images, 9 classes
- Duplicates and train/test leakage removed with perceptual hashing (`scripts/dedupe_check.py`, results cached for fast re-runs)
- **`normal` comes from Kaggle only.** Images that the import script placed in `normal` from the Roboflow sources (purvi, fyp) turned out to be unlabeled failures (spaghetti, stringing, etc.), because every image without a bounding box was treated as normal. They were moved to `quarantine_normal/` on Drive (notebook cell 6.1).

## Classes
| Class | Meaning | System action |
|---|---|---|
| `normal` | print progressing correctly | continue |
| `spaghetti` | filament extruded into the air | stop |
| `detachment` | print came off the bed | stop |
| `warping` | corners lifting (early sign of detachment) | stop |
| `poor_first_layer` | bad first-layer adhesion | stop |
| `stringing` | thin strands between parts | log |
| `zits` | small blobs on the surface | log |
| `over_extrusion` | too much material | log |
| `under_extrusion` | too little material, gaps | log |

## How to run
1. Open `notebooks/data_pipeline_and_baseline.ipynb` in Colab (GPU runtime)
2. Run sections 1–8 to build the dataset, 9 to train, 10 to evaluate
   - Cell 4.1 can skip the duplicate check when the cache is up to date
   - Cell 6.1 moves the unlabeled Roboflow "normal" images to quarantine
   - Sections 7, 8 and 10 run without a GPU
3. Trained weights: `runs/cls9_v2/weights/best.pt` · evaluation results: `runs/evaluation` (Drive)

## Results
| Date | Model | Classes | Test accuracy | Macro F1 | Stop recall | False-stop rate |
|---|---|---|---|---|---|---|
| 2026-11-24 | yolov8n-cls (`cls9_v2`, mapped to groups) | 9 | 92.3% | 88.4% | 94.6% | 7.7% |
| 2026-11-24 | yolov8n-cls (`cls3_v2`) | 3 | 92.0% | 89.5% | 92.3% | 7.7% |

Test set: ~680 images, ~30 normal (single source, Kaggle). Per-frame metrics; the deployed system will require several consecutive "stop" frames before halting.

**Main model: `cls9_v2`.** Both models perform almost the same, and the 9-class model also tells which failure occurred.

Cleaning the `normal` class lowered the false-stop rate from ~28% (val, before cleaning) to 7.7%. The earlier `cls9_baseline` run is not valid: it was trained before cleaning and before the test set was re-split.

### Known limitations
- The false-stop rate is based on ~30 normal test images (2–3 errors), so it is indicative rather than precise.
- All normal images come from one source, so the model may partly learn that source's background, angle and lighting. Images from our own printer are needed to confirm the results.
- 7.7% per frame is still too high for real time (one frame every 5 s ≈ one false stop per minute). The planned fix is to halt only after several consecutive "stop" frames (e.g. 3 of 4).

## Next steps
1. Record normal prints and induced failures on our own printer (real-world test set)
2. Consecutive-frame stop logic, measured on real recordings
3. Raspberry Pi capture script (frame every 5 s + audio)
4. FastAPI `/predict` endpoint serving `cls9_v2`
