# Edge-Cloud Anomaly Detection for Additive Manufacturing

## Project Overview
This project is an autonomous, distributed AI system designed to detect physical failures in 3D printing processes (e.g., layer shifting, spaghetti failures) in real-time. It utilizes an Edge device (Raspberry Pi) for local acoustic processing and machine control via active G-code injection, paired with a Cloud server running a fine-tuned YOLO computer vision model.

## Team Members
* Lior Avital
* Matan Yehezkel (Group 102)

## Advisors
* Nir Sweed (Advisor)
* Gal Fiebelman (Mentor)

## Setup Instructions
*(Detailed setup instructions for the Edge client and Cloud server environments will be added in upcoming milestones).*
## Repository structure
| Folder | Contents |
|---|---|
| `notebooks/` | Data pipeline and model training (Google Colab) |
| `scripts/` | Dataset import and duplicate-detection tools |
| `cloud/` | FastAPI inference server |
| `edge/` | Raspberry Pi capture and printer control |
| `docs/` | Data sources and licenses |

## Data
Images are stored in Google Drive (`Final_Project/DataSet`), not in this repo.
Sources and licenses: [docs/data_sources.md](docs/data_sources.md)

### Classes
| Class | Meaning | System action |
|---|---|---|
| normal | print progressing correctly | continue |
| spaghetti | filament extruded into the air | stop |
| detachment | print came off the bed | stop |
| warping | corners lifting (early sign of detachment) | stop |
| poor_first_layer | bad first-layer adhesion | stop |
| stringing | thin strands between parts | log |
| zits | small blobs on the surface | log |
| over_extrusion | too much material | log |
| under_extrusion | too little material, gaps | log |

## How to run
1. Open `notebooks/data_pipeline_and_baseline.ipynb` in Colab (GPU runtime)
2. Run sections 1–8 to build the dataset, 9 to train, 10 to evaluate

## Results
| Date | Model | Classes | Test accuracy | Stop recall | False-stop rate |
|---|---|---|---|---|---|
| 2026-11-22 | yolov8n-cls | 9 | – | – | – |
| 2026-11-22 | yolov8n-cls | 3 | – | – | – |
