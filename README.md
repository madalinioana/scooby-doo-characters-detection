# Scooby-Doo Characters Detection

Face detection and character classification project for Scooby-Doo characters using deep learning.

## Required Libraries

```
numpy==2.4.1
opencv-python==4.10.0.84
torch==2.5.1
torchvision==0.20.1
matplotlib==3.10.8
Pillow==12.1.0
ultralytics==8.3.41
PyYAML==6.0.2
```

## How to Run

**Important:** The code is designed to run on **Kaggle Notebook** with GPU enabled.

### Main Script

Script: `scripts/cnn.py`

This script uses a custom detector based on ResNet18 for face detection and character classification.

**On Kaggle:**

1. Upload the dataset as input to the notebook
2. Run:

```python
python scripts/cnn.py
```

**Output:**

- Output directory is `/kaggle/working/` (on Kaggle)
- Results are saved in:
  - `cnn_results/task1/` - detections for all faces
  - `cnn_results/task2/` - separate detections for each character (daphne, fred, shaggy, velma)

Generated files:

- Task 1: `detections_all_faces.npy`, `scores_all_faces.npy`, `file_names_all_faces.npy`
- Task 2: `detections_{character}.npy`, `scores_{character}.npy`, `file_names_{character}.npy` for each character

### Bonus: YOLO Implementation

Script: `scripts/yolo.py`

Alternative implementation using YOLOv8 for character detection.

**On Kaggle:**

```python
python scripts/yolo.py
```

Results are saved in `yolo_results/task1/` and `yolo_results/task2/`.

## Project Structure

```
.
├── scripts/
│   ├── cnn.py              # Main script with custom model
│   └── yolo.py             # YOLO implementation (bonus)
├── antrenare/              # Training data
├── validare/               # Validation data
├── cnn_results/            # Results from CNN model
│   ├── task1/             # All faces detections
│   └── task2/             # Per-character detections
├── yolo_results/           # Results from YOLO model
└── evaluare/               # Evaluation scripts
    └── cod_evaluare/
        └── evalueaza_solutie.py
```
