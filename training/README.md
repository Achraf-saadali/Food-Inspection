# Food Detection Training Experiments

This directory contains the complete training record for the two food-detection experiments used by the Food Inspection system. **Experiment 1** is the original YOLOv9c baseline. **Experiment 2** is the later YOLO11m experiment with dataset auditing, class-taxonomy cleanup, explicit optimization settings, resumable checkpointing, and a longer training schedule.

The repository keeps both experiments side by side so that the original baseline remains reproducible and auditable while the second experiment is available as the current preferred detector. The runtime application uses the Experiment 2 checkpoint copied to `models/best.pt`.

## Directory layout

```text
training/
├── README.md
├── data/
│   └── lvis_fruits_61class/
│       └── data_local.yaml                 # processed 61-class training manifest
├── notebooks/
│   └── food_detection.ipynb                # canonical Experiment 2 pipeline
├── experiments/
│   ├── experiment_1_yolov9c/
│   │   ├── food_detection_first_experiment.ipynb
│   │   └── weights/
│   │       ├── best.pt
│   │       └── last.pt
│   └── experiment_2_yolo11m/
│       ├── food_detection_second_experiment.ipynb
│       └── weights/
│           ├── best.pt
│           └── last.pt
└── runs/
    ├── train4/                             # original Experiment 1 run artifacts
    └── lvis_fruits_yolo11m_80_v1/           # Experiment 2 metrics and manifests
```

The large `.pt` files are tracked with **Git LFS**. The raw image dataset is intentionally not committed because it is large and is already available through the sources described below.

## 1. Dataset sources and structure

### 1.1 Experiment 1 source

Experiment 1 downloads the LVIS Fruits and Vegetables dataset through KaggleHub using the dataset identifier `henningheyen/lvis-fruits-and-vegetables-dataset`. The public dataset page is available at [Kaggle: LVIS Fruits and Vegetables Dataset][1]. The original notebook then trains against a separately specified YAML path:

```text
/kaggle/input/newdata3/data (5).yaml
```

This is a weakness of the first experiment: the notebook downloads one dataset but trains from a hard-coded path whose manifest is not stored beside the notebook. The original training log reports 6,721 training images and 1,500 validation images, with 33,695 validation instances. The raw image and label tree is not committed here.

The expected YOLO detection layout is conceptually:

```text
LVIS_Fruits_And_Vegetables/
├── images/
│   ├── train/...
│   ├── val/...
│   └── test/...
├── labels/
│   ├── train/...
│   ├── val/...
│   └── test/...
└── *.yaml
```

Each label file is a text file containing one object per line:

```text
<class_id> <x_center> <y_center> <width> <height>
```

The four coordinates are normalized to the interval `[0, 1]`. The class ID is an integer index into the `names` list in the dataset YAML.

### 1.2 Experiment 2 source

Experiment 2 uses the Google Drive folder supplied for this project: [FoodDetection Google Drive folder][2]. The connected account used for the folder is `saadaliachraf06@gmail.com`. The Drive folder contains the source dataset, processed dataset, training runs, recovery checkpoints, exports, and experiment artifacts.

The second notebook mounts Drive and reads the source YOLO dataset from:

```text
/content/drive/MyDrive/LVIS_Fruits_And_Vegetables
```

Before training, it copies the source folder to writable local storage:

```text
/content/lvis_fruits_yolo
```

This local copy is important because training caches and generated metadata should not be written into the original Drive source tree. The notebook auto-detects the dataset YAML unless `DATA_YAML_HINT` is set explicitly.

The processed dataset is versioned as:

```text
lvis_fruits_61class_drop30_35_noempty_v1
```

Its effective structure is:

```text
/content/lvis_fruits_61class/
├── images/
│   ├── train/...
│   └── val/...
├── labels/
│   ├── train/...
│   └── val/...
└── data_local.yaml
```

The committed processed manifest is `training/data/lvis_fruits_61class/data_local.yaml`. It declares `nc: 61` and a contiguous class index range from `0` through `60`.

The processed dataset audit records 4,707 training images, 1,532 validation images, 119,422 training instances, and 33,695 validation instances. These counts are recorded in `training/runs/lvis_fruits_yolo11m_80_v1/dataset_audit.json` and `training/runs/lvis_fruits_yolo11m_80_v1/training_config.json`.

## 2. Class taxonomy cleanup and annotation reindexing

The source taxonomy contains 63 classes and has case-only duplicate concepts:

| Original ID | Original name | Decision |
|---:|---|---|
| 30 | `Strawberry` | Drop |
| 35 | `Tomato` | Drop |
| 57 | `strawberry` | Retain |
| 59 | `tomato` | Retain |

Experiment 2 does not merely delete those names from the YAML. It performs a complete label transformation. First, it creates the retained-ID sequence and an old-to-new mapping:

```python
kept_old_ids = [old_id for old_id in range(NC) if old_id not in DROP_CLASS_IDS]
old_to_new = {old_id: new_id for new_id, old_id in enumerate(kept_old_ids)}
new_class_names = [class_names[old_id] for old_id in kept_old_ids]
```

Then every retained annotation row is rewritten using the new class ID while preserving its bounding-box coordinates:

```python
new_class_id = old_to_new[old_class_id]
cleaned_rows.append(' '.join([str(new_class_id), *fields[1:]]))
```

The rewritten labels are saved in the processed `labels/train` and `labels/val` directories, and the processed YAML uses the matching 61-name list. This prevents the class-index misalignment that would occur if the YAML were reduced without rewriting the label files.

Images containing either dropped class are removed in full. This is deliberate: keeping an image in which a visible Strawberry or Tomato has had its annotation deleted would create a false-negative target. Images with missing labels or no remaining annotations are also removed. The notebook checks that the new class names are unique case-insensitively and that no image occurs in both the train and validation sets.

The Drive artifact `class_id_map.csv` is the authoritative mapping table. For example, original ID 36 becomes new ID 34 because IDs 30 and 35 were removed earlier in the sequence; original ID 57 becomes new ID 55; original ID 59 becomes new ID 57; and original ID 62 becomes new ID 60.

## 3. Experiment 1: YOLOv9c baseline

### 3.1 Data ingestion and preparation

The first notebook installs `roboflow` and `ultralytics`, imports the required libraries, downloads the LVIS dataset with KaggleHub, and initializes a pretrained YOLOv9c model. It does not implement a formal dataset audit, explicit class-taxonomy validation, missing-label policy, duplicate-name policy, overlap check, or versioned processed dataset.

The training invocation is:

```python
model = YOLO('yolov9c.pt')
results = model.train(
    data='/kaggle/input/newdata3/data (5).yaml',
    epochs=30,
    imgsz=640,
    batch=8,
    lr0=0.001,
)
```

Ultralytics supplied the standard detection augmentations shown in the recorded `args.yaml`. The run log also shows an important implementation detail: `optimizer=auto` ignored the requested `lr0=0.001` and selected AdamW with an effective learning rate of approximately `0.000149`. Thus, the code request and the actual optimizer configuration are not identical.

### 3.2 Model and training parameters

| Parameter | Recorded value | Engineering meaning |
|---|---:|---|
| Model | `yolov9c.pt` | Pretrained YOLOv9 compact model used as the baseline |
| Task | `detect` | Bounding-box object detection |
| Epochs | 30 | Short baseline schedule |
| Input size | 640 | Images are resized/trained at 640 px square resolution |
| Batch size | 8 | Fixed batch size; less portable across GPU memory configurations |
| Optimizer | `auto` | Ultralytics selected AdamW rather than using a fully explicit optimizer configuration |
| Requested initial LR | 0.001 | Ignored by optimizer auto-selection in the recorded run |
| Effective optimizer LR | 0.000149 | Recorded AdamW learning rate |
| Momentum | 0.9 effective | AdamW momentum-like beta setting in the recorded run |
| Weight decay | 0.0005 | Regularization against excessive parameter growth |
| Warmup | 3 epochs | Gradual start to stabilize early optimization |
| Seed | 0 | Deterministic seed setting |
| Workers | 8 recorded, 2 observed in log | Data-loader configuration was not fully stable across the run environment |
| Validation | `split: val` | Validation was run during training |
| AMP | Enabled | Mixed precision for lower memory use and faster GPU execution |
| Mosaic | 1.0 | Standard YOLO mosaic augmentation enabled |
| Auto augmentation | `randaugment` | Additional appearance augmentation |
| Erasing | 0.4 | Random erasing configuration |
| Loss gains | box 7.5, cls 0.5, dfl 1.5 | Standard Ultralytics detection-loss balance |

### 3.3 Baseline results

The best values in the recorded 30-row `results.csv` are:

| Metric | Best recorded value | Logged epoch |
|---|---:|---:|
| Precision (B) | 0.46271 | 12 |
| Recall (B) | 0.26704 | 26 |
| mAP@50 (B) | 0.23750 | 30 |
| mAP@50–95 (B) | 0.17418 | 30 |

The baseline checkpoint is available at `training/experiments/experiment_1_yolov9c/weights/best.pt`. The original run artifacts remain under `training/runs/train4/`.

## 4. Experiment 2: YOLO11m cleaned-data pipeline

### 4.1 Data ingestion

The second notebook mounts the supplied Drive account, creates dedicated `FoodDetection/source`, `FoodDetection/experiments`, and `FoodDetection/exports` directories, and copies the immutable source dataset to local storage. It supports either a Drive folder or a Drive ZIP, with the folder mode selected for this experiment.

The notebook resolves relative and stale absolute YAML paths against the local copy. It requires a YAML containing `train`, `val`, and `names`, and writes a stable local manifest before training. This eliminates the first experiment’s dependence on an unrelated hard-coded Kaggle path.

### 4.2 Preprocessing and quality gates

Before any GPU training begins, the notebook audits both splits. It validates class IDs, checks the YOLO label row shape, validates normalized coordinates, counts images and instances by class, reports missing labels, detects case-only duplicate names, and writes the audit and class-distribution files.

The second preprocessing stage then creates a versioned 61-class dataset. It removes the two duplicate-case classes, rewrites all retained annotation class IDs, removes images with dropped classes, removes missing or empty annotations, writes a new YAML and mapping CSV, and asserts that the output class list is exactly 61 classes. It also checks that the source train and validation image sets do not overlap.

### 4.3 Augmentation and training controls

Experiment 2 retains Ultralytics’ standard detection augmentations but improves the explicit training controls. The recorded arguments include HSV augmentation, translation, scale, horizontal flipping, mosaic, RandAugment, erasing, and the same YOLO loss gains as Experiment 1. The important changes are the explicit optimizer, cosine schedule, automatic batch sizing, local disk caching, longer schedule, early stopping patience, and checkpoint persistence.

| Parameter | Experiment 2 value | Justification |
|---|---:|---|
| Model | `yolo11m.pt` | Higher-capacity pretrained detector selected for the final experiment |
| Epoch target | 80 | Allows more convergence than the 30-epoch baseline |
| Input size | 640 | Preserves the baseline resolution for a more interpretable comparison |
| Batch size | `-1` automatic; observed 13 | Uses available GPU memory while remaining portable across Colab GPU variants |
| Optimizer | AdamW | Explicitly records the optimizer instead of relying on `auto` |
| Initial LR | 0.000149 | Matches the effective AdamW rate observed in the baseline run |
| Momentum | 0.9 | Matches the effective baseline AdamW setting |
| Weight decay | 0.0005 | Keeps the baseline regularization level |
| Warmup | 3 epochs | Stabilizes transfer-learning initialization |
| LR schedule | `cos_lr=true`, `lrf=0.01` | Provides controlled decay toward the final learning rate |
| Patience | 20 | Stops after sustained validation non-improvement while allowing recovery |
| Cache | `disk` | Enables writable local caching without filling Drive storage |
| Workers | 2 | More compatible with typical Colab CPU allocations |
| Checkpoint period | Every 5 epochs | Limits lost work after a runtime interruption |
| AMP | Enabled | Reduces GPU memory use and improves throughput |
| Seed | 0, deterministic | Improves repeatability within the same software and hardware conditions |
| Loss gains | box 7.5, cls 0.5, dfl 1.5 | Preserves the validated Ultralytics detection-loss balance |

### 4.4 Resume semantics

The notebook distinguishes three operations. `fresh` starts a new run from pretrained YOLO11m weights. `resume_interrupted` continues an interrupted run from its matching `last.pt` and should not change the model, dataset, optimizer, or epoch target. `extend_completed` starts a new fine-tuning stage from a completed checkpoint and must use a new experiment identity because it is not an exact resume.

A 63-class YOLOv9c checkpoint must not be used for an exact resume of the 61-class YOLO11m experiment. The detection head has a different class dimension, and the class-index mapping is different. Experiment 2 starts from pretrained YOLO11m weights instead.

### 4.5 Experiment 2 results

The best values in the available logged rows are:

| Metric | Best recorded value | Logged epoch |
|---|---:|---:|
| Precision (B) | 0.40785 | 64 |
| Recall (B) | 0.26546 | 60 |
| mAP@50 (B) | 0.24364 | 60 |
| mAP@50–95 (B) | 0.17677 | 60 |

The epoch-80 final row is lower than the best checkpoint: precision `0.35160`, recall `0.24722`, mAP@50 `0.23402`, and mAP@50–95 `0.16956`. The correct deployment choice is therefore `best.pt`, not `last.pt`.

The final checkpoint is available at `training/experiments/experiment_2_yolo11m/weights/best.pt`. The second experiment’s metrics and manifests are under `training/runs/lvis_fruits_yolo11m_80_v1/`.

## 5. Side-by-side engineering comparison

| Dimension | Experiment 1 | Experiment 2 | Interpretation |
|---|---|---|---|
| Architecture | YOLOv9c | YOLO11m | Experiment 2 has greater model capacity but higher compute cost |
| Taxonomy | 63-class source space | Cleaned 61-class space | Experiment 2 removes case-only duplicate concepts and requires remapped labels |
| Dataset manifest | Hard-coded Kaggle path | Stable local YAML generated from Drive source | Experiment 2 is easier to reproduce |
| Label validation | Not explicit | Explicit class, coordinate, missing-label, and overlap checks | Experiment 2 has stronger data integrity controls |
| Training length | 30 epochs | 80 epochs | Experiment 2 provides a longer optimization budget |
| Optimizer | Auto-selected | Explicit AdamW | Experiment 2 removes optimizer ambiguity |
| Batch | Fixed 8 | Automatic sizing | Experiment 2 adapts to GPU memory |
| Recovery | Standard checkpoints | Drive persistence plus periodic checkpoints | Experiment 2 is more robust to notebook disconnections |
| Best mAP@50 | 0.23750 | 0.24364 | Experiment 2 is modestly higher |
| Best mAP@50–95 | 0.17418 | 0.17677 | Experiment 2 is modestly higher |
| Best precision | 0.46271 | 0.40785 | Experiment 1 is higher on this metric |
| Best recall | 0.26704 | 0.26546 | Approximately similar |

Experiment 2 is the stronger **engineering pipeline**, not an unequivocally superior scientific model under a controlled ablation. Multiple variables changed at once: model family, class taxonomy, image counts, dataset processing, optimizer configuration, schedule, and checkpointing. The metric differences should therefore be reported as directional evidence.

## 6. Reproducibility and deployment checklist

To reproduce Experiment 1, open `experiments/experiment_1_yolov9c/food_detection_first_experiment.ipynb`, make the dataset YAML available at the path expected by the notebook, install the recorded Ultralytics environment, and run the YOLOv9c training cell.

To reproduce Experiment 2, open `experiments/experiment_2_yolo11m/food_detection_second_experiment.ipynb` or the canonical copy at `notebooks/food_detection.ipynb`, mount the supplied Drive dataset, run the setup and audit cells, run the versioned 61-class processing cell, and leave `RUN_MODE='fresh'` for a new run. The local YAML must point to the processed images and rewritten labels. For an interruption, use the same experiment name and matching `last.pt` with `RUN_MODE='resume_interrupted'`.

For deployment, use `models/best.pt` or the Experiment 2 copy under `experiments/experiment_2_yolo11m/weights/best.pt`. Do not substitute `last.pt` merely because it was created later. Do not pair a 63-class YAML with the 61-class YOLO11m checkpoint, and do not pair the old YOLOv9c checkpoint with the processed 61-class taxonomy.

A complete final evaluation should still include an independent held-out test split, per-class AP and support, confidence-threshold selection on validation data, checks for near-duplicate leakage, and latency measurements on the target deployment hardware. The current CSV metrics are validation metrics from the recorded training runs, not a production acceptance test.

## References

[1]: https://www.kaggle.com/datasets/henningheyen/lvis-fruits-and-vegetables-dataset "Kaggle: LVIS Fruits and Vegetables Dataset"

[2]: https://drive.google.com/drive/folders/1XQBPUJiyUarhytjy25NKCi_AcU-dEZO8?usp=drive_link "FoodDetection Google Drive folder"

[3]: https://docs.ultralytics.com/modes/train/ "Ultralytics train mode documentation"

[4]: https://docs.ultralytics.com/datasets/detect/ "Ultralytics YOLO detection dataset format"
