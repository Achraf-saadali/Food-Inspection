# Food Detection Experiment Audit and Integration

## Decision

Experiment 2, `lvis_fruits_yolo11m_80_v1`, was integrated as the repository’s canonical training pipeline and runtime detector. It is the stronger engineering pipeline, although its metric improvement is modest and the comparison is not a controlled ablation.

## Pipeline differences

| Area | Experiment 1 | Experiment 2 | Engineering assessment |
|---|---|---|---|
| Model | YOLOv9c | YOLO11m | Larger/newer backbone; likely higher capacity, but slower and more memory-intensive |
| Dataset | Hard-coded Kaggle YAML; 63-class ambiguity | Drive-backed dataset copied locally, audited, cleaned to 61 classes | Experiment 2 is more reproducible and avoids known case-duplicate classes |
| Data integrity | No explicit overlap or label validation in notebook | Class-ID validation, missing-label accounting, duplicate-name warning, train/val overlap assertion | Major pipeline improvement |
| Schedule | 30 epochs | 80-epoch target with patience 20 | More opportunity to converge; best checkpoint must be selected |
| Batch | 8 | Automatic GPU sizing (`-1`; observed 13) | More portable across Colab GPU variants |
| Optimizer | `auto`; recorded AdamW with `lr0` request ignored | Explicit AdamW, `lr0=0.000149`, momentum `0.9` | Removes ambiguity from optimizer auto-selection |
| Learning-rate schedule | Default cosine disabled | `cos_lr=true`, `lrf=0.01`, 3 warmup epochs | Better-controlled schedule |
| Caching | Disabled/read-only cache warning | Local disk cache | Better throughput and fewer cache warnings |
| Checkpointing | Standard `best.pt`/`last.pt` | Periodic immutable checkpoints plus Drive persistence callback | Better recovery from Colab interruption |
| Taxonomy | Duplicate `Strawberry`/`strawberry` and `Tomato`/`tomato` classes remain | Upper-case duplicate classes removed; contiguous 61-class mapping | Better semantic consistency, but class-space changed |

## Recorded metrics

| Metric | Experiment 1 best | Experiment 2 best | Difference |
|---|---:|---:|---:|
| Precision | 0.46271 | 0.40785 | -0.05486 |
| Recall | 0.26704 | 0.26546 | -0.00158 |
| mAP@50 | 0.23750 | 0.24364 | +0.00614 |
| mAP@50–95 | 0.17418 | 0.17677 | +0.00259 |

Experiment 2 achieved its best recorded mAP around logged epoch 60. Its final epoch-80 metrics fell to mAP@50 `0.23402` and mAP@50–95 `0.16956`, so the integrated runtime uses `best.pt`, not `last.pt`.

These metrics are directional evidence only. Architecture, class taxonomy, image counts, split composition, and training schedule all changed together. A proper scientific comparison would hold the dataset and split constant and vary one factor at a time.

## Dataset evidence

The processed Experiment 2 dataset contains 4,707 training images and 1,532 validation images. It contains 119,422 training instances and 33,695 validation instances. The processed taxonomy contains 61 classes after dropping source IDs 30 (`Strawberry`) and 35 (`Tomato`) and remapping the retained labels.

## Repository changes

The following changes were pushed to `Achraf-saadali/Food-Inspection` on `main` in commit `a6729ca`:

- Replaced `training/notebooks/food_detection.ipynb` with the second experiment’s data-audited YOLO11m pipeline.
- Configured the notebook for a fresh 80-epoch run by default, with explicit resume and extension modes.
- Added the processed 61-class YAML under `training/data/lvis_fruits_61class/`.
- Added the second experiment’s arguments, audit manifest, class distribution, training configuration, checkpoint status, and results under `training/runs/lvis_fruits_yolo11m_80_v1/`.
- Replaced `models/best.pt` with the Experiment 2 YOLO11m best checkpoint and tracked it with Git LFS.
- Updated backend model metadata and runtime references from `train4` to `lvis_fruits_yolo11m_80_v1`.
- Updated the README with the pipeline rationale, metrics, and limitations.

## Validation

The integrated notebook is valid JSON, backend Python files compile successfully, the processed YAML declares `nc: 61`, and `git diff --check` passed before commit. The branch was pushed successfully and is synchronized with `origin/main`.

## Remaining engineering work

The raw dataset is still external to the repository. Before making final performance claims, evaluate the committed `best.pt` on a locked independent test split, report per-class AP and class support, verify no near-duplicate leakage across splits, tune the confidence threshold on validation data, and measure inference latency on the deployment hardware.
