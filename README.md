# Roulette Predictor

A video-based, end-to-end deep learning approach to predicting roulette.

Given a short clip of the ball spinning around the wheel, an `r3d_18` 3D CNN (pretrained on Kinetics-400) outputs a probability distribution over the 37 pockets. The bet on the model's top-*k* pockets clears the house edge on a held-out set.

Full talk: [Predicting-Roulette.pdf](Predicting-Roulette.pdf).

## Motivation

Roulette has been beaten before, always by modelling the physics:

- **Thorp & Shannon (1961)** — first wearable computer, measured ball/wheel velocities by hand, ~44% edge.
- **Small & Tse (2012)** — deterministic-chaos model, ~18% expected edge on a perfect wheel.
- **Kanhirun et al. (2008)** — image processing + a neural network, ~40% returns.

All three hand-engineer the physical parameters (velocity, tilt, deceleration) and feed them to a predictor. The question here is different: **can we skip the physics and let a video model learn the relevant dynamics end-to-end?**

## Task

37-class classification (0–36) from a clip of the ball spinning *before* it drops.

| Strategy | Break-even accuracy | Random baseline |
| --- | --- | --- |
| Top-1 (1 unit on most likely) | 2.78% (1/36) | 2.70% (1/37) |
| Top-5 equal (1 unit each on top 5) | 13.89% (5/36) | 13.51% (5/37) |

House edge on a European wheel is **−2.7%**. Anything above break-even is a positive-expectation strategy.

## Results

Best held-out run ([01aaefb](../../commit/01aaefb)):

| Strategy | Test accuracy | Player edge |
| --- | --- | --- |
| Top-1 betting | 3.12% | **+12.0%** |
| Top-5 equal betting | 14.84% | **+6.8%** |
| Top-5 proportional betting (stake split by model confidence) | — | **+8.4%** |

These are small edges on a small held-out set, and they are not stable across runs — see [Challenges](#challenges).

## Approach

- **Backbone:** `torchvision.models.video.r3d_18` pretrained on Kinetics-400 (400 human-action classes). Chosen because it already encodes motion/temporal features from video, which is exactly the relevant signal here — ball and wheel dynamics — even though the source domain is unrelated.
- **Head:** Dropout → Linear(512→256) → ReLU → Dropout → Linear(256→37).
- **Input:** first 16 frames of the input clip, resized to 112×112, Kinetics-400 normalized. Shape `(3, 16, 112, 112)`.
- **Fine-tuning:** backbone frozen except `layer4`, which is unfrozen at LR `1e-5` while the head trains at LR `1e-3`.
- **Class imbalance:** `WeightedRandomSampler` in the train loader (46 samples per class on average, slight imbalance).
- **Augmentation:** random start-frame offset (0–7) and horizontal flip, training split only.
- **Splits:** 76.5% train / 8.5% val / 15% test, held constant via saved indices in `runs/run_*/train_test_split.pkl`.

Config lives in the `CONFIG` dict in [train_model.ipynb](train_model.ipynb).

## Dataset

[RouletteVision](https://huggingface.co/datasets/mp-coder/RouletteVision-Dataset) — 1,703 roulette-spin videos, each split into:

- **Input:** ball spinning around the wheel (2–5+ seconds)
- **Output:** ball falling into a pocket (used only for labelling)

Landing pockets were manually labelled across the four subsets — see `dataset_labels_manual_S{1..4}.csv`, merged into [dataset_labels_combined.csv](dataset_labels_combined.csv).

## Challenges

- **Limited data per class.** ~46 videos per number over 37 classes; risk of overfitting to the specific wheel/camera/lighting captured in the dataset. Whether the model transfers to another table is untested.
- **Statistical significance.** The edge over break-even is on the order of a single percentage point, on a ~256-video test set. The measured edge is real for this held-out split but noisy — variance across runs is comparable to the effect size.
- **Single-table dataset.** All videos come from one physical wheel in an uncontrolled environment. Any learned bias may be table-specific rather than a general roulette-prediction ability.
- **Class imbalance.** Mitigated with `WeightedRandomSampler`, but the tail classes still see the fewest gradient updates.

## Repository layout

| Path | Description |
| --- | --- |
| [Predicting-Roulette.pdf](Predicting-Roulette.pdf) | Presentation slides |
| [train_model.ipynb](train_model.ipynb) | Main training notebook — dataset loader, `r3d_18` model, training loop, evaluation, inference |
| [dataset-exploration.ipynb](dataset-exploration.ipynb) | Inspects video properties, label distribution, sample frames |
| [label_distribution_analysis.ipynb](label_distribution_analysis.ipynb) | Sanity-checks class balance across the four sets |
| [dataset_labels_combined.csv](dataset_labels_combined.csv) | Merged labels for all 1703 videos |
| `dataset_labels_manual_S{1..4}.csv` | Per-set manual labels |
| [RouletteVision/](RouletteVision/) | Upstream CV analysis code (ball/wheel tracking with OpenCV + NanoTrack ONNX). Not required for the training pipeline. |
| [RouletteVision-Dataset/](RouletteVision-Dataset/) | Video dataset (git-ignored; clone from HuggingFace) |
| [runs/](runs/) | Per-run training history and split indices |
| [weights/](weights/) | Saved `.pth` checkpoints |

## Getting started

1. Clone the dataset alongside this repo:
   ```
   git clone https://huggingface.co/datasets/mp-coder/RouletteVision-Dataset
   ```
2. Install dependencies (PyTorch with CUDA, torchvision, opencv-python, pandas, tqdm, scikit-learn, seaborn, matplotlib).
3. Open [train_model.ipynb](train_model.ipynb) and run top-to-bottom. Weights save per-epoch as `roulette_model_epoch_N.pth`.

Trained on an NVIDIA RTX 2070 Max-Q — roughly 9 minutes per epoch at batch size 8.

## Credits

- Dataset and upstream OpenCV analysis code: [@mp_coder / mpcodingdev/RouletteVision](https://github.com/mpcodingdev/RouletteVision).

---

*Research/educational purposes only. This project does not promote gambling.*
