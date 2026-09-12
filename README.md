# Blood Cell Type Classification on BloodMNIST


Comparison of four deep learning architectures for classifying 8 types of
peripheral blood cells from microscopy images, plus one additional ablation
experiment.

## Overview

This project classifies peripheral blood cells into 8 types using the
[BloodMNIST](https://medmnist.com/) dataset (part of MedMNIST v2). Four
architecturally different approaches are trained and compared:

1. **Baseline CNN** – a standard convolutional neural network (3 conv
   blocks, 32/64/128 filters).
2. **CNN + Squeeze-and-Excitation** – the same backbone with a
   channel-attention mechanism added after each conv block.
3. **Patch-based decision fusion CNN** – the same backbone trained on
   image quadrants, with predictions averaged at test time.
4. **Spiking Neural Network (SNN)** – a leaky integrate-and-fire network
   (PyTorch + snnTorch), the only architecture using PyTorch, as required
   by the course rules.

An additional **center-crop ablation** tests whether a single center crop
of the input can match the benefit of patch-based fusion, without its
extra training cost or test-time averaging.

## Dataset

- **BloodMNIST** (MedMNIST v2): 17,092 microscopy images of single blood
  cells, 8 classes (basophil, eosinophil, erythroblast, immature
  granulocytes, lymphocyte, monocyte, neutrophil, platelet).
- Official split: 11,959 train / 1,712 validation / 3,421 test images.
- Source: Acevedo et al., *Data in Brief*, 2020; standardized by
  Yang et al., *Scientific Data*, 2023.

## Results

Test-set accuracy for all model variants (single training run, no fixed
random seed unless noted):

| Model                          | Accuracy | Parameters |
|---------------------------------|---------:|-----------:|
| Baseline CNN                    | 91.58%   | 111,240    |
| SE-CNN (first attempt)          | 89.27%   | 116,868    |
| SE-CNN (tuned)                  | **96.99%** | 116,868  |
| Patch fusion CNN                | 96.87%   | 111,240    |
| SNN (28x28)                     | 64.45%   | 14,648     |
| SNN (64x64)                     | 87.26%   | 53,816     |
| Center-crop CNN (ablation)      | 96.84%   | 111,240    |

Notes:
- The tuned SE-CNN, patch fusion, and the center-crop ablation all land
  within 0.15 points of each other. Since none of the training scripts
  use a fixed random seed, this ranking can change between runs - see
  the report for a full discussion of this finding.
- The first SE-CNN attempt and the first SNN (64x64) attempt both showed
  training instability, diagnosed and corrected by adjusting the
  learning rate and EarlyStopping patience. See the report, Section
  VI-B, for details.

## Repository structure

```
.
├── notebooks/
│   └── ml4hd_project.ipynb        # full pipeline: EDA -> training -> demo
├── scripts/
│   ├── eda.py                     # exploratory data analysis
│   ├── three_architectures.py     # baseline, SE-CNN (v1), patch fusion
│   ├── se_cnn_retry.py            # SE-CNN with tuned hyperparameters
│   ├── snn_28x28.py               # SNN, native resolution
│   ├── snn_64x64.py               # SNN, matched resolution
│   ├── center_crop.py             # ablation experiment
│   └── live_demo.py               # loads trained models, predicts live
├── figures/                       # pipeline/architecture diagrams, plots
├── report/
│   └── ML4HD_report.pdf           # full written report (IEEE-style)
└── README.md
```

## Requirements

- Python 3.10+
- TensorFlow / Keras (baseline, SE-CNN, patch fusion, center-crop)
- PyTorch + [snnTorch](https://snntorch.readthedocs.io/) (SNN only, per
  course rules)
- `medmnist`, `numpy`, `scikit-learn`, `matplotlib`

```bash
pip install tensorflow torch snntorch medmnist scikit-learn matplotlib
```

## How to run

1. Run `scripts/eda.py` first to download the dataset and generate the
   exploratory analysis figures.
2. Run `scripts/three_architectures.py` to train the baseline, the first
   SE-CNN attempt, and patch fusion.
3. Run `scripts/se_cnn_retry.py` for the tuned SE-CNN.
4. Run `scripts/snn_28x28.py` and `scripts/snn_64x64.py` for the two SNN
   variants (requires PyTorch).
5. Run `scripts/center_crop.py` for the ablation experiment.
6. Run `scripts/live_demo.py` to load the trained models and see live
   predictions on test-set examples.

All scripts were developed and run on Kaggle (T4/P100 GPU). Trained
models and result files are saved under `models/` and `reports/`
respectively when run there.

## Live demo

The live demo loads the trained models and classifies unseen test-set
images in real time, showing each model's prediction and confidence for
one example per class.

## Report and presentation

The full written report, following an IEEE-style template, is available
in `report/`. It covers the dataset, preprocessing, architecture theory
and implementation, training diagnostics, and a full results discussion,
including the ablation study and its limitations.

## Author

Jelena Nikolic – MSc Data Science, University of Padova / ULB Brussels

## References

- Acevedo, A. et al. (2020). *A dataset of microscopic peripheral blood
  cell images for development of automatic recognition systems.* Data in
  Brief, 30, 105474.
- Yang, J. et al. (2023). *MedMNIST v2 – a large-scale lightweight
  benchmark for 2D and 3D biomedical image classification.* Scientific
  Data, 10(1), 41.
- Hu, J., Shen, L., & Sun, G. (2018). *Squeeze-and-Excitation Networks.*
  CVPR.
