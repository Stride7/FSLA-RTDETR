# FSLA-RTDETR
Official implementation of FSLA-RTDETR for dense small-object seabed pockmark detection in shaded-relief bathymetric imagery.

Environment
The main experimental environment was:
- Python 3.10.20
- PyTorch 2.7.0+cu128
- CUDA 12.8
- Ultralytics 8.4.26
- NVIDIA GeForce RTX 5090
Dataset
The dataset was constructed from BOEM shaded-relief bathymetric imagery.
- Image size: 512 × 512 pixels
- Patch stride: 512 pixels
- Total images: 880
- Pockmark instances: 8013
- Train / validation / test: 704 / 88 / 88
- Number of classes: 1 (pockmark)
Representative image patches and corresponding YOLO-format annotations are provided in sample_data/.
The complete derived annotation dataset, precise patch-location information, and full train/validation/test file lists are not publicly released at the present stage.
Model Weights
The trained FSLA-RTDETR weights are available in the GitHub Releases section.
- Release: v1.0
- Weight file: FSLA.pt
- Input size: 512 × 512
- Class: pockmark

Data Availability
The original bathymetric data are publicly available from the U.S. Bureau of Ocean Energy Management (BOEM). This repository provides the model configuration, core implementation code, trained model weights, and representative annotated samples.

## Overview

FSLA-RTDETR is an improved four-scale RT-DETR framework that combines:

- SAC-C2f enhanced backbone
- LPEA Encoder
- DGCST-based bidirectional feature fusion
- High-resolution P2 feature level
- Four-scale F2-F5 RT-DETR decoder

## Repository Structure

```text
FSLA-RTDETR/
├── configs/
│   └── FSLA-RTDETR.yaml
├── models/
│   ├── SAC_C2f.py
│   ├── LPEA.py
│   └── DGCST.py
├── sample_data/
│   ├── images/
│   └── labels/
├── requirements.txt
└── README.md

