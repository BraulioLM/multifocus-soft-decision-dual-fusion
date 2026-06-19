# Tenengrad Multifocus Fusion

This repository implements a lightweight multi-focus image fusion method based on Tenengrad focus measures and adaptive pixel-wise blending.

## Overview

The method computes a Tenengrad focus measure for each source image, generates a soft decision map, and performs image fusion using two strategies:

- ADD: weighted additive fusion
- EXP: weighted exponential fusion

## Datasets

### MFI-WHU

Expected structure:

```text
MFI-WHU/
├── source_1/
├── source_2/
└── full_clear/
```

### Lytro

Expected structure:

```text
Lytro/
└── sourceimages/
    └── color/
```

## Usage

### MFI-WHU

```bash
python run_mfi_whu.py
```

### Lytro

```bash
python run_lytro.py
```

## Requirements

```bash
pip install -r requirements.txt
```

## Citation

If you use this code, please cite the associated publication.
