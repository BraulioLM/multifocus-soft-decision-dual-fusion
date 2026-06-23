# Multi-Focus Image Fusion Using Soft Decision Maps and Dual Fusion Rules

This repository implements a lightweight multi-focus image fusion method based on Tenengrad focus measures and soft decision maps.

The method generates pixel-wise focus weights from the source images and performs image fusion using two complementary fusion rules:

- ADD: weighted additive fusion
- EXP: weighted exponential fusion

## Repository Structure

```text
.
├── data
│   ├── mfi_whu
│   │   ├── source_1
│   │   ├── source_2
│   │   └── full_clear
│   └── lytro
│       └── color
│
├── outputs
│
├── run_mfi_whu.py
├── run_lytro.py
├── requirements.txt
└── README.md
```

## Sample Data

A small set of sample images is included to verify that the implementation runs correctly.

Users may replace the sample images with the complete MFI-WHU or Lytro datasets.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### MFI-WHU

Run:

```bash
python run_mfi_whu.py
```

Results will be saved in:

```text
outputs/
```

### Lytro

Run:

```bash
python run_lytro.py
```

Results will be saved in:

```text
outputs/
```

## Method Overview

The proposed method follows four main steps:

1. Compute Tenengrad focus measures for both source images.
2. Generate a soft decision map from the focus responses.
3. Apply adaptive pixel-wise weighting.
4. Fuse the source images using ADD or EXP fusion rules.

## Citation

If you use this code in your research, please cite the associated publication:

```text
Citation information will be added after publication.
```

## License

This repository is released for academic and research purposes.
