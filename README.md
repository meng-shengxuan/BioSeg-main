# BioSeg

BioSeg is a semantic segmentation project for marine biofouling / underwater biological images. The repository contains image-mask pairs, LabelMe-style annotations, comparison visualizations, a pretrained BioSeg checkpoint, and training code based on PyTorch and MMSegmentation.

The current implementation uses a MambaVision-style backbone with a UPerNet segmentation head. The dataset labels include algal, barnacle, composite, other, sea anemone, and sea squirt categories, plus background.

The dataset will be made publicly available upon official acceptance of the paper.

## Repository Structure

```text
BioSeg-main/
|-- BioSeg.py                 # MambaVision backbone/model implementation
|-- BioSeg.pth                # pretrained BioSeg checkpoint
|-- config.py                 # MMSegmentation config
|-- train.py                  # training entry point
|-- requirements.txt          # Python dependencies
|-- images/                   # 64 input images
|-- masks/                    # 64 segmentation masks
|-- .json/                    # 64 LabelMe annotation files
|-- results/                  # visual comparison results on the main set
|   |-- BEiT/
|   |-- BioSeg/
|   |-- Ground Truth/
|   |-- MambaVision/
|   |-- ResNet-101/
|   |-- Swin/
|   |-- Twin/
|   `-- ViT/
`-- challenge/
    |-- challenge images/     # 5 challenge images
    `-- challenge result/     # prediction results from different methods
```

## Environment

The project was prepared with CUDA-enabled PyTorch and MMSegmentation.

```bash
conda create -n bioseg python=3.10 -y
conda activate bioseg

pip install -r requirements.txt
```

Main dependencies:

- PyTorch 2.6.0 with CUDA 12.4
- TorchVision 0.21.0 with CUDA 12.4
- MMCV 2.1.0
- MMEngine 0.10.7
- MMSegmentation 1.2.2
- Mamba-SSM 2.2.4
- timm, einops, transformers, OpenCV, Pillow

If your CUDA version is different, install the PyTorch/MMCV versions that match your local GPU environment.

We recommend placing this project under the MambaVision directory to ensure proper execution and compatibility with the original environment: https://github.com/NVlabs/MambaVision

## Data

The dataset in this repository contains:

- `images/`: original RGB images
- `masks/`: semantic segmentation masks
- `.json/`: polygon annotations in LabelMe JSON format

Classes observed in the annotation files:

| ID | Class |
|---:|-------|
| 0 | background |
| 1 | algal |
| 2 | barnacle |
| 3 | composite |
| 4 | other |
| 5 | sea anemone |
| 6 | sea squirt |

The color palette used in `config.py` is:

| Class ID | RGB Color |
|---:|-----------|
| 0 | `(0, 0, 0)` |
| 1 | `(0, 255, 0)` |
| 2 | `(255, 0, 0)` |
| 3 | `(255, 255, 0)` |
| 4 | `(0, 255, 255)` |
| 5 | `(255, 0, 255)` |
| 6 | `(128, 128, 128)` |

## Training

Training is launched through the MMSegmentation runner:

```bash
python train.py config.py --work-dir work_dirs/bioseg
```

To enable automatic mixed precision:

```bash
python train.py config.py --work-dir work_dirs/bioseg --amp
```

To resume training from the latest checkpoint in the work directory:

```bash
python train.py config.py --work-dir work_dirs/bioseg --resume
```

Important training settings in `config.py`:

- Crop size: `512 x 512`
- Number of classes: `7`
- Optimizer: AdamW
- Learning rate: `5e-5`
- Weight decay: `0.01`
- Max iterations: `10000`
- Validation interval: `500`
- Metrics: `mIoU` and `mDice`
- Best checkpoint selection: highest `mIoU`

## Pretrained Checkpoint

The repository includes:

```text
BioSeg.pth
```

This file can be used as a pretrained or trained checkpoint for BioSeg experiments. If the checkpoint is too large for direct GitHub upload, consider using Git LFS:

```bash
git lfs install
git lfs track "*.pth"
```

## Results

The `results/` directory provides qualitative comparisons among several segmentation methods:

- BioSeg
- MambaVision
- BEiT
- ResNet-101
- Swin
- Twin
- ViT
- Ground Truth

The `challenge/` directory contains 5 challenge images and corresponding visual prediction results from the compared methods.

## Notes Before Public Release

Before uploading this repository to GitHub, please check the following items:

1. `config.py` references MMSegmentation base configs such as `../_base_/models/upernet_swin.py`, `../_base_/datasets/myseg.py`, `../_base_/default_runtime.py`, and `../_base_/schedules/schedule_160k.py`. Make sure these files are included in the final repository or update the paths.
2. `train.py` imports `semantic_segmentation.mmseg_mambavision_backbone`, and `config.py` imports `semantic_segmentation.hooks.save_color_mask_hook`. Make sure the `semantic_segmentation/` package is included or adjust the imports to match the released code structure.
3. `BioSeg.py` contains NVIDIA MambaVision-derived code. Please verify the license requirements before public release.
4. Add a `LICENSE` file if the code is intended to be reused by others.
5. If this repository accompanies a paper, add the paper title, authors, venue, and citation information once available.

## Citation

If you use this code or dataset, please cite the associated paper:
'BioSeg: A lightweight Mamba-based semantic segmentation method for biofouling severity grading toward hull cleaning robots.'

## Acknowledgements

This project builds on the PyTorch, MMSegmentation, MMEngine, MMCV, timm, and MambaVision ecosystems.
