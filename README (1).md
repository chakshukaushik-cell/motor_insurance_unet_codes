# Motor Insurance U-Net Codes

This repository contains Python scripts for car detection, car-part
segmentation, defect-mask generation, U-Net training, fine-tuning, and
inference.

Repository:
https://github.com/chakshukaushik-cell/motor_insurance_unet_codes

## 1. Project Components

  -----------------------------------------------------------------------
  Component                           Purpose
  ----------------------------------- -----------------------------------
  PyTorch                             U-Net training, fine-tuning, model
                                      evaluation, and inference

  Segmentation Models PyTorch (SMP)   U-Net architecture with pretrained
                                      encoders such as ResNet34

  NumPy                               Numerical operations and
                                      image-array processing

  Pillow (PIL)                        Image loading, resizing,
                                      conversion, and saving

  OpenCV                              Image processing, mask generation,
                                      and color conversion

  Matplotlib                          Training plots and visualizations

  SAM2                                Prompt-based image segmentation for
                                      car parts and defects

  Python standard library             File handling, JSON/CSV processing,
                                      CLI arguments, randomization, and
                                      paths
  -----------------------------------------------------------------------

## 2. Dependencies

### Core Python packages

The following third-party packages are referenced by the repository
scripts:

-   `numpy`
-   `Pillow`
-   `torch`
-   `segmentation-models-pytorch`
-   `opencv-python`
-   `matplotlib`

SAM2 is an additional dependency for `prepare_sam2_part_dataset.py`. It
is generally installed separately from the official Meta SAM2
repository.

### Python standard-library modules

These modules are included with Python and do not need separate
installation:

-   `argparse`
-   `collections`
-   `csv`
-   `json`
-   `pathlib`
-   `random`
-   `re`
-   `shutil`
-   `sys`
-   `time`

## 3. Installation

Python 3.10 or 3.11 is a practical starting point. Check the
compatibility requirements of the PyTorch and SAM2 versions you install.

### Step 1: Create and activate a virtual environment

``` bash
python -m venv .venv
```

Windows:

``` bash
.venv\Scripts\activate
```

Linux/macOS:

``` bash
source .venv/bin/activate
```

### Step 2: Install PyTorch

Install a PyTorch build compatible with your system and CUDA version, if
using an NVIDIA GPU.

Official installation selector:

https://pytorch.org/get-started/locally/

For a CPU-only environment, use the CPU installation command provided by
the official PyTorch selector.

### Step 3: Install the other Python packages

``` bash
pip install numpy Pillow segmentation-models-pytorch opencv-python matplotlib
```

### Step 4: Install SAM2 (only if using the SAM2 dataset-generation script)

Follow the official installation instructions:

https://github.com/facebookresearch/sam2

SAM2 requires its own compatible PyTorch/torchvision setup and model
configuration/checkpoint files. Follow the versions and installation
steps specified by the official SAM2 repository.

## 4. Script-wise Dependency Reference

  -----------------------------------------------------------------------
  Python file                         Main dependencies
  ----------------------------------- -----------------------------------
  `Car_detection_fine_tune.py`        `numpy`, `Pillow`, `torch`,
                                      `segmentation-models-pytorch`

  `build_partwise_from_sam2.py`       `numpy`, `Pillow`

  `fine_tune_41_carparts.py`          `numpy`, `Pillow`, `torch`,
                                      `segmentation-models-pytorch`

  `infer_defect_unet.py`              `numpy`, `Pillow`, `torch`, local
                                      model module

  `infer_unseen_image_carparts.py`    `numpy`, `Pillow`, `torch`,
                                      `segmentation-models-pytorch`

  `inspect_sam_masks_defect.py`       `opencv-python`, `numpy`

  `prepare_car_parts_dataset.py`      `numpy`, `Pillow`

  `prepare_sam2_part_dataset.py`      `opencv-python`, `numpy`, `Pillow`,
                                      `torch`, `SAM2`

  `train_car_parts_unet.py`           `numpy`, `Pillow`, `torch`,
                                      `segmentation-models-pytorch`,
                                      `matplotlib`

  `train_defect_unet.py`              `numpy`, `Pillow`, `torch`

  `train_unet_car_detection.py`       `opencv-python`, `numpy`, `torch`
  -----------------------------------------------------------------------

Note: `infer_defect_unet.py` imports `UNet` from `train_unet.py`. That
module is not present in the repository file list. The script will need
the matching model-definition file or an import-path correction (for
example, if the intended class is in `train_defect_unet.py`).

## 5. Suggested requirements.txt

For convenience, the following is a starting dependency list for the
non-SAM2 scripts:

``` text
numpy
Pillow
segmentation-models-pytorch
opencv-python
matplotlib
```

Install PyTorch separately using the official selector so that the
correct CPU/CUDA build is selected:

``` bash
pip install -r requirements.txt
```

Then install PyTorch using the command generated at
https://pytorch.org/get-started/locally/.

If you use `prepare_sam2_part_dataset.py`, install SAM2 separately as
described above.

## 6. External Files and Model Assets

The Python packages alone are not sufficient to run every script.
Depending on the script, you may also need:

-   Training images and corresponding masks.
-   LabelMe JSON annotations.
-   YOLO-format defect annotation text files.
-   A car-parts class mapping JSON file.
-   Trained U-Net checkpoint files (`.pth`).
-   SAM2 checkpoint and model configuration files.
-   The expected directory structure and file naming conventions.

Update hard-coded paths in the scripts to match your local environment
before execution.

## 7. Hardware

-   CPU: Supported for basic execution and smaller workloads.
-   NVIDIA GPU: Recommended for model training and SAM2 processing.
-   CUDA: Optional, but the installed PyTorch build must be compatible
    with the system's CUDA setup.
-   RAM/VRAM: Requirements depend on image resolution, batch size, model
    architecture, and dataset size.

## 8. Important Notes

1.  No exact package versions are pinned in this README because the
    scripts do not specify a complete, consistent version lock.
2.  Install compatible PyTorch, torchvision, and SAM2 versions together.
3.  Some scripts use machine-specific absolute paths. Update these paths
    before running them.
4.  The SAM2 script needs external checkpoint/configuration assets.
5.  `infer_defect_unet.py` currently references a local `train_unet.py`
    module that is not listed in this repository.
6.  This README describes dependencies inferred from the Python imports;
    it does not guarantee that every script has been tested in a single
    environment.

## 9. References

-   Project repository:
    https://github.com/chakshukaushik-cell/motor_insurance_unet_codes
-   PyTorch installation: https://pytorch.org/get-started/locally/
-   Segmentation Models PyTorch:
    https://github.com/qubvel-org/segmentation_models.pytorch
-   SAM2: https://github.com/facebookresearch/sam2
-   OpenCV: https://opencv.org/
-   Pillow: https://python-pillow.org/
-   NumPy: https://numpy.org/
-   Matplotlib: https://matplotlib.org/
