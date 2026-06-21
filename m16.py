#!/usr/bin/env python3
"""
Multi-model fire/nofire training script.

Supported --model values:
resnet18, resnet34, resnet50, mobilenet_v2, mobilenet_v3_large,
efficientnet_b0, efficientnet_b1, vgg16_bn, vgg19_bn, densenet121

Auto features:
- --img_size auto reads image size from data_root folder name, e.g. Dataset/128x128 -> 128x128.
- --runs_root auto creates Runs/<ModelName>_<Size>_01/Run_XX.
- Every run contains exactly two main folders: weights/ and results/.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import copy
import sys
import csv
import json
import gc
import math
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFile
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_ALIASES = {
    "resnet18": "resnet18", "resnet_18": "resnet18", "resnet-18": "resnet18",
    "resnet34": "resnet34", "resnet_34": "resnet34", "resnet-34": "resnet34",
    "resnet50": "resnet50", "resnet_50": "resnet50", "resnet-50": "resnet50",
    "mobilenet_v2": "mobilenet_v2", "mobilenetv2": "mobilenet_v2", "mobilenet-v2": "mobilenet_v2",
    "mobilenet_v3_large": "mobilenet_v3_large", "mobilenetv3large": "mobilenet_v3_large", "mobilenet-v3-large": "mobilenet_v3_large",
    "efficientnet_b0": "efficientnet_b0", "efficientnetb0": "efficientnet_b0", "efficientnet-b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1", "efficientnetb1": "efficientnet_b1", "efficientnet-b1": "efficientnet_b1",
    "vgg16_bn": "vgg16_bn", "vgg16": "vgg16_bn", "vgg-16-bn": "vgg16_bn",
    "vgg19_bn": "vgg19_bn", "vgg19": "vgg19_bn", "vgg-19-bn": "vgg19_bn",
    "densenet121": "densenet121", "densenet_121": "densenet121", "densenet-121": "densenet121",
}

MODEL_DISPLAY = {
    "resnet18": "ResNet18", "resnet34": "ResNet34", "resnet50": "ResNet50",
    "mobilenet_v2": "MobileNetV2", "mobilenet_v3_large": "MobileNetV3Large",
    "efficientnet_b0": "EfficientNetB0", "efficientnet_b1": "EfficientNetB1",
    "vgg16_bn": "VGG16BN", "vgg19_bn": "VGG19BN", "densenet121": "DenseNet121",
}


DETAILED_HELP_TEXT = r"""
================================================================================
FIRE / NO-FIRE MULTI-MODEL TRAINING SCRIPT HELP
================================================================================

WHAT THIS SCRIPT DOES
---------------------
This script trains a pretrained CNN model for a 2-class fire/nofire dataset.

It supports:
    - Multiple model architectures
    - Automatic image-size detection from folder name
    - Automatic run-folder creation
    - Separate weights/ and results/ folders
    - Strong augmentation
    - Regularization
    - Freeze-then-finetune training
    - Weighted sampler for class imbalance
    - Validation-based model selection
    - Separate best weights for train, val, test, and test_e
    - Repeat training runs using --repeat X for average results

DATASET STRUCTURE
-----------------
Your dataset should look like this:

    Dataset/128x128/
        train/
            fire/
            nofire/
        val/
            fire/
            nofire/
        test/
            fire/
            nofire/
        test_e/
            fire/
            nofire/

The same structure also works for:

    Dataset/32x32/
    Dataset/64x64/
    Dataset/128x128/

BASIC COMMAND
-------------
For normal use, write:

    python train_fire_multimodel.py --model resnet18 --data_root "Dataset/128x128"

The script automatically detects the image size from the folder name.

So:

    --data_root "Dataset/128x128"

automatically means:

    image size = 128x128

And:

    --data_root "Dataset/32x32"

automatically means:

    image size = 32x32

AUTO DATASET FOLDER SCAN
------------------------
If you do NOT give --data_root, the script scans --dataset_base.

Default:

    --dataset_base "Dataset"

So this command:

    python train_fire_multimodel.py --model resnet18 --repeat 3

will look inside:

    Dataset/

and automatically find folders such as:

    Dataset/32x32/
    Dataset/64x64/
    Dataset/128x128/

Then it will train one by one:

    32x32 repeat 3 times
    64x64 repeat 3 times
    128x128 repeat 3 times

If --data_root is given, it trains only that folder:

    python train_fire_multimodel.py --model resnet18 --data_root "Dataset/128x128" --repeat 3


SUPPORTED MODELS
----------------
Use one of these names with --model:

    resnet18
    resnet34
    resnet50
    mobilenet_v2
    mobilenet_v3_large
    efficientnet_b0
    efficientnet_b1
    vgg16_bn
    vgg19_bn
    densenet121

MODEL ALIASES
-------------
These aliases also work:

    resnet-18           -> resnet18
    resnet_18           -> resnet18
    resnet-34           -> resnet34
    resnet_34           -> resnet34
    resnet-50           -> resnet50
    resnet_50           -> resnet50
    mobilenetv2         -> mobilenet_v2
    mobilenet-v2        -> mobilenet_v2
    mobilenetv3large    -> mobilenet_v3_large
    mobilenet-v3-large  -> mobilenet_v3_large
    efficientnetb0      -> efficientnet_b0
    efficientnet-b0     -> efficientnet_b0
    efficientnetb1      -> efficientnet_b1
    efficientnet-b1     -> efficientnet_b1
    vgg16               -> vgg16_bn
    vgg-16-bn           -> vgg16_bn
    vgg19               -> vgg19_bn
    vgg-19-bn           -> vgg19_bn
    densenet-121        -> densenet121

RECOMMENDED COMMANDS FOR 128x128
--------------------------------

ResNet-18:

    python train_fire_multimodel.py --model resnet18 --data_root "Dataset/128x128" --batch_size 32 --epochs 100 --early_stop_patience 20

ResNet-34:

    python train_fire_multimodel.py --model resnet34 --data_root "Dataset/128x128" --batch_size 32 --epochs 100 --early_stop_patience 20

ResNet-50:

    python train_fire_multimodel.py --model resnet50 --data_root "Dataset/128x128" --batch_size 16 --epochs 100 --early_stop_patience 20

MobileNetV2:

    python train_fire_multimodel.py --model mobilenet_v2 --data_root "Dataset/128x128" --batch_size 32 --epochs 100 --early_stop_patience 20

MobileNetV3-Large:

    python train_fire_multimodel.py --model mobilenet_v3_large --data_root "Dataset/128x128" --batch_size 32 --epochs 100 --early_stop_patience 20

EfficientNet-B0:

    python train_fire_multimodel.py --model efficientnet_b0 --data_root "Dataset/128x128" --batch_size 32 --epochs 100 --early_stop_patience 20

EfficientNet-B1:

    python train_fire_multimodel.py --model efficientnet_b1 --data_root "Dataset/128x128" --batch_size 24 --epochs 100 --early_stop_patience 20

VGG16-BN:

    python train_fire_multimodel.py --model vgg16_bn --data_root "Dataset/128x128" --batch_size 8 --epochs 100 --early_stop_patience 20

VGG19-BN:

    python train_fire_multimodel.py --model vgg19_bn --data_root "Dataset/128x128" --batch_size 8 --epochs 100 --early_stop_patience 20

DenseNet-121:

    python train_fire_multimodel.py --model densenet121 --data_root "Dataset/128x128" --batch_size 16 --epochs 100 --early_stop_patience 20

RECOMMENDED COMMANDS FOR 32x32
------------------------------

ResNet-18:

    python train_fire_multimodel.py --model resnet18 --data_root "Dataset/32x32" --batch_size 32 --epochs 100 --early_stop_patience 20

EfficientNet-B0:

    python train_fire_multimodel.py --model efficientnet_b0 --data_root "Dataset/32x32" --batch_size 32 --epochs 100 --early_stop_patience 20

MobileNetV2:

    python train_fire_multimodel.py --model mobilenet_v2 --data_root "Dataset/32x32" --batch_size 32 --epochs 100 --early_stop_patience 20

IMPORTANT ARGUMENTS
-------------------

Model selection:

    --model resnet18

Dataset path:

    --data_root "Dataset/128x128"

Image size:

    --img_size auto       Automatically detects size from folder name.
    --img_size 128        Forces resize to 128x128.
    --img_height 128 --img_width 256

Run folder:

    --runs_root auto
        Automatically creates:
        Runs/<ModelName>_<ImageSize>_01/Run_XX/

    --runs_base Runs
        Base folder for automatic run folders.

Training:

    --epochs 100
    --batch_size 32
    --early_stop_patience 20
    --repeat 5
        Runs the complete training process 5 separate times.

Learning rate:

    --lr 1e-4             Learning rate for frozen-backbone classifier-head training.
    --finetune_lr 5e-5    Learning rate after unfreezing the backbone.

Freeze then fine-tune:

    --freeze_epochs 5
        First 5 epochs train only classifier head.
        After that, the full model is unfrozen and fine-tuned.

Regularization:

    --weight_decay 5e-4
    --dropout 0.30
    --label_smoothing 0.05

Augmentation:

    --augment strong
    --augment basic
    --augment off

Weighted sampler:

    --use_weighted_sampler auto
    --use_weighted_sampler yes
    --use_weighted_sampler no

Device:

    --device auto     Uses GPU if CUDA is available, otherwise CPU.
    --device cuda     Forces GPU.
    --device cpu      Forces CPU.

FP16:

    --use_fp16 1      Uses mixed precision when CUDA is available.
    --use_fp16 0      Disables mixed precision.

Progress bar:

    --progress_bar 1
    --progress_bar 0

DataLoader workers:

    --num_workers 0
        Safest option for Windows/CUDA.

    --num_workers 2 or --num_workers 4
        May be faster on Linux, but can crash on Windows with CUDA/pagefile errors.

For Windows, keep:

    --num_workers 0

OUTPUT FOLDER STRUCTURE
-----------------------
The script automatically creates one model-size folder, then separates weights and results:

    Runs/ResNet18_128x128_01/
        weights/
            Run_01/
            Run_02/
            Run_03/
        results/
            Run_01/
            Run_02/
            Run_03/

For another model:

    Runs/EfficientNetB0_128x128_01/
        weights/
            Run_01/
        results/
            Run_01/

This makes it easy to download all weights separately or all results separately.

WEIGHTS FOLDER
--------------
The weights/ folder contains:

    best_val_model.pth
    best_val_model_state_dict_only.pth

    best_train_model.pth
    best_train_model_state_dict_only.pth

    best_test_model.pth
    best_test_model_state_dict_only.pth

    best_test_e_model.pth
    best_test_e_model_state_dict_only.pth

    last_model.pth
    last_model_state_dict_only.pth

For final reporting, use:

    best_val_model_state_dict_only.pth

REPEAT TRAINING
---------------
To train the same model multiple times for better average reporting:

    python train_fire_multimodel.py --model resnet18 --data_root "Dataset/128x128" --repeat 5

This creates matching subfolders inside weights and results:

    weights/Run_01/
    weights/Run_02/
    weights/Run_03/
    weights/Run_04/
    weights/Run_05/

    results/Run_01/
    results/Run_02/
    results/Run_03/
    results/Run_04/
    results/Run_05/

The seed changes automatically for each repeat:

    repeat 1 -> seed 42
    repeat 2 -> seed 43
    repeat 3 -> seed 44

After all repeats finish, the script saves:

    results/repeat_summary.json
    results/repeat_summary.csv

These contain mean and standard deviation for train/val/test/test_e results.

FULL RESEARCH RESULTS ADDED
---------------------------
Each run now also saves:

    model_structure.txt
    model_summary.json
    model_recreate_info.json
    model_recreate_code.txt

    predictions_train.csv
    predictions_val.csv
    predictions_test.csv
    predictions_test_e.csv

    detailed_metrics_train.json
    detailed_metrics_val.json
    detailed_metrics_test.json
    detailed_metrics_test_e.json

    roc_curve_train.csv
    roc_curve_val.csv
    roc_curve_test.csv
    roc_curve_test_e.csv

    precision_recall_curve_train.csv
    precision_recall_curve_val.csv
    precision_recall_curve_test.csv
    precision_recall_curve_test_e.csv

The plots/ folder also saves:

    roc_curve_train.png
    roc_curve_val.png
    roc_curve_test.png
    roc_curve_test_e.png

    precision_recall_curve_train.png
    precision_recall_curve_val.png
    precision_recall_curve_test.png
    precision_recall_curve_test_e.png

The global Runs folder also saves a master file across all models/runs:

    Runs/master_summary.csv
    Runs/master_summary.jsonl

These files are useful for final thesis/report tables and comparing all trained models.

RESULTS FOLDER
--------------
The results/ folder contains:

    config.json
    final_results.json
    training_history.json
    training_log.csv
    class_to_idx.json
    dataset_balance_report.json
    plots/

The plots/ folder contains:

    loss_curves.png
    accuracy_curves.png
    learning_rate_curve.png
    combined_training_curves.png

    confusion_matrix_train_best_val_model.png
    confusion_matrix_val_best_val_model.png
    confusion_matrix_test_best_val_model.png
    confusion_matrix_test_e_best_val_model.png

TOO MANY OPEN FILES NOTE
------------------------
If you see this Linux error:

    OSError: [Errno 24] Too many open files

it usually happens during final prediction export when too many DataLoader worker
processes/file handles are open.

This script now uses:

    --final_num_workers 0

by default for final prediction export, ROC curve export, PR curve export, and
detailed prediction CSV generation.

If the error still happens, also run training with:

    --num_workers 0


WINERROR 1455 NOTE
------------------
If you see this error:

    OSError: [WinError 1455] The paging file is too small

it usually means Windows DataLoader worker processes are loading PyTorch/CUDA DLLs repeatedly.

This script now uses:

    --num_workers 0

by default, and also forces num_workers=0 on Windows if a higher value is requested.

This is safer for repeated training on Windows with CUDA.

QUICK HELP
----------
To show this helper:

    python train_fire_multimodel.py

or:

    python train_fire_multimodel.py --help

================================================================================
"""


WEIGHTS_CLASS = {
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "mobilenet_v2": "MobileNet_V2_Weights",
    "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
    "efficientnet_b0": "EfficientNet_B0_Weights",
    "efficientnet_b1": "EfficientNet_B1_Weights",
    "vgg16_bn": "VGG16_BN_Weights",
    "vgg19_bn": "VGG19_BN_Weights",
    "densenet121": "DenseNet121_Weights",
}


def str_to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    x = str(x).strip().lower()
    if x in ["1", "true", "yes", "y", "on"]:
        return True
    if x in ["0", "false", "no", "n", "off"]:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected: use 1/0 or true/false.")


def normalize_model_name(x: str) -> str:
    key = str(x).strip().lower().replace(" ", "_")
    if key not in MODEL_ALIASES:
        raise ValueError(f"Unsupported model '{x}'. Use one of: {', '.join(sorted(set(MODEL_ALIASES.values())))}")
    return MODEL_ALIASES[key]


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def save_json(data: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def append_csv_row(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def count_images(folder: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in exts)


def get_lr(optimizer: optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def num_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def num_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_device(device_arg: str) -> torch.device:
    device_arg = device_arg.lower()
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was selected, but torch.cuda.is_available() is False. Install CUDA-enabled PyTorch.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError("--device must be auto, cuda, or cpu")


def device_info(device: torch.device) -> Dict[str, Any]:
    info = {
        "selected_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "pytorch_version": torch.__version__,
        "pytorch_cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update({
            "cuda_current_device_index": idx,
            "cuda_device_name": torch.cuda.get_device_name(idx),
            "cuda_total_memory_gb": round(props.total_memory / (1024 ** 3), 3),
            "cuda_capability": f"{props.major}.{props.minor}",
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        })
    return info


def resolve_data_root(args: argparse.Namespace) -> Path:
    """
    Resolve one concrete dataset root.

    This is used inside train_model(), where args.data_root is always set
    to one exact folder by run_repeated_training().
    """
    if args.data_root:
        return Path(args.data_root)

    if str(args.dataset_size).lower() != "auto":
        return Path(args.dataset_base) / str(args.dataset_size)

    raise ValueError(
        "No single dataset root was selected. This should be handled by run_repeated_training()."
    )


def is_valid_dataset_root(path: Path) -> bool:
    """
    A valid dataset root must contain:
        train/
        val/
        test/
        test_e/

    Each split should usually contain:
        fire/
        nofire/
    """
    if not path.exists() or not path.is_dir():
        return False

    required = ["train", "val", "test", "test_e"]
    return all((path / split).is_dir() for split in required)


def dataset_sort_key(path: Path) -> Tuple[int, str]:
    """
    Sort folders naturally by image size:
        32x32 before 64x64 before 128x128
    """
    detected = detect_size_from_folder(path)

    if detected is None:
        return (10**9, path.name.lower())

    h, w = detected
    return (max(h, w), path.name.lower())


def discover_dataset_roots(args: argparse.Namespace) -> List[Path]:
    """
    If --data_root is provided:
        return only that folder.

    If --data_root is not provided:
        scan --dataset_base for subfolders named like:
            32x32
            64x64
            128x128

        and containing train/val/test/test_e.
    """
    if args.data_root:
        root = Path(args.data_root)

        if not is_valid_dataset_root(root):
            raise FileNotFoundError(
                f"--data_root was given but does not look like a valid dataset root: {root}\n"
                "Expected subfolders: train, val, test, test_e"
            )

        return [root]

    if str(args.dataset_size).lower() != "auto":
        root = Path(args.dataset_base) / str(args.dataset_size)

        if not is_valid_dataset_root(root):
            raise FileNotFoundError(
                f"--dataset_size selected this folder, but it is not a valid dataset root: {root}\n"
                "Expected subfolders: train, val, test, test_e"
            )

        return [root]

    base = Path(args.dataset_base)

    if not base.exists():
        raise FileNotFoundError(
            f"--data_root was not given, so I tried to scan --dataset_base, but it does not exist: {base}"
        )

    candidates = []

    for child in base.iterdir():
        if not child.is_dir():
            continue

        # Only auto-pick folders whose names contain image size information.
        # Examples: 32x32, 64x64, 128x128, 224
        if detect_size_from_folder(child) is None:
            continue

        if is_valid_dataset_root(child):
            candidates.append(child)

    candidates = sorted(candidates, key=dataset_sort_key)

    if not candidates:
        raise FileNotFoundError(
            f"--data_root was not given, and no valid size folders were found inside: {base}\n\n"
            "Expected something like:\n"
            "  Dataset/32x32/train, val, test, test_e\n"
            "  Dataset/64x64/train, val, test, test_e\n"
            "  Dataset/128x128/train, val, test, test_e\n\n"
            "Or give one folder directly:\n"
            "  --data_root Dataset/128x128"
        )

    return candidates


def detect_size_from_folder(data_root: Path) -> Optional[Tuple[int, int]]:
    name = data_root.name.lower().strip()
    m = re.search(r"(\d+)\s*x\s*(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"^(\d+)$", name)
    if m:
        s = int(m.group(1))
        return s, s
    return None


def resolve_image_size(args: argparse.Namespace, data_root: Path) -> Tuple[int, int, str, str]:
    if args.img_height is not None or args.img_width is not None:
        if args.img_height is None or args.img_width is None:
            raise ValueError("Use both --img_height and --img_width for non-square images.")
        h, w = int(args.img_height), int(args.img_width)
        return h, w, f"{h}x{w}", "manual_img_height_width"

    img_size_arg = str(args.img_size).strip().lower()
    if img_size_arg != "auto":
        s = int(img_size_arg)
        return s, s, f"{s}x{s}", "manual_img_size"

    detected = detect_size_from_folder(data_root)
    if detected is None:
        raise ValueError(f"Could not detect image size from folder name '{data_root.name}'. Use --img_size 128.")
    h, w = detected
    return h, w, f"{h}x{w}", "auto_from_data_root_folder_name"


def resolve_small_image_mode(args: argparse.Namespace, h: int, w: int) -> bool:
    mode = args.small_image_mode.lower()
    if mode == "yes":
        return True
    if mode == "no":
        return False
    if mode == "auto":
        return max(h, w) <= 64
    raise ValueError("--small_image_mode must be auto, yes, or no")


def create_next_run_name(root: Path) -> str:
    """
    Create matching run names for the organized structure:

        root/
            weights/
                Run_01/
            results/
                Run_01/

    The next run number is detected from both weights/ and results/.
    """
    root.mkdir(parents=True, exist_ok=True)

    weights_root = root / "weights"
    results_root = root / "results"

    weights_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    nums = []

    for parent in [weights_root, results_root]:
        for p in parent.glob("Run_*"):
            if p.is_dir():
                try:
                    nums.append(int(p.name.split("_")[-1]))
                except ValueError:
                    pass

    run_name = f"Run_{max(nums, default=0) + 1:02d}"
    return run_name


def resolve_run_paths(args: argparse.Namespace, model_name: str, size_tag: str) -> Tuple[Path, str, Path, Path, Path]:
    """
    Organized output structure:

        Runs/ResNet18_128x128_01/
            weights/
                Run_01/
            results/
                Run_01/
                    plots/

    Returns:
        root, run_name, weights_dir, results_dir, plots_dir
    """
    if str(args.runs_root).lower() == "auto":
        root = Path(args.runs_base) / f"{MODEL_DISPLAY[model_name]}_{size_tag}_01"
    else:
        root = Path(args.runs_root)

    run_name = create_next_run_name(root)

    weights_dir = root / "weights" / run_name
    results_dir = root / "results" / run_name
    plots_dir = results_dir / "plots"

    weights_dir.mkdir(parents=True, exist_ok=False)
    plots_dir.mkdir(parents=True, exist_ok=True)

    return root, run_name, weights_dir, results_dir, plots_dir



def get_safe_num_workers(requested_workers: int) -> Tuple[int, str]:
    """
    Windows + CUDA + repeated training can crash with:
        OSError: [WinError 1455] The paging file is too small...

    Reason:
        DataLoader multiprocessing starts extra Python worker processes.
        On Windows, every worker process imports torch again and may load CUDA DLLs again.
        This can consume a lot of RAM/pagefile.

    Safe fix:
        Use num_workers=0 on Windows.
    """
    requested_workers = int(requested_workers)

    if requested_workers < 0:
        requested_workers = 0

    if os.name == "nt" and requested_workers > 0:
        return 0, (
            f"Windows detected. Requested num_workers={requested_workers}, "
            "but changed to num_workers=0 to avoid WinError 1455 / CUDA DLL paging-file errors."
        )

    return requested_workers, "Using requested num_workers."


def progress_iterator(iterable, total: int, desc: str, enabled: bool):
    if enabled and TQDM_AVAILABLE:
        return tqdm(iterable, total=total, desc=desc, leave=False, dynamic_ncols=True)
    return iterable


def safe_set_postfix(iterator, values: Dict[str, Any]) -> None:
    if TQDM_AVAILABLE and hasattr(iterator, "set_postfix"):
        iterator.set_postfix(values)


def build_transforms(h: int, w: int, mode: str) -> Tuple[transforms.Compose, transforms.Compose]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mode = mode.lower()
    if mode == "off":
        train_t = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor(), transforms.Normalize(mean, std)])
    elif mode == "basic":
        train_t = transforms.Compose([
            transforms.Resize((h, w)), transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.20, hue=0.05),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
    elif mode == "strong":
        train_t = transforms.Compose([
            transforms.RandomResizedCrop((h, w), scale=(0.70, 1.00), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.30, hue=0.05)], p=0.85),
            transforms.RandomAutocontrast(p=0.25),
            transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.20),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.90, 1.10)),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.20),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.15),
            transforms.ToTensor(), transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.30, 3.30), value="random"),
        ])
    else:
        raise ValueError("--augment must be off, basic, or strong")
    eval_t = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor(), transforms.Normalize(mean, std)])
    return train_t, eval_t


def build_datasets(data_root: Path, h: int, w: int, augment: str) -> Dict[str, datasets.ImageFolder]:
    train_t, eval_t = build_transforms(h, w, augment)
    tfms = {"train": train_t, "val": eval_t, "test": eval_t, "test_e": eval_t}
    out = {}
    for split in ["train", "val", "test", "test_e"]:
        p = data_root / split
        if not p.exists():
            raise FileNotFoundError(f"Missing split folder: {p}")
        out[split] = datasets.ImageFolder(str(p), transform=tfms[split])
    base_classes = out["train"].class_to_idx
    if len(base_classes) != 2:
        raise ValueError(f"Expected exactly 2 classes but found: {base_classes}")
    for split, ds in out.items():
        if ds.class_to_idx != base_classes:
            raise ValueError(f"Class mismatch in {split}: {ds.class_to_idx} vs train {base_classes}")
    return out


def class_counts(ds: datasets.ImageFolder) -> Dict[int, int]:
    counts = {idx: 0 for idx in ds.class_to_idx.values()}
    for _, y in ds.samples:
        counts[int(y)] += 1
    return counts


def balance_report(dsets: Dict[str, datasets.ImageFolder], data_root: Path) -> Dict[str, Any]:
    class_to_idx = dsets["train"].class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    report = {"data_root": str(data_root), "class_to_idx": class_to_idx, "idx_to_class": {str(k): v for k, v in idx_to_class.items()}, "splits": {}, "domain_checks": {}, "warnings": []}
    for split, ds in dsets.items():
        counts = class_counts(ds)
        total = sum(counts.values())
        ratio = max(counts.values()) / max(min(counts.values()), 1)
        report["splits"][split] = {
            "total": total,
            "counts": {idx_to_class[i]: c for i, c in counts.items()},
            "percentages": {idx_to_class[i]: c / max(total, 1) for i, c in counts.items()},
            "imbalance_ratio_max_over_min": ratio,
        }
        if ratio >= 1.5:
            report["warnings"].append(f"{split} is imbalanced. max/min ratio = {ratio:.3f}")
    train_counts = class_counts(dsets["train"])
    te_counts = class_counts(dsets["test_e"])
    train_total = sum(train_counts.values())
    te_total = sum(te_counts.values())
    diffs = {}
    for i in sorted(train_counts):
        tr = train_counts[i] / max(train_total, 1)
        te = te_counts[i] / max(te_total, 1)
        diff = abs(tr - te)
        diffs[idx_to_class[i]] = {"train_percentage": tr, "test_e_percentage": te, "absolute_difference": diff}
        if diff >= 0.10:
            report["warnings"].append(f"Class distribution differs between train and test_e for class '{idx_to_class[i]}': abs diff = {diff:.3f}")
    report["domain_checks"]["train_vs_test_e_class_distribution"] = diffs
    return report


def make_sampler(ds: datasets.ImageFolder, mode: str, threshold: float) -> Tuple[Optional[WeightedRandomSampler], Dict[str, Any]]:
    counts = class_counts(ds)
    ratio = max(counts.values()) / max(min(counts.values()), 1)
    use = mode == "yes" or (mode == "auto" and ratio >= threshold)
    info = {"requested_mode": mode, "class_counts": {str(k): v for k, v in counts.items()}, "imbalance_ratio_max_over_min": ratio, "imbalance_threshold": threshold, "weighted_sampler_used": use}
    if not use:
        return None, info
    weights_by_class = {k: 1.0 / max(v, 1) for k, v in counts.items()}
    sample_weights = [weights_by_class[int(y)] for _, y in ds.samples]
    return WeightedRandomSampler(torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True), info


def build_loaders(dsets: Dict[str, datasets.ImageFolder], args: argparse.Namespace) -> Tuple[Dict[str, DataLoader], Dict[str, Any]]:
    sampler, sampler_info = make_sampler(dsets["train"], args.use_weighted_sampler, args.weighted_sampler_threshold)

    safe_workers, workers_note = get_safe_num_workers(args.num_workers)
    args.num_workers = safe_workers

    sampler_info["num_workers_effective"] = safe_workers
    sampler_info["num_workers_note"] = workers_note

    common = {
        "batch_size": args.batch_size,
        "num_workers": safe_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": safe_workers > 0
    }

    loaders = {
        "train": DataLoader(dsets["train"], shuffle=sampler is None, sampler=sampler, **common),
        "val": DataLoader(dsets["val"], shuffle=False, **common),
        "test": DataLoader(dsets["test"], shuffle=False, **common),
        "test_e": DataLoader(dsets["test_e"], shuffle=False, **common),
    }
    return loaders, sampler_info


def get_weights(model_name: str):
    cls = getattr(models, WEIGHTS_CLASS[model_name])
    return cls.DEFAULT


def make_classifier(in_features: int, num_classes: int, dropout: float) -> nn.Module:
    if dropout > 0:
        return nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    return nn.Linear(in_features, num_classes)


def adapt_resnet_small(model: nn.Module, pretrained: bool) -> nn.Module:
    old = model.conv1
    new = nn.Conv2d(3, old.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    if pretrained:
        with torch.no_grad():
            new.weight.copy_(old.weight[:, :, 2:5, 2:5])
    model.conv1 = new
    model.maxpool = nn.Identity()
    return model


def replace_head(model: nn.Module, model_name: str, num_classes: int, dropout: float) -> nn.Module:
    if model_name.startswith("resnet"):
        model.fc = make_classifier(model.fc.in_features, num_classes, dropout)
    elif model_name in ["mobilenet_v2", "mobilenet_v3_large", "efficientnet_b0", "efficientnet_b1", "vgg16_bn", "vgg19_bn"]:
        model.classifier[-1] = make_classifier(model.classifier[-1].in_features, num_classes, dropout)
    elif model_name == "densenet121":
        model.classifier = make_classifier(model.classifier.in_features, num_classes, dropout)
    else:
        raise ValueError(f"Unsupported head replacement: {model_name}")
    return model


def build_model(model_name: str, num_classes: int, pretrained: bool, small_image_mode: bool, dropout: float) -> nn.Module:
    weights = get_weights(model_name) if pretrained else None
    builders = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "mobilenet_v2": models.mobilenet_v2,
        "mobilenet_v3_large": models.mobilenet_v3_large,
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b1": models.efficientnet_b1,
        "vgg16_bn": models.vgg16_bn,
        "vgg19_bn": models.vgg19_bn,
        "densenet121": models.densenet121,
    }
    model = builders[model_name](weights=weights)
    if small_image_mode and model_name.startswith("resnet"):
        model = adapt_resnet_small(model, pretrained)
    return replace_head(model, model_name, num_classes, dropout)


def is_head_param(model_name: str, name: str) -> bool:
    if model_name.startswith("resnet"):
        return name.startswith("fc.")
    return name.startswith("classifier.")


def set_backbone_trainable(model: nn.Module, model_name: str, trainable: bool) -> None:
    for name, p in model.named_parameters():
        p.requires_grad = True if is_head_param(model_name, name) else trainable


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters.")
    return optim.Adam(params, lr=lr, weight_decay=weight_decay)


def build_scheduler(args: argparse.Namespace, optimizer: optim.Optimizer):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience, threshold=args.lr_threshold, min_lr=args.min_lr)


def confusion_matrix(y_true: List[int], y_pred: List[int], n: int) -> List[List[int]]:
    cm = [[0 for _ in range(n)] for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1
    return cm


def per_class_metrics(cm: List[List[int]]) -> Dict[str, Dict[str, float]]:
    n = len(cm)
    out = {}
    for c in range(n):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(n) if r != c)
        fn = sum(cm[c][r] for r in range(n) if r != c)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        out[str(c)] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(cm[c])}
    return out


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, amp_enabled: bool, num_classes: int, split_name: str, epoch: int, total_epochs: int, use_progress_bar: bool) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_true, y_pred = [], []
    desc = f"{split_name} {epoch}/{total_epochs}" if epoch > 0 else split_name
    bar = progress_iterator(loader, len(loader), desc, use_progress_bar)
    for x, y in bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            out = model(x)
            loss = criterion(out, y)
        pred = torch.argmax(out, dim=1)
        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_correct += int((pred == y).sum().item())
        total_samples += bs
        safe_set_postfix(bar, {"loss": f"{total_loss / max(total_samples, 1):.4f}", "acc": f"{total_correct / max(total_samples, 1):.4f}"})
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())
    cm = confusion_matrix(y_true, y_pred, num_classes)
    return {"loss": total_loss / max(total_samples, 1), "accuracy": total_correct / max(total_samples, 1), "correct": total_correct, "total": total_samples, "confusion_matrix": cm, "per_class": per_class_metrics(cm)}


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, scaler: torch.amp.GradScaler, device: torch.device, amp_enabled: bool, epoch: int, total_epochs: int, use_progress_bar: bool) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    bar = progress_iterator(loader, len(loader), f"Train {epoch}/{total_epochs}", use_progress_bar)
    for x, y in bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            out = model(x)
            loss = criterion(out, y)
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        pred = torch.argmax(out.detach(), dim=1)
        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_correct += int((pred == y).sum().item())
        total_samples += bs
        safe_set_postfix(bar, {"loss": f"{total_loss / max(total_samples, 1):.4f}", "acc": f"{total_correct / max(total_samples, 1):.4f}"})
    return {"loss": total_loss / max(total_samples, 1), "accuracy": total_correct / max(total_samples, 1)}


def save_best_checkpoint(monitor: str, epoch: int, model_name: str, model: nn.Module, optimizer: optim.Optimizer, scheduler, scaler: torch.amp.GradScaler, amp_enabled: bool, weights_dir: Path, metrics_all: Dict[str, Any], class_to_idx: Dict[str, int], idx_to_class: Dict[str, str], config: Dict[str, Any], history: Dict[str, Any], metric_value: float) -> None:
    ckpt_path = weights_dir / f"best_{monitor}_model.pth"
    sd_path = weights_dir / f"best_{monitor}_model_state_dict_only.pth"
    ckpt = {"epoch": epoch, "model_name": model_name, "monitor": f"{monitor}_accuracy", "monitor_accuracy": metric_value, "metrics_at_save_time": metrics_all, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict() if amp_enabled else None, "class_to_idx": class_to_idx, "idx_to_class": idx_to_class, "config": config}
    torch.save(ckpt, ckpt_path)
    torch.save(model.state_dict(), sd_path)
    history["best"][monitor] = {"epoch": epoch, "accuracy": metric_value, "loss": metrics_all[monitor]["loss"], "checkpoint_path": str(ckpt_path), "state_dict_path": str(sd_path)}
    if monitor == "val":
        torch.save(ckpt, weights_dir / "best_model.pth")
        torch.save(model.state_dict(), weights_dir / "best_model_state_dict_only.pth")


def plot_training_curves(history: Dict[str, Any], plots_dir: Path) -> None:
    data = history.get("epochs", [])
    if not data:
        return
    epochs = [e["epoch"] for e in data]
    lrs = [e["lr"] for e in data]
    for metric, ylabel, fname in [("loss", "Loss", "loss_curves.png"), ("accuracy", "Accuracy", "accuracy_curves.png")]:
        plt.figure(figsize=(10, 6))
        for split in ["train", "val", "test", "test_e"]:
            plt.plot(epochs, [e[split][metric] for e in data], marker="o", label=split if split != "test_e" else "Test_E")
        plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(f"{ylabel} Curves"); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig(plots_dir / fname, dpi=200); plt.close()
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lrs, marker="o", label="Learning rate")
    plt.xlabel("Epoch"); plt.ylabel("Learning rate"); plt.title("Learning Rate Schedule"); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig(plots_dir / "learning_rate_curve.png", dpi=200); plt.close()
    fig, axes = plt.subplots(3, 1, figsize=(11, 15))
    for split in ["train", "val", "test", "test_e"]:
        label = split if split != "test_e" else "Test_E"
        axes[0].plot(epochs, [e[split]["loss"] for e in data], marker="o", label=label)
        axes[1].plot(epochs, [e[split]["accuracy"] for e in data], marker="o", label=label)
    axes[2].plot(epochs, lrs, marker="o", label="Learning rate")
    for ax, title, ylabel in zip(axes, ["Loss", "Accuracy", "Learning Rate"], ["Loss", "Accuracy", "Learning rate"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(plots_dir / "combined_training_curves.png", dpi=200); plt.close(fig)



# =========================
# Full research result helpers
# =========================

def class_name_from_idx(idx_to_class: Dict[str, str], idx: int) -> str:
    return idx_to_class.get(str(idx), str(idx))


def get_positive_class_idx(class_to_idx: Dict[str, int]) -> int:
    lower_map = {str(k).lower(): int(v) for k, v in class_to_idx.items()}
    if "fire" in lower_map:
        return lower_map["fire"]
    if len(class_to_idx) == 2:
        return 1
    return 0


def balanced_accuracy_from_cm(cm: List[List[int]]) -> float:
    recalls = []
    for i in range(len(cm)):
        support = sum(cm[i])
        if support > 0:
            recalls.append(cm[i][i] / support)
    if not recalls:
        return 0.0
    return sum(recalls) / len(recalls)


def trapezoid_auc(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    area = 0.0
    for i in range(1, len(xs)):
        area += (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2.0
    return float(area)


def binary_roc_pr_curves(y_true: List[int], scores: List[float], positive_idx: int) -> Dict[str, Any]:
    labels = [1 if int(y) == int(positive_idx) else 0 for y in y_true]
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos

    if total_pos == 0 or total_neg == 0:
        return {
            "valid": False,
            "reason": "ROC/PR needs both positive and negative samples.",
            "roc_auc": None,
            "pr_auc_trapezoid": None,
            "average_precision": None,
            "roc_points": [],
            "pr_points": []
        }

    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)

    tp = 0
    fp = 0

    roc_points = [{"threshold": "inf", "fpr": 0.0, "tpr": 0.0}]
    pr_points = [{"threshold": "inf", "recall": 0.0, "precision": 1.0}]

    for i in order:
        score = float(scores[i])
        label = labels[i]

        if label == 1:
            tp += 1
        else:
            fp += 1

        fn = total_pos - tp
        tn = total_neg - fp

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tpr

        roc_points.append({"threshold": score, "fpr": fpr, "tpr": tpr})
        pr_points.append({"threshold": score, "recall": recall, "precision": precision})

    if roc_points[-1]["fpr"] != 1.0 or roc_points[-1]["tpr"] != 1.0:
        roc_points.append({"threshold": "-inf", "fpr": 1.0, "tpr": 1.0})

    roc_auc = trapezoid_auc([p["fpr"] for p in roc_points], [p["tpr"] for p in roc_points])
    pr_auc = trapezoid_auc([p["recall"] for p in pr_points], [p["precision"] for p in pr_points])

    ap = 0.0
    for i in range(1, len(pr_points)):
        delta_recall = pr_points[i]["recall"] - pr_points[i - 1]["recall"]
        if delta_recall > 0:
            ap += delta_recall * pr_points[i]["precision"]

    return {
        "valid": True,
        "positive_class_index": int(positive_idx),
        "num_positive": int(total_pos),
        "num_negative": int(total_neg),
        "roc_auc": float(roc_auc),
        "pr_auc_trapezoid": float(pr_auc),
        "average_precision": float(ap),
        "roc_points": roc_points,
        "pr_points": pr_points
    }


def save_curve_csv(points: List[Dict[str, Any]], path: Path, fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in points:
            writer.writerow({k: row.get(k) for k in fieldnames})


def plot_roc_curve(roc_points: List[Dict[str, Any]], roc_auc: Optional[float], title: str, save_path: Path) -> None:
    if not roc_points:
        return
    fpr = [p["fpr"] for p in roc_points]
    tpr = [p["tpr"] for p in roc_points]
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, marker="o", label=f"ROC AUC = {roc_auc:.4f}" if roc_auc is not None else "ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_pr_curve(pr_points: List[Dict[str, Any]], ap: Optional[float], title: str, save_path: Path) -> None:
    if not pr_points:
        return
    recall = [p["recall"] for p in pr_points]
    precision = [p["precision"] for p in pr_points]
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, marker="o", label=f"AP = {ap:.4f}" if ap is not None else "PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def build_eval_datasets(data_root: Path, h: int, w: int) -> Dict[str, datasets.ImageFolder]:
    # Use evaluation transform only. No random augmentation for final prediction export.
    _, eval_t = build_transforms(h, w, "off")
    return {
        split: datasets.ImageFolder(str(data_root / split), transform=eval_t)
        for split in ["train", "val", "test", "test_e"]
    }


@torch.no_grad()
def evaluate_and_export_predictions(
    model: nn.Module,
    dataset: datasets.ImageFolder,
    split_name: str,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    batch_size: int,
    num_workers: int,
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[str, str],
    results_dir: Path,
    plots_dir: Path,
    use_progress_bar: bool,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    model.eval()

    # Final prediction export can open many files because it loops through
    # train/val/test/test_e and writes prediction/ROC/PR files.
    # Keep this single-process by default to avoid:
    #   OSError: [Errno 24] Too many open files
    safe_workers, _ = get_safe_num_workers(num_workers)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=safe_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )

    num_classes = len(class_to_idx)
    class_names = [idx_to_class[str(i)] for i in range(num_classes)]
    positive_idx = get_positive_class_idx(class_to_idx)
    positive_class_name = class_name_from_idx(idx_to_class, positive_idx)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    y_true: List[int] = []
    y_pred: List[int] = []
    positive_scores: List[float] = []
    rows: List[Dict[str, Any]] = []

    forward_time_sec = 0.0
    start_index = 0

    bar = progress_iterator(loader, len(loader), f"Final {split_name}", use_progress_bar)

    for x, y in bar:
        paths = [dataset.samples[i][0] for i in range(start_index, start_index + y.size(0))]
        start_index += y.size(0)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        conf = torch.max(probs, dim=1).values

        if device.type == "cuda":
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        forward_time_sec += (t1 - t0)

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_correct += int((pred == y).sum().item())
        total_samples += bs

        y_cpu = y.detach().cpu().tolist()
        pred_cpu = pred.detach().cpu().tolist()
        probs_cpu = probs.detach().cpu().tolist()
        conf_cpu = conf.detach().cpu().tolist()

        y_true.extend(y_cpu)
        y_pred.extend(pred_cpu)
        positive_scores.extend([float(p[positive_idx]) for p in probs_cpu])

        for i in range(bs):
            row = {
                "split": split_name,
                "image_path": paths[i],
                "true_index": int(y_cpu[i]),
                "true_label": class_name_from_idx(idx_to_class, int(y_cpu[i])),
                "pred_index": int(pred_cpu[i]),
                "pred_label": class_name_from_idx(idx_to_class, int(pred_cpu[i])),
                "correct": int(y_cpu[i]) == int(pred_cpu[i]),
                "confidence": float(conf_cpu[i]),
                "positive_class": positive_class_name,
                "positive_probability": float(probs_cpu[i][positive_idx]),
            }
            for class_idx, class_name in enumerate(class_names):
                row[f"prob_{class_name}"] = float(probs_cpu[i][class_idx])
            rows.append(row)

        safe_set_postfix(bar, {
            "loss": f"{total_loss / max(total_samples, 1):.4f}",
            "acc": f"{total_correct / max(total_samples, 1):.4f}"
        })

    cm = confusion_matrix(y_true, y_pred, num_classes)
    per_cls = per_class_metrics(cm)
    bal_acc = balanced_accuracy_from_cm(cm)
    curve_data = binary_roc_pr_curves(y_true, positive_scores, positive_idx)

    prediction_csv = results_dir / f"predictions_{split_name}.csv"
    pred_fields = list(rows[0].keys()) if rows else [
        "split", "image_path", "true_index", "true_label", "pred_index",
        "pred_label", "correct", "confidence", "positive_class", "positive_probability"
    ]

    with open(prediction_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pred_fields)
        writer.writeheader()
        writer.writerows(rows)

    roc_csv = results_dir / f"roc_curve_{split_name}.csv"
    pr_csv = results_dir / f"precision_recall_curve_{split_name}.csv"

    if curve_data["valid"]:
        save_curve_csv(curve_data["roc_points"], roc_csv, ["threshold", "fpr", "tpr"])
        save_curve_csv(curve_data["pr_points"], pr_csv, ["threshold", "recall", "precision"])
        plot_roc_curve(curve_data["roc_points"], curve_data["roc_auc"], f"ROC curve - {split_name}", plots_dir / f"roc_curve_{split_name}.png")
        plot_pr_curve(curve_data["pr_points"], curve_data["average_precision"], f"Precision-Recall curve - {split_name}", plots_dir / f"precision_recall_curve_{split_name}.png")

    inference = {
        "forward_time_sec": float(forward_time_sec),
        "total_images": int(total_samples),
        "avg_forward_time_ms_per_image": float((forward_time_sec / max(total_samples, 1)) * 1000.0),
        "images_per_second_forward_only": float(total_samples / max(forward_time_sec, 1e-12)),
        "note": "Measured model forward pass time only, including CUDA synchronization when using GPU."
    }

    metrics = {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "balanced_accuracy": bal_acc,
        "correct": total_correct,
        "total": total_samples,
        "confusion_matrix": cm,
        "per_class": per_cls,
        "positive_class_index": int(positive_idx),
        "positive_class_name": positive_class_name,
        "roc_auc": curve_data.get("roc_auc"),
        "pr_auc_trapezoid": curve_data.get("pr_auc_trapezoid"),
        "average_precision": curve_data.get("average_precision"),
        "inference": inference,
        "prediction_csv": str(prediction_csv),
        "roc_curve_csv": str(roc_csv) if curve_data["valid"] else None,
        "precision_recall_curve_csv": str(pr_csv) if curve_data["valid"] else None,
        "roc_curve_plot": str(plots_dir / f"roc_curve_{split_name}.png") if curve_data["valid"] else None,
        "precision_recall_curve_plot": str(plots_dir / f"precision_recall_curve_{split_name}.png") if curve_data["valid"] else None,
    }

    metrics_json = results_dir / f"detailed_metrics_{split_name}.json"
    save_json(metrics, metrics_json)

    output_paths = {
        "prediction_csv": str(prediction_csv),
        "detailed_metrics_json": str(metrics_json),
        "roc_curve_csv": str(roc_csv) if curve_data["valid"] else "",
        "precision_recall_curve_csv": str(pr_csv) if curve_data["valid"] else "",
        "roc_curve_plot": str(plots_dir / f"roc_curve_{split_name}.png") if curve_data["valid"] else "",
        "precision_recall_curve_plot": str(plots_dir / f"precision_recall_curve_{split_name}.png") if curve_data["valid"] else "",
    }

    return metrics, output_paths


def model_size_mb_from_state_dict(model: nn.Module) -> float:
    total_bytes = 0
    for tensor in model.state_dict().values():
        total_bytes += tensor.numel() * tensor.element_size()
    return float(total_bytes / (1024 ** 2))


def save_model_recreation_artifacts(
    model: nn.Module,
    model_name: str,
    display: str,
    num_classes: int,
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[str, str],
    args: argparse.Namespace,
    small_image_mode: bool,
    img_h: int,
    img_w: int,
    size_tag: str,
    results_dir: Path
) -> Dict[str, Any]:
    structure_path = results_dir / "model_structure.txt"
    summary_path = results_dir / "model_summary.json"
    recreate_path = results_dir / "model_recreate_info.json"
    code_path = results_dir / "model_recreate_code.txt"

    with open(structure_path, "w", encoding="utf-8") as f:
        f.write(str(model))

    total_params = num_total_params(model)
    trainable_params = num_trainable_params(model)

    summary = {
        "model": model_name,
        "model_display": display,
        "num_classes": num_classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "total_parameters": int(total_params),
        "trainable_parameters_current": int(trainable_params),
        "non_trainable_parameters_current": int(total_params - trainable_params),
        "estimated_state_dict_size_mb": model_size_mb_from_state_dict(model),
        "pretrained": bool(args.pretrained),
        "dropout": float(args.dropout),
        "small_image_mode": bool(small_image_mode),
        "image_height": int(img_h),
        "image_width": int(img_w),
        "size_tag": size_tag,
    }

    recreate = {
        "purpose": "Use this information to recreate the same model architecture before loading the saved state_dict.",
        "model_name": model_name,
        "num_classes": num_classes,
        "pretrained_for_recreation": False,
        "pretrained_used_during_training": bool(args.pretrained),
        "small_image_mode": bool(small_image_mode),
        "dropout": float(args.dropout),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "image_resize": {"height": int(img_h), "width": int(img_w), "size_tag": size_tag},
        "recommended_weight_file": "best_val_model_state_dict_only.pth",
        "build_call": f"model = build_model('{model_name}', num_classes={num_classes}, pretrained=False, small_image_mode={bool(small_image_mode)}, dropout={float(args.dropout)})",
        "load_call": "model.load_state_dict(torch.load(PATH_TO_STATE_DICT, map_location=device))"
    }

    code = (
        "# Recreate model architecture for this run\n"
        "# Then load saved weights from weights/Run_XX/best_val_model_state_dict_only.pth\n\n"
        "import torch\n\n"
        f"model_name = \"{model_name}\"\n"
        f"num_classes = {num_classes}\n"
        f"small_image_mode = {bool(small_image_mode)}\n"
        f"dropout = {float(args.dropout)}\n\n"
        "# This assumes you are using the same train_fire_multimodel script.\n"
        "model = build_model(\n"
        "    model_name=model_name,\n"
        "    num_classes=num_classes,\n"
        "    pretrained=False,\n"
        "    small_image_mode=small_image_mode,\n"
        "    dropout=dropout\n"
        ")\n\n"
        "state = torch.load(\"PATH_TO/best_val_model_state_dict_only.pth\", map_location=\"cpu\")\n"
        "model.load_state_dict(state)\n"
        "model.eval()\n"
    )

    save_json(summary, summary_path)
    save_json(recreate, recreate_path)

    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code)

    return {
        "model_structure_txt": str(structure_path),
        "model_summary_json": str(summary_path),
        "model_recreate_info_json": str(recreate_path),
        "model_recreate_code_txt": str(code_path),
        "summary": summary,
    }


def append_master_summary(final_results: Dict[str, Any], master_base: Path) -> None:
    master_base.mkdir(parents=True, exist_ok=True)

    csv_path = master_base / "master_summary.csv"
    jsonl_path = master_base / "master_summary.jsonl"

    row = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": final_results.get("model"),
        "model_display": final_results.get("model_display"),
        "image_size": _get_nested_value(final_results, ["image_resize", "size_tag"]),
        "run_collection_dir": final_results.get("run_collection_dir"),
        "run_name": final_results.get("run_name"),
        "weights_dir": final_results.get("weights_dir"),
        "results_dir": final_results.get("results_dir"),
        "seed": _get_nested_value(final_results, ["config_snapshot", "seed"]),
        "total_parameters": _get_nested_value(final_results, ["model_summary", "total_parameters"]),
        "estimated_state_dict_size_mb": _get_nested_value(final_results, ["model_summary", "estimated_state_dict_size_mb"]),
        "final_train_accuracy": _get_nested_value(final_results, ["final_evaluation_using_best_val_model", "train", "accuracy"]),
        "final_val_accuracy": _get_nested_value(final_results, ["final_evaluation_using_best_val_model", "val", "accuracy"]),
        "final_test_accuracy": _get_nested_value(final_results, ["final_evaluation_using_best_val_model", "test", "accuracy"]),
        "final_test_e_accuracy": _get_nested_value(final_results, ["final_evaluation_using_best_val_model", "test_e", "accuracy"]),
        "final_train_balanced_accuracy": _get_nested_value(final_results, ["final_evaluation_using_best_val_model", "train", "balanced_accuracy"]),
        "final_val_balanced_accuracy": _get_nested_value(final_results, ["final_evaluation_using_best_val_model", "val", "balanced_accuracy"]),
        "final_test_balanced_accuracy": _get_nested_value(final_results, ["final_evaluation_using_best_val_model", "test", "balanced_accuracy"]),
        "final_test_e_balanced_accuracy": _get_nested_value(final_results, ["final_evaluation_using_best_val_model", "test_e", "balanced_accuracy"]),
        "final_train_roc_auc": _get_nested_value(final_results, ["final_evaluation_using_best_val_model", "train", "roc_auc"]),
        "final_val_roc_auc": _get_nested_value(final_results, ["final_evaluation_using_best_val_model", "val", "roc_auc"]),
        "final_test_roc_auc": _get_nested_value(final_results, ["final_evaluation_using_best_val_model", "test", "roc_auc"]),
        "final_test_e_roc_auc": _get_nested_value(final_results, ["final_evaluation_using_best_val_model", "test_e", "roc_auc"]),
        "final_train_average_precision": _get_nested_value(final_results, ["final_evaluation_using_best_val_model", "train", "average_precision"]),
        "final_val_average_precision": _get_nested_value(final_results, ["final_evaluation_using_best_val_model", "val", "average_precision"]),
        "final_test_average_precision": _get_nested_value(final_results, ["final_evaluation_using_best_val_model", "test", "average_precision"]),
        "final_test_e_average_precision": _get_nested_value(final_results, ["final_evaluation_using_best_val_model", "test_e", "average_precision"]),
        "test_inference_ms_per_image": _get_nested_value(final_results, ["final_evaluation_using_best_val_model", "test", "inference", "avg_forward_time_ms_per_image"]),
        "test_e_inference_ms_per_image": _get_nested_value(final_results, ["final_evaluation_using_best_val_model", "test_e", "inference", "avg_forward_time_ms_per_image"]),
        "best_val_epoch": _get_nested_value(final_results, ["best_epochs", "val"]),
        "best_val_accuracy": _get_nested_value(final_results, ["best_accuracies", "val"]),
    }

    fieldnames = list(row.keys())

    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists or csv_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)

    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def plot_confusion_matrix(cm: List[List[int]], class_names: List[str], title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right"); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label"); ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    fig.colorbar(im, ax=ax); plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close(fig)


def train_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    model_name = normalize_model_name(args.model)
    display = MODEL_DISPLAY[model_name]
    data_root = resolve_data_root(args)
    img_h, img_w, size_tag, size_source = resolve_image_size(args, data_root)
    small_image_mode = resolve_small_image_mode(args, img_h, img_w)
    runs_root, run_name, weights_dir, results_dir, plots_dir = resolve_run_paths(args, model_name, size_tag)
    device = get_device(args.device)
    info = device_info(device)

    safe_workers, workers_note = get_safe_num_workers(args.num_workers)
    args.num_workers = safe_workers

    if device.type == "cuda":
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass
    amp_enabled = bool(args.use_fp16) and device.type == "cuda"

    print("\n" + "="*80)
    print("Fire/Nofire Multi-Model Training")
    print("="*80)
    print(f"Model              : {display} ({model_name})")
    print(f"Data root          : {data_root}")
    print(f"Image resize       : {img_h}x{img_w} ({size_source})")
    print(f"Run collection     : {runs_root}")
    print(f"Run name           : {run_name}")
    print(f"Weights directory  : {weights_dir}")
    print(f"Results directory  : {results_dir}")
    print(f"Device             : {device}")
    if device.type == "cuda":
        print(f"GPU name           : {info.get('cuda_device_name')}")
        print(f"GPU memory         : {info.get('cuda_total_memory_gb')} GB")
    print(f"FP16 enabled       : {amp_enabled}")
    print(f"DataLoader workers : {args.num_workers}")
    print(f"Workers note       : {workers_note}")
    print(f"Freeze epochs      : {args.freeze_epochs}")
    print("="*80 + "\n")

    dsets = build_datasets(data_root, img_h, img_w, args.augment)
    bal_report = balance_report(dsets, data_root)
    save_json(bal_report, results_dir / "dataset_balance_report.json")
    loaders, sampler_info = build_loaders(dsets, args)
    class_to_idx = dsets["train"].class_to_idx
    idx_to_class = {str(v): k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    model = build_model(model_name, num_classes, args.pretrained, small_image_mode, args.dropout).to(device)
    if args.freeze_epochs > 0:
        set_backbone_trainable(model, model_name, trainable=False)
        phase = "frozen_backbone"
        start_lr = args.lr
    else:
        set_backbone_trainable(model, model_name, trainable=True)
        phase = "full_finetune"
        start_lr = args.lr

    model_artifacts = save_model_recreation_artifacts(
        model=model,
        model_name=model_name,
        display=display,
        num_classes=num_classes,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        args=args,
        small_image_mode=small_image_mode,
        img_h=img_h,
        img_w=img_w,
        size_tag=size_tag,
        results_dir=results_dir
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(model, start_lr, args.weight_decay)
    scheduler = build_scheduler(args, optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    config = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script": "train_fire_multimodel.py",
        "model": model_name,
        "model_display": display,
        "pretrained": args.pretrained,
        "num_classes": num_classes,
        "supported_models": sorted(set(MODEL_ALIASES.values())),
        "image_resize": {"height": img_h, "width": img_w, "size_tag": size_tag, "source": size_source},
        "augmentation": args.augment,
        "regularization": {"weight_decay": args.weight_decay, "dropout": args.dropout, "label_smoothing": args.label_smoothing},
        "freeze_then_finetune": {"freeze_epochs": args.freeze_epochs, "head_only_lr": args.lr, "finetune_lr": args.finetune_lr, "reset_scheduler_on_unfreeze": True},
        "small_image_mode": small_image_mode,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": "Adam",
        "scheduler": {"name": "ReduceLROnPlateau", "mode": "max", "monitor": "val_accuracy", "factor": args.lr_factor, "patience": args.lr_patience, "threshold": args.lr_threshold, "min_lr": args.min_lr},
        "early_stopping": {"monitor": "val_accuracy", "patience": args.early_stop_patience},
        "weighted_sampler": sampler_info,
        "device": str(device),
        "device_info": info,
        "amp_enabled": amp_enabled,
        "seed": args.seed,
        "model_summary": model_artifacts["summary"],
        "model_artifacts": {k: v for k, v in model_artifacts.items() if k != "summary"},
        "paths": {"runs_root": str(runs_root), "run_name": run_name, "weights_dir": str(weights_dir), "results_dir": str(results_dir), "plots_dir": str(plots_dir), "organization": "root/weights/Run_XX and root/results/Run_XX"},
        "dataset_info": {"data_root": str(data_root), "class_to_idx": class_to_idx, "idx_to_class": idx_to_class, "splits": {s: {"folder": str(data_root / s), "num_images_by_imagefolder": len(dsets[s]), "num_images_counted": count_images(data_root / s)} for s in ["train", "val", "test", "test_e"]}},
    }
    save_json(config, results_dir / "config.json")
    save_json(class_to_idx, results_dir / "class_to_idx.json")

    history = {"config": config, "epochs": [], "best": {m: {"epoch": None, "accuracy": -1.0, "loss": None, "checkpoint_path": str(weights_dir / f"best_{m}_model.pth"), "state_dict_path": str(weights_dir / f"best_{m}_model_state_dict_only.pth")} for m in ["val", "train", "test", "test_e"]}}
    best = {"val": -1.0, "train": -1.0, "test": -1.0, "test_e": -1.0}
    best_epoch = {"val": 0, "train": 0, "test": 0, "test_e": 0}
    no_improve = 0
    total_params = num_total_params(model)
    csv_path = results_dir / "training_log.csv"
    csv_fields = ["epoch", "phase", "lr", "trainable_params", "total_params", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "test_loss", "test_accuracy", "test_e_loss", "test_e_accuracy", "epoch_time_sec", "is_best_val", "is_best_train", "is_best_test", "is_best_test_e", "epochs_without_improvement"]

    for epoch in range(1, args.epochs + 1):
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            print(f"\nUnfreezing backbone at epoch {epoch}. Switching to full fine-tuning. LR={args.finetune_lr}\n")
            set_backbone_trainable(model, model_name, trainable=True)
            phase = "full_finetune"
            optimizer = build_optimizer(model, args.finetune_lr, args.weight_decay)
            scheduler = build_scheduler(args, optimizer)
        t0 = time.time()
        train_m = train_one_epoch(model, loaders["train"], criterion, optimizer, scaler, device, amp_enabled, epoch, args.epochs, args.progress_bar)
        val_m = evaluate(model, loaders["val"], criterion, device, amp_enabled, num_classes, "Val", epoch, args.epochs, args.progress_bar)
        scheduler.step(val_m["accuracy"])
        test_m = evaluate(model, loaders["test"], criterion, device, amp_enabled, num_classes, "Test", epoch, args.epochs, args.progress_bar)
        teste_m = evaluate(model, loaders["test_e"], criterion, device, amp_enabled, num_classes, "Test_E", epoch, args.epochs, args.progress_bar)
        metrics_all = {"train": train_m, "val": val_m, "test": test_m, "test_e": teste_m}
        flags = {"val": val_m["accuracy"] > best["val"], "train": train_m["accuracy"] > best["train"], "test": test_m["accuracy"] > best["test"], "test_e": teste_m["accuracy"] > best["test_e"]}
        if flags["val"]:
            best["val"] = val_m["accuracy"]; best_epoch["val"] = epoch; no_improve = 0
        else:
            no_improve += 1
        for mon, metrics in [("train", train_m), ("val", val_m), ("test", test_m), ("test_e", teste_m)]:
            if flags[mon]:
                best[mon] = metrics["accuracy"]; best_epoch[mon] = epoch
                save_best_checkpoint(mon, epoch, model_name, model, optimizer, scheduler, scaler, amp_enabled, weights_dir, metrics_all, class_to_idx, idx_to_class, config, history, best[mon])
        epoch_time = time.time() - t0
        row = {"epoch": epoch, "phase": phase, "lr": get_lr(optimizer), "trainable_params": num_trainable_params(model), "total_params": total_params, "train_loss": train_m["loss"], "train_accuracy": train_m["accuracy"], "val_loss": val_m["loss"], "val_accuracy": val_m["accuracy"], "test_loss": test_m["loss"], "test_accuracy": test_m["accuracy"], "test_e_loss": teste_m["loss"], "test_e_accuracy": teste_m["accuracy"], "epoch_time_sec": epoch_time, "is_best_val": flags["val"], "is_best_train": flags["train"], "is_best_test": flags["test"], "is_best_test_e": flags["test_e"], "epochs_without_improvement": no_improve}
        history["epochs"].append({"epoch": epoch, "phase": phase, "lr": row["lr"], "trainable_params": row["trainable_params"], "total_params": total_params, "train": train_m, "val": val_m, "test": test_m, "test_e": teste_m, "epoch_time_sec": epoch_time, "is_best_val": flags["val"], "is_best_train": flags["train"], "is_best_test": flags["test"], "is_best_test_e": flags["test_e"], "epochs_without_improvement": no_improve})
        save_json(history, results_dir / "training_history.json")
        append_csv_row(csv_path, row, csv_fields)
        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] Model: {display} | Phase: {phase} | LR: {row['lr']:.2e} | Train Acc: {train_m['accuracy']:.4f} | Val Acc: {val_m['accuracy']:.4f} | Test Acc: {test_m['accuracy']:.4f} | Test_E Acc: {teste_m['accuracy']:.4f} | " + ("BEST_VAL" if flags["val"] else f"No val improve: {no_improve}") + (" | BEST_TRAIN" if flags["train"] else "") + (" | BEST_TEST" if flags["test"] else "") + (" | BEST_TEST_E" if flags["test_e"] else ""))
        plot_training_curves(history, plots_dir)
        if no_improve >= args.early_stop_patience:
            print(f"\nEarly stopping: validation accuracy did not improve for {args.early_stop_patience} consecutive epochs.")
            break

    torch.save({"epoch": history["epochs"][-1]["epoch"], "model_name": model_name, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict() if amp_enabled else None, "class_to_idx": class_to_idx, "idx_to_class": idx_to_class, "config": config, "history": history}, weights_dir / "last_model.pth")
    torch.save(model.state_dict(), weights_dir / "last_model_state_dict_only.pth")

    best_val_path = weights_dir / "best_val_model_state_dict_only.pth"
    if best_val_path.exists():
        state = torch.load(best_val_path, map_location=device)
    elif (weights_dir / "best_model_state_dict_only.pth").exists():
        state = torch.load(weights_dir / "best_model_state_dict_only.pth", map_location=device)
    else:
        ckpt = torch.load(weights_dir / "best_model.pth", map_location=device, weights_only=False)
        state = ckpt["model_state_dict"]
    model.load_state_dict(state)

    final_dsets = build_eval_datasets(data_root, img_h, img_w)
    final_eval = {}
    prediction_exports = {}

    for split in ["train", "val", "test", "test_e"]:
        final_eval[split], prediction_exports[split] = evaluate_and_export_predictions(
            model=model,
            dataset=final_dsets[split],
            split_name=split,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            batch_size=args.batch_size,
            num_workers=args.final_num_workers,
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
            results_dir=results_dir,
            plots_dir=plots_dir,
            use_progress_bar=args.progress_bar,
        )

    class_names = [idx_to_class[str(i)] for i in range(num_classes)]

    for split in ["train", "val", "test", "test_e"]:
        plot_confusion_matrix(final_eval[split]["confusion_matrix"], class_names, f"{split} confusion matrix using best validation model", plots_dir / f"confusion_matrix_{split}_best_val_model.png")

    final_results = {
        "final_reporting_model": "best validation accuracy model",
        "model": model_name,
        "model_display": display,
        "best_epochs": best_epoch,
        "best_accuracies": best,
        "final_evaluation_using_best_val_model": final_eval,
        "prediction_exports": prediction_exports,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "run_collection_dir": str(runs_root),
        "run_name": run_name,
        "weights_dir": str(weights_dir),
        "results_dir": str(results_dir),
        "output_organization": "root/weights/Run_XX and root/results/Run_XX",
        "best_weight_paths": {m: {"checkpoint": str(weights_dir / f"best_{m}_model.pth"), "state_dict": str(weights_dir / f"best_{m}_model_state_dict_only.pth")} for m in ["val", "train", "test", "test_e"]},
        "training_history_path": str(results_dir / "training_history.json"),
        "training_log_csv_path": str(csv_path),
        "config_path": str(results_dir / "config.json"),
        "dataset_balance_report_path": str(results_dir / "dataset_balance_report.json"),
        "model_summary": model_artifacts["summary"],
        "model_artifacts": {k: v for k, v in model_artifacts.items() if k != "summary"},
        "image_resize": {"height": img_h, "width": img_w, "size_tag": size_tag, "source": size_source},
        "device_info": info,
        "config_snapshot": config,
        "master_summary_paths": {
            "csv": str(Path(args.runs_base) / "master_summary.csv"),
            "jsonl": str(Path(args.runs_base) / "master_summary.jsonl")
        }
    }

    save_json(final_results, results_dir / "final_results.json")
    append_master_summary(final_results, Path(args.runs_base))
    plot_training_curves(history, plots_dir)
    print("\nFinal results using BEST VALIDATION model:")
    for split in ["train", "val", "test", "test_e"]:
        print(f"{split:6s} Acc: {final_eval[split]['accuracy']:.4f}, Bal Acc: {final_eval[split]['balanced_accuracy']:.4f}, ROC AUC: {final_eval[split]['roc_auc'] if final_eval[split]['roc_auc'] is not None else 'N/A'}, Loss: {final_eval[split]['loss']:.4f}")
    print("\n" + "="*80)
    print("Training finished.")
    print(f"Model          : {display}")
    print(f"Image size     : {size_tag}")
    print(f"Run collection : {runs_root}")
    print(f"Run name       : {run_name}")
    print(f"Weights folder : {weights_dir}")
    print(f"Results folder : {results_dir}")
    print(f"Best val epoch : {best_epoch['val']}, Acc: {best['val']:.4f}")
    print(f"Final results  : {results_dir / 'final_results.json'}")
    print("="*80 + "\n")

    # Cleanup local heavy objects before the next repeat/dataset starts.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_results


# =========================
# Repeat training helpers
# =========================

def _get_nested_value(data: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _mean(values: List[float]) -> float:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return float("nan")
    return sum(clean) / len(clean)


def _std(values: List[float]) -> float:
    clean = [float(v) for v in values if v is not None]
    n = len(clean)
    if n <= 1:
        return 0.0
    m = _mean(clean)
    return math.sqrt(sum((x - m) ** 2 for x in clean) / (n - 1))


def save_repeat_summary(repeat_results: List[Dict[str, Any]], summary_dir: Path) -> None:
    """
    Save repeat_summary.csv and repeat_summary.json after all repeats finish.
    The summary is saved inside the common results folder that contains Run_01, Run_02, ...
    """
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for idx, result in enumerate(repeat_results, start=1):
        row = {
            "repeat": idx,
            "model": result.get("model"),
            "model_display": result.get("model_display"),
            "run_collection_dir": result.get("run_collection_dir"),
            "run_name": result.get("run_name"),
            "weights_dir": result.get("weights_dir"),
            "results_dir": result.get("results_dir"),

            "best_val_epoch": _get_nested_value(result, ["best_epochs", "val"]),
            "best_train_epoch": _get_nested_value(result, ["best_epochs", "train"]),
            "best_test_epoch": _get_nested_value(result, ["best_epochs", "test"]),
            "best_test_e_epoch": _get_nested_value(result, ["best_epochs", "test_e"]),

            "best_val_accuracy": _get_nested_value(result, ["best_accuracies", "val"]),
            "best_train_accuracy": _get_nested_value(result, ["best_accuracies", "train"]),
            "best_test_accuracy": _get_nested_value(result, ["best_accuracies", "test"]),
            "best_test_e_accuracy": _get_nested_value(result, ["best_accuracies", "test_e"]),

            "final_train_accuracy": _get_nested_value(result, ["final_evaluation_using_best_val_model", "train", "accuracy"]),
            "final_val_accuracy": _get_nested_value(result, ["final_evaluation_using_best_val_model", "val", "accuracy"]),
            "final_test_accuracy": _get_nested_value(result, ["final_evaluation_using_best_val_model", "test", "accuracy"]),
            "final_test_e_accuracy": _get_nested_value(result, ["final_evaluation_using_best_val_model", "test_e", "accuracy"]),

            "final_train_balanced_accuracy": _get_nested_value(result, ["final_evaluation_using_best_val_model", "train", "balanced_accuracy"]),
            "final_val_balanced_accuracy": _get_nested_value(result, ["final_evaluation_using_best_val_model", "val", "balanced_accuracy"]),
            "final_test_balanced_accuracy": _get_nested_value(result, ["final_evaluation_using_best_val_model", "test", "balanced_accuracy"]),
            "final_test_e_balanced_accuracy": _get_nested_value(result, ["final_evaluation_using_best_val_model", "test_e", "balanced_accuracy"]),

            "final_train_roc_auc": _get_nested_value(result, ["final_evaluation_using_best_val_model", "train", "roc_auc"]),
            "final_val_roc_auc": _get_nested_value(result, ["final_evaluation_using_best_val_model", "val", "roc_auc"]),
            "final_test_roc_auc": _get_nested_value(result, ["final_evaluation_using_best_val_model", "test", "roc_auc"]),
            "final_test_e_roc_auc": _get_nested_value(result, ["final_evaluation_using_best_val_model", "test_e", "roc_auc"]),

            "final_train_average_precision": _get_nested_value(result, ["final_evaluation_using_best_val_model", "train", "average_precision"]),
            "final_val_average_precision": _get_nested_value(result, ["final_evaluation_using_best_val_model", "val", "average_precision"]),
            "final_test_average_precision": _get_nested_value(result, ["final_evaluation_using_best_val_model", "test", "average_precision"]),
            "final_test_e_average_precision": _get_nested_value(result, ["final_evaluation_using_best_val_model", "test_e", "average_precision"]),

            "test_inference_ms_per_image": _get_nested_value(result, ["final_evaluation_using_best_val_model", "test", "inference", "avg_forward_time_ms_per_image"]),
            "test_e_inference_ms_per_image": _get_nested_value(result, ["final_evaluation_using_best_val_model", "test_e", "inference", "avg_forward_time_ms_per_image"]),

            "total_parameters": _get_nested_value(result, ["model_summary", "total_parameters"]),
            "estimated_state_dict_size_mb": _get_nested_value(result, ["model_summary", "estimated_state_dict_size_mb"]),

            "final_train_loss": _get_nested_value(result, ["final_evaluation_using_best_val_model", "train", "loss"]),
            "final_val_loss": _get_nested_value(result, ["final_evaluation_using_best_val_model", "val", "loss"]),
            "final_test_loss": _get_nested_value(result, ["final_evaluation_using_best_val_model", "test", "loss"]),
            "final_test_e_loss": _get_nested_value(result, ["final_evaluation_using_best_val_model", "test_e", "loss"]),
        }
        rows.append(row)

    csv_path = summary_dir / "repeat_summary.csv"

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    metrics_for_stats = [
        "best_val_accuracy",
        "best_train_accuracy",
        "best_test_accuracy",
        "best_test_e_accuracy",
        "final_train_accuracy",
        "final_val_accuracy",
        "final_test_accuracy",
        "final_test_e_accuracy",
        "final_train_balanced_accuracy",
        "final_val_balanced_accuracy",
        "final_test_balanced_accuracy",
        "final_test_e_balanced_accuracy",
        "final_train_roc_auc",
        "final_val_roc_auc",
        "final_test_roc_auc",
        "final_test_e_roc_auc",
        "final_train_average_precision",
        "final_val_average_precision",
        "final_test_average_precision",
        "final_test_e_average_precision",
        "test_inference_ms_per_image",
        "test_e_inference_ms_per_image",
        "total_parameters",
        "estimated_state_dict_size_mb",
        "final_train_loss",
        "final_val_loss",
        "final_test_loss",
        "final_test_e_loss",
    ]

    stats = {}

    for metric in metrics_for_stats:
        values = [row.get(metric) for row in rows if row.get(metric) is not None]
        stats[metric] = {
            "mean": _mean(values),
            "std": _std(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "values": values
        }

    summary = {
        "num_repeats": len(repeat_results),
        "summary_dir": str(summary_dir),
        "csv_path": str(csv_path),
        "rows": rows,
        "statistics": stats,
        "note": (
            "final_* metrics use the best validation model from each repeat. "
            "Mean/std values are calculated across independent repeat runs."
        )
    }

    save_json(summary, summary_dir / "repeat_summary.json")

    print("\n" + "=" * 80)
    print("Repeat summary saved.")
    print(f"Summary folder : {summary_dir}")
    print(f"CSV summary    : {csv_path}")
    print(f"JSON summary   : {summary_dir / 'repeat_summary.json'}")
    print("-" * 80)

    for metric in ["final_val_accuracy", "final_test_accuracy", "final_test_e_accuracy"]:
        mean_val = stats[metric]["mean"]
        std_val = stats[metric]["std"]
        print(f"{metric:22s}: mean={mean_val:.4f}, std={std_val:.4f}")

    print("=" * 80 + "\n")


def save_all_dataset_summary(all_dataset_summaries: List[Dict[str, Any]], summary_base: Path, model_display: str) -> None:
    """
    Save one overall summary when multiple dataset-size folders are trained.
    """
    summary_base.mkdir(parents=True, exist_ok=True)

    csv_path = summary_base / "all_dataset_summary.csv"
    json_path = summary_base / "all_dataset_summary.json"

    rows = []

    for item in all_dataset_summaries:
        row = {
            "dataset_root": item.get("dataset_root"),
            "size_tag": item.get("size_tag"),
            "repeat_count": item.get("repeat_count"),
            "summary_dir": item.get("summary_dir"),
            "repeat_summary_csv": item.get("repeat_summary_csv"),
            "repeat_summary_json": item.get("repeat_summary_json"),
            "final_val_accuracy_mean": _get_nested_value(item, ["statistics", "final_val_accuracy", "mean"]),
            "final_val_accuracy_std": _get_nested_value(item, ["statistics", "final_val_accuracy", "std"]),
            "final_test_accuracy_mean": _get_nested_value(item, ["statistics", "final_test_accuracy", "mean"]),
            "final_test_accuracy_std": _get_nested_value(item, ["statistics", "final_test_accuracy", "std"]),
            "final_test_e_accuracy_mean": _get_nested_value(item, ["statistics", "final_test_e_accuracy", "mean"]),
            "final_test_e_accuracy_std": _get_nested_value(item, ["statistics", "final_test_e_accuracy", "std"]),
            "best_val_accuracy_mean": _get_nested_value(item, ["statistics", "best_val_accuracy", "mean"]),
            "best_test_accuracy_mean": _get_nested_value(item, ["statistics", "best_test_accuracy", "mean"]),
            "best_test_e_accuracy_mean": _get_nested_value(item, ["statistics", "best_test_e_accuracy", "mean"]),
            "final_test_balanced_accuracy_mean": _get_nested_value(item, ["statistics", "final_test_balanced_accuracy", "mean"]),
            "final_test_e_balanced_accuracy_mean": _get_nested_value(item, ["statistics", "final_test_e_balanced_accuracy", "mean"]),
            "final_test_roc_auc_mean": _get_nested_value(item, ["statistics", "final_test_roc_auc", "mean"]),
            "final_test_e_roc_auc_mean": _get_nested_value(item, ["statistics", "final_test_e_roc_auc", "mean"]),
            "test_inference_ms_per_image_mean": _get_nested_value(item, ["statistics", "test_inference_ms_per_image", "mean"]),
            "test_e_inference_ms_per_image_mean": _get_nested_value(item, ["statistics", "test_e_inference_ms_per_image", "mean"]),
            "total_parameters": _get_nested_value(item, ["statistics", "total_parameters", "mean"]),
            "estimated_state_dict_size_mb": _get_nested_value(item, ["statistics", "estimated_state_dict_size_mb", "mean"]),
        }
        rows.append(row)

    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    save_json(
        {
            "model_display": model_display,
            "num_dataset_roots": len(all_dataset_summaries),
            "summary_base": str(summary_base),
            "csv_path": str(csv_path),
            "rows": rows,
            "dataset_summaries": all_dataset_summaries,
            "note": (
                "This file summarizes all dataset-size folders trained in one command. "
                "Each individual size folder also has its own results/repeat_summary.csv/json."
            )
        },
        json_path
    )

    print("\n" + "=" * 80)
    print("All-dataset summary saved.")
    print(f"Summary folder : {summary_base}")
    print(f"CSV summary    : {csv_path}")
    print(f"JSON summary   : {json_path}")
    print("=" * 80 + "\n")


def run_repeats_for_one_dataset(args: argparse.Namespace, dataset_root: Path, dataset_index: int, total_datasets: int) -> Dict[str, Any]:
    """
    Run all repeats for one concrete dataset root.
    """
    repeat_count = int(args.repeat)

    if repeat_count < 1:
        raise ValueError("--repeat must be >= 1")

    base_seed = int(args.seed)
    repeat_results = []

    # Set the concrete dataset root for train_model().
    base_args = copy.deepcopy(args)
    base_args.data_root = str(dataset_root)

    # Detect size tag for printing and summary.
    img_h, img_w, size_tag, size_source = resolve_image_size(base_args, dataset_root)

    print("\n" + "=" * 80)
    print(f"DATASET {dataset_index}/{total_datasets}")
    print(f"Dataset root : {dataset_root}")
    print(f"Image size   : {size_tag} ({size_source})")
    print(f"Repeats      : {repeat_count}")
    print("=" * 80 + "\n")

    for repeat_idx in range(1, repeat_count + 1):
        repeat_args = copy.deepcopy(base_args)
        repeat_args.seed = base_seed + repeat_idx - 1

        print("\n" + "#" * 80)
        print(f"STARTING DATASET {dataset_index}/{total_datasets} | REPEAT {repeat_idx}/{repeat_count}")
        print(f"Dataset root: {dataset_root}")
        print(f"Seed        : {repeat_args.seed}")
        print("#" * 80 + "\n")

        result = train_model(repeat_args)
        repeat_results.append(result)

        # Cleanup between repeats to reduce open file handles and GPU memory pressure.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\n" + "#" * 80)
        print(f"FINISHED DATASET {dataset_index}/{total_datasets} | REPEAT {repeat_idx}/{repeat_count}")
        print("#" * 80 + "\n")

    first_results_dir = Path(repeat_results[0]["results_dir"])
    summary_dir = first_results_dir.parent

    save_repeat_summary(repeat_results, summary_dir)

    repeat_summary_json = summary_dir / "repeat_summary.json"
    repeat_summary_csv = summary_dir / "repeat_summary.csv"

    summary_data = {}
    if repeat_summary_json.exists():
        with open(repeat_summary_json, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

    return {
        "dataset_root": str(dataset_root),
        "size_tag": size_tag,
        "repeat_count": repeat_count,
        "summary_dir": str(summary_dir),
        "repeat_summary_csv": str(repeat_summary_csv),
        "repeat_summary_json": str(repeat_summary_json),
        "statistics": summary_data.get("statistics", {}),
        "rows": summary_data.get("rows", []),
    }


def run_repeated_training(args: argparse.Namespace) -> None:
    """
    Main training launcher.

    If --data_root is given:
        train only that dataset root, repeat X times.

    If --data_root is not given:
        scan --dataset_base for folders like 32x32, 64x64, 128x128,
        then train each folder one by one, repeating each X times.
    """
    model_name = normalize_model_name(args.model)
    model_display = MODEL_DISPLAY[model_name]

    dataset_roots = discover_dataset_roots(args)

    print("\n" + "=" * 80)
    print("Resolved dataset folders for training")
    print("=" * 80)
    for i, root in enumerate(dataset_roots, start=1):
        print(f"{i}. {root}")
    print("=" * 80 + "\n")

    all_dataset_summaries = []

    for dataset_index, dataset_root in enumerate(dataset_roots, start=1):
        dataset_summary = run_repeats_for_one_dataset(
            args=args,
            dataset_root=dataset_root,
            dataset_index=dataset_index,
            total_datasets=len(dataset_roots)
        )
        all_dataset_summaries.append(dataset_summary)

    # Save combined summary only when multiple dataset roots were trained.
    if len(all_dataset_summaries) > 1:
        if str(args.runs_root).lower() == "auto":
            summary_base = Path(args.runs_base) / f"{model_display}_ALL_SIZES_summary"
        else:
            summary_base = Path(args.runs_root) / f"{model_display}_ALL_SIZES_summary"

        save_all_dataset_summary(
            all_dataset_summaries=all_dataset_summaries,
            summary_base=summary_base,
            model_display=model_display
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multiple pretrained CNN models on a fire/nofire dataset.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=DETAILED_HELP_TEXT
    )

    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="Model to train. Options: resnet18, resnet34, resnet50, mobilenet_v2, mobilenet_v3_large, efficientnet_b0, efficientnet_b1, vgg16_bn, vgg19_bn, densenet121."
    )

    parser.add_argument("--dataset_base", type=str, default="Dataset", help="Base dataset folder to scan when --data_root is not given. Default: Dataset")
    parser.add_argument("--dataset_size", type=str, default="auto", help="Dataset size folder inside dataset_base, e.g. 128x128. Usually use --data_root instead.")
    parser.add_argument("--data_root", type=str, default=None, help="Optional direct dataset path containing train/val/test/test_e, e.g. Dataset/128x128. If omitted, scans --dataset_base for 32x32/64x64/128x128 folders.")

    parser.add_argument("--img_size", type=str, default="auto", help="Image size. Default auto detects from folder name, e.g. Dataset/128x128 -> 128.")
    parser.add_argument("--img_height", type=int, default=None, help="Optional manual image height. Use with --img_width.")
    parser.add_argument("--img_width", type=int, default=None, help="Optional manual image width. Use with --img_height.")

    parser.add_argument("--runs_base", type=str, default="Runs", help="Base folder for automatic run folders. Default: Runs")
    parser.add_argument("--runs_root", type=str, default="auto", help="Default auto creates Runs/<Model>_<Size>_01/Run_XX. Or give a custom root folder.")

    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers. Default 0 is safest on Windows/CUDA. Linux users may try 2 or 4.")
    parser.add_argument("--final_num_workers", type=int, default=0, help="Workers for final prediction/ROC/PR export. Default 0 prevents Too many open files errors.")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="auto uses GPU if available, otherwise CPU.")
    parser.add_argument("--use_fp16", type=str_to_bool, default=True, help="Use FP16 mixed precision on CUDA. Use 1/0.")

    parser.add_argument("--pretrained", type=str_to_bool, default=True, help="Use pretrained ImageNet weights. Use 1/0.")
    parser.add_argument("--small_image_mode", type=str, default="auto", choices=["auto", "yes", "no"], help="For small images, modifies ResNet stem when auto/yes.")

    parser.add_argument("--augment", type=str, default="strong", choices=["off", "basic", "strong"], help="Training augmentation strength.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Adam weight decay.")
    parser.add_argument("--dropout", type=float, default=0.30, help="Dropout before final classifier.")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="CrossEntropy label smoothing.")

    parser.add_argument("--freeze_epochs", type=int, default=5, help="Train classifier head only for this many epochs, then unfreeze all.")
    parser.add_argument("--lr", type=float, default=1e-4, help="LR for classifier-head/frozen-backbone phase.")
    parser.add_argument("--finetune_lr", type=float, default=5e-5, help="LR after unfreezing full model.")

    parser.add_argument("--use_weighted_sampler", type=str, default="auto", choices=["no", "yes", "auto"], help="Weighted sampler for class imbalance.")
    parser.add_argument("--weighted_sampler_threshold", type=float, default=1.5, help="Use sampler in auto mode if class imbalance ratio exceeds this.")

    parser.add_argument("--lr_patience", type=int, default=5, help="Reduce LR if val accuracy does not improve for this many epochs.")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="LR reduction factor.")
    parser.add_argument("--lr_threshold", type=float, default=1e-4, help="LR scheduler improvement threshold.")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="Minimum LR.")
    parser.add_argument("--early_stop_patience", type=int, default=20, help="Stop if val accuracy does not improve for this many epochs.")
    parser.add_argument("--repeat", type=int, default=1, help="Run the complete training process X times. Example: --repeat 5")

    parser.add_argument("--progress_bar", type=str_to_bool, default=True, help="Show tqdm progress bars. Use 1/0.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    return parser.parse_args()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(DETAILED_HELP_TEXT)
        sys.exit(0)

    run_repeated_training(parse_args())
