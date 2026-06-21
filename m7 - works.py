#!/usr/bin/env python3
# ============================================================
# WINDOWS OPENMP FIX
# ============================================================
# This must run BEFORE importing torch, torchvision, numpy, matplotlib, etc.
# It fixes Windows error:
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ============================================================
# WINDOWS OPENMP FIX
# ============================================================
# This must run BEFORE importing torch, torchvision, numpy, matplotlib, etc.
# It fixes Windows error:
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

"""
Fix included:
- Sets KMP_DUPLICATE_LIB_OK=TRUE before imports to avoid Windows libiomp5md.dll duplicate OpenMP crash.
- Uses matplotlib Agg backend to save plots without GUI backend conflicts.
- Uses --device auto/cuda/cpu for GPU control.

Train ResNet-18 on a 2-class fire/nofire dataset.

Expected dataset structure:

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

Outputs are saved automatically as:
Runs/Resnet18_01/Run_01
Runs/Resnet18_01/Run_02
...

Example:
python train_resnet18_fire.py --data_root "Dataset/128x128" --runs_root "Runs/Resnet18_01" --epochs 50 --batch_size 32 --use_fp16 1
"""

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False

# This helps training continue even if one image file is slightly truncated.
ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================
# Utility functions
# =========================

def set_seed(seed: int) -> None:
    """Make the run more reproducible."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # More reproducible, slightly slower.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def str_to_bool(x: Any) -> bool:
    """Accept 1/0, true/false, yes/no from command line."""
    if isinstance(x, bool):
        return x
    x = str(x).strip().lower()
    if x in ["1", "true", "yes", "y", "on"]:
        return True
    if x in ["0", "false", "no", "n", "off"]:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected: use 1/0 or true/false.")


def create_next_run_dir(runs_root: Path) -> Path:
    """
    Create a new run folder:
    Runs/Resnet18_01/Run_01, Run_02, Run_03, ...
    """
    runs_root.mkdir(parents=True, exist_ok=True)

    existing_numbers = []
    for p in runs_root.glob("Run_*"):
        if p.is_dir():
            try:
                existing_numbers.append(int(p.name.split("_")[-1]))
            except ValueError:
                pass

    next_number = max(existing_numbers, default=0) + 1
    run_dir = runs_root / f"Run_{next_number:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save dictionary as pretty JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def append_csv_row(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    """Append one row to CSV, creating header if needed."""
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_lr(optimizer: optim.Optimizer) -> float:
    """Return current learning rate from first optimizer group."""
    return optimizer.param_groups[0]["lr"]


def count_images(folder: Path) -> int:
    """Count common image files recursively."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in exts)


# =========================
# Dataset and dataloaders
# =========================


def get_training_device(device_arg: str) -> torch.device:
    """
    Select training device.

    --device auto : use NVIDIA GPU/CUDA if available, otherwise CPU
    --device cuda : force CUDA/GPU, error if CUDA is not available
    --device cpu  : force CPU
    """
    device_arg = str(device_arg).strip().lower()

    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "You selected --device cuda, but CUDA is not available. "
                "Install the CUDA version of PyTorch and make sure your NVIDIA GPU driver is working."
            )
        return torch.device("cuda")

    if device_arg == "cpu":
        return torch.device("cpu")

    raise ValueError("--device must be one of: auto, cuda, cpu")


def get_device_info(device: torch.device) -> Dict[str, Any]:
    """Return useful device/GPU information for printing and JSON logging."""
    info = {
        "selected_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "pytorch_version": torch.__version__,
        "pytorch_cuda_version": torch.version.cuda,
    }

    if device.type == "cuda":
        current_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(current_index)

        info.update({
            "cuda_current_device_index": current_index,
            "cuda_device_name": torch.cuda.get_device_name(current_index),
            "cuda_total_memory_gb": round(props.total_memory / (1024 ** 3), 3),
            "cuda_capability": f"{props.major}.{props.minor}",
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        })

    return info


# =========================
# Dataset and dataloaders
# =========================

def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    ImageNet normalization is used because ResNet-18 pretrained weights
    were trained using ImageNet-style normalization.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.20,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_tfms, eval_tfms


def build_datasets(data_root: Path, img_size: int) -> Dict[str, datasets.ImageFolder]:
    """
    Build ImageFolder datasets for:
    train, val, test, test_e
    """
    required_splits = ["train", "val", "test", "test_e"]
    train_tfms, eval_tfms = build_transforms(img_size)

    split_to_transform = {
        "train": train_tfms,
        "val": eval_tfms,
        "test": eval_tfms,
        "test_e": eval_tfms,
    }

    dataset_dict = {}
    for split in required_splits:
        split_path = data_root / split
        if not split_path.exists():
            raise FileNotFoundError(f"Missing dataset split folder: {split_path}")

        dataset_dict[split] = datasets.ImageFolder(
            root=str(split_path),
            transform=split_to_transform[split]
        )

    # Check class consistency across all splits.
    train_class_to_idx = dataset_dict["train"].class_to_idx
    for split in required_splits:
        if dataset_dict[split].class_to_idx != train_class_to_idx:
            raise ValueError(
                f"Class folder mismatch in split '{split}'.\n"
                f"train class_to_idx = {train_class_to_idx}\n"
                f"{split} class_to_idx = {dataset_dict[split].class_to_idx}\n"
                "Make sure each split has the same class folders, e.g., fire and nofire."
            )

    if len(train_class_to_idx) != 2:
        raise ValueError(
            f"This script expects exactly 2 classes, but found {len(train_class_to_idx)}: "
            f"{list(train_class_to_idx.keys())}"
        )

    return dataset_dict


def build_loaders(
    dataset_dict: Dict[str, datasets.ImageFolder],
    batch_size: int,
    num_workers: int,
    seed: int
) -> Dict[str, DataLoader]:
    """Build DataLoader objects."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    loaders = {
        "train": DataLoader(
            dataset_dict["train"],
            shuffle=True,
            generator=generator,
            **common_kwargs
        ),
        "val": DataLoader(
            dataset_dict["val"],
            shuffle=False,
            **common_kwargs
        ),
        "test": DataLoader(
            dataset_dict["test"],
            shuffle=False,
            **common_kwargs
        ),
        "test_e": DataLoader(
            dataset_dict["test_e"],
            shuffle=False,
            **common_kwargs
        ),
    }

    return loaders


# =========================
# Model
# =========================

def build_resnet18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Load ResNet-18 with pretrained ImageNet weights and replace the final FC layer
    for a 2-class fire/nofire dataset.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model



# =========================
# Progress bar helper
# =========================

def progress_iterator(iterable, total: int, desc: str, enabled: bool):
    """
    Return tqdm progress iterator if tqdm is installed and enabled.
    Otherwise return the original iterable.
    """
    if enabled and TQDM_AVAILABLE:
        return tqdm(
            iterable,
            total=total,
            desc=desc,
            leave=False,
            dynamic_ncols=True
        )

    return iterable


def safe_set_postfix(iterator, values: Dict[str, Any]) -> None:
    """
    Update tqdm postfix safely.
    If tqdm is not available or progress bar is disabled, this does nothing.
    """
    if TQDM_AVAILABLE and hasattr(iterator, "set_postfix"):
        iterator.set_postfix(values)


# =========================
# Metrics
# =========================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    num_classes: int,
    split_name: str = "eval",
    epoch: int = 0,
    total_epochs: int = 0,
    use_progress_bar: bool = True
) -> Dict[str, Any]:
    """Evaluate model on one split with optional terminal progress bar."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    y_true = []
    y_pred = []

    if epoch > 0 and total_epochs > 0:
        desc = f"{split_name} {epoch}/{total_epochs}"
    else:
        desc = split_name

    progress_loader = progress_iterator(
        loader,
        total=len(loader),
        desc=desc,
        enabled=use_progress_bar
    )

    for images, labels in progress_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds == labels).sum().item())
        total_samples += batch_size

        running_loss = total_loss / max(total_samples, 1)
        running_acc = total_correct / max(total_samples, 1)

        safe_set_postfix(
            progress_loader,
            {
                "loss": f"{running_loss:.4f}",
                "acc": f"{running_acc:.4f}"
            }
        )

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)

    cm = confusion_matrix_from_lists(y_true, y_pred, num_classes=num_classes)
    per_class = per_class_metrics(cm)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "correct": total_correct,
        "total": total_samples,
        "confusion_matrix": cm,
        "per_class": per_class,
    }


def confusion_matrix_from_lists(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int
) -> List[List[int]]:
    """Simple confusion matrix without sklearn dependency."""
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1
    return cm


def per_class_metrics(cm: List[List[int]]) -> Dict[str, Dict[str, float]]:
    """
    Compute precision, recall, F1 for each class index.
    cm[row=true][col=pred]
    """
    num_classes = len(cm)
    out = {}

    for c in range(num_classes):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(num_classes) if r != c)
        fn = sum(cm[c][r] for r in range(num_classes) if r != c)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        out[str(c)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(cm[c]),
        }

    return out


# =========================
# Training
# =========================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    epoch: int,
    total_epochs: int,
    use_progress_bar: bool = True
) -> Dict[str, float]:
    """Train for one epoch with terminal progress bar."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_loader = progress_iterator(
        loader,
        total=len(loader),
        desc=f"Train {epoch}/{total_epochs}",
        enabled=use_progress_bar
    )

    for images, labels in progress_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        preds = torch.argmax(outputs.detach(), dim=1)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds == labels).sum().item())
        total_samples += batch_size

        running_loss = total_loss / max(total_samples, 1)
        running_acc = total_correct / max(total_samples, 1)

        safe_set_postfix(
            progress_loader,
            {
                "loss": f"{running_loss:.4f}",
                "acc": f"{running_acc:.4f}"
            }
        )

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def train_model(args: argparse.Namespace) -> None:
    """Main training function."""
    set_seed(args.seed)

    data_root = Path(args.data_root)
    runs_root = Path(args.runs_root)
    run_dir = create_next_run_dir(runs_root)

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    device = get_training_device(args.device)
    device_info = get_device_info(device)

    if device.type == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # FP16 is only enabled on CUDA here. On CPU it will safely use FP32.
    amp_enabled = bool(args.use_fp16) and device.type == "cuda"

    print("\n" + "=" * 80)
    print("ResNet-18 Fire/Nofire Training")
    print("=" * 80)
    print(f"Data root      : {data_root}")
    print(f"Run directory  : {run_dir}")
    print(f"Device         : {device}")
    print(f"CUDA available : {torch.cuda.is_available()}")
    print(f"CUDA devices   : {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA   : {torch.version.cuda}")
    if device.type == "cuda":
        print(f"GPU name       : {device_info.get('cuda_device_name')}")
        print(f"GPU memory     : {device_info.get('cuda_total_memory_gb')} GB")
    else:
        print("GPU note       : Running on CPU because CUDA is not available to this Python/PyTorch environment.")
        print("                 Try --device cuda to force an error with details.")
    print(f"FP16 enabled   : {amp_enabled}")
    print(f"Initial LR     : {args.lr}")
    print("=" * 80 + "\n")

    dataset_dict = build_datasets(data_root=data_root, img_size=args.img_size)
    loaders = build_loaders(
        dataset_dict=dataset_dict,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )

    class_to_idx = dataset_dict["train"].class_to_idx
    idx_to_class = {str(v): k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    dataset_info = {
        "data_root": str(data_root),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "splits": {
            split: {
                "folder": str(data_root / split),
                "num_images_by_imagefolder": len(dataset_dict[split]),
                "num_images_counted": count_images(data_root / split),
            }
            for split in ["train", "val", "test", "test_e"]
        },
        "note": {
            "train_val_test": "Same-domain dataset splits.",
            "test_e": "External-domain / other-domain test split."
        }
    }

    model = build_resnet18(num_classes=num_classes, pretrained=args.pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=args.lr_threshold,
        min_lr=args.min_lr
    )

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    config = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script": "train_resnet18_fire.py",
        "model": "resnet18",
        "pretrained": args.pretrained,
        "num_classes": num_classes,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "progress_bar": args.progress_bar,
        "tqdm_available": TQDM_AVAILABLE,
        "lr": args.lr,
        "optimizer": "Adam",
        "weight_decay": args.weight_decay,
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "mode": "max",
            "monitor": "val_accuracy",
            "factor": args.lr_factor,
            "patience": args.lr_patience,
            "threshold": args.lr_threshold,
            "min_lr": args.min_lr
        },
        "early_stopping": {
            "monitor": "val_accuracy",
            "patience": args.early_stop_patience
        },
        "use_fp16_requested": args.use_fp16,
        "amp_enabled": amp_enabled,
        "device": str(device),
        "device_argument": args.device,
        "device_info": device_info,
        "omp_fix": {
            "KMP_DUPLICATE_LIB_OK": os.environ.get("KMP_DUPLICATE_LIB_OK"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS")
        },
        "seed": args.seed,
        "dataset_info": dataset_info,
    }

    save_json(config, run_dir / "config.json")
    save_json(class_to_idx, run_dir / "class_to_idx.json")

    history = {
        "config": config,
        "epochs": [],
        "best": {
            "val": {
                "epoch": None,
                "accuracy": -1.0,
                "loss": None,
                "checkpoint_path": str(checkpoints_dir / "best_val_model.pth"),
                "state_dict_path": str(checkpoints_dir / "best_val_model_state_dict_only.pth")
            },
            "train": {
                "epoch": None,
                "accuracy": -1.0,
                "loss": None,
                "checkpoint_path": str(checkpoints_dir / "best_train_model.pth"),
                "state_dict_path": str(checkpoints_dir / "best_train_model_state_dict_only.pth")
            },
            "test": {
                "epoch": None,
                "accuracy": -1.0,
                "loss": None,
                "checkpoint_path": str(checkpoints_dir / "best_test_model.pth"),
                "state_dict_path": str(checkpoints_dir / "best_test_model_state_dict_only.pth")
            },
            "test_e": {
                "epoch": None,
                "accuracy": -1.0,
                "loss": None,
                "checkpoint_path": str(checkpoints_dir / "best_test_e_model.pth"),
                "state_dict_path": str(checkpoints_dir / "best_test_e_model_state_dict_only.pth")
            }
        }
    }

    csv_path = run_dir / "training_log.csv"
    csv_fields = [
        "epoch",
        "lr",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "test_loss",
        "test_accuracy",
        "test_e_loss",
        "test_e_accuracy",
        "epoch_time_sec",
        "is_best_val",
        "is_best_train",
        "is_best_test",
        "is_best_test_e",
        "epochs_without_improvement"
    ]

    best_val_acc = -1.0
    best_train_acc = -1.0
    best_test_acc = -1.0
    best_test_e_acc = -1.0

    best_val_epoch = 0
    best_train_epoch = 0
    best_test_epoch = 0
    best_test_e_epoch = 0

    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            epoch=epoch,
            total_epochs=args.epochs,
            use_progress_bar=args.progress_bar
        )

        val_metrics = evaluate(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            num_classes=num_classes,
            split_name="Val",
            epoch=epoch,
            total_epochs=args.epochs,
            use_progress_bar=args.progress_bar
        )

        # Main LR update based on validation accuracy.
        scheduler.step(val_metrics["accuracy"])

        # Evaluate test and external-domain test_e after every epoch.
        # These results are logged for monitoring, but model selection still uses val accuracy only.
        test_metrics = evaluate(
            model=model,
            loader=loaders["test"],
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            num_classes=num_classes,
            split_name="Test",
            epoch=epoch,
            total_epochs=args.epochs,
            use_progress_bar=args.progress_bar
        )

        test_e_metrics = evaluate(
            model=model,
            loader=loaders["test_e"],
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            num_classes=num_classes,
            split_name="Test_E",
            epoch=epoch,
            total_epochs=args.epochs,
            use_progress_bar=args.progress_bar
        )

        epoch_time = time.time() - epoch_start
        current_lr = get_lr(optimizer)

        is_best_val = val_metrics["accuracy"] > best_val_acc
        is_best_train = train_metrics["accuracy"] > best_train_acc
        is_best_test = test_metrics["accuracy"] > best_test_acc
        is_best_test_e = test_e_metrics["accuracy"] > best_test_e_acc

        def save_best_checkpoint(monitor_name: str, metrics: Dict[str, Any], metric_value: float) -> None:
            """
            Save a full checkpoint and a state_dict-only file for the selected best monitor.
            monitor_name must be one of: val, train, test, test_e
            """
            checkpoint_path = checkpoints_dir / f"best_{monitor_name}_model.pth"
            state_dict_path = checkpoints_dir / f"best_{monitor_name}_model_state_dict_only.pth"

            best_ckpt = {
                "epoch": epoch,
                "model_name": "resnet18",
                "monitor": f"{monitor_name}_accuracy",
                "monitor_accuracy": metric_value,
                "metrics_at_save_time": {
                    "train": train_metrics,
                    "val": val_metrics,
                    "test": test_metrics,
                    "test_e": test_e_metrics,
                },
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if amp_enabled else None,
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "config": config,
            }

            torch.save(best_ckpt, checkpoint_path)
            torch.save(model.state_dict(), state_dict_path)

            history["best"][monitor_name] = {
                "epoch": epoch,
                "accuracy": metric_value,
                "loss": metrics["loss"],
                "checkpoint_path": str(checkpoint_path),
                "state_dict_path": str(state_dict_path)
            }

            # Compatibility aliases for the validation-best model.
            # Older code expected best_model.pth and best_model_state_dict_only.pth.
            if monitor_name == "val":
                torch.save(best_ckpt, checkpoints_dir / "best_model.pth")
                torch.save(model.state_dict(), checkpoints_dir / "best_model_state_dict_only.pth")

        if is_best_val:
            best_val_acc = val_metrics["accuracy"]
            best_val_epoch = epoch
            epochs_without_improvement = 0
            save_best_checkpoint("val", val_metrics, best_val_acc)
        else:
            epochs_without_improvement += 1

        if is_best_train:
            best_train_acc = train_metrics["accuracy"]
            best_train_epoch = epoch
            save_best_checkpoint("train", train_metrics, best_train_acc)

        if is_best_test:
            best_test_acc = test_metrics["accuracy"]
            best_test_epoch = epoch
            save_best_checkpoint("test", test_metrics, best_test_acc)

        if is_best_test_e:
            best_test_e_acc = test_e_metrics["accuracy"]
            best_test_e_epoch = epoch
            save_best_checkpoint("test_e", test_e_metrics, best_test_e_acc)

        epoch_record = {
            "epoch": epoch,
            "lr": current_lr,
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "test_e": test_e_metrics,
            "epoch_time_sec": epoch_time,
            "is_best_val": is_best_val,
            "is_best_train": is_best_train,
            "is_best_test": is_best_test,
            "is_best_test_e": is_best_test_e,
            "epochs_without_improvement": epochs_without_improvement
        }

        history["epochs"].append(epoch_record)

        # Save after every epoch so progress is not lost if training stops/crashes.
        save_json(history, run_dir / "training_history.json")

        append_csv_row(
            csv_path,
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_e_loss": test_e_metrics["loss"],
                "test_e_accuracy": test_e_metrics["accuracy"],
                "epoch_time_sec": epoch_time,
                "is_best_val": is_best_val,
                "is_best_train": is_best_train,
                "is_best_test": is_best_test,
                "is_best_test_e": is_best_test_e,
                "epochs_without_improvement": epochs_without_improvement,
            },
            fieldnames=csv_fields
        )

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Test Acc: {test_metrics['accuracy']:.4f} | "
            f"Test_E Acc: {test_e_metrics['accuracy']:.4f} | "
            f"{'BEST_VAL' if is_best_val else f'No val improve: {epochs_without_improvement}'}"
            f"{' | BEST_TRAIN' if is_best_train else ''}"
            f"{' | BEST_TEST' if is_best_test else ''}"
            f"{' | BEST_TEST_E' if is_best_test_e else ''}"
        )

        # Update plots every epoch.
        plot_training_curves(history, plots_dir)

        if epochs_without_improvement >= args.early_stop_patience:
            print(
                f"\nEarly stopping triggered: validation accuracy did not improve for "
                f"{args.early_stop_patience} consecutive epochs."
            )
            break

    # Save last checkpoint.
    last_ckpt = {
        "epoch": history["epochs"][-1]["epoch"],
        "model_name": "resnet18",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if amp_enabled else None,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "config": config,
        "history": history,
    }
    torch.save(last_ckpt, checkpoints_dir / "last_model.pth")

    # Load the best model weights safely for final test/test_e evaluation.
    # We load the state_dict-only file instead of the full checkpoint dictionary.
    # This avoids PyTorch 2.6+ torch.load(weights_only=True) unpickling issues.
    best_state_dict_path = checkpoints_dir / "best_val_model_state_dict_only.pth"
    if best_state_dict_path.exists():
        best_state_dict = torch.load(best_state_dict_path, map_location=device)
        model.load_state_dict(best_state_dict)
    elif (checkpoints_dir / "best_model_state_dict_only.pth").exists():
        best_state_dict = torch.load(checkpoints_dir / "best_model_state_dict_only.pth", map_location=device)
        model.load_state_dict(best_state_dict)
    else:
        # Fallback only for old runs/scripts that did not save state_dict-only weights.
        # This checkpoint was created by this training script, so weights_only=False is acceptable here.
        best_checkpoint = torch.load(
            checkpoints_dir / "best_model.pth",
            map_location=device,
            weights_only=False
        )
        model.load_state_dict(best_checkpoint["model_state_dict"])

    print("\nLoading best model and evaluating final test sets only...")
    final_eval = {}
    for split in ["test", "test_e"]:
        final_eval[split] = evaluate(
            model=model,
            loader=loaders[split],
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            num_classes=num_classes,
            split_name=f"Final {split}",
            epoch=0,
            total_epochs=0,
            use_progress_bar=args.progress_bar
        )

    final_results = {
        "best_epochs": {
            "val": best_val_epoch,
            "train": best_train_epoch,
            "test": best_test_epoch,
            "test_e": best_test_e_epoch
        },
        "best_accuracies": {
            "val": best_val_acc,
            "train": best_train_acc,
            "test": best_test_acc,
            "test_e": best_test_e_acc
        },
        "final_test_evaluation_using_best_model": final_eval,
        "note": "train/val/test/test_e were evaluated after every epoch. Separate best weights were saved for train accuracy, val accuracy, test accuracy, and test_e accuracy. Final test evaluation still uses the best validation-accuracy model.",
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "run_dir": str(run_dir),
        "best_model_path": str(checkpoints_dir / "best_model.pth"),
        "best_model_state_dict_only_path": str(checkpoints_dir / "best_model_state_dict_only.pth"),
        "best_weight_paths": {
            "val": {
                "checkpoint": str(checkpoints_dir / "best_val_model.pth"),
                "state_dict": str(checkpoints_dir / "best_val_model_state_dict_only.pth")
            },
            "train": {
                "checkpoint": str(checkpoints_dir / "best_train_model.pth"),
                "state_dict": str(checkpoints_dir / "best_train_model_state_dict_only.pth")
            },
            "test": {
                "checkpoint": str(checkpoints_dir / "best_test_model.pth"),
                "state_dict": str(checkpoints_dir / "best_test_model_state_dict_only.pth")
            },
            "test_e": {
                "checkpoint": str(checkpoints_dir / "best_test_e_model.pth"),
                "state_dict": str(checkpoints_dir / "best_test_e_model_state_dict_only.pth")
            }
        },
        "last_model_path": str(checkpoints_dir / "last_model.pth"),
        "training_history_path": str(run_dir / "training_history.json"),
        "training_log_csv_path": str(csv_path),
    }

    save_json(final_results, run_dir / "final_results.json")

    # Save final confusion matrix plots using best model.
    for split in ["test", "test_e"]:
        plot_confusion_matrix(
            cm=final_eval[split]["confusion_matrix"],
            class_names=[idx_to_class[str(i)] for i in range(num_classes)],
            title=f"{split} confusion matrix",
            save_path=plots_dir / f"confusion_matrix_{split}.png"
        )

    # Final plots.
    plot_training_curves(history, plots_dir)

    print("\nFinal test results using best model:")
    print(f"Test   Acc: {final_eval['test']['accuracy']:.4f}, Loss: {final_eval['test']['loss']:.4f}")
    print(f"Test_E Acc: {final_eval['test_e']['accuracy']:.4f}, Loss: {final_eval['test_e']['loss']:.4f}")

    print("\n" + "=" * 80)
    print("Training finished.")
    print(f"Best val epoch    : {best_val_epoch}, Acc: {best_val_acc:.4f}")
    print(f"Best train epoch  : {best_train_epoch}, Acc: {best_train_acc:.4f}")
    print(f"Best test epoch   : {best_test_epoch}, Acc: {best_test_acc:.4f}")
    print(f"Best test_e epoch : {best_test_e_epoch}, Acc: {best_test_e_acc:.4f}")
    print(f"Run folder        : {run_dir}")
    print(f"Best val weights  : {checkpoints_dir / 'best_val_model.pth'}")
    print(f"Best train weights: {checkpoints_dir / 'best_train_model.pth'}")
    print(f"Best test weights : {checkpoints_dir / 'best_test_model.pth'}")
    print(f"Best test_e weights: {checkpoints_dir / 'best_test_e_model.pth'}")
    print(f"JSON history      : {run_dir / 'training_history.json'}")
    print(f"Final results     : {run_dir / 'final_results.json'}")
    print("=" * 80 + "\n")


# =========================
# Plotting
# =========================

def plot_training_curves(history: Dict[str, Any], plots_dir: Path) -> None:
    """
    Save training plots.

    This version logs:
    - train
    - val
    - test
    - test_e

    Model selection and early stopping still use validation accuracy only.
    """
    epochs_data = history.get("epochs", [])
    if not epochs_data:
        return

    epochs = [x["epoch"] for x in epochs_data]

    train_loss = [x["train"]["loss"] for x in epochs_data]
    val_loss = [x["val"]["loss"] for x in epochs_data]
    test_loss = [x["test"]["loss"] for x in epochs_data]
    test_e_loss = [x["test_e"]["loss"] for x in epochs_data]

    train_acc = [x["train"]["accuracy"] for x in epochs_data]
    val_acc = [x["val"]["accuracy"] for x in epochs_data]
    test_acc = [x["test"]["accuracy"] for x in epochs_data]
    test_e_acc = [x["test_e"]["accuracy"] for x in epochs_data]

    lrs = [x["lr"] for x in epochs_data]

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, marker="o", label="Train")
    plt.plot(epochs, val_loss, marker="o", label="Val")
    plt.plot(epochs, test_loss, marker="o", label="Test")
    plt.plot(epochs, test_e_loss, marker="o", label="Test_E")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_curves.png", dpi=200)
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, marker="o", label="Train")
    plt.plot(epochs, val_acc, marker="o", label="Val")
    plt.plot(epochs, test_acc, marker="o", label="Test")
    plt.plot(epochs, test_e_acc, marker="o", label="Test_E")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "accuracy_curves.png", dpi=200)
    plt.close()

    # LR plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lrs, marker="o", label="Learning rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "learning_rate_curve.png", dpi=200)
    plt.close()

    # Combined plot
    fig, axes = plt.subplots(3, 1, figsize=(11, 15))

    axes[0].plot(epochs, train_loss, marker="o", label="Train")
    axes[0].plot(epochs, val_loss, marker="o", label="Val")
    axes[0].plot(epochs, test_loss, marker="o", label="Test")
    axes[0].plot(epochs, test_e_loss, marker="o", label="Test_E")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", label="Train")
    axes[1].plot(epochs, val_acc, marker="o", label="Val")
    axes[1].plot(epochs, test_acc, marker="o", label="Test")
    axes[1].plot(epochs, test_e_acc, marker="o", label="Test_E")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, lrs, marker="o", label="Learning rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning rate")
    axes[2].set_title("Learning Rate")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(plots_dir / "combined_training_curves.png", dpi=200)
    plt.close()


def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    title: str,
    save_path: Path
) -> None:
    """Save confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# =========================
# Argument parser
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained ResNet-18 on fire/nofire dataset."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="Dataset/128x128",
        help="Path containing train, val, test, test_e folders."
    )
    parser.add_argument(
        "--runs_root",
        type=str,
        default="Runs/Resnet18_01",
        help="Root folder where Run_01, Run_02, ... will be created."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of epochs. Early stopping may stop earlier."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size."
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=128,
        help="Input image size. Your dataset is 128x128, so default is 128."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Adam weight decay."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers. Use 0 if Windows gives multiprocessing issues."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Training device. auto uses GPU/CUDA if available, otherwise CPU."
    )
    parser.add_argument(
        "--progress_bar",
        type=str_to_bool,
        default=True,
        help="Show tqdm progress bars in terminal. Use 1/0 or true/false."
    )
    parser.add_argument(
        "--use_fp16",
        type=str_to_bool,
        default=True,
        help="Use FP16 mixed precision on CUDA. Use 1/0 or true/false."
    )
    parser.add_argument(
        "--pretrained",
        type=str_to_bool,
        default=True,
        help="Use pretrained ImageNet weights."
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=5,
        help="Reduce LR if val accuracy does not improve for this many epochs."
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=0.5,
        help="LR reduction factor for ReduceLROnPlateau."
    )
    parser.add_argument(
        "--lr_threshold",
        type=float,
        default=1e-4,
        help="Improvement threshold for ReduceLROnPlateau."
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-7,
        help="Minimum learning rate."
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=10,
        help="Stop if val accuracy does not improve for this many consecutive epochs."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args)