#!/usr/bin/env python3
"""
Train ResNet-18 on a 2-class fire/nofire dataset.

Updated version includes:
- Strong task-relevant augmentations for fire/nofire images.
- Regularization: weight decay, dropout, label smoothing.
- Freeze-then-finetune strategy:
    * First N epochs: freeze ResNet backbone, train classifier head only.
    * Then unfreeze all layers and fine-tune full model.
- Class/domain balancing checks saved to JSON.
- Optional weighted sampler for imbalanced training set.
- Train/val/test/test_e checked after every epoch.
- Separate best weights saved for:
    * best validation accuracy
    * best training accuracy
    * best test accuracy
    * best test_e accuracy
- Final reporting uses the best validation model.
- GPU/CUDA support with FP16 option.
- tqdm progress bars.
- Windows OpenMP fix.

Expected dataset structure:

Dataset/32x32/
    train/fire
    train/nofire
    val/fire
    val/nofire
    test/fire
    test/nofire
    test_e/fire
    test_e/nofire
"""

# ============================================================
# WINDOWS OPENMP FIX
# ============================================================
# Must run before importing torch / torchvision / numpy / matplotlib.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFile
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================
# Basic utilities
# =========================

def str_to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x

    x = str(x).strip().lower()

    if x in ["1", "true", "yes", "y", "on"]:
        return True

    if x in ["0", "false", "no", "n", "off"]:
        return False

    raise argparse.ArgumentTypeError("Boolean value expected: use 1/0 or true/false.")


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
    file_exists = path.exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def count_images(folder: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in exts)


def create_next_run_dir(runs_root: Path) -> Path:
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


def get_lr(optimizer: optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def get_num_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_num_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# =========================
# Device utilities
# =========================

def get_training_device(device_arg: str) -> torch.device:
    device_arg = str(device_arg).strip().lower()

    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "\nCUDA/GPU was forced using --device cuda, but PyTorch cannot see CUDA.\n\n"
                "Check with:\n"
                "python -c \"import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)\"\n"
            )
        return torch.device("cuda")

    if device_arg == "cpu":
        return torch.device("cpu")

    raise ValueError("--device must be one of: auto, cuda, cpu")


def get_device_info(device: torch.device) -> Dict[str, Any]:
    info = {
        "selected_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "pytorch_version": torch.__version__,
        "pytorch_cuda_version": torch.version.cuda,
    }

    if torch.cuda.is_available():
        gpu_list = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_list.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(props.total_memory / (1024 ** 3), 3),
                "cuda_capability": f"{props.major}.{props.minor}",
            })
        info["available_gpus"] = gpu_list

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
# Argument/path utilities
# =========================

def resolve_data_root(args: argparse.Namespace) -> Path:
    if args.data_root is not None and str(args.data_root).strip() != "":
        return Path(args.data_root)

    return Path(args.dataset_base) / args.dataset_size


def resolve_image_size(args: argparse.Namespace) -> Tuple[int, int, str]:
    if args.img_height is not None or args.img_width is not None:
        if args.img_height is None or args.img_width is None:
            raise ValueError("For non-square resizing, give both --img_height and --img_width.")
        h = int(args.img_height)
        w = int(args.img_width)
    else:
        h = int(args.img_size)
        w = int(args.img_size)

    if h <= 0 or w <= 0:
        raise ValueError("Image height and width must be positive.")

    return h, w, f"{h}x{w}"


def resolve_small_image_mode(args: argparse.Namespace, img_h: int, img_w: int) -> bool:
    mode = str(args.small_image_mode).strip().lower()

    if mode == "yes":
        return True

    if mode == "no":
        return False

    if mode == "auto":
        return max(img_h, img_w) <= 64

    raise ValueError("--small_image_mode must be one of: auto, yes, no")


# =========================
# Progress bar helpers
# =========================

def progress_iterator(iterable, total: int, desc: str, enabled: bool):
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
    if TQDM_AVAILABLE and hasattr(iterator, "set_postfix"):
        iterator.set_postfix(values)


# =========================
# Dataset / transforms
# =========================

def build_transforms(img_h: int, img_w: int, augment_mode: str) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Strong augmentation is designed for fire/nofire generalization:
    - brightness/contrast/color variation
    - small perspective/rotation/translation changes
    - blur/sharpness changes
    - random erasing to reduce over-reliance on one region
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    augment_mode = str(augment_mode).strip().lower()

    if augment_mode == "off":
        train_tfms = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

    elif augment_mode == "basic":
        train_tfms = transforms.Compose([
            transforms.Resize((img_h, img_w)),
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

    elif augment_mode == "strong":
        train_tfms = transforms.Compose([
            transforms.RandomResizedCrop(
                size=(img_h, img_w),
                scale=(0.70, 1.00),
                ratio=(0.85, 1.15)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.35,
                    contrast=0.35,
                    saturation=0.30,
                    hue=0.05
                )
            ], p=0.85),
            transforms.RandomAutocontrast(p=0.25),
            transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.20),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.90, 1.10)
            ),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.20),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
            ], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.12),
                ratio=(0.30, 3.30),
                value="random"
            ),
        ])

    else:
        raise ValueError("--augment must be one of: off, basic, strong")

    eval_tfms = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_tfms, eval_tfms


def build_datasets(data_root: Path, img_h: int, img_w: int, augment_mode: str) -> Dict[str, datasets.ImageFolder]:
    required_splits = ["train", "val", "test", "test_e"]

    train_tfms, eval_tfms = build_transforms(
        img_h=img_h,
        img_w=img_w,
        augment_mode=augment_mode
    )

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
            raise FileNotFoundError(
                f"Missing dataset split folder: {split_path}\n"
                f"Expected train, val, test, test_e inside: {data_root}"
            )

        dataset_dict[split] = datasets.ImageFolder(
            root=str(split_path),
            transform=split_to_transform[split]
        )

    train_class_to_idx = dataset_dict["train"].class_to_idx

    for split in required_splits:
        if dataset_dict[split].class_to_idx != train_class_to_idx:
            raise ValueError(
                f"Class folder mismatch in split '{split}'.\n"
                f"train class_to_idx = {train_class_to_idx}\n"
                f"{split} class_to_idx = {dataset_dict[split].class_to_idx}"
            )

    if len(train_class_to_idx) != 2:
        raise ValueError(
            f"This script expects exactly 2 classes, but found {len(train_class_to_idx)}: "
            f"{list(train_class_to_idx.keys())}"
        )

    return dataset_dict


def get_class_counts(dataset: datasets.ImageFolder) -> Dict[int, int]:
    counts = {idx: 0 for idx in dataset.class_to_idx.values()}

    for _, label in dataset.samples:
        counts[int(label)] += 1

    return counts


def build_balance_report(
    dataset_dict: Dict[str, datasets.ImageFolder],
    data_root: Path
) -> Dict[str, Any]:
    class_to_idx = dataset_dict["train"].class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    report: Dict[str, Any] = {
        "data_root": str(data_root),
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
        "splits": {},
        "domain_checks": {},
        "warnings": []
    }

    for split, dataset in dataset_dict.items():
        counts_idx = get_class_counts(dataset)
        total = sum(counts_idx.values())

        counts_name = {
            idx_to_class[idx]: count
            for idx, count in counts_idx.items()
        }

        percentages_name = {
            idx_to_class[idx]: (count / total if total > 0 else 0.0)
            for idx, count in counts_idx.items()
        }

        max_count = max(counts_idx.values()) if counts_idx else 0
        min_count = min(counts_idx.values()) if counts_idx else 0
        imbalance_ratio = max_count / max(min_count, 1)

        report["splits"][split] = {
            "total": total,
            "counts": counts_name,
            "percentages": percentages_name,
            "imbalance_ratio_max_over_min": imbalance_ratio
        }

        if imbalance_ratio >= 1.5:
            report["warnings"].append(
                f"{split} is imbalanced. max/min ratio = {imbalance_ratio:.3f}"
            )

    # Domain check: same-domain train/val/test vs external test_e.
    if "test_e" in dataset_dict:
        train_counts = get_class_counts(dataset_dict["train"])
        test_e_counts = get_class_counts(dataset_dict["test_e"])

        train_total = sum(train_counts.values())
        test_e_total = sum(test_e_counts.values())

        domain_diff = {}

        for idx in sorted(train_counts.keys()):
            train_pct = train_counts[idx] / max(train_total, 1)
            test_e_pct = test_e_counts[idx] / max(test_e_total, 1)
            diff = abs(train_pct - test_e_pct)

            domain_diff[idx_to_class[idx]] = {
                "train_percentage": train_pct,
                "test_e_percentage": test_e_pct,
                "absolute_difference": diff
            }

            if diff >= 0.10:
                report["warnings"].append(
                    f"Class distribution differs between train and test_e for class '{idx_to_class[idx]}': "
                    f"abs diff = {diff:.3f}"
                )

        report["domain_checks"]["train_vs_test_e_class_distribution"] = domain_diff
        report["domain_checks"]["note"] = (
            "This checks class balance only. Visual domain gap may still exist because of lighting, camera, "
            "resolution, background, smoke/fire appearance, or compression differences."
        )

    return report


def make_train_sampler(
    dataset: datasets.ImageFolder,
    mode: str,
    imbalance_threshold: float
) -> Tuple[Optional[WeightedRandomSampler], Dict[str, Any]]:
    """
    Optional weighted sampler to reduce class imbalance in the training loader.

    mode:
    - no: never use
    - yes: always use
    - auto: use only if max/min class-count ratio >= imbalance_threshold
    """
    mode = str(mode).strip().lower()

    if mode not in ["no", "yes", "auto"]:
        raise ValueError("--use_weighted_sampler must be one of: no, yes, auto")

    counts = get_class_counts(dataset)
    max_count = max(counts.values())
    min_count = min(counts.values())
    imbalance_ratio = max_count / max(min_count, 1)

    use_sampler = False

    if mode == "yes":
        use_sampler = True
    elif mode == "auto":
        use_sampler = imbalance_ratio >= imbalance_threshold

    sampler_info = {
        "requested_mode": mode,
        "class_counts": {str(k): v for k, v in counts.items()},
        "imbalance_ratio_max_over_min": imbalance_ratio,
        "imbalance_threshold": imbalance_threshold,
        "weighted_sampler_used": use_sampler
    }

    if not use_sampler:
        return None, sampler_info

    class_weights = {label: 1.0 / max(count, 1) for label, count in counts.items()}
    sample_weights = [class_weights[int(label)] for _, label in dataset.samples]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler, sampler_info


def build_loaders(
    dataset_dict: Dict[str, datasets.ImageFolder],
    batch_size: int,
    num_workers: int,
    seed: int,
    use_weighted_sampler: str,
    weighted_sampler_threshold: float
) -> Tuple[Dict[str, DataLoader], Dict[str, Any]]:
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_sampler, sampler_info = make_train_sampler(
        dataset=dataset_dict["train"],
        mode=use_weighted_sampler,
        imbalance_threshold=weighted_sampler_threshold
    )

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    loaders = {
        "train": DataLoader(
            dataset_dict["train"],
            shuffle=train_sampler is None,
            sampler=train_sampler,
            generator=generator if train_sampler is None else None,
            **common_kwargs
        ),
        "val": DataLoader(dataset_dict["val"], shuffle=False, **common_kwargs),
        "test": DataLoader(dataset_dict["test"], shuffle=False, **common_kwargs),
        "test_e": DataLoader(dataset_dict["test_e"], shuffle=False, **common_kwargs),
    }

    return loaders, sampler_info


# =========================
# Model
# =========================

def adapt_resnet_for_small_images(model: nn.Module, pretrained: bool) -> nn.Module:
    """
    For 32x32/64x64 images:
    - use 3x3 conv stride 1
    - remove initial maxpool

    This avoids too much early downsampling.
    """
    old_conv = model.conv1

    new_conv = nn.Conv2d(
        in_channels=3,
        out_channels=old_conv.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    if pretrained:
        with torch.no_grad():
            # Copy center 3x3 from pretrained 7x7 kernel.
            new_conv.weight.copy_(old_conv.weight[:, :, 2:5, 2:5])

    model.conv1 = new_conv
    model.maxpool = nn.Identity()

    return model


def build_resnet18(
    num_classes: int,
    pretrained: bool,
    small_image_mode: bool,
    dropout: float
) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if small_image_mode:
        model = adapt_resnet_for_small_images(model, pretrained=pretrained)

    in_features = model.fc.in_features

    if dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)

    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """
    Freeze/unfreeze all ResNet layers except final classifier head.
    The classifier head remains trainable.
    """
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            param.requires_grad = True
        else:
            param.requires_grad = trainable


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]

    if len(params) == 0:
        raise RuntimeError("No trainable parameters found for optimizer.")

    return optim.Adam(
        params,
        lr=lr,
        weight_decay=weight_decay
    )


def build_scheduler(args: argparse.Namespace, optimizer: optim.Optimizer) -> optim.lr_scheduler.ReduceLROnPlateau:
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=args.lr_threshold,
        min_lr=args.min_lr
    )


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
    split_name: str,
    epoch: int,
    total_epochs: int,
    use_progress_bar: bool
) -> Dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    y_true: List[int] = []
    y_pred: List[int] = []

    desc = f"{split_name} {epoch}/{total_epochs}" if epoch > 0 and total_epochs > 0 else split_name

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
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1

    return cm


def per_class_metrics(cm: List[List[int]]) -> Dict[str, Dict[str, float]]:
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
    use_progress_bar: bool
) -> Dict[str, float]:
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


def save_best_checkpoint(
    monitor_name: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    checkpoints_dir: Path,
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    test_e_metrics: Dict[str, Any],
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[str, str],
    config: Dict[str, Any],
    history: Dict[str, Any],
    metric_value: float,
    metrics_for_monitor: Dict[str, Any]
) -> None:
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
        "loss": metrics_for_monitor["loss"],
        "checkpoint_path": str(checkpoint_path),
        "state_dict_path": str(state_dict_path)
    }

    # Compatibility alias: best_model.pth always means best validation model.
    if monitor_name == "val":
        torch.save(best_ckpt, checkpoints_dir / "best_model.pth")
        torch.save(model.state_dict(), checkpoints_dir / "best_model_state_dict_only.pth")


def train_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    data_root = resolve_data_root(args)
    img_h, img_w, image_size_tag = resolve_image_size(args)
    small_image_mode = resolve_small_image_mode(args, img_h, img_w)

    runs_root = Path(args.runs_root)
    run_dir = create_next_run_dir(runs_root)

    plots_dir = run_dir / "plots"
    checkpoints_dir = run_dir / "checkpoints"
    plots_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    device = get_training_device(args.device)
    device_info = get_device_info(device)

    if device.type == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    amp_enabled = bool(args.use_fp16) and device.type == "cuda"

    print("\n" + "=" * 80)
    print("ResNet-18 Fire/Nofire Training")
    print("=" * 80)
    print(f"Data root          : {data_root}")
    print(f"Image resize       : {img_h}x{img_w}")
    print(f"Augmentation       : {args.augment}")
    print(f"Small image mode   : {small_image_mode}")
    print(f"Run directory      : {run_dir}")
    print(f"Device             : {device}")

    if device.type == "cuda":
        print(f"GPU name           : {device_info.get('cuda_device_name')}")
        print(f"GPU memory         : {device_info.get('cuda_total_memory_gb')} GB")
        print(f"PyTorch CUDA       : {device_info.get('pytorch_cuda_version')}")
    else:
        print("GPU note           : Running on CPU because CUDA is not available to this Python/PyTorch environment.")

    print(f"FP16 enabled       : {amp_enabled}")
    print(f"Initial LR         : {args.lr}")
    print(f"Fine-tune LR       : {args.finetune_lr}")
    print(f"Freeze epochs      : {args.freeze_epochs}")
    print(f"Weight decay       : {args.weight_decay}")
    print(f"Dropout            : {args.dropout}")
    print(f"Label smoothing    : {args.label_smoothing}")
    print(f"Progress bar       : {args.progress_bar}")

    if args.progress_bar and not TQDM_AVAILABLE:
        print("WARNING: tqdm is not installed. Install using: pip install tqdm")

    print("=" * 80 + "\n")

    dataset_dict = build_datasets(
        data_root=data_root,
        img_h=img_h,
        img_w=img_w,
        augment_mode=args.augment
    )

    balance_report = build_balance_report(dataset_dict, data_root=data_root)
    save_json(balance_report, run_dir / "dataset_balance_report.json")

    print("Class/domain balance check:")
    for split, info in balance_report["splits"].items():
        print(f"  {split:6s}: total={info['total']}, counts={info['counts']}, imbalance_ratio={info['imbalance_ratio_max_over_min']:.3f}")

    if balance_report["warnings"]:
        print("Balance/domain warnings:")
        for w in balance_report["warnings"]:
            print(f"  - {w}")
    else:
        print("  No major class-count imbalance detected.")

    loaders, sampler_info = build_loaders(
        dataset_dict=dataset_dict,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_weighted_sampler=args.use_weighted_sampler,
        weighted_sampler_threshold=args.weighted_sampler_threshold
    )

    print(f"Weighted sampler   : {sampler_info['weighted_sampler_used']} ({sampler_info['requested_mode']})")

    class_to_idx = dataset_dict["train"].class_to_idx
    idx_to_class = {str(v): k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    model = build_resnet18(
        num_classes=num_classes,
        pretrained=args.pretrained,
        small_image_mode=small_image_mode,
        dropout=args.dropout
    )
    model = model.to(device)

    if args.freeze_epochs > 0:
        set_backbone_trainable(model, trainable=False)
        current_phase = "frozen_backbone"
        current_lr = args.lr
    else:
        set_backbone_trainable(model, trainable=True)
        current_phase = "full_finetune"
        current_lr = args.lr

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = build_optimizer(
        model=model,
        lr=current_lr,
        weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(args=args, optimizer=optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    dataset_info = {
        "data_root": str(data_root),
        "image_resize_height": img_h,
        "image_resize_width": img_w,
        "image_size_tag": image_size_tag,
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
        "domain_note": {
            "train_val_test": "Same-domain dataset splits.",
            "test_e": "External-domain / other-domain test split."
        }
    }

    config = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script": "train_resnet18_fire_strongaug_regularized.py",
        "model": "resnet18",
        "pretrained": args.pretrained,
        "num_classes": num_classes,
        "image_resize": {
            "height": img_h,
            "width": img_w,
            "size_tag": image_size_tag
        },
        "augmentation": args.augment,
        "regularization": {
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "label_smoothing": args.label_smoothing
        },
        "freeze_then_finetune": {
            "freeze_epochs": args.freeze_epochs,
            "head_only_lr": args.lr,
            "finetune_lr": args.finetune_lr,
            "reset_scheduler_on_unfreeze": True
        },
        "small_image_mode": small_image_mode,
        "small_image_mode_argument": args.small_image_mode,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": "Adam",
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
        "final_reporting": {
            "model_selection": "best validation accuracy",
            "best_val_state_dict": str(checkpoints_dir / "best_val_model_state_dict_only.pth")
        },
        "weighted_sampler": sampler_info,
        "use_fp16_requested": args.use_fp16,
        "amp_enabled": amp_enabled,
        "device": str(device),
        "device_argument": args.device,
        "device_info": device_info,
        "seed": args.seed,
        "progress_bar": args.progress_bar,
        "tqdm_available": TQDM_AVAILABLE,
        "dataset_info": dataset_info,
        "balance_report_file": str(run_dir / "dataset_balance_report.json"),
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
        "phase",
        "lr",
        "trainable_params",
        "total_params",
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

    total_params = get_num_total_params(model)

    for epoch in range(1, args.epochs + 1):
        # Switch from frozen-backbone training to full fine-tuning.
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            print("\n" + "-" * 80)
            print(f"Unfreezing backbone at epoch {epoch}. Switching to full fine-tuning.")
            print(f"Fine-tune LR: {args.finetune_lr}")
            print("-" * 80 + "\n")

            set_backbone_trainable(model, trainable=True)
            current_phase = "full_finetune"

            optimizer = build_optimizer(
                model=model,
                lr=args.finetune_lr,
                weight_decay=args.weight_decay
            )
            scheduler = build_scheduler(args=args, optimizer=optimizer)

        epoch_start = time.time()
        trainable_params = get_num_trainable_params(model)

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

        # LR scheduler uses validation accuracy only.
        scheduler.step(val_metrics["accuracy"])

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

        if is_best_val:
            best_val_acc = val_metrics["accuracy"]
            best_val_epoch = epoch
            epochs_without_improvement = 0
            save_best_checkpoint(
                monitor_name="val",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                amp_enabled=amp_enabled,
                checkpoints_dir=checkpoints_dir,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                test_e_metrics=test_e_metrics,
                class_to_idx=class_to_idx,
                idx_to_class=idx_to_class,
                config=config,
                history=history,
                metric_value=best_val_acc,
                metrics_for_monitor=val_metrics
            )
        else:
            epochs_without_improvement += 1

        if is_best_train:
            best_train_acc = train_metrics["accuracy"]
            best_train_epoch = epoch
            save_best_checkpoint(
                monitor_name="train",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                amp_enabled=amp_enabled,
                checkpoints_dir=checkpoints_dir,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                test_e_metrics=test_e_metrics,
                class_to_idx=class_to_idx,
                idx_to_class=idx_to_class,
                config=config,
                history=history,
                metric_value=best_train_acc,
                metrics_for_monitor=train_metrics
            )

        if is_best_test:
            best_test_acc = test_metrics["accuracy"]
            best_test_epoch = epoch
            save_best_checkpoint(
                monitor_name="test",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                amp_enabled=amp_enabled,
                checkpoints_dir=checkpoints_dir,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                test_e_metrics=test_e_metrics,
                class_to_idx=class_to_idx,
                idx_to_class=idx_to_class,
                config=config,
                history=history,
                metric_value=best_test_acc,
                metrics_for_monitor=test_metrics
            )

        if is_best_test_e:
            best_test_e_acc = test_e_metrics["accuracy"]
            best_test_e_epoch = epoch
            save_best_checkpoint(
                monitor_name="test_e",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                amp_enabled=amp_enabled,
                checkpoints_dir=checkpoints_dir,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                test_e_metrics=test_e_metrics,
                class_to_idx=class_to_idx,
                idx_to_class=idx_to_class,
                config=config,
                history=history,
                metric_value=best_test_e_acc,
                metrics_for_monitor=test_e_metrics
            )

        epoch_record = {
            "epoch": epoch,
            "phase": current_phase,
            "lr": current_lr,
            "trainable_params": trainable_params,
            "total_params": total_params,
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
        save_json(history, run_dir / "training_history.json")

        append_csv_row(
            csv_path,
            {
                "epoch": epoch,
                "phase": current_phase,
                "lr": current_lr,
                "trainable_params": trainable_params,
                "total_params": total_params,
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
            f"Phase: {current_phase} | "
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
    torch.save(model.state_dict(), checkpoints_dir / "last_model_state_dict_only.pth")

    # Final reporting must use best validation model.
    print("\nLoading best validation model for final reporting...")
    best_val_state_dict_path = checkpoints_dir / "best_val_model_state_dict_only.pth"

    if best_val_state_dict_path.exists():
        best_state_dict = torch.load(best_val_state_dict_path, map_location=device)
    elif (checkpoints_dir / "best_model_state_dict_only.pth").exists():
        best_state_dict = torch.load(checkpoints_dir / "best_model_state_dict_only.pth", map_location=device)
    else:
        best_checkpoint = torch.load(
            checkpoints_dir / "best_model.pth",
            map_location=device,
            weights_only=False
        )
        best_state_dict = best_checkpoint["model_state_dict"]

    model.load_state_dict(best_state_dict)

    final_eval = {}
    for split in ["train", "val", "test", "test_e"]:
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
        "final_reporting_model": "best validation accuracy model",
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
        "final_evaluation_using_best_val_model": final_eval,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "run_dir": str(run_dir),
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
            },
            "last": {
                "checkpoint": str(checkpoints_dir / "last_model.pth"),
                "state_dict": str(checkpoints_dir / "last_model_state_dict_only.pth")
            }
        },
        "training_history_path": str(run_dir / "training_history.json"),
        "training_log_csv_path": str(csv_path),
        "dataset_balance_report_path": str(run_dir / "dataset_balance_report.json"),
        "image_resize": {
            "height": img_h,
            "width": img_w,
            "size_tag": image_size_tag
        },
        "small_image_mode": small_image_mode,
        "device_info": device_info,
        "note": (
            "Model selection and final reporting use best validation accuracy. "
            "Separate train/test/test_e best weights are saved for analysis only."
        )
    }

    save_json(final_results, run_dir / "final_results.json")

    class_names = [idx_to_class[str(i)] for i in range(num_classes)]

    for split in ["train", "val", "test", "test_e"]:
        plot_confusion_matrix(
            cm=final_eval[split]["confusion_matrix"],
            class_names=class_names,
            title=f"{split} confusion matrix using best validation model",
            save_path=plots_dir / f"confusion_matrix_{split}_best_val_model.png"
        )

    plot_training_curves(history, plots_dir)

    print("\nFinal results using BEST VALIDATION model:")
    print(f"Train  Acc: {final_eval['train']['accuracy']:.4f}, Loss: {final_eval['train']['loss']:.4f}")
    print(f"Val    Acc: {final_eval['val']['accuracy']:.4f}, Loss: {final_eval['val']['loss']:.4f}")
    print(f"Test   Acc: {final_eval['test']['accuracy']:.4f}, Loss: {final_eval['test']['loss']:.4f}")
    print(f"Test_E Acc: {final_eval['test_e']['accuracy']:.4f}, Loss: {final_eval['test_e']['loss']:.4f}")

    print("\n" + "=" * 80)
    print("Training finished.")
    print(f"Best val epoch     : {best_val_epoch}, Acc: {best_val_acc:.4f}")
    print(f"Best train epoch   : {best_train_epoch}, Acc: {best_train_acc:.4f}")
    print(f"Best test epoch    : {best_test_epoch}, Acc: {best_test_acc:.4f}")
    print(f"Best test_e epoch  : {best_test_e_epoch}, Acc: {best_test_e_acc:.4f}")
    print(f"Run folder         : {run_dir}")
    print(f"Best val weights   : {checkpoints_dir / 'best_val_model.pth'}")
    print(f"Best train weights : {checkpoints_dir / 'best_train_model.pth'}")
    print(f"Best test weights  : {checkpoints_dir / 'best_test_model.pth'}")
    print(f"Best test_e weights: {checkpoints_dir / 'best_test_e_model.pth'}")
    print(f"JSON history       : {run_dir / 'training_history.json'}")
    print(f"Final results      : {run_dir / 'final_results.json'}")
    print(f"Balance report     : {run_dir / 'dataset_balance_report.json'}")
    print("=" * 80 + "\n")


# =========================
# Plotting
# =========================

def plot_training_curves(history: Dict[str, Any], plots_dir: Path) -> None:
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
    plt.close(fig)


def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    title: str,
    save_path: Path
) -> None:
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
    plt.close(fig)


# =========================
# Arguments
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained ResNet-18 on fire/nofire dataset with stronger augmentation and regularization."
    )

    parser.add_argument("--dataset_base", type=str, default="Dataset")
    parser.add_argument("--dataset_size", type=str, default="128x128")
    parser.add_argument("--data_root", type=str, default=None)

    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--img_height", type=int, default=None)
    parser.add_argument("--img_width", type=int, default=None)

    parser.add_argument("--runs_root", type=str, default="Runs/Resnet18_01")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--use_fp16", type=str_to_bool, default=True)

    parser.add_argument("--pretrained", type=str_to_bool, default=True)
    parser.add_argument("--small_image_mode", type=str, default="auto", choices=["auto", "yes", "no"])

    # Strong augmentation and regularization
    parser.add_argument("--augment", type=str, default="strong", choices=["off", "basic", "strong"])
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--label_smoothing", type=float, default=0.05)

    # Freeze then fine-tune
    parser.add_argument("--freeze_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4, help="LR for classifier-head phase.")
    parser.add_argument("--finetune_lr", type=float, default=5e-5, help="LR after unfreezing backbone.")

    # Sampler / balance
    parser.add_argument("--use_weighted_sampler", type=str, default="auto", choices=["no", "yes", "auto"])
    parser.add_argument("--weighted_sampler_threshold", type=float, default=1.5)

    # Scheduler and early stopping
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_threshold", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--early_stop_patience", type=int, default=20)

    parser.add_argument("--progress_bar", type=str_to_bool, default=True)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
