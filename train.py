# Imports
from google.colab import drive

from pathlib import Path
import os
import random
import time
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy import ndimage as ndi
from tqdm.auto import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

class LiveDeadDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["image_path"]).convert("L")
        label = torch.tensor(row["label_idx"], dtype=torch.float32)

        if self.transform is not None:
            img = self.transform(img)

        return img, label
    

def evaluate_binary_predictions(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if len(np.unique(y_true)) == 2:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
        metrics["auprc"] = average_precision_score(y_true, y_prob)
    else:
        metrics["auroc"] = np.nan
        metrics["auprc"] = np.nan

    return metrics, y_pred


def print_metrics(metrics, title):
    print(f"\n{title}")
    print("-" * len(title))
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.floating)):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


def per_group_metrics_from_predictions(eval_df, y_prob, y_pred=None, group_col="assay_id", threshold=0.5):
    temp = eval_df.copy()
    temp["y_prob"] = np.asarray(y_prob)

    if y_pred is None:
        temp["y_pred"] = (temp["y_prob"] >= threshold).astype(int)
    else:
        temp["y_pred"] = np.asarray(y_pred).astype(int)

    rows = []

    for group_value, group_df in temp.groupby(group_col):
        y_true_g = group_df["label_idx"].values.astype(int)
        y_prob_g = group_df["y_prob"].values
        y_pred_g = group_df["y_pred"].values.astype(int)

        row = {
            group_col: group_value,
            "n": len(group_df),
            "dead_fraction": y_true_g.mean(),
            "accuracy": accuracy_score(y_true_g, y_pred_g),
            "precision": precision_score(y_true_g, y_pred_g, zero_division=0),
            "recall": recall_score(y_true_g, y_pred_g, zero_division=0),
            "f1": f1_score(y_true_g, y_pred_g, zero_division=0),
        }

        if len(np.unique(y_true_g)) == 2:
            row["auroc"] = roc_auc_score(y_true_g, y_prob_g)
            row["auprc"] = average_precision_score(y_true_g, y_prob_g)
        else:
            row["auroc"] = np.nan
            row["auprc"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def get_gpu_utilization():
    """
    Returns current GPU utilization using nvidia-smi.
    If unavailable, returns None.
    """
    if not torch.cuda.is_available():
        return None

    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        return float(result.strip().split("\n")[0])
    except Exception:
        return None


def reset_gpu_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_gpu_memory_stats():
    if not torch.cuda.is_available():
        return {
            "peak_memory_allocated_mb": None,
            "peak_memory_reserved_mb": None,
        }

    return {
        "peak_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_memory_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }


def get_hardware_info(device):
    return {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
    }


def build_resnet18_grayscale(pretrained=True):
    try:
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
    except Exception as e:
        print("Could not load pretrained weights. Falling back to random initialization.")
        print(e)
        model = resnet18(weights=None)

    # Adapt first convolution from RGB to grayscale
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    with torch.no_grad():
        if old_conv.weight.shape[1] == 3:
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

    model.conv1 = new_conv

    # Binary classifier: one logit
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)

    return model


def run_one_epoch(model, loader, criterion, scaler, device, optimizer=None, use_amp=True, desc=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_probs = []
    all_labels = []

    start_time = time.time()
    n_samples_seen = 0

    if desc is None:
        desc = "Train" if is_train else "Eval"

    progress_bar = tqdm(loader, desc=desc, leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1, 1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = images.size(0)
        n_samples_seen += batch_size
        total_loss += loss.item() * batch_size

        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        labels_np = labels.detach().cpu().numpy().reshape(-1)

        all_probs.extend(probs)
        all_labels.extend(labels_np)

        running_loss = total_loss / n_samples_seen
        elapsed = time.time() - start_time
        samples_per_sec = n_samples_seen / elapsed if elapsed > 0 else 0

        progress_bar.set_postfix({
            "loss": f"{running_loss:.4f}",
            "samples/s": f"{samples_per_sec:.1f}",
        })

    elapsed = time.time() - start_time
    n_samples = len(loader.dataset)

    avg_loss = total_loss / n_samples
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels).astype(int)

    metrics, _ = evaluate_binary_predictions(all_labels, all_probs)

    metrics["loss"] = avg_loss
    metrics["epoch_time_sec"] = elapsed
    metrics["samples_per_sec"] = n_samples / elapsed

    return metrics, all_probs, all_labels

def main():
    # Paths
    DATA_DIR = Path("/content/drive/MyDrive/TeraCyte_HA")  # folder containing metadata.csv and images/
    METADATA_PATH = DATA_DIR / "metadata.csv"
    IMAGE_DIR = DATA_DIR / "images"
    OUTPUT_DIR = DATA_DIR / "outputs"
    OUTPUT_DIR.mkdir(exist_ok=True)

    assert DATA_DIR.exists(), f"DATA_DIR does not exist: {DATA_DIR}"
    assert METADATA_PATH.exists(), f"metadata.csv not found: {METADATA_PATH}"
    assert IMAGE_DIR.exists(), f"images folder not found: {IMAGE_DIR}"

    print("Data directory:", DATA_DIR)
    print("Metadata path:", METADATA_PATH)
    print("Image directory:", IMAGE_DIR)

    df = pd.read_csv(METADATA_PATH)
    LABEL_COL = "label"
    label_to_idx = {"live": 0, "dead": 1}
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    unknown_labels = set(df[LABEL_COL].dropna().unique()) - set(label_to_idx.keys())
    assert len(unknown_labels) == 0, f"Unexpected labels found: {unknown_labels}"
    df["label_idx"] = df[LABEL_COL].map(label_to_idx)
    df["image_path"] = df["filepath"].apply(lambda p: DATA_DIR / p)
    TRAIN_ASSAYS = ["assay-20260204103330-9793"]
    VAL_ASSAYS   = ["assay-20260204103236-0469"]
    TEST_ASSAYS  = [
        "assay-20260310120944-3366",
        "assay-20260310120948-7091",
    ]
    df["split"] = None

    df.loc[df["assay_id"].isin(TRAIN_ASSAYS), "split"] = "train"
    df.loc[df["assay_id"].isin(VAL_ASSAYS), "split"] = "val"
    df.loc[df["assay_id"].isin(TEST_ASSAYS), "split"] = "test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)

    train_split_df = df[df["split"] == "train"].copy()
    val_split_df   = df[df["split"] == "val"].copy()
    test_split_df  = df[df["split"] == "test"].copy()

    print("Train:", len(train_split_df))
    print("Val:", len(val_split_df))
    print("Test:", len(test_split_df))

    sample_for_norm = train_split_df.sample(
        min(1000, len(train_split_df)),
        random_state=SEED
    )

    train_image_means = []
    train_image_stds = []

    for _, row in tqdm(sample_for_norm.iterrows(), total=len(sample_for_norm)):
        arr = np.array(Image.open(row["image_path"]).convert("L")).astype(np.float32) / 255.0
        train_image_means.append(arr.mean())
        train_image_stds.append(arr.std())

    train_mean = float(np.mean(train_image_means))
    train_std = float(np.mean(train_image_stds))

    print("Train mean:", train_mean)
    print("Train std:", train_std)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        # transforms.ColorJitter(brightness=0.35, contrast=0.35),
        # transforms.RandomAutocontrast(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[train_mean], std=[train_std]),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[train_mean], std=[train_std]),
    ])

    BATCH_SIZE = 64
    NUM_WORKERS = 0 # For colab errors, can use more..

    train_dataset = LiveDeadDataset(train_split_df, transform=train_transform)
    val_dataset   = LiveDeadDataset(val_split_df, transform=eval_transform)
    test_dataset  = LiveDeadDataset(test_split_df, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_resnet18_grayscale(pretrained=True).to(device)
    print("Model ready")

    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    USE_AMP = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    EPOCHS = 20
    PATIENCE = 5

    CHECKPOINT_PATH = OUTPUT_DIR / "best_resnet18_full_image.pt"
    WEIGHTS_PATH = OUTPUT_DIR / "resnet18_full_image_weights.pt"

    history = []
    best_val_f1 = -np.inf
    best_epoch = None
    epochs_without_improvement = 0

    total_start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        reset_gpu_memory_stats()
        gpu_util_start = get_gpu_utilization()

        train_metrics, _, _ = run_one_epoch(
            model,
            train_loader,
            criterion,
            scaler,
            device,
            optimizer=optimizer,
            use_amp=USE_AMP,
        )

        val_metrics, val_probs_epoch, val_labels_epoch = run_one_epoch(
            model,
            val_loader,
            criterion,
            scaler,
            device,
            optimizer=None,
            use_amp=USE_AMP,
        )

        scheduler.step(val_metrics["f1"])

        gpu_util_end = get_gpu_utilization()
        gpu_mem = get_gpu_memory_stats()
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,

            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_auroc": train_metrics["auroc"],
            "train_auprc": train_metrics["auprc"],
            "train_samples_per_sec": train_metrics["samples_per_sec"],

            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],

            "epoch_time_sec": train_metrics["epoch_time_sec"] + val_metrics["epoch_time_sec"],
            "peak_memory_allocated_mb": gpu_mem["peak_memory_allocated_mb"],
            "peak_memory_reserved_mb": gpu_mem["peak_memory_reserved_mb"],
            "gpu_util_start": gpu_util_start,
            "gpu_util_end": gpu_util_end,
            "batch_size": BATCH_SIZE,
            "mixed_precision": USE_AMP,
        }

        history.append(row)

        print(f"Train loss: {train_metrics['loss']:.4f} | Train F1: {train_metrics['f1']:.4f}")
        print(f"Val loss:   {val_metrics['loss']:.4f} | Val F1:   {val_metrics['f1']:.4f} | Val AUROC: {val_metrics['auroc']:.4f}")
        print(f"LR: {current_lr:.2e}")
        print(f"Train samples/sec: {train_metrics['samples_per_sec']:.2f}")
        print(f"Peak allocated MB: {gpu_mem['peak_memory_allocated_mb']}")
        print(f"Peak reserved MB: {gpu_mem['peak_memory_reserved_mb']}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(model.state_dict(), WEIGHTS_PATH)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val_f1": best_val_f1,
                    "train_mean": train_mean,
                    "train_std": train_std,
                    "batch_size": BATCH_SIZE,
                    "mixed_precision": USE_AMP,
                },
                CHECKPOINT_PATH,
            )

            print(f"Saved new best checkpoint: {CHECKPOINT_PATH}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    total_training_time = time.time() - total_start_time

    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / "training_history_resnet18.csv", index=False)

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Total training time: {total_training_time:.2f} sec")


if __name__ == "__main__":
    main()
