#!/usr/bin/env python3

"""
Simple U-Net Cloud Segmentation Training
========================================

Train a lightweight U-Net for cloud segmentation on Landsat 8 imagery.

Dataset:
- 38-Cloud dataset
- RGB + NIR bands
- Binary cloud masks

Features
--------
- Simple U-Net architecture
- PyTorch training pipeline
- GPU support
- mixed precision (optional)
- validation metrics
- checkpoint saving
- argparse configuration

Example
-------
python3 train_unet_clouds.py \
    --data-dir ./38-Cloud \
    --epochs 50 \
    --batch-size 16 \
    --learning-rate 1e-3

Code directory structure
-------------------
examples/python/pytorch/cloud_segmentation/
│
├── train_unet_clouds.py
├── model.py
├── dataset.py
├── utils.py
└── results/

Data Directory Structure
-------------------
38-Cloud/
├── train/
│   ├── red/
│   ├── green/
│   ├── blue/
│   ├── nir/
│   └── gt/
│
├── test/
│   ├── red/
│   ├── green/
│   ├── blue/
│   ├── nir/
│   └── gt/

"""

#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from dataset import CloudDataset
from model import UNET
from utils import pixel_accuracy


def train_epoch(
        model,
        loader,
        optimizer,
        criterion,
        device,
        scaler=None
):

    model.train()

    running_loss = 0.0
    running_acc = 0.0

    for x, y in tqdm(loader, desc="Training"):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        if scaler:

            with torch.cuda.amp.autocast():

                outputs = model(x)

                loss = criterion(outputs, y)

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

        else:

            outputs = model(x)

            loss = criterion(outputs, y)

            loss.backward()

            optimizer.step()

        acc = pixel_accuracy(outputs, y)

        running_loss += loss.item()
        running_acc += acc

    return (
        running_loss / len(loader),
        running_acc / len(loader)
    )


@torch.no_grad()
def validate(
        model,
        loader,
        criterion,
        device
):

    model.eval()

    running_loss = 0.0
    running_acc = 0.0

    for x, y in tqdm(loader, desc="Validation"):

        x = x.to(device)
        y = y.to(device)

        outputs = model(x)

        loss = criterion(outputs, y)

        acc = pixel_accuracy(outputs, y)

        running_loss += loss.item()
        running_acc += acc

    return (
        running_loss / len(loader),
        running_acc / len(loader)
    )


def main(args):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    dataset = CloudDataset(
        Path(args.data_dir) / "train"
    )

    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=args.val_split,
        random_state=42
    )

    train_ds = torch.utils.data.Subset(
        dataset,
        train_idx
    )

    val_ds = torch.utils.data.Subset(
        dataset,
        val_idx
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = UNET(
        in_channels=4,
        out_channels=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    scaler = None

    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()

    history = []

    best_acc = 0.0

    for epoch in range(args.epochs):

        print("\n" + "=" * 60)
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("=" * 60)

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(
                model.state_dict(),
                args.output_dir / "best_model.pt"
            )

            print("Saved best model")

    with open(
            args.output_dir / "training_history.json",
            "w"
    ) as f:

        json.dump(history, f, indent=4)

    print("\nTraining completed.")


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results")
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--fp16",
        action="store_true"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    main(args)