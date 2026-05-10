#!/usr/bin/env python3

"""
GPU Benchmark for GeoRisk GPU Workstation
=========================================

Benchmark GPU computation performance using PyTorch.

Features
--------
- synthetic dataset generation
- configurable neural network size
- configurable batch size
- configurable epochs
- wall-clock timing
- GPU memory reporting
- FP16 Tensor Core benchmark

Example
-------
python gpu_benchmark.py \
    --batch-size 4096 \
    --samples 500000 \
    --input-dim 4096 \
    --hidden-dim 8192 \
    --epochs 5
------
GeoRisk GPU Computing
"""

import argparse
import time

import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader



# Argument Parser
parser = argparse.ArgumentParser(
    description="PyTorch GPU benchmark"
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=4096,
    help="Batch size"
)

parser.add_argument(
    "--samples",
    type=int,
    default=500_000,
    help="Number of synthetic samples"
)

parser.add_argument(
    "--input-dim",
    type=int,
    default=4096,
    help="Input feature dimension"
)

parser.add_argument(
    "--hidden-dim",
    type=int,
    default=8192,
    help="Hidden layer dimension"
)

parser.add_argument(
    "--output-dim",
    type=int,
    default=1000,
    help="Output classes"
)

parser.add_argument(
    "--epochs",
    type=int,
    default=5,
    help="Number of training epochs"
)

parser.add_argument(
    "--num-workers",
    type=int,
    default=4,
    help="DataLoader workers"
)

parser.add_argument(
    "--learning-rate",
    type=float,
    default=1e-3,
    help="Learning rate"
)

parser.add_argument(
    "--fp16",
    action="store_true",
    help="Enable FP16 benchmark"
)

parser.add_argument(
    "--results-dir",
    type=str,
    default="./results",
    help="Directory for benchmark JSON results"
)

args = parser.parse_args()


# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE != "cuda":
    raise RuntimeError("CUDA GPU not available.")


# GPU Information
print("=" * 60)
print("GeoRisk GPU Benchmark")
print("=" * 60)

print(f"PyTorch version : {torch.__version__}")
print(f"GPU device      : {torch.cuda.get_device_name(0)}")

gpu_mem = (
    torch.cuda.get_device_properties(0).total_memory / 1e9
)

print(f"GPU memory      : {gpu_mem:.2f} GB")

print("=" * 60)


# Synthetic Dataset
print("Generating synthetic dataset...")

X = torch.randn(args.samples, args.input_dim)

y = torch.randint(
    0,
    args.output_dim,
    (args.samples,)
)

dataset = TensorDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
)

print(f"Samples         : {args.samples:,}")
print(f"Batch size      : {args.batch_size}")
print(f"Input dimension : {args.input_dim}")
print(f"Hidden dimension: {args.hidden_dim}")
print(f"Epochs          : {args.epochs}")

print("=" * 60)


# Model
model = nn.Sequential(
    nn.Linear(args.input_dim, args.hidden_dim),
    nn.ReLU(),

    nn.Linear(args.hidden_dim, args.hidden_dim),
    nn.ReLU(),

    nn.Linear(args.hidden_dim, args.hidden_dim),
    nn.ReLU(),

    nn.Linear(args.hidden_dim, args.output_dim),
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
)


# Benchmark
print("Starting benchmark...")
print("=" * 60)

torch.cuda.reset_peak_memory_stats()

total_start = time.perf_counter()

for epoch in range(args.epochs):

    epoch_start = time.perf_counter()

    running_loss = 0.0

    for batch_X, batch_y in loader:

        batch_X = batch_X.to(
            DEVICE,
            non_blocking=True
        )

        batch_y = batch_y.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    torch.cuda.synchronize()

    epoch_time = time.perf_counter() - epoch_start

    print(
        f"Epoch {epoch + 1}/{args.epochs} | "
        f"Loss: {running_loss:.4f} | "
        f"Time: {epoch_time:.2f} s"
    )

total_time = time.perf_counter() - total_start


# Results
peak_memory = (
    torch.cuda.max_memory_allocated() / 1e9
)

samples_per_second = (
    (args.samples * args.epochs) / total_time
)

print("=" * 60)
print("Benchmark Results")
print("=" * 60)

print(f"Total time           : {total_time:.2f} s")
print(f"Samples/sec          : {samples_per_second:,.0f}")
print(f"Peak GPU memory      : {peak_memory:.2f} GB")

print("=" * 60)


# FP16 Benchmark
if args.fp16:

    print("Running FP16 Tensor Core benchmark...")

    model.half()

    fp16_start = time.perf_counter()

    with torch.cuda.amp.autocast():

        for batch_X, _ in loader:

            batch_X = batch_X.to(
                DEVICE,
                non_blocking=True,
                dtype=torch.float16,
            )

            outputs = model(batch_X)

            del outputs

    torch.cuda.synchronize()

    fp16_time = (
        time.perf_counter() - fp16_start
    )

    print(f"FP16 inference time  : {fp16_time:.2f} s")

    print("=" * 60)


# Save Results
results_dir = Path(args.results_dir)
results_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

result_file = results_dir / f"gpu_benchmark_{timestamp}.json"

results = {
    "timestamp": timestamp,
    "gpu_name": torch.cuda.get_device_name(0),
    "pytorch_version": torch.__version__,
    "cuda_version": torch.version.cuda,

    "benchmark_config": {
        "batch_size": args.batch_size,
        "samples": args.samples,
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim,
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "fp16": args.fp16,
    },

    "results": {
        "total_time_sec": total_time,
        "samples_per_second": samples_per_second,
        "peak_gpu_memory_gb": peak_memory,
    }
}

if args.fp16:
    results["results"]["fp16_inference_time_sec"] = fp16_time

with open(result_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to: {result_file}")


print("Benchmark completed.")
print("=" * 60)
print('END OF COMPUTATION!')