# GeoRisk GPU Computing

# GPU Computing

This document explains how to run **GPU-accelerated scientific workloads** on the GeoRisk GPU workstation.



## Python GPU libraries

Common GPU-enabled libraries:
* CUDA
* PyTorch
* To be later installed TensorFlow (TF does not yet support CIDA 13!) 


# Verify GPU Availability

Check GPU visibility:

```bash id="s7jlwm"
nvidia-smi
```

Check CUDA availability in PyTorch:

```python id="c1l8rj"
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Expected output:

```id="f9l2gw"
True
NVIDIA RTX PRO 6000 Blackwell Max-Q
```


---

# Example: GPU Benchmark

This repository contains a benchmark example demonstrating GPU computation with PyTorch.

Example location:

```id="l6df6p"
examples/pytorch_gpu/gpu_benchmark.py
```

The benchmark:

* generates large synthetic datasets
* trains GPU-intensive neural networks
* measures wall-clock execution time
* reports throughput and GPU memory usage

---

# Running the Benchmark

```bash id="wt0d9g"
python examples/python/pytorch/gpu_benchmark.py
```

Example:

```bash id="gcrhkg"
python examples/pytorch_gpu/gpu_benchmark.py \
    --batch-size 4096 \
    --samples 500000 \
    --hidden-dim 8192 \
    --epochs 5
```

FP16 Tensor Core benchmark:

```bash id="lmv1yy"
python examples/python/pytorch/gpu_benchmark.py --fp16
```

---

# Basic PyTorch GPU Usage

Move tensors to GPU:

```python id="xq7l93"
import torch

device = "cuda"

x = torch.randn(1000, 1000).to(device)
```

Move models to GPU:

```python id="hm1g9m"
model = model.to("cuda")
```

---

# Best Practice: Detect GPU Automatically

Recommended approach:

```python id="vphk7s"
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)
```


# GPU Memory Monitoring

Monitor GPU usage:

```bash id="g4n03z"
watch -n 1 nvidia-smi
```

PyTorch memory reporting: 

```python id="u6q0gx"
import torch

print(torch.cuda.memory_allocated() / 1e9)
```

Peak memory:

```python id="2aqg7g"
print(torch.cuda.max_memory_allocated() / 1e9)
```

---

