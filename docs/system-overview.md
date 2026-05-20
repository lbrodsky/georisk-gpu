# GeoRisk GPU System Overview

## 1. Hardware Overview 
| Component               | Specification                                          |
| ----------------------- | ------------------------------------------------------ |
| CPU                     | AMD EPYC 9334 (32 cores, 2.7 GHz base / 3.9 GHz boost) |
| GPU                     | NVIDIA RTX PRO 6000 Blackwell Max-Q  (96 GB)           |
| RAM                     | 768 GB DDR5                                            |
| Storage (OS / software) | 2 TB NVMe                                              |
| Storage (data)          | 2 × 10 TB                                              |
| Network                 | Gigabit / high-speed LAN                               |
| Architecture            | Single-node workstation                                |


## 2. Operating Systems 
The workstation is configured as a dual-boot system:

| OS         | Purpose                                   |
| ---------- | ----------------------------------------- |
| Ubuntu 24  | Primary scientific computing environment  |
| Windows 11 | Visualization, desktop workflows, testing |

## 3. Linux Environment (Ubuntu 24)
Ubuntu is the primary environment for multiuser scientific workloads
Check who is logged! 
```
$ who
```

**Key software**
- CUDA Toolkit
- NVIDIA drivers
- Python3 (environments)
- PyTorch (TODO: TensorFlow)
- NumPy / SciPy 
- OpenMP 
- GCC 
- Git

## 4. Windows Environment (Windows 11) 
Windows provides a secondary desktop environment for:
- CUDA Toolkit
- NVIDIA drivers
- Python (environments)
- PyTorch
- NumPy / SciPy
- Metashape? 

