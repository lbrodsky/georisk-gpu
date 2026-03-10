# CPU computing 

## Python scientific libraries automatically use multiple CPU cores.
- NumPy, SciPy, scikit-learn, PyTorch (CPU mode)

Parallelization: 
- TODO: joblib, OpenMP

## Example: Multicore CPU Benchmark
This repository contains a benchmark example demonstrating CPU parallelization.

Example location:
```
examples/numpy_sklearn/multicore_cpu_benchmark.py
```

Scikit-learn Parallelization: 
```
RandomForestClassifier(n_jobs=-1) 
```

Best practice in Python: 
```
import os

n_jobs = max(1, os.cpu_count() - 1)
print(f"Using {n_jobs} CPU cores")
```

```
from sklearn.ensemble import RandomForestClassifier
import os

n_jobs = max(1, os.cpu_count() - 1)

model = RandomForestClassifier(
    n_estimators=200,
    n_jobs=n_jobs
)
```

## Multiprocessing
```
import multiprocessing

n_workers = max(1, multiprocessing.cpu_count() - 1)

pool = multiprocessing.Pool(n_workers) 
```
