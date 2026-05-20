# CPU computing 

This document explains how to run **multi-core CPU scientific workloads** on the GeoRisk GPU workstation.

Python Scientific Libraries: NumPy, SciPy, scikit-learn, PyTorch (CPU mode), XGBoost, pandas, rasterio / GDAL 


# Verify CPU Resources

Check CPU configuration:

```bash id="4f0z7j"
lscpu
```

Check memory:

```bash id="0e0m1k"
free -h
```

Monitor CPU usage:

```bash id="uk7fxs"
htop
```

---

# Example: Multicore CPU Benchmark

This repository contains a benchmark example demonstrating:

* NumPy-heavy workloads
* scikit-learn multicore parallelization 
* PCA + RandomForest pipelines 
* cross-validation scaling
* CPU speedup measurements

Example location:

```id="z9w6ka"
examples/numpy_sklearn/multicore_cpu_benchmark.py
```

The benchmark:

* generates large synthetic datasets
* uses CPU-intensive models
* explicitly controls CPU cores
* measures wall-clock execution time
* compares single-core vs multicore performance


**Running the Benchmark** 

```bash id="v4v3xj"
python3 multicore_cpu_benchmark.py \
  --n-samples 2000000 \
  --n-features 50 \
  --noise 0.1 \
  --pca-components 20 \
  --n-estimators 200 \
  --cv-folds 3 \
  --n-jobs 32 \
  --random-state 42 
```

**Single benchmark run** 

```bash id="eqx0h4"
python3 multicore_cpu_benchmark.py \
    --single-run
```

**Benchmark Parameters**

| Parameter          | Description               |
| ------------------ | ------------------------- |
| `--n-samples`      | synthetic dataset size    |
| `--n-features`     | number of input variables |
| `--pca-components` | PCA dimensionality        |
| `--n-estimators`   | RandomForest trees        |
| `--cv-folds`       | cross-validation folds    |
| `--n-jobs`         | CPU cores used            |
| `--blas-threads`   | BLAS/OpenMP threads       |



# Scikit-learn Parallelization

Many scikit-learn models support multicore execution.

Example:

```python id="tmx96m"
RandomForestRegressor(
    n_estimators=200,
    n_jobs=32
)
```


# joblib Parallelization

scikit-learn internally uses:

* joblib
* multiprocessing

Example:

```python id="nhl4n5"
from joblib import cpu_count

n_jobs = cpu_count() - 1
```

---

# Multiprocessing

Python multiprocessing example:

```python id="rphc2n"
import multiprocessing

n_workers = max(1, multiprocessing.cpu_count() - 1)

pool = multiprocessing.Pool(n_workers)
```

Useful for:

* independent tasks
* raster tiling
* Monte Carlo simulations
* preprocessing pipelines

--- 

# OpenMPI paralelization 
```
mpirun --version
```

Create a simple MPI test:
```
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    printf("Hello from MPI!\n");
    MPI_Finalize();
    return 0;
}
```

Compile & run:

```
mpicc test.c -o test
mpirun -np 4 ./test
```

