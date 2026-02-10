#!/usr/bin/env Python3

"""
The idea:
- Generate a large synthetic dataset
- Use CPU-intensive models that parallelize well
- Explicitly control and test number of CPU cores
- Measure wall-clock time

Models used:
- RandomForestRegressor → embarrassingly parallel
- StandardScaler + PCA → linear algebra heavy
- Cross-validation → multiplies CPU load

Usage:
$ python multicore_cpu_benchmark.py \
  --n-samples 2000000 \
  --n-features 50 \
  --noise 0.1 \
  --pca-components 20 \
  --n-estimators 200 \
  --cv-folds 3 \
  --n-jobs 32 \
  --blas-threads 32 \
  --random-state 42

# TODO: pip install tqdm
"""


import os
import time
import argparse
import json
import socket
from datetime import datetime
from pathlib import Path
import platform
import psutil
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from joblib import cpu_count
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def get_hardware_info():
    """
    Collect basic hardware and system information.
    """
    cpu_freq = psutil.cpu_freq()
    mem = psutil.virtual_memory()

    return {
        "cpu_model": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "cpu_max_frequency_mhz": cpu_freq.max if cpu_freq else None,
        "total_memory_gb": round(mem.total / (1024 ** 3), 2),
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version()
    }

def set_blas_threads(n_threads: int):
    """
    Control BLAS / OpenMP thread usage to avoid oversubscription.
    """
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)


def run_benchmark(args, n_jobs: int):
    """
    Run CPU-intensive NumPy + scikit-learn benchmark.
    """

    print("\n" + "=" * 70)
    print(f"Running benchmark with n_jobs = {n_jobs}")
    print(f"Samples={args.n_samples:,}, Features={args.n_features}, "
          f"Estimators={args.n_estimators}, PCA={args.pca_components}")
    print("=" * 70)

    # ---------------------------
    # Data generation (NumPy heavy)
    # ---------------------------
    t0 = time.time()
    X, y = make_regression(
        n_samples=args.n_samples,
        n_features=args.n_features,
        noise=args.noise,
        random_state=args.random_state
    )
    print(f"Data generation time: {time.time() - t0:.2f} s")

    # ------------------------------------
    # Pipeline: Scaling -> PCA -> RandomForest
    # ------------------------------------
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=args.pca_components)),
        ("rf", RandomForestRegressor(
            n_estimators=args.n_estimators,
            n_jobs=n_jobs,
            random_state=args.random_state
        ))
    ])

    # ---------------------------
    # Cross-validation (CPU heavy)
    # ---------------------------
    t0 = time.time()
    kf = KFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state
    )

    scores = []

    t0 = time.time()
    for fold, (train_idx, test_idx) in enumerate(
            tqdm(kf.split(X), total=args.cv_folds, desc="Cross-validation folds")):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = r2_score(y_test, y_pred)
        scores.append(score)

    elapsed = time.time() - t0
    scores = np.array(scores)

    print(f"CV R² scores: {np.round(scores, 4)}")
    print(f"Mean R²: {scores.mean():.4f}")
    print(f"Total training time: {elapsed:.2f} s")

    return {
        "n_jobs": n_jobs,
        "cv_scores": scores.tolist(),
        "mean_r2": float(scores.mean()),
        "elapsed_seconds": elapsed
    }


def save_report(args, results):
    """
    Save benchmark report to file with hostname and timestamp.
    """
    hostname = socket.gethostname()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    base_name = f"benchmark_{hostname}_{timestamp}"

    # ---------------------------
    # Save JSON (machine-readable)
    # ---------------------------
    json_path = results_dir / f"{base_name}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    # ---------------------------
    # Save TXT (human-readable)
    # ---------------------------
    txt_path = results_dir / f"{base_name}.txt"
    with open(txt_path, "w") as f:
        f.write("CPU MULTI-CORE BENCHMARK REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Hostname: {hostname}\n")
        f.write(f"Timestamp: {timestamp}\n\n")

        f.write("Configuration:\n")
        for k, v in vars(args).items():
            f.write(f"  {k}: {v}\n")

        f.write("\nResults:\n")
        for k, v in results.items():
            f.write(f"  {k}: {v}\n")

    print(f"\n Report saved to:")
    print(f"  {json_path}")
    print(f"  {txt_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-core CPU benchmark using NumPy + scikit-learn"
    )

    # ---------------------------
    # Data parameters
    # ---------------------------
    parser.add_argument("--n-samples", type=int, default=2_000_000,
                        help="Number of samples")
    parser.add_argument("--n-features", type=int, default=50,
                        help="Number of input features")
    parser.add_argument("--noise", type=float, default=0.1,
                        help="Noise level for synthetic regression")
    parser.add_argument("--pca-components", type=int, default=20,
                        help="Number of PCA components")

    # ---------------------------
    # Model parameters
    # ---------------------------
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="Number of trees in RandomForest")
    parser.add_argument("--cv-folds", type=int, default=3,
                        help="Number of cross-validation folds")

    # ---------------------------
    # CPU / threading parameters
    # ---------------------------
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Number of CPU cores (-1 = all cores)")
    parser.add_argument("--blas-threads", type=int, default=None,
                        help="Threads for BLAS/OpenMP (default = n_jobs)")

    # ---------------------------
    # Runtime options
    # ---------------------------
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--single-run", action="store_true",
                        help="Run only once with --n-jobs")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    total_cores = cpu_count()
    print(f"Detected CPU cores: {total_cores}")

    # Resolve n_jobs
    if args.n_jobs == -1:
        n_jobs = total_cores
    else:
        n_jobs = args.n_jobs

    # Resolve BLAS threads
    blas_threads = args.blas_threads or n_jobs
    set_blas_threads(blas_threads)

    print(f"Using n_jobs={n_jobs}")

    # ---------------------------
    # Run benchmarks
    # ---------------------------
    results = {
        "hostname": socket.gethostname(),
        "hardware": get_hardware_info(),
        "total_cpu_cores": total_cores,
        "n_jobs_used": n_jobs,
        "blas_threads": blas_threads,
        "parameters": vars(args),
        "runs": {}
    }

    if args.single_run:
        run_result = run_benchmark(args, n_jobs=n_jobs)
        results["runs"]["single"] = run_result
    else:
        single = run_benchmark(args, n_jobs=1)
        multi = run_benchmark(args, n_jobs=n_jobs)

        results["runs"]["single_core"] = single
        results["runs"]["multi_core"] = multi
        results["speedup"] = single["elapsed_seconds"] / multi["elapsed_seconds"]

        print("\n" + "=" * 70)
        print(f"Speedup using {n_jobs} cores: {results['speedup']:.2f}×")
        print("=" * 70)

    save_report(args, results)


