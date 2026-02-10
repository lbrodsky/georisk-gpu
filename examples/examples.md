# EXAMPLE SCRIPTS 

### Multi-core processing in Python
Default (full stress test)
```
$ python multicore_cpu_benchmark.py
```

Explicit 16-core run
```
python multicore_cpu_benchmark.py --n-jobs 16
```

Memory-safe test (smaller dataset)
```
python multicore_cpu_benchmark.py \
  --n-samples 500000 \
  --n-estimators 100
```

Single configuration only (no comparison)
```
python multicore_cpu_benchmark.py --single-run --n-jobs 16
```

Full example: run with all parameters specified
```
python multicore_cpu_benchmark.py \
  --n-samples 2000000 \
  --n-features 50 \
  --noise 0.1 \
  --pca-components 20 \
  --n-estimators 200 \
  --cv-folds 3 \
  --single-run \
  --n-jobs 32 \
  --random-state 42
```