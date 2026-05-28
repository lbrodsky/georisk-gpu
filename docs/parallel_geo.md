# Parallel geocompotuation

### 1. Multiprocessing (CPU parallelism)
Case: 
- independent raster tiles,
- vector feature batches,
- per-scene satellite processing

**Parallel raster tile processing** 
```python id="tmx96m"
from multiprocessing import Pool
import rasterio
import numpy as np

def compute_mean(tile_path):
    with rasterio.open(tile_path) as src:
        data = src.read(1)
        return np.nanmean(data)

tiles = [
    "tile1.tif",
    "tile2.tif",
    "tile3.tif"
]

with Pool(processes=4) as pool:
    results = pool.map(compute_mean, tiles)

print(results)
```

Limitation of this approach is memory duplication. 

### 2. Rasterio windows for parallel raster processing

Case of: 
- large rasters processed by windows/tiles.

```python id="tmx96m"
import rasterio
from rasterio.windows import Window
from concurrent.futures import ProcessPoolExecutor

def process_window(args):
    path, window = args

    with rasterio.open(path) as src:
        data = src.read(1, window=window)
        return data.mean()

windows = [
    Window(0, 0, 1024, 1024),
    Window(1024, 0, 1024, 1024)
]

tasks = [("large.tif", w) for w in windows]

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_window, tasks))

print(results)
```

### 3. Joblib parallel computing 
Case of: 
- large rasters

See:  
Brodský, L., Landa, M., Bouček, T., Pešek, O., & Halounová, L. (2026). The LUCAS dataset revisited: enhancing spatial representativeness for machine learning land cover mapping. International Journal of Digital Earth, 19(1). https://doi.org/10.1080/17538947.2026.2644671
 
 Full code with parallel processing of tiles over Europe: https://github.com/lbrodsky/LUCAS_representativeness/tree/main 

```python id="tmx96m"
from joblib import Parallel, delayed 
import multiprocessing

# process info
logging.info(f'Number of CPUs: {multiprocessing.cpu_count()}')
logging.info(f'Number of workers to run: {args.workers}')

# Parallel jobs 
if args.workers > 1:
    # joblib parallelization
    logging.info(f'Number of tiles to process: {len(tiles)}')
    Parallel(n_jobs=args.workers, backend="threading", verbose=10)(
        delayed(process_single_tile)([t, args.lucas_points, args.lucas_thr_points, dst_dir, selected_points, lc_mappings, args]) for t in tiles)
else:
    # process using one core
    for t in tiles:
        process_single_tile([t, args.lucas_points, args.lucas_thr_points, dst_dir, selected_points, lc_mappings, args])
```

### 4. Dask large-scale parallel arrays
Case of: 
- large rasters,
- xarray datasets,
- chunked satellite imagery

```python id="tmx96m"
import rioxarray
import dask.array as da

nir = rioxarray.open_rasterio(
    "nir.tif",
    chunks={"x": 2048, "y": 2048}
)

red = rioxarray.open_rasterio(
    "red.tif",
    chunks={"x": 2048, "y": 2048}
)

ndvi = (nir - red) / (nir + red)

mean_ndvi = ndvi.mean().compute()

print(mean_ndvi.values)
```

### 5. Parallel machine learning 
Case of: 
- any geospatial data tabulated 

```python id="tmx96m"
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1
)
``` 
Beaware: n_jobs=-1 uses all CPU cores! 

### 6. GPU Parallelism with PyTorch 
Case of: 
- semantic segmentation,
- satellite image classification,
- deep learning on raster data. 

PyTorch automatically parallelizes tensor operations on the GPU. 

```python id="tmx96m"
import torch

# Select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create large tensor on GPU
x = torch.rand((10000, 10000), device=device)

# Parallel GPU computation
result = torch.mean(x)

print(result)
print("Running on:", device)
```
