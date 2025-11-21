# VDPC (Python)

Python reimplementation of the MATLAB scripts in `experiments/VDPC`. The goal is a drop-in, NumPy-first version of the original `main.m` workflow so it can be used directly inside this repository without MATLAB.

## Files

- `vdpc.py`: translated functions (`compute_dc`, `get_local_density`, `get_distance_to_higher_density`, `DBSCAN`, `mknn`, etc.) and an `run_vdpc` helper that mirrors `main.m` end-to-end.
- `__init__.py`: exports the public API.

## Quick start

```bash
python - <<'PY'
import numpy as np
from experiments.vdpc_py import run_vdpc

# toy data: two blobs
rng = np.random.default_rng(0)
data = np.concatenate([
    rng.normal(loc=(-2, -2), scale=0.4, size=(40, 2)),
    rng.normal(loc=(2, 2), scale=0.5, size=(40, 2)),
])

result = run_vdpc(data, percent=0.4, min_delta=0.5, min_rho=0)
print("labels shape:", result.labels.shape)
print("unique labels:", np.unique(result.labels))
PY
```

To reuse an existing MATLAB `.mat` dataset, load it and feed the `data` matrix:

```python
from scipy.io import loadmat
mat = loadmat("pathbased.mat")
data = mat["data"]           # adjust the key to your file
labels = mat.get("label")    # optional, used only for your own evaluation
result = run_vdpc(data)
```

## Alignment with MATLAB

If you need to compare against the MATLAB scripts:

`run_vdpc` defaults are chosen to match MATLAB more closely (`center_rho_eps=1e-12` allows rho≈0 points to become centers). When you need 1:1 comparison, load the MATLAB labels and majority-map Python IDs to MATLAB IDs (see snippet below).

## Notes and differences from MATLAB

- Noise/outlier points are still marked with `100` to match the original DBSCAN code; everything else is 1-based cluster IDs.
- Plotting the decision graph is optional: pass `plot=True` to `run_vdpc` if `matplotlib` is available.
- The MATLAB script referenced helper functions (`drawgraph`, `Evaluation`, `showShapeSet`) that were not present in the repository; their responsibilities are not part of this port. You can plug in your own visualisation or evaluation routine using `VDPCResult.labels`.

## Parameters (run_vdpc / vdpc_cluster)

- `data`: `np.ndarray` (n_samples, n_features), required input.
- `percent` (default 0.4): percentage for cutoff distance `dc` in `compute_dc` (0.4 means 0.4%).
- `min_delta` (default 3.5): delta threshold for picking density peaks.
- `min_rho` (default 0.0): rho threshold for picking density peaks.
- `plot` (default False): plot the decision graph if Matplotlib is available.
- `rng`: optional `np.random.Generator` for deterministic DBSCAN visiting order.
- `continue_on_start_100` (default False): force running downstream steps even if no low-density gap is found.
- `center_rho_eps` (default 1e-12): small epsilon applied to rho when deciding centers so extremely small rho values are retained (helps match MATLAB).

### 调参提示

- `percent`（默认 0.4）：用于计算 cutoff 距离 `dc` 的百分位数。取值越大，`dc` 越大，等效于加粗邻域，密度曲线更平滑，中心数通常减少；取值偏小，`dc` 越小，局部差异被放大，容易出现更多密度峰（甚至碎簇）。
- `min_delta`（默认 3.5）：决策图上 delta 的阈值，越低会选出更多峰（簇数增加，易把近邻峰拆开）；越高则更保守（簇数减少，易把弱峰并入大簇）。
- `min_rho`（默认 0.0）：决策图上 rho 的阈值，越高会滤掉低密度峰（弱簇/离群峰被合并或丢弃）；越低则保留更多低密度峰。通常与 `min_delta` 配合调节。
- 交互影响：`percent` 控制密度估计的“粗/细”，`min_delta`/`min_rho` 控制峰的选择。若希望找到更多细粒度簇，可减小 `percent` 或降低 `min_delta`/`min_rho`；若希望稳健少簇，可增大 `percent` 或提高 `min_delta`/`min_rho`。
