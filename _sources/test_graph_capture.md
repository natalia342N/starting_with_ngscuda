# Graph Capture with NGSCuda on the device

In the same folder where NGSolve with NGSCuda is installed, create a script `test_graph_capture.py` with the following content:

```python
import sys, os
from time import time

print("Python:", sys.version)
print("Executable:", sys.executable)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

import ngsolve
import ngsolve.ngscuda as ngscuda
from ngsolve import BaseVector

print("ngsolve:", ngsolve.__file__)
print("ngscuda:", ngscuda.__file__)
print("Has CudaGraph:", hasattr(ngscuda, "CudaGraph"))
print("Imports OK")

# Create a vector on host
n = 1_000_000
v = BaseVector(n)
v[:] = 1.0

# Move to device
dv = v.CreateDeviceVector(copy=True)

# Warm-up execution
dv *= 1.000001

# Sync before capture/timing if available
if hasattr(ngscuda, "DeviceSynchronize"):
    ngscuda.DeviceSynchronize()
else:
    _ = float(dv[0])

# Capture + instantiate graph
g = ngscuda.CudaGraph()
g.BeginCapture()
dv *= 1.000001      # should be captured
g.EndCapture()

# Timing
runs = 1000

# Sync before timing
if hasattr(ngscuda, "DeviceSynchronize"):
    ngscuda.DeviceSynchronize()
else:
    _ = float(dv[0])

ts = time()
for _ in range(runs):
    g.Launch()

# Sync after launches 
if hasattr(ngscuda, "DeviceSynchronize"):
    ngscuda.DeviceSynchronize()
else:
    _ = float(dv[0])
te = time()

print("Graph replay time per launch (s):", (te - ts) / runs)

# Correctness check 
print("dv[0] (after replays):", float(dv[0]))

```

i dont think this yet passes the correctness check..

but to run the script use:

```bash
#!/bin/bash
#SBATCH --job-name=ngscuda_graph
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --threads-per-core=1
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

ml --force purge
ml load ASC/2023.06
ml load buildenv/default-foss-2023a
ml load CMake/3.26.3-GCCcore-12.3.0
ml load CUDA/12.9.0
ml load SciPy-bundle/2023.07-gfbf-2023a
ml load occt/7.8.0-GCCcore-12.3.0
ml unload pybind11/2.11.1-GCCcore-12.3.0 || true

# Paths
WORKING_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
VENV="$WORKING_DIR/ngs"
PREFIX="$WORKING_DIR/install"

source "$VENV/bin/activate"

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

if [ -n "${EBROOTCUDA:-}" ] && [ -d "$EBROOTCUDA/lib64" ]; then
  export LD_LIBRARY_PATH="$EBROOTCUDA/lib64:${LD_LIBRARY_PATH:-}"
fi

PYDIR=$(find "$PREFIX" -maxdepth 6 -type d \( -name site-packages -o -name dist-packages \) | head -n 1)
export PYTHONPATH="$PYDIR:${PYTHONPATH:-}"
export PATH="$PREFIX/bin:$PATH"


which python
python -V
nvidia-smi
echo

echo "Job started at $(date)"
echo "Running on host $(hostname)"
echo

python test_graph_capture.py

echo
python - <<'PY'
import ngsolve, ngsolve.ngscuda as ngscuda
print("NGSolve version:", getattr(ngsolve, "__version__", "unknown"))
print("Has CudaGraph:", hasattr(ngscuda, "CudaGraph"))
PY

echo
echo "Job finished at $(date)"


```

