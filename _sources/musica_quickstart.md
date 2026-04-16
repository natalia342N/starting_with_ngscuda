# NEW_Running DevCGSolver on Musica

This page provides instructions for running the `DevCGSolver` 
CUDA graph benchmark on the Musica cluster.

## Step 1 — Build and install NGSolve with NGSCuda

Create `build.sh` with the following content:

```bash
#!/bin/sh
set -e

ml --force purge
ml load ASC/2023.06
ml load buildenv/default-foss-2023a
ml load CMake/3.26.3-GCCcore-12.3.0 CUDA/12.9.0 \
    SciPy-bundle/2023.07-gfbf-2023a occt/7.8.0-GCCcore-12.3.0
ml unload pybind11/2.11.1-GCCcore-12.3.0 || true

WORKING_DIR=$(realpath "${PWD}")
VENV="${WORKING_DIR}/ngs"
SOURCES="${WORKING_DIR}/src/ngsolve"

virtualenv "${VENV}"
source ${VENV}/bin/activate

if [ ! -d "${SOURCES}/.git" ]; then
    mkdir -p "$(dirname "${SOURCES}")"
    git clone --recurse-submodules \
        git@gitlab.tuwien.ac.at:ngsolve/ngsolve.git "${SOURCES}"
fi

BUILD_DIR="${WORKING_DIR}/build/ngsolve"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${SOURCES}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_SUPERBUILD=ON \
  -DUSE_OCC=ON \
  -DUSE_CCACHE=ON \
  -DCMAKE_INSTALL_PREFIX="${WORKING_DIR}/install" \
  -DUSE_CUDA=ON \
  -DUSE_GUI=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="90" \
  -DUSE_UMFPACK=OFF \
  -DBUILD_STUB_FILES=OFF

make -j 8
make install

echo "=== Build done ==="
echo "Install prefix: ${WORKING_DIR}/install"
echo "Venv:           ${VENV}"
```

```bash
rm -rf build install ngs
bash build.sh
```
This takes approximately 20-30 minutes on a login node. After completion:

```bash
source ~/ngs/bin/activate
pip install numpy scipy matplotlib
```

---

## Timing benchmark (graph vs no-graph)

Create the test script `test_devcg.py`:

```python
import ngsolve
from ngsolve import *
import ngsolve.ngscuda as ngscuda
from netgen.geom2d import unit_square
import time
import os

mesh = Mesh(unit_square.GenerateMesh(maxh=0.01))
fes = H1(mesh, order=2, dirichlet=".*")
u, v = fes.TnT()
a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
f = LinearForm(x*y*v*dx).Assemble()

blocks = fes.CreateSmoothingBlocks()
pre    = a.mat.CreateBlockSmoother(blocks)
adev   = a.mat.CreateDeviceMatrix()
predev = pre.CreateDeviceMatrix()

solver = ngscuda.DevCGSolver(
    adev, predev,
    adev_raw=adev,
    cdev_raw=predev,
    precision=1e-10,
    maxsteps=1000,
    printrates=False)

gfu = GridFunction(fes)

# warm up
solver.Mult(f.vec, gfu.vec)

# timed solve
t0 = time.perf_counter()
solver.Mult(f.vec, gfu.vec)
t1 = time.perf_counter()

use_graph = os.environ.get("NO_CUDA_GRAPH", "0") == "0"
print(f"ndof       = {fes.ndof}")
print(f"|sol|      = {Norm(gfu.vec):.8e}")
print(f"use_graph  = {use_graph}")
print(f"elapsed    = {(t1-t0)*1000:.3f} ms")
```

Create the submit script `submit_timing.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=ngscuda_timing
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ml --force purge
ml load ASC/2023.06
ml load buildenv/default-foss-2023a
ml load CMake/3.26.3-GCCcore-12.3.0 CUDA/12.9.0
ml load SciPy-bundle/2023.07-gfbf-2023a
ml load occt/7.8.0-GCCcore-12.3.0
ml unload pybind11/2.11.1-GCCcore-12.3.0 || true

source ~/starting_with_ngscuda/ngs/bin/activate
export PYTHONPATH=$(find ~/starting_with_ngscuda/install \
    -maxdepth 6 -type d -name site-packages | head -1)

echo "=== NO GRAPH ==="
NO_CUDA_GRAPH=1 srun --ntasks=1 \
    python ~/starting_with_ngscuda/test_devcg.py

echo "=== GRAPH ==="
srun --ntasks=1 \
    python ~/starting_with_ngscuda/test_devcg.py
```

Submit:

```bash
sbatch submit_timing.sh
```

### Expected output

```
=== NO GRAPH ===
CUDA Device 0: NVIDIA H100, cap 9.0
ndof       = 46741
|sol|      = 1.15654611e+00
use_graph  = False
elapsed    = 30.307 ms

=== GRAPH ===
CUDA Device 0: NVIDIA H100, cap 9.0
[CudaGraph] captured nodes: 22
ndof       = 46741
|sol|      = 1.15654611e+00
use_graph  = True
elapsed    = 23.532 ms
```

Speedup: **~1.29×**, with 22 kernel nodes captured in the graph.

---

## Scaling 

Create `test_devcg_scaling.py`:

```python
import ngsolve
from ngsolve import *
import ngsolve.ngscuda as ngscuda
from netgen.geom2d import unit_square
import time
import os

use_graph = os.environ.get("NO_CUDA_GRAPH", "0") == "0"

for maxh in [0.8, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07,
             0.05, 0.03, 0.02, 0.01, 0.007, 0.005]:
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
    fes = H1(mesh, order=2, dirichlet=".*")
    u, v = fes.TnT()
    a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
    f = LinearForm(x*y*v*dx).Assemble()

    blocks = fes.CreateSmoothingBlocks()
    pre    = a.mat.CreateBlockSmoother(blocks)
    adev   = a.mat.CreateDeviceMatrix()
    predev = pre.CreateDeviceMatrix()

    solver = ngscuda.DevCGSolver(
        adev, predev,
        adev_raw=adev,
        cdev_raw=predev,
        precision=1e-10,
        maxsteps=1000,
        printrates=False)

    gfu = GridFunction(fes)

    # warm up (2 runs)
    solver.Mult(f.vec, gfu.vec)
    solver.Mult(f.vec, gfu.vec)

    # average over 3 timed runs
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        solver.Mult(f.vec, gfu.vec)
        t1 = time.perf_counter()
        times.append((t1-t0)*1000)
    avg = sum(times)/len(times)

    print(f"maxh={maxh:.3f}  ndof={fes.ndof:8d}  "
          f"use_graph={use_graph}  elapsed={avg:.3f} ms")
```

Create `submit_scaling.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=cg_scaling
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ml --force purge
ml load ASC/2023.06
ml load buildenv/default-foss-2023a
ml load CMake/3.26.3-GCCcore-12.3.0 CUDA/12.9.0
ml load SciPy-bundle/2023.07-gfbf-2023a
ml load occt/7.8.0-GCCcore-12.3.0
ml unload pybind11/2.11.1-GCCcore-12.3.0 || true

source ~/starting_with_ngscuda/ngs/bin/activate
export PYTHONPATH=$(find ~/starting_with_ngscuda/install \
    -maxdepth 6 -type d -name site-packages | head -1)

echo "=== NO GRAPH ==="
NO_CUDA_GRAPH=1 srun --ntasks=1 \
    python ~/starting_with_ngscuda/test_devcg_scaling.py

echo "=== GRAPH ==="
srun --ntasks=1 \
    python ~/starting_with_ngscuda/test_devcg_scaling.py
```

Submit:

```bash
sbatch submit_scaling.sh
```

### Expected output

| ndof | no-graph (ms) | graph (ms) | speedup |
|------|-------------|----------|---------|
| 9 | 0.225 | 0.319 | 0.71× |
| 21 | 0.392 | 0.421 | 0.93× |
| 45 | 0.991 | 0.844 | 1.17× |
| 125 | 1.495 | 1.193 | 1.25× |
| 1,961 | 5.649 | 4.217 | 1.34× |
| 11,825 | 14.153 | 10.478 | 1.35× |
| 46,741 | 30.189 | 23.143 | 1.30× |
| 185,809 | 91.170 | 76.780 | 1.19× |

The graph version becomes beneficial above approximately **ndof = 30**,
with peak speedup of **~1.35×** at ndof ≈ 5,000–12,000.

---

## Step 5 — Nsight Systems profiling

Create `submit_nsys.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=ngscuda_nsys
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ml --force purge
ml load ASC/2023.06
ml load buildenv/default-foss-2023a
ml load CMake/3.26.3-GCCcore-12.3.0 CUDA/12.9.0
ml load SciPy-bundle/2023.07-gfbf-2023a
ml load occt/7.8.0-GCCcore-12.3.0
ml unload pybind11/2.11.1-GCCcore-12.3.0 || true

source ~/starting_with_ngscuda/ngs/bin/activate
export PYTHONPATH=$(find ~/starting_with_ngscuda/install \
    -maxdepth 6 -type d -name site-packages | head -1)

OUTBASE="nsys_devcg_${SLURM_JOB_ID}"

nsys profile \
    -t cuda,nvtx,osrt \
    --cuda-graph-trace=node \
    --cuda-event-trace=false \
    --stats=true \
    --force-overwrite=true \
    --sample=none \
    --cpuctxsw=none \
    -o "$OUTBASE" \
    python ~/starting_with_ngscuda/test_devcg2.py
```

Submit:

```bash
sbatch submit_nsys.sh
```

This generates a `.nsys-rep` file which can be opened in
**NVIDIA Nsight Systems** locally.
With `--cuda-graph-trace=node`, individual kernels inside 
the graph are visible in the timeline.


From the `cuda_gpu_kern_sum` report:

| Kernel | % GPU time | Avg duration |
|--------|-----------|-------------|
| `BlockJacobiKernel` | 17.2% | 6,003 ns |
| `cusparse csrmv_v3` | 13.1% | 4,552 ns |
| `reduce_1Block` (×2) | 11.2% | 1,950 ns |
| `dot_kernel` (×2) | 10.0% | 1,745 ns |
| `csr_partition_kernel` | 9.9% | 3,465 ns |

<!-- From the `cuda_api_sum` report:

| API call | Graph path | No-graph path |
|----------|-----------|--------------|
| Launch overhead | 928 × 15µs = **14ms** | 16,732 × 3µs = **53ms** |
| Sync (convergence) | 938 × 44µs | 936 × 5µs |

The graph eliminates **~38ms** of launch overhead across ~930 CG iterations,
explaining the wall-clock speedup. -->

---

## Disabling CUDA graph capture

Set the environment variable `NO_CUDA_GRAPH=1` before running:

```bash
NO_CUDA_GRAPH=1 python test_devcg.py
```

Or in a submit script:

```bash
NO_CUDA_GRAPH=1 srun --ntasks=1 python test_devcg.py
```

