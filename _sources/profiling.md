# Profiling GPU Overhead with NVIDIA Nsight Systems

To motivate the CUDA WHILE graph approach, we first measured the actual
synchronization overhead in a standard (no-graph) CG solve using
**NVIDIA Nsight Systems (nsys)**.

---

## What we measured

Two profiling runs on H100 (Musica), ndof = 46,741:

| Run | Description | Script |
|---|---|---|
| No-graph CG | 3 iterations, `NO_CUDA_GRAPH=1` | `test_cg_nsys_nograph.py` |
| WHILE graph CG | Full solve (~460 iterations) | `test_cg_nsys_while.py` |

Both scripts use `cuProfilerStart` / `cuProfilerStop` to bracket exactly the
region of interest, captured with `--capture-range=cudaProfilerApi`.

---

## No-graph CG — per-iteration D2H stalls

```python
# test_cg_nsys_nograph.py (key excerpt)
import os
os.environ["NO_CUDA_GRAPH"] = "1"   # disable all graph capture

# ... setup mesh, assemble, move to device ...

cuda.cuProfilerStart()
solver.Mult(fdev, gfu.vec)          # 3 iterations only
cuda.cuProfilerStop()
```

In Nsight Systems, the no-graph timeline shows:

- One `cudaStreamSynchronize` **per iteration** — the CPU stalls waiting for
  the convergence norm to be copied back from device to host
- Each sync interrupts the GPU pipeline and forces a host–device round-trip
- Over 460 iterations this adds up to measurable wasted time

---

## WHILE graph CG — one launch, zero syncs

```python
# test_cg_nsys_while.py (key excerpt)

# Warmup: graph is built and instantiated
solver.Mult(fdev, gfu.vec)

cuda.cuProfilerStart()
solver.Mult(fdev, gfu.vec)          # full solve, WHILE graph
cuda.cuProfilerStop()
```

NSys confirmed (full solve, ~460 iterations):

| Event | Count |
|---|---|
| `cudaGraphLaunch` | **2** (warmup + timed run) |
| `cudaStreamSynchronize` | **14** (setup only, none per-iteration) |
| `ConvergenceCheckKernel` | **928** (all GPU-side) |
| Per-iteration D2H transfers | **0**  |

The entire CG loop — including the convergence check — runs without CPU involvement.

---

## How to run the profiling yourself

Submit on a Musica GPU node:

```bash
sbatch submit_cg_nsys.sh
```

This produces two Nsight Systems report files:

```
cg_nsys_nograph-<jobid>.nsys-rep
cg_nsys_while-<jobid>.nsys-rep
```

Open them in **Nsight Systems** (GUI, on your local machine):

```bash
nsys-ui cg_nsys_nograph-<jobid>.nsys-rep
```

Look for the `CUDA API` row — in the no-graph profile you will see
repeated `cudaStreamSynchronize` calls between each kernel group.
In the WHILE graph profile, only a single `cudaGraphLaunch` is visible
for the entire solve.

---

## Submit script

```bash
#!/bin/bash
#SBATCH --job-name=cg_nsys
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ... module loads and env setup (see installation.md) ...

NSYS_ARGS=(profile -t cuda,nvtx
           --capture-range=cudaProfilerApi
           --stats=true --force-overwrite=true)

nsys "${NSYS_ARGS[@]}" -o "cg_nsys_nograph-${SLURM_JOB_ID}" \
    env ${PY_ENV} python3 test_cg_nsys_nograph.py

nsys "${NSYS_ARGS[@]}" -o "cg_nsys_while-${SLURM_JOB_ID}" \
    env ${PY_ENV} python3 test_cg_nsys_while.py
```

> **Note:** `LD_LIBRARY_PATH` must **not** be exported globally when running nsys —
> it causes nsys itself to crash. Pass it only to the Python child via `env VAR=val python3 ...`.
> This was discovered during development and is already handled in `submit_cg_nsys.sh`.
