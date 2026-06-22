## CUDA Graph Acceleration for Krylov Solvers

NGSolve already supports GPU execution via `ngscuda`, but for matrix-free operators
(like the convection operator) performance can be limited by **kernel launch and
synchronization overhead**, especially when the same operator is applied repeatedly.

CUDA Graphs offer a way to:
- **capture** a fixed sequence of GPU operations once
- **replay** it efficiently many times with lower CPU overhead

**The bottleneck in Krylov solvers:**

```
iteration 1:  [GPU kernels] → D2H sync → converged? no → ...
iteration 2:  [GPU kernels] → D2H sync → converged? no → ...
   ...
iteration N:  [GPU kernels] → D2H sync → converged? yes
```

~500 iterations × one host–device sync = measurable overhead,
especially at small–medium problem sizes where GPU compute is fast.

> The GPU is not the bottleneck — the **synchronization** is.


## CUDA Graphs — capture once, replay many times

CUDA Graphs capture a fixed sequence of GPU operations and replay them with minimal CPU overhead.

```python
import ngsolve.ngscuda as ngscuda

g = ngscuda.CudaGraph()
g.BeginCapture()
dv *= 1.000001          # any GPU kernel(s)
g.EndCapture()

for _ in range(1000):
    g.Launch()          # replay
```

**For a kernel to be captured, it must run on the capture stream.**  
NGSolve routes all GPU work through `ngs_cuda_stream` — `BeginCapture()` redirects this stream automatically, so kernels (SpMV, preconditioner, dot products) are all captured without changes to user code.

---

## CUDA WHILE graph — the convergence loop on GPU

Standard graphs replay a **fixed** sequence. But a Krylov solver must loop until convergence — the number of iterations is not known in advance.

**CUDA WHILE graph** (CUDA ≥ 12.4): the entire convergence loop is a single conditional graph node.

```
Traditional:   [kernels] → D2H sync → converged? → [kernels] → D2H sync → ...
WHILE graph:   [capture loop once] → Launch()     ← zero per-iteration syncs
```

`ConvergenceCheckKernel` runs GPU-side and calls `cudaGraphSetConditional` to break the loop — the CPU is not involved until the solve is complete.

| Mode | Requirement | Syncs per solve |
|---|---|---|
| WHILE graph | CUDA ≥ 12.4 | **0** |
| Per-iteration graph | any CUDA | 1 per iteration |
| No graph | `NO_CUDA_GRAPH=1` | 1 per iteration |

Selected automatically — no user configuration needed.


## Part 1: DevCGSolver — Poisson (symmetric SPD)
# H100, ndof = 460,033, 5 runs averaged

adev   = a.mat.CreateDeviceMatrix()         # ← 3 lines added
jacdev = jac.CreateDeviceMatrix()
fdev   = f.vec.CreateDeviceVector(copy=True)

# Traditional GPU CG (Python-level loop, per-iteration sync)
inv = CGSolver(adev, jacdev, maxiter=2000)
gfu.vec.data = (inv * fdev).Evaluate()

# WHILE graph CG (entire loop on GPU, zero syncs)
gpu_solver = DevCGSolver(mat=adev, pre=jacdev,
                         adev_raw=adev, cdev_raw=jacdev, maxsteps=2000)
gfu.vec.data = (gpu_solver * fdev).Evaluate()
