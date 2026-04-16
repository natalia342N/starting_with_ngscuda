# OLD_Graph Capture with NGSCuda

CUDA Graphs allow capturing a sequence of GPU operations once and replaying them with lower overhead. This is useful when the same kernel sequence is executed many times, as in Krylov solvers.

## Basic Example

```python
import ngsolve.ngscuda as ngscuda
from ngsolve import BaseVector

n = 1_000_000
v = BaseVector(n)
v[:] = 1.0
dv = v.CreateDeviceVector(copy=True)

# Warm-up
dv *= 1.000001

# Capture
g = ngscuda.CudaGraph()
g.BeginCapture()
dv *= 1.000001
g.EndCapture()

# Replay 1000 times
for _ in range(1000):
    g.Launch()

# Correctness: 1.000001^1000 ≈ 1.001001
print("dv[0]:", float(dv[0]))
```

Submit via `sbatch` on a GPU node — see `installation.md` for the job script template.

## Stream Routing

For a kernel to be recorded into a graph, it must be submitted on the **capture stream** — the stream passed to `cudaStreamBeginCapture`. 

In NGSolve, all GPU work runs on `ngs_cuda_stream`. `CudaGraph::BeginCapture()` redirects this global stream to the capture stream, so raw CUDA kernels (such as the block-Jacobi preconditioner) are captured automatically.

<!-- Library calls such as `cusparseSpMV` use a separate handle with its own stream assignment and require explicit `cusparseSetStream` to participate in capture. -->
