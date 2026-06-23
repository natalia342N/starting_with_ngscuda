# Profiling with NVIDIA Nsight Systems

## No graph CG

:::{figure} pictures/nograph1.png
:alt: No-graph CG timeline showing per-iteration DtoH spikes
:width: 100%
20 iterations: each red spike is one convergence check transfer (8 bytes).
:::


One iteration kernel sequence:

| Step | Kernel |
|---|---|
| SpMV | `csrmv_v3_kernel` |
| Preconditioner | `BlockJacobiKernel` |
| Inner products | `dot_kernel` + `reduce_1Block_kernel` |
| Scalar updates | `CUDA_forall` (α, β, x, r, p) |
| **Convergence check** | **`Memcpy DtoH (Pageable)`** |

![Timeline overview](pictures/nograph2.png)
![Timeline overview](pictures/nograph3.png)



---

## CUDA graph CG
