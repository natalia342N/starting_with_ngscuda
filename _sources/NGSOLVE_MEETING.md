---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# GPU-Accelerated Krylov Solvers in NGSolve

**NGSolve User Meeting 2026 — Natalia Tylek, TU Wien**

This notebook shows how to use the new GPU Krylov solvers in NGSolve:
- **`DevCGSolver`** — for symmetric problems (Poisson, elasticity)
- **`DevTFQMRSolver`** — for non-symmetric problems (convection-diffusion, Navier–Stokes)

```{admonition} GPU required
:class: important
To run this notebook on GPU in Colab: **Runtime → Change runtime type → T4 GPU**
```

```{code-cell} ipython3
:tags: [hide-cell]
import sys
if 'google.colab' in sys.modules:
    !pip install --upgrade --pre ngsolve anywidget -q
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/natalia342N/starting_with_ngscuda/blob/main/docs/ngsolve_meeting_tutorial.ipynb)

```{code-cell} ipython3
# Check GPU is available
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                        capture_output=True, text=True)
if result.returncode == 0:
    print("GPU found:", result.stdout.strip())
else:
    print("No GPU detected — please switch to a GPU runtime.")
```

```{code-cell} ipython3
from ngsolve import *
from ngsolve import la
from netgen.occ import unit_square
from netgen.csg import unit_cube
import ngsolve.ngscuda as ngscuda
import time

def timed(solver, rhs, sol, runs=5):
    """Average solver.Mult time over several runs (ms)."""
    solver.Mult(rhs, sol)   # warm-up
    t0 = time.perf_counter()
    for _ in range(runs):
        solver.Mult(rhs, sol)
    return (time.perf_counter() - t0) / runs * 1000

print("NGSolve version:", ngsolve.__version__)
```

---

## Part 1: Symmetric problem — Poisson equation

$$-\Delta u = f \quad \text{in } \Omega, \qquad u = 0 \quad \text{on } \partial\Omega$$

The stiffness matrix is **symmetric positive definite** → use Conjugate Gradient.

```{code-cell} ipython3
# Standard NGSolve setup — nothing new here
mesh = Mesh(unit_cube.GenerateMesh(maxh=0.08))
fes  = H1(mesh, order=2, dirichlet=".*")
u, v = fes.TnT()

a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
f = LinearForm(x*y*z*v*dx).Assemble()
gfu = GridFunction(fes)

print(f"ndof = {fes.ndof}")
```

```{code-cell} ipython3
# CPU solve — as usual
pre = Projector(fes.FreeDofs(), True)

cpu_solver = la.CGSolver(mat=a.mat, pre=pre, precision=1e-8, maxsteps=1000, printrates=False)

t0 = time.perf_counter()
for _ in range(5):
    gfu.vec.data = cpu_solver * f.vec
cpu_ms = (time.perf_counter() - t0) / 5 * 1000

ref_norm = Norm(gfu.vec)
print(f"CPU CG:  {cpu_ms:.1f} ms   |sol| = {ref_norm:.6e}   iters = {cpu_solver.GetSteps()}")
```

### Moving to GPU — two extra lines

Everything else stays the same. You get the same result, faster.

```{code-cell} ipython3
# Move matrices to GPU  ← new
adev = a.mat.CreateDeviceMatrix()
pdev = pre.CreateDeviceMatrix()
fdev = f.vec.CreateDeviceVector(copy=True)

# DevCGSolver — same interface as CGSolver  ← new
gpu_solver = ngscuda.DevCGSolver(
    mat=adev, pre=pdev,
    adev_raw=adev, cdev_raw=pdev,
    precision=1e-8, maxsteps=1000, printrates=False
)

gpu_ms = timed(gpu_solver, fdev, gfu.vec)
err = abs(Norm(gfu.vec) - ref_norm)

print(f"GPU CG:  {gpu_ms:.1f} ms   |sol| = {Norm(gfu.vec):.6e}   err vs CPU = {err:.1e}")
print(f"Speedup: {cpu_ms/gpu_ms:.2f}x")
```

```{note}
The solver automatically selects the best available GPU execution mode:
**WHILE graph** (CUDA ≥ 12.3) → **per-iteration graph** → **no-graph fallback**.
No configuration needed.
```

```{code-cell} ipython3
# Speedup across problem sizes
print(f"{'ndof':>8}  {'CPU (ms)':>10}  {'GPU (ms)':>10}  {'speedup':>8}")
print("-" * 42)

for maxh in [0.15, 0.10, 0.08, 0.06, 0.04]:
    m  = Mesh(unit_cube.GenerateMesh(maxh=maxh))
    fs = H1(m, order=2, dirichlet=".*")
    uu, vv = fs.TnT()
    aa = BilinearForm(grad(uu)*grad(vv)*dx).Assemble()
    ff = LinearForm(x*y*z*vv*dx).Assemble()
    pr = Projector(fs.FreeDofs(), True)

    cs = la.CGSolver(mat=aa.mat, pre=pr, precision=1e-8, maxsteps=1000, printrates=False)
    t0 = time.perf_counter()
    for _ in range(3): _ = cs * ff.vec
    c_ms = (time.perf_counter()-t0)/3*1000

    ad = aa.mat.CreateDeviceMatrix()
    pd = pr.CreateDeviceMatrix()
    fd = ff.vec.CreateDeviceVector(copy=True)
    gf = GridFunction(fs)
    gs = ngscuda.DevCGSolver(mat=ad, pre=pd, adev_raw=ad, cdev_raw=pd,
                              precision=1e-8, maxsteps=1000, printrates=False)
    g_ms = timed(gs, fd, gf.vec, runs=3)

    print(f"{fs.ndof:>8}  {c_ms:>10.1f}  {g_ms:>10.1f}  {c_ms/g_ms:>7.2f}x")
```

---

## Part 2: Non-symmetric problem — convection-diffusion

$$-\Delta u + \mathbf{b} \cdot \nabla u = f$$

The transport term makes the stiffness matrix **non-symmetric**.
CG does not apply — we use **TFQMR** (Transpose-Free QMR), which handles non-symmetric
systems without storing a growing Krylov basis. This is the relevant solver for
**Navier–Stokes** linearizations.

```{code-cell} ipython3
mesh2 = Mesh(unit_square.GenerateMesh(maxh=0.01))
fes2  = H1(mesh2, order=1, dirichlet=".*")
u2, v2 = fes2.TnT()
b = CoefficientFunction((1, 0))

a2 = BilinearForm(grad(u2)*grad(v2)*dx + b*grad(u2)*v2*dx).Assemble()
f2 = LinearForm(v2*dx).Assemble()
gfu2 = GridFunction(fes2)

print(f"ndof = {fes2.ndof}")
```

```{code-cell} ipython3
# CPU TFQMR reference
from ngsolve.krylovspace import TFQMR

pre2  = a2.mat.CreateSmoother(fes2.FreeDofs())
pre2p = Projector(fes2.FreeDofs(), True)

t0 = time.perf_counter()
for _ in range(5):
    gfu2.vec.data = TFQMR(mat=pre2@a2.mat, pre=pre2p,
                           rhs=(pre2*f2.vec).Evaluate(),
                           maxsteps=800, printrates=False, tol=1e-8)
cpu2_ms = (time.perf_counter()-t0)/5*1000
ref_norm2 = Norm(gfu2.vec)
print(f"CPU TFQMR: {cpu2_ms:.1f} ms   |sol| = {ref_norm2:.6e}")
```

```{code-cell} ipython3
# GPU DevTFQMRSolver
adev2  = a2.mat.CreateDeviceMatrix()
pdev2  = pre2.CreateDeviceMatrix()
pdev2p = pre2p.CreateDeviceMatrix()
fdev2  = f2.vec.CreateDeviceVector(copy=True)

pa_dev2  = pdev2 @ adev2                  # preconditioned system on device
rhs_pre2 = (pdev2 * fdev2).Evaluate()     # preconditioned rhs on device

tfqmr_solver = ngscuda.DevTFQMRSolver(
    mat=pre2@a2.mat, pre=pre2p,           # CPU forms for solver setup
    adev_raw=pa_dev2, cdev_raw=pdev2p,    # GPU operators for execution
    precision=1e-8, maxsteps=800, printrates=False
)

gpu2_ms = timed(tfqmr_solver, rhs_pre2, gfu2.vec)
err2 = abs(Norm(gfu2.vec) - ref_norm2)

print(f"GPU TFQMR: {gpu2_ms:.1f} ms   |sol| = {Norm(gfu2.vec):.6e}   err vs CPU = {err2:.1e}")
print(f"Speedup over CPU: {cpu2_ms/gpu2_ms:.1f}x")
```

---

## Summary

| Solver | Problem type | Typical use case |
|---|---|---|
| `DevCGSolver` | Symmetric SPD | Poisson, elasticity |
| `DevTFQMRSolver` | Non-symmetric | Convection-diffusion, Navier–Stokes |

Both solvers require no CUDA knowledge, produce numerically identical results to
their CPU counterparts, and automatically select the best GPU execution strategy.

**Minimal changes to an existing script:**

```python
import ngsolve.ngscuda as ngscuda

adev = a.mat.CreateDeviceMatrix()   # ← add
pdev = pre.CreateDeviceMatrix()     # ← add

solver = ngscuda.DevCGSolver(mat=adev, pre=pdev, adev_raw=adev, cdev_raw=pdev,
                              precision=1e-8, maxsteps=1000)
solver.Mult(f.vec.CreateDeviceVector(copy=True), gfu.vec)
```
