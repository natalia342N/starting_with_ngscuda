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

# GPU Accelerated Krylov Solvers in NGSolve

**NGSolve User Meeting 2026 — Natalia Tylek, TU Wien**

This notebook shows how to use the GPU Krylov solvers in NGSolve's `ngscuda` module:
- **`DevCGSolver`** — for symmetric positive-definite problems (Poisson, elasticity)
- **`DevTFQMRSolver`** — for non-symmetric problems (convection-diffusion, Navier–Stokes) (TO BE ADDED AFTER MR IS APPROVED)

```{admonition} GPU required
:class: important
To run on GPU in Colab: **Runtime → Change runtime type → T4 GPU**
```

```{code-cell} ipython3
:tags: [hide-cell]
import sys
if 'google.colab' in sys.modules:
    !pip install --upgrade --pre ngsolve anywidget -q
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/natalia342N/starting_with_ngscuda/blob/main/docs/ngsolve_meeting_tutorial.ipynb)

```{code-cell} ipython3
# Check GPU availability
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
import ngsolve.ngscuda as ngscuda
from time import perf_counter

print("NGSolve version:", ngsolve.__version__)
```

---

## Part 1: Symmetric problem — Poisson equation

$$-\Delta u + u = f \quad \text{in } \Omega, \qquad u = 0 \quad \text{on } \partial\Omega$$

The stiffness matrix is **symmetric positive definite** → Conjugate Gradient.

This extends the setup from
[5.5.1 Solving the Poisson Equation on devices](https://docu.ngsolve.org/latest/i-tutorials/unit-5.5-cuda/poisson_cuda.html)
with the new **WHILE graph** execution mode.

```{code-cell} ipython3
# Same setup as the official 5.5.1 tutorial
mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
for l in range(5): mesh.Refine()

fes = H1(mesh, order=2, dirichlet=".*")
u, v = fes.TnT()

with TaskManager():
    a = BilinearForm(grad(u)*grad(v)*dx + u*v*dx).Assemble()
    f = LinearForm(x*v*dx).Assemble()

gfu = GridFunction(fes)
jac = a.mat.CreateSmoother(fes.FreeDofs())

print(f"ndof = {fes.ndof}")
```

```{code-cell} ipython3
# CPU solve (reference)
with TaskManager():
    cpu_solver = la.CGSolver(mat=a.mat, pre=jac, maxsteps=2000, printrates=False)
    gfu.vec.data = cpu_solver * f.vec   # warm-up
    ts = perf_counter()
    for _ in range(5): gfu.vec.data = cpu_solver * f.vec
    cpu_ms = (perf_counter() - ts) / 5 * 1000

ref_norm = Norm(gfu.vec)
print(f"CPU CG:  {cpu_ms:.0f} ms   |sol| = {ref_norm:.6e}   steps = {cpu_solver.GetSteps()}")
```

### GPU acceleration

```{code-cell} ipython3
adev   = a.mat.CreateDeviceMatrix()       # move matrix to GPU
jacdev = jac.CreateDeviceMatrix()         # move preconditioner to GPU
fdev   = f.vec.CreateDeviceVector(copy=True)

gpu_solver = ngscuda.DevCGSolver(
    mat=adev, pre=jacdev,
    adev_raw=adev, cdev_raw=jacdev,
    maxsteps=2000, printrates=False
)

gfu.vec.data = (gpu_solver * fdev).Evaluate()   # warm-up
ts = perf_counter()
for _ in range(5): gfu.vec.data = (gpu_solver * fdev).Evaluate()
gpu_ms = (perf_counter() - ts) / 5 * 1000

err = abs(Norm(gfu.vec) - ref_norm)
print(f"GPU CG:  {gpu_ms:.0f} ms   |sol| = {Norm(gfu.vec):.6e}   err = {err:.1e}")
print(f"Speedup: {cpu_ms/gpu_ms:.2f}x")
```

```{note}
`DevCGSolver` automatically selects the best available GPU execution mode:

- **WHILE graph** (CUDA ≥ 12.3) — entire convergence loop on device, zero per-iteration host–device sync
- **per-iteration graph** — fallback for older CUDA
- **no-graph** — set `NO_CUDA_GRAPH=1` to disable graph capture

No configuration required from the user.
```

```{code-cell} ipython3
# Scaling with problem size
print(f"{'ndof':>8}  {'CPU (ms)':>10}  {'GPU (ms)':>10}  {'speedup':>8}")
print("-" * 42)

for nref in [3, 4, 5, 6]:
    m = Mesh(unit_square.GenerateMesh(maxh=0.1))
    for _ in range(nref): m.Refine()
    fs = H1(m, order=2, dirichlet=".*")
    uu, vv = fs.TnT()
    with TaskManager():
        aa = BilinearForm(grad(uu)*grad(vv)*dx + uu*vv*dx).Assemble()
        ff = LinearForm(x*vv*dx).Assemble()
    jj = aa.mat.CreateSmoother(fs.FreeDofs())
    gf = GridFunction(fs)

    cs = la.CGSolver(mat=aa.mat, pre=jj, maxsteps=2000, printrates=False)
    gf.vec.data = cs * ff.vec
    ts = perf_counter()
    for _ in range(3): gf.vec.data = cs * ff.vec
    c_ms = (perf_counter() - ts) / 3 * 1000

    ad = aa.mat.CreateDeviceMatrix()
    jd = jj.CreateDeviceMatrix()
    fd = ff.vec.CreateDeviceVector(copy=True)
    gs = ngscuda.DevCGSolver(mat=ad, pre=jd, adev_raw=ad, cdev_raw=jd,
                              maxsteps=2000, printrates=False)
    gf.vec.data = (gs * fd).Evaluate()
    ts = perf_counter()
    for _ in range(3): gf.vec.data = (gs * fd).Evaluate()
    g_ms = (perf_counter() - ts) / 3 * 1000

    print(f"{fs.ndof:>8}  {c_ms:>10.0f}  {g_ms:>10.0f}  {c_ms/g_ms:>7.2f}x")
```

---

## Part 2: Non-symmetric problem — convection-diffusion

$$-\varepsilon\,\Delta u + \mathbf{b} \cdot \nabla u = f$$

The convection term makes the stiffness matrix **non-symmetric** — CG does not apply.
**TFQMR** (Transpose-Free QMR) handles non-symmetric systems without storing a growing
Krylov basis, making it suitable for large problems. This is the relevant solver for
**Navier–Stokes** linearizations.

```{code-cell} ipython3
mesh2 = Mesh(unit_square.GenerateMesh(maxh=0.02))
fes2  = H1(mesh2, order=2, dirichlet=".*")
u2, v2 = fes2.TnT()

eps = 1e-3
b   = CF((1, 0))
with TaskManager():
    a2 = BilinearForm(eps*grad(u2)*grad(v2)*dx + b*grad(u2)*v2*dx).Assemble()
    f2 = LinearForm(1*v2*dx).Assemble()

gfu2 = GridFunction(fes2)
jac2 = a2.mat.CreateSmoother(fes2.FreeDofs())

print(f"ndof = {fes2.ndof}")
```

```{code-cell} ipython3
# CPU TFQMR reference
from ngsolve.krylovspace import TFQMR

with TaskManager():
    gfu2.vec.data = TFQMR(mat=a2.mat, pre=jac2, rhs=f2.vec,
                           maxsteps=2000, printrates=False, tol=1e-8)
    ts = perf_counter()
    for _ in range(3):
        gfu2.vec.data = TFQMR(mat=a2.mat, pre=jac2, rhs=f2.vec,
                               maxsteps=2000, printrates=False, tol=1e-8)
    cpu2_ms = (perf_counter() - ts) / 3 * 1000

ref_norm2 = Norm(gfu2.vec)
print(f"CPU TFQMR: {cpu2_ms:.0f} ms   |sol| = {ref_norm2:.6e}")
```

```{code-cell} ipython3
# GPU DevTFQMRSolver
adev2  = a2.mat.CreateDeviceMatrix()
jdev2  = jac2.CreateDeviceMatrix()
fdev2  = f2.vec.CreateDeviceVector(copy=True)

tfqmr_solver = ngscuda.DevTFQMRSolver(
    mat=adev2, pre=jdev2,
    adev_raw=adev2, cdev_raw=jdev2,
    maxsteps=2000, printrates=False, precision=1e-8
)

tfqmr_solver.Mult(fdev2, gfu2.vec)   # warm-up
ts = perf_counter()
for _ in range(3): tfqmr_solver.Mult(fdev2, gfu2.vec)
gpu2_ms = (perf_counter() - ts) / 3 * 1000

err2 = abs(Norm(gfu2.vec) - ref_norm2)
print(f"GPU TFQMR: {gpu2_ms:.0f} ms   |sol| = {Norm(gfu2.vec):.6e}   err = {err2:.1e}")
print(f"Speedup:   {cpu2_ms/gpu2_ms:.1f}x")
```

---

## Summary

| Solver | Problem type | Typical use case |
|---|---|---|
| `DevCGSolver` | Symmetric SPD | Poisson, elasticity |
| `DevTFQMRSolver` | Non-symmetric | Convection-diffusion, Navier–Stokes |

Both solvers require no CUDA knowledge, produce numerically identical results to CPU
counterparts, and automatically select the best GPU execution strategy.

**Minimal changes to an existing NGSolve script:**

```python
import ngsolve.ngscuda as ngscuda

adev = a.mat.CreateDeviceMatrix()
jdev = jac.CreateDeviceMatrix()    

solver = ngscuda.DevCGSolver(mat=adev, pre=jdev, adev_raw=adev, cdev_raw=jdev,
                              maxsteps=2000)
gfu.vec.data = (solver * f.vec.CreateDeviceVector(copy=True)).Evaluate()
```
