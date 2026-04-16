# OLD_TFQMR Solver on Device

The Transpose-Free Quasi-Minimal Residual (TFQMR) method is a Krylov solver for non-symmetric linear systems. Compared to other Krylov methods available in NGSolve, TFQMR offers:

- no transpose matrix-vector product required (unlike QMR)
- good convergence on convection-dominated problems
- fastest GPU solve time among tested methods on H100

## Usage

```python
from ngsolve.solvers import TFQMR

gfu.vec.data = TFQMR(mat=predev@adev, pre=pre1dev, rhs=rhs_dev,
                     maxsteps=400, tol=1e-12)
```

## Example

```python
from ngsolve import *
from ngsolve.solvers import TFQMR

mesh = Mesh(unit_cube.GenerateMesh(maxh=0.05))
fes  = L2(mesh, order=2, dgjumps=True)
u, v = fes.TnT()
wind = CF((1, 0.2, 0.3))
n    = specialcf.normal(mesh.dim)
dS   = dx(element_boundary=True)

a = BilinearForm(fes)
a += -20*u*v*dx
a += u*wind*grad(v)*dx
a += -(wind*n)*IfPos(wind*n, u, u.Other(bnd=0))*v*dS
with TaskManager():
    a.Assemble()

blocks = fes.CreateSmoothingBlocks(blocktype="element")
pre    = a.mat.CreateBlockSmoother(blocks)
pre1   = Projector(fes.FreeDofs(), True)
f      = LinearForm(1*v*dx).Assemble()
gfu    = GridFunction(fes)

fdev    = f.vec.CreateDeviceVector(copy=True)
adev    = a.mat.CreateDeviceMatrix()
predev  = pre.CreateDeviceMatrix()
pre1dev = pre1.CreateDeviceMatrix()
rhs_dev = (predev * fdev).Evaluate()

gfu.vec.data = TFQMR(mat=predev@adev, pre=pre1dev, rhs=rhs_dev,
                     maxsteps=400, tol=1e-12)
print(f"|sol| = {Norm(gfu.vec):.8e}")
```

## Performance on H100 (Musica, ndof = 340370)

| Solver | Platform | Time (s) |
|---|---|---|
| NGSolve TFQMR | CPU | 5.2 |
| NGSolve GMRES | GPU | 0.145 |
| NGSolve TFQMR | GPU | **0.032** |
