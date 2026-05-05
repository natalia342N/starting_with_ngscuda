from ngsolve import *
from netgen.occ import unit_square
from ngsolve.solvers import LOBPCG
import ngsolve.ngscuda as ngscuda
import math

mesh = Mesh(unit_square.GenerateMesh(maxh=0.05))
fes  = H1(mesh, order=2, dirichlet=".*")
u, v = fes.TnT()
a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
m = BilinearForm(u*v*dx).Assemble()

NUM = 4
expected = sorted(math.pi**2 * (i**2 + j**2) for i in range(1,5) for j in range(1,5))[:NUM]
print(f"Analytical:      {[f'{l:.6f}' for l in expected]}")

pre_chol  = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky")
lams_chol, _ = LOBPCG(a.mat, m.mat, pre_chol, num=NUM, maxit=30, printrates=False)
print(f"CPU LOBPCG (Cholesky, 30 it):  {[f'{float(l):.6f}' for l in lams_chol]}")

pre_jac = a.mat.CreateSmoother(fes.FreeDofs())
lams_jac, _ = LOBPCG(a.mat, m.mat, pre_jac, num=NUM, maxit=100, printrates=False)
print(f"CPU LOBPCG (Jacobi,   100 it): {[f'{float(l):.6f}' for l in lams_jac]}")

adev   = a.mat.CreateDeviceMatrix()
mdev   = m.mat.CreateDeviceMatrix()
predev = pre_jac.CreateDeviceMatrix()
solver = ngscuda.DevLOBPCGSolver(a=adev, m=mdev, pre=predev, num=NUM, maxit=100)
lams_gpu, _ = solver.Solve()
print(f"GPU LOBPCG (Jacobi,   100 it): {[f'{float(l):.6f}' for l in lams_gpu]}")
