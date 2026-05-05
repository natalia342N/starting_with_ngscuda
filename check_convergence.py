from ngsolve import *
from netgen.occ import unit_square
import ngsolve.ngscuda as ngscuda
import math

mesh = Mesh(unit_square.GenerateMesh(maxh=0.05))
fes  = H1(mesh, order=2, dirichlet=".*")
u, v = fes.TnT()

a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
m = BilinearForm(u*v*dx).Assemble()
pre = a.mat.CreateSmoother(fes.FreeDofs())

adev   = a.mat.CreateDeviceMatrix()
mdev   = m.mat.CreateDeviceMatrix()
predev = pre.CreateDeviceMatrix()

NUM = 4
expected = sorted(math.pi**2 * (i**2 + j**2) for i in range(1,5) for j in range(1,5))[:NUM]
print("Expected:", [f"{l:.5f}" for l in expected])

for maxit in [30, 100, 200, 500]:
    solver = ngscuda.DevLOBPCGSolver(a=adev, m=mdev, pre=predev, num=NUM, maxit=maxit)
    lams, _ = solver.Solve()
    errs = [abs(lams[i]-expected[i])/expected[i] for i in range(NUM)]
    print(f"maxit={maxit:4d}: lams={[f'{l:.5f}' for l in lams]}  rel_err={[f'{e:.2e}' for e in errs]}")
