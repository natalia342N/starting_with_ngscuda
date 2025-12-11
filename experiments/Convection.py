# ssh cerbsim 'source ~/.bashrc; python -' < Convection.py
# scp cerbsim:gpu.trace .


print ("Convection GPU demo")

from time import time
from ngsolve import *
mesh = Mesh(unit_cube.GenerateMesh(maxh=0.1))
mesh.Refine()
# mesh.Refine()

fes_wind = HDiv(mesh, order=2)
fes_u = VectorL2(mesh, order=2, piola=True)
fes_facet = VectorFacetFESpace(mesh, order=2, dirichlet="inflow")
fes = fes_wind * fes_u * fes_facet

print ("fes_u.ndof: ", fes_u.ndof)

w,u,uhat = fes.TrialFunction()
v = fes_u.TestFunction()

embw, embu, embuhat = fes.embeddings
restw, restu, restuhat = fes.restrictions


Omega = CF( (0,0,1) )

conv = BilinearForm(trialspace=fes, testspace=fes_u, nonlinear_matrix_free_bdb=True)
# conv += Cross(Omega, u)*v*dx
conv += Grad(v)*w*u * dx
# wn = w.Operator("normalcomponent")
# conv += -wn*(IfPos(wn, u, 2*uhat-u)*v)*dx(element_boundary=True)

conv.Assemble()
convop = conv.mat

# print (convop.GetOperatorInfo())

diag = BaseVector(fes_facet.ndof)
diag[:]=1  
diag[~fes_facet.FreeDofs()]=0.5
halfinflowbnd = DiagonalMatrix(diag)

traceop = ConvertOperator(fes_u, fes_facet)
convop = convop @ (IdentityMatrix() + embuhat@halfinflowbnd@traceop@restu)
# input vector:  wind, volume-u, 1/2 boundary-values



gfu = GridFunction(fes)
gfu.components[0].Set(CF((1,0,0)))
gfu.components[1].Set(CF((1,0,0)))

gfv = GridFunction(fes_u)
gfv.Set(CF((x,1,0)))

runs = 10


hv = BaseVector(fes_u.ndof)
SetNumThreads(4)
with TaskManager():
    ts = time()
    for k in range(runs):
        hv.data = convop * gfu.vec
    te = time()
print ("time = ", (te-ts)/runs)
print (InnerProduct(hv, gfv.vec))



    
try:
    import ngsolve.ngscuda
    ngsolve.ngscuda.SetSyncKernels(False)
except:
    print ("no cuda")
    

devuvec = gfu.vec.CreateDeviceVector(copy=True)
devvvec = gfv.vec.CreateDeviceVector(copy=True)

devconv = convop.CreateDeviceMatrix()
devhv = hv.CreateDeviceVector(copy=False)

runs = 10
ts = time()
for k in range(runs):
    devhv.data = devconv * devuvec
te = time()
print ("time = ", (te-ts)/runs)
print (InnerProduct(devhv, devvvec))


try:
    ngsolve.ngscuda.SetCudaTimer(True)
except:
    pass

with PajeTrace("gpu"):
    devhv.data = devconv * devuvec
    print (InnerProduct(devhv, devvvec))    
print ("writing trace file 'gpu'")


with TaskManager(pajetrace=10**8):
    hv.data = convop * gfu.vec
