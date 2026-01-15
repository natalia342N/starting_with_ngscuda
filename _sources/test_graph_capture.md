# Graph Capture with NGSCuda on the device

In the same folder where NGSolve with NGSCuda is installed, create a script `test_graph_capture.py` with the following content:

```python
import sys, os
from time import time

print("Python:", sys.version)
print("Executable:", sys.executable)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

import ngsolve
import ngsolve.ngscuda as ngscuda
from ngsolve import BaseVector

print("ngsolve:", ngsolve.__file__)
print("ngscuda:", ngscuda.__file__)
print("Has CudaGraph:", hasattr(ngscuda, "CudaGraph"))
print("Imports OK")

# Create a vector on host
n = 1_000_000
v = BaseVector(n)
v[:] = 1.0

# Move to device
dv = v.CreateDeviceVector(copy=True)

# Warm-up execution
dv *= 1.000001

# Sync before capture/timing if available
if hasattr(ngscuda, "DeviceSynchronize"):
    ngscuda.DeviceSynchronize()
else:
    _ = float(dv[0])

# Capture + instantiate graph
g = ngscuda.CudaGraph()
g.BeginCapture()
dv *= 1.000001      # should be captured
g.EndCapture()

# Timing
runs = 1000

# Sync before timing
if hasattr(ngscuda, "DeviceSynchronize"):
    ngscuda.DeviceSynchronize()
else:
    _ = float(dv[0])

ts = time()
for _ in range(runs):
    g.Launch()

# Sync after launches 
if hasattr(ngscuda, "DeviceSynchronize"):
    ngscuda.DeviceSynchronize()
else:
    _ = float(dv[0])
te = time()

print("Graph replay time per launch (s):", (te - ts) / runs)

# Correctness sanity check 
print("dv[0] (after replays):", float(dv[0]))

