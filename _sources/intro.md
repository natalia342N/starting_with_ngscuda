# Reducing GPU kernel launch overhead in NGSolve using CUDA Graphs

## NGSCuda on Supercomputer

#### Idea

NGSolve already supports GPU execution via ngscuda, but for matrix-free operators (like the convection operator) performance can be limited by kernel launch and synchronization overhead, especially when the same operator is applied repeatedly.



CUDA Graphs offer a way to:
- capture a fixed sequence of GPU operations once,
- replay it efficiently many times with lower CPU overhead.
