import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("timing_matmat.txt")

N = data[:, 0] 
gflops_cols = data[:, 1:]  

labels = ["m=20", "m=50", "m=100", "m=200", "m=500"] 

plt.figure(figsize=(8,6))

for idx in range(gflops_cols.shape[1]):
    plt.plot(N, gflops_cols[:, idx], "-x", label=labels[idx])

plt.xscale('log')    
# plt.yscale('log')     
plt.xlabel('Matrix width n')
plt.ylabel('GFlops')
plt.title('Runtime performance mat-mat')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('timing_matmat.png')
# plt.show()

print("Saved: timing_matmat.png")
