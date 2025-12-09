import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import LogLocator, ScalarFormatter

m_values = [20, 50, 100, 200, 500]

plt.figure(figsize=(10,6))

for m in m_values:
    fname = f"timings_matmat_m{m}.txt"
    
    if not os.path.exists(fname):
        print(f"WARNING: file {fname} not found, skipping.")
        continue
    
    data = np.loadtxt(fname, comments="#")
    N = data[:,0]
    gflops = data[:,2]
    
    plt.plot(N, gflops, marker="x", label=f"m={m}")

plt.xscale("log")

plt.gca().xaxis.set_major_locator(LogLocator(base=10, numticks=10))
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.gca().xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2,10)*0.1))

plt.xlabel("matrix width n")
plt.ylabel("GFLOPs")
plt.title("Runtime performance mat-mat ")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

plt.savefig("timings_matmat_all.png", dpi=200)
print("Saved plot to timings_matmat_all.png")

plt.show()
