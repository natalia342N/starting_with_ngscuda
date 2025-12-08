import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("timings_matmat_m50.txt", comments="#")
N = data[:,0]
gflops = data[:,2]

plt.figure(figsize=(10,6))

plt.plot(N, gflops, marker="x", label="m=50")

plt.xlabel("matrix width n")
plt.ylabel("GFLOPs")
plt.title("Runtime performance mat-mat")
plt.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)

plt.legend()
plt.tight_layout()

plt.savefig("timings_matmat_m50.png", dpi=200)
print("Saved plot to timings_matmat_m50.png")

plt.show()
