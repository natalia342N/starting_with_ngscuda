import numpy as np
import matplotlib.pyplot as plt

data_copy = np.loadtxt('timing_copyvec.txt', usecols=(0, 1))
data_daxpy = np.loadtxt('timing_daxpy.txt', usecols=(0, 1))






plt.figure(figsize=(8,6))

sizes, GFlops = zip(*data_copy)
# plt.plot(sizes, GFlops, marker='o', linestyle='-', color='b', label='copy')
plt.plot(sizes, GFlops, "-x", label='copy')

sizes, GFlops = zip(*data_daxpy)
# plt.plot(sizes, GFlops, marker='o', linestyle='-', color='b', label='daxpy')
plt.plot(sizes, GFlops, "-x", label='daxpy')



plt.xscale('log') # , base=2)
plt.yscale('log')

plt.xlabel('Vector length (N)')
plt.ylabel('GFlops')
plt.title('Runtime performance vector copy')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('timing_copyvec.png')
# plt.show()







