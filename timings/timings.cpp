// ngs_nvcc -x cu -c  timings.cpp ; ngs_nvcc -dlink timings.o -o timings_device.o -L/opt/cuda/lib64 -L/home/jschoeberl/local_install/lib -lngcore ;  ngsld timings.o timings_device.o  -L/opt/cuda/lib64 -lcudart -lcudadevrt -lngcore

#include <fstream>
#include <ngstd.hpp>
#include <cuda_ngbla.hpp>

using namespace ngs_cuda;
using namespace ngbla;

int main()
{
  ofstream copyvec("timing_copyvec.txt");
  for (int n = 1024; n <= 256*1024*1024; n*=2)
    {
      Vector<double> x(n), y(n);
      x = 1; y = 2;

      Vector<Dev<double>> devx(x), devy(y);

      double est_time = 5e-6 + n*1e-10;
      size_t runs = 0.1 / est_time;

      Timer t("vec"+ToString(n));

      cudaDeviceSynchronize();
      t.Start();
      for (size_t k = 0; k < runs; k++)
        devy.Range(0,n) = 2.7*devx;
      cudaDeviceSynchronize();
      t.Stop();
      t.AddFlops (double(runs)*n);
      cout << "n = " << n << ", GFlops = " << t.GetMFlops()*1e-3 << endl;
      copyvec << n << " " << t.GetMFlops()*1e-3  << endl;
    }
}
